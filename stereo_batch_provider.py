from os import listdir, path
from threading import Thread, Lock
import time

import numpy as np
import torch
from imageio import imread, imwrite
from skimage.transform import rescale
from matplotlib import pyplot as plt


def rgb2gray(I):
    """
    Convert rgb image to grayscale
    """
    return np.dot(I[...,:3], [0.299, 0.587, 0.114])


class KITTIDisparityDataset(object):
    
    def __init__(self, image_dir, disparity_dir=None, downsample=True):
        # folder containing disparity images
        self.disparity_dir = disparity_dir
        # whether downsample by factor of 2 or not
        self.downsample = downsample

        # paths for images of camera 2 and 3 in KITTI dataset 
        left_dir = path.join(image_dir, "image_2")
        right_dir = path.join(image_dir, "image_3")

        # name of images for camera 2 and 3 and check both have same number of images
        self._left_images = sorted([
            path.join(left_dir, img) for img in listdir(left_dir) if "_10." in img
        ])
        self._right_images = sorted([
            path.join(right_dir, img) for img in listdir(right_dir) if "_10." in img
        ])
        assert len(self._left_images) == len(self._right_images)

        # if disparities is provided
        if disparity_dir is not None:
            # name of disparity images
            self._disp_images = sorted([
                path.join(disparity_dir, img) for img in listdir(disparity_dir)
            ])
            assert len(self._left_images) == len(self._disp_images)
        else:
            self._disp_images = []

        print('KITTI data loaded (%d images)!' % len(self._left_images))

    def __len__(self):
        return len(self._left_images)

    def __getitem__(self, i):
        # for better neural network training, convert pixel values to a float number [0,1].
        img_l = imread(self._left_images[i]).astype(np.float32) / 255.
        img_r = imread(self._right_images[i]).astype(np.float32) / 255.
        
        # convert images from color to gray scale with shape of (Height, Width, 1).
        img_l = rgb2gray(img_l)[..., np.newaxis]
        img_r = rgb2gray(img_r)[..., np.newaxis]

        if self.downsample:
            img_l = rescale(img_l, 0.5, mode='reflect', anti_aliasing=True, multichannel=True)
            img_r = rescale(img_r, 0.5, mode='reflect', anti_aliasing=True, multichannel=True)

        if self.disparity_dir is not None:
            # for better neural network training, convert pixel values to a float number [0,1].
            disp = imread(self._disp_images[i]).astype(np.float32) / 256.
        
            if self.downsample:
                # when images are subsampled by factor of 2, disparity for a pixel will be halfed.
                H, W = disp.shape
                disp = disp[np.arange(0, H, 2), :]
                disp = disp[:, np.arange(0, W, 2)]
                disp = disp / 2. 

            # for pixels in sparity map that disparity value does not exist, assign value -1.
            disp[disp <= 0] = -1
    
            return img_l, img_r, disp
        else:
            return img_l, img_r


class PatchProvider(object):
    """
        Provide training patches.
    """
    def __init__(self, data, patch_size=(7, 7), N=(4, 10), P=1):
        """_summary_

        Args:
            data: KITTIDisparityDataset object for reading training left and right images and their disparity.
            patch_size: size of neighbourhood around each pixel.
            N: in a pair in form of (a, b). If the positve patch is located in pixel location (row_i, col_j), the 
               negative patch will be from one of these pixels: (row_i, col_j-b), (row_i, col_j-b-1),..., (row_i, col_j-a) or (row_i, col_j+a), (row_i, col_j+a+1),..., (row_i, col_j+b) 
            P: If the ground truth positve patch is located in pixel location (row_i, col_j), the 
               positive patch will be one of these pixels: (row_i, col_j-p), ... ,(row_i, col_j), (row_i, col_j+p) 
        """
        self._data = data
        self._patch_size = patch_size
        self._N = N
        self._P = P
        self.idxs = None

        self._stop = False
        self._cache = 5
        self._lock = Lock()


    def _get_neg_idx(self, col, W):

        # local copy for convenience
        half_patch = self._patch_size[1]//2
        N = self._N

        # determine a negative matched pixel aroud ground truth positive match according input argument N.
        neg_offset = np.random.randint(N[0], N[1] + 1)
        neg_offset = neg_offset * np.sign(np.random.rand() - 0.5).astype(np.int32)

        # slice for a patch around the selected pixel
        if half_patch <= col + neg_offset < W-half_patch:
            return slice(col+neg_offset-half_patch, col+neg_offset+half_patch+1)
        else:
            # try again if the patch goes of the image
            return self._get_neg_idx(col, W)

    def _get_pos_idx(self, col, W):
        
        # local copy for convenience
        half_patch = self._patch_size[1]//2
        P = self._P

        # get a random pixel in neighbourhood of ground truth positive match according input argument P
        pos_offset = np.random.randint(-P, P+1)
        
        # slice for a patch around the selected pixel
        if half_patch <= col + pos_offset < W-half_patch:
            return slice(col+pos_offset-half_patch, col+pos_offset+half_patch+1)
        else:
            # try again if the patch goes of the image
            return self._get_pos_idx(col, W)
        
    def random_patch(self):

        # local copy for convenience
        patch_size = self._patch_size
        half_patch = np.array(patch_size)//2

        # read a random image from KITTI disparity dataset
        img_l, img_r, disp = self._data[int(np.random.rand()*len(self._data))]
        H, W = img_l.shape[:2]
        
        # select a pixel in left image (reference image), if the patch around it goes off the image,
        # or ground truth disparity for it does not exist, select another point
        while True:
            half_p = patch_size[0] // 2
            row = np.random.randint(half_p, H - half_p)
            col = np.random.randint(half_p, W - half_p)
            d = disp[row, col]
            if d > 0 and (col - d) > half_p and (col - d) < W - half_p:
                break
        
        # for the selected pixel, return its patch in reference image, and find a positive
        # and negative patch in the right image.
        ref_idx = (
            slice(row-half_patch[0], row+half_patch[0]+1),
            slice(col-half_patch[1], col+half_patch[1]+1)
        )
        neg_idx = (
            slice(row-half_patch[0], row+half_patch[0]+1),
            self._get_neg_idx(int(col - disp[row, col]), W)
        )
        pos_idx = (
            slice(row-half_patch[0], row+half_patch[0]+1),
            self._get_pos_idx(int(col - disp[row, col]), W)
        )

        # return patches: ref, positive, negative
        return img_l[ref_idx], img_r[pos_idx], img_r[neg_idx]

    def iterate_batches(self, batch_size):
        
        # Get a patch to infer the image shape
        patch = self.random_patch()
        channels = patch[0].shape[-1]

        # create three np.array with shape of (cache size * batch size, patch size, patch size, number of channel)
        # for reference, positive and negative patches
        ref_batch = np.zeros(
            (self._cache*batch_size, ) + self._patch_size + (channels,),
            dtype="float32"
        )
        pos_batch = np.zeros_like(ref_batch)
        neg_batch = np.zeros_like(ref_batch)

        # start the thread for constantly filling patches with fresh data
        self._thread = Thread(
            target=self.fill_batches,
            args=(ref_batch, pos_batch, neg_batch)
        )
        self._stop = False
        self._thread.start()

        # wait for the buffers to fill
        while True:
            time.sleep(1)
            with self._lock:
                if ref_batch[-1].sum() == 0:
                    pass
                else:
                    break

        # start generating batches
        while True:
            # from (batch * cache size), select (batch size) of them
            self.idxs = np.random.choice(len(ref_batch), batch_size)
            with self._lock:
                yield torch.Tensor(ref_batch[self.idxs]), torch.Tensor(pos_batch[self.idxs]), torch.Tensor(neg_batch[self.idxs])

    def fill_batches(self, ref, pos, neg):
        idx = 0
        while not self._stop:
            patch = self.random_patch()
            with self._lock:
                ref[idx] = patch[0]
                pos[idx] = patch[1]
                neg[idx] = patch[2]
            idx += 1
            idx = idx % len(ref)

    def stop(self):
        self._stop = True
        self._thread.join()

def upsample_disparity_map(disparity_map, output_shape, sampling_factor=2.):
    ''' Upsamples the disparity map to the provided output shape.

    Please note that when upsampling the disparity, the value need to be adjusted
    with regard to the downsampling factor of the images for which the disparity
    was calculated. For example, when we half the resolution before calculating
    the disparity, we need to multiply the disparity by 2 if we want to obtain
    the disparity map for the original size. 
    
    Arguments:
    ----------
        disparity_map: disparity map 
        output_shape: desired output shape
        sampling_factor: sampling factor by which the upsampled disparity map is
            multiplied (default: 2.)
    '''
    disparity_map = imresize(disparity_map, output_shape, interp='nearest')
    disparity_map = disparity_map * sampling_factor
    return disparity_map


def return_accuracy(pred_disparity, gt_disparity, threshold=1.5):
    """ Returns the accuracy for the predicted and GT disparity maps.

    Arguments:
    ----------
        pred_disparity: predicted disparity map 
        gt_disparity: ground truth disparity map 
        threshold: threshold value defining which maximum difference should be considered correct (default 3)
        half_resolution: whether the disparity was calculated on half resolution images. If so, the predicted 
            disparity map needs to be upsampled and multiplied by 2 before comparing against the ground truth
    """

    diff = np.abs(pred_disparity - gt_disparity)
    mask = gt_disparity > 0.

    diff[mask == 0] = 0.

    correct = (np.abs(gt_disparity[mask] - pred_disparity[mask]) < threshold).sum()
    total = mask.sum()    
    acc = correct / total

    return diff, acc
