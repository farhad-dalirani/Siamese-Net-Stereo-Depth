import torch
from model import SiameseNetStereoMatching
from stereo_batch_provider import KITTIDisparityDataset
import matplotlib.pyplot as plt
from disparity_map import disparity_map_nn
import cv2


if __name__ == '__main__':
    
    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    # load Siamese Neural Network from the file
    model_weights = torch.load('./saved-model-2023-08-19-01-39-50/model.pth')
    model = SiameseNetStereoMatching(in_channel=1)
    model.load_state_dict(model_weights)

    # the designed Siamese neural netwrok get an image of size H*W and return an image of size (H-8)*(W-8)
    padding_size = 4

    # KITTI test disparity dataset
    dataset_test = KITTIDisparityDataset(image_dir='./KITTI_disparity_subset/data_scene_flow/testing')

    for sample_i in range(0, 5):
        img_l, img_r = dataset_test[sample_i]

        # obtain disparity map by Siamses matching
        disparity_map_siamese = disparity_map_nn(model=model, img_l=img_l, img_r=img_r, padding_size=padding_size, max_disparity=64, device=device)
        disparity_map_siamese = disparity_map_siamese.cpu().numpy()

        # Convert the float images to 8-bit format (0-255)
        left_image = cv2.normalize(img_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        right_image = cv2.normalize(img_r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # obtain disparity map by OpencvStereo matching
        # Create a StereoBM object
        stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=5)
        # Compute the disparity map
        disparity_map_5_opencv = stereo_bm.compute(left_image, right_image)
        # Create a StereoBM object
        stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=9)
        # Compute the disparity map
        disparity_map_9_opencv = stereo_bm.compute(left_image, right_image)

        plt.figure(figsize=(12, 5))
        # Create the first subplot for img_l
        plt.subplot(2, 2, 1)
        plt.imshow(img_l.squeeze(), cmap='gray')
        plt.title('Left Image')
        plt.axis('off')  
        # Create the second subplot for disparity_map
        plt.subplot(2, 2, 2)
        plt.imshow(disparity_map_siamese)
        plt.title('Disparity Map (Siamese CNN)')
        plt.axis('off')  
        plt.subplot(2, 2, 3)
        plt.imshow(disparity_map_5_opencv)
        plt.title('Disparity Map (OpenCV StereoBM_create), block size 5')
        plt.axis('off')  
        plt.subplot(2, 2, 4)
        plt.imshow(disparity_map_9_opencv)
        plt.title('Disparity Map (OpenCV StereoBM_create), block size 9')
        plt.axis('off')  
        plt.tight_layout()
        plt.savefig('./readme-image/Disparity-Siamse-opencv-{}.png'.format(sample_i))
    plt.show()