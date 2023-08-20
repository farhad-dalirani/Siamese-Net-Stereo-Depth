import torch
from model import SiameseNetStereoMatching
from stereo_batch_provider import KITTIDisparityDataset
import matplotlib.pyplot as plt
from disparity_map import disparity_map_nn
import cv2

if __name__ == '__main__':
    
    # KITTI test disparity dataset
    dataset_test = KITTIDisparityDataset(image_dir='./KITTI_disparity_subset/data_scene_flow/testing')

    for sample_i in range(0, 5):
        left_image, right_image = dataset_test[sample_i]

        # Convert the float images to 8-bit format (0-255)
        left_image = cv2.normalize(left_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        right_image = cv2.normalize(right_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


        # Create a StereoBM object
        stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=9)
        # Compute the disparity map
        disparity_map_9 = stereo_bm.compute(left_image, right_image)

        # Create a StereoBM object
        stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=5)
        # Compute the disparity map
        disparity_map_5 = stereo_bm.compute(left_image, right_image)

        plt.figure(figsize=(10, 5))
        # Create the first subplot for img_l
        plt.subplot(3, 1, 1)
        plt.imshow(left_image.squeeze(), cmap='gray')
        plt.title('Left Image')
        plt.axis('off')  
        # Create the second subplot for disparity_map
        plt.subplot(3, 1, 2)
        plt.imshow(disparity_map_5)
        plt.title('Disparity Map (OpenCV StereoBM_create), block size 5')
        plt.axis('off')  
        plt.subplot(3, 1, 3)
        plt.imshow(disparity_map_9)
        plt.title('Disparity Map (OpenCV StereoBM_create), block size 9')
        plt.axis('off')  
        plt.tight_layout()
    
    plt.show()