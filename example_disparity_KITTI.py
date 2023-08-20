import torch
from model import SiameseNetStereoMatching
from stereo_batch_provider import KITTIDisparityDataset
import matplotlib.pyplot as plt
from disparity_map import disparity_map_nn


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

        # obtain disparity map
        disparity_map = disparity_map_nn(model=model, img_l=img_l, img_r=img_r, padding_size=padding_size, max_disparity=50, device=device)
        disparity_map = disparity_map.cpu().numpy()

        plt.figure(figsize=(10, 5))
        # Create the first subplot for img_l
        plt.subplot(2, 1, 1)
        plt.imshow(img_l.squeeze(), cmap='gray')
        plt.title('Left Image')
        plt.axis('off')  
        # Create the second subplot for disparity_map
        plt.subplot(2, 1, 2)
        plt.imshow(disparity_map)
        plt.title('Disparity Map')
        plt.axis('off')  
        plt.tight_layout()
    
    plt.show()