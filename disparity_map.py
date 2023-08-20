import torch

def disparity_map_nn(model, img_l, img_r, padding_size, max_disparity, device):
    """Calcualte disparity of two rectified images by patch matching. patch matching 
       is done by using Siamese convolutional neural netwrok.

    Args:
        model: Siamese neural network
        img_l: rectified left image
        img_r: rectified right image
        padding_size: pad size to avoid changing spatial resolution change by Siamese network 
        max_disparity: maximum possible disparity to get considered
        device: device cpu/cuda
    """
    # model to device
    model.to(device)

    # put model in evaluation
    model.eval()

    # convert to tensor
    I_l = torch.tensor(img_l, dtype=torch.float32)
    I_l = I_l.to(device)
    I_l = I_l.unsqueeze(0)
    I_r = torch.tensor(img_r, dtype=torch.float32)
    I_r = I_r.to(device)
    I_r = I_r.unsqueeze(0)

    # convert (batch, height, width, channel) to (batch, channel, height, width) 
    I_l = I_l.permute(0, 3, 1, 2)
    I_r = I_r.permute(0, 3, 1, 2)

    # pad images to keep spatial feature size the same as input image to the model.  
    pad = torch.nn.ZeroPad2d(padding_size)
    I_l_padded = pad(I_l)
    I_r_padded = pad(I_r)

    # extract feature for each pixel
    F_l = model(I_l_padded)
    F_r = model(I_r_padded)

    best_similarity = torch.ones(size=(F_l.shape[0], 1, F_l.shape[2], F_l.shape[3]), device=device) * -torch.inf
    disparity_map = torch.zeros(size=(F_l.shape[0], 1, F_l.shape[2], F_l.shape[3]), device=device, dtype=torch.int)

    # for each pixel in the left image at row i and column j, compare similarity with
    # all pixels [i, j-d], [i, j-d+1], [i, j-d+2], ..., [i, j]  
    for disparity_i in range(0, max_disparity+1):

        if disparity_i > 0:
            # shif values of feature by d for right image
            F_r_shifted = torch.zeros_like(F_r)
            F_r_shifted[:,:,:,disparity_i:] = F_r[:,:,:,:-disparity_i]
        elif disparity_i == 0:
            F_r_shifted = F_r 
        
        # calculate similarity of pixels in left feature map to the shifted right feature map
        similarity = torch.sum(F_l * F_r_shifted, dim=1, keepdim=True)

        # positions that a better match has found
        better_match = similarity > best_similarity
        best_similarity[better_match] = similarity[better_match]
        disparity_map[better_match] = disparity_i

    disparity_map = disparity_map.squeeze()

    return disparity_map


