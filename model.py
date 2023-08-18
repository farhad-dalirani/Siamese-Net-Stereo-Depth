import torch
import torch.nn as nn


class SiameseNetStereoMatching(nn.Module):
    """
    Create a Siamese Neural network for patch matching in stereo matching
    """
    def __init__(self, in_channel, out_channel=64):
        super(SiameseNetStereoMatching, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3)
        
    def forward(self, x: torch.Tensor):
        """

        Args:
            x: tensor in shape of (batch size, image channel, hight, width)

        Returns:
            the architecture with respect to batch size should be in a way that the output become
            a tensor in shape of (batch, channel, 1, 1)
        """
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))

        x = self.act3(self.conv3(x))
        x = self.conv4(x) # batch, channel, hight, width

        # nomalize features for each pixel
        features = torch.nn.functional.normalize(x, dim=1, p=2)

        return features


def similarity_score(model, x_l, x_r):
    """This function get features calculated by the neural network and calculates the cosine similarity

    Args:
        model: Siamese network's that generate features two input patches
        x_l: patch in the left image, (batch size, feature channel, 1, 1)
        x_r: patch in right image, (batch size, feature channel, 1, 1)
    """
    
    # siamese network feature for left image 
    f_l = model(x_l) # (batch size, feature channel, 1, 1)
    
    # siamese network feature for right image
    f_r = model(x_r) # (batch size, feature channel, 1, 1)

    # similarity score of the patches
    similarity = torch.sum(f_l * f_r, dim=1).squeeze() # (batch size,)

    return similarity


if __name__ == '__main__':
    in_channel = 3  # Example input channel
    batch_size = 3  # Example batch size
    height = 9     # Example height
    width = 9      # Example width
    model = SiameseNetStereoMatching(in_channel)
    input_data = torch.randn(batch_size, in_channel, height, width)
    output = model(input_data)
    
    print(input_data.shape)
    print(output.shape)
    print(output)

    x_l = torch.randn(6, 3, 9, 9)
    x_r = torch.randn(6, 3, 9, 9)
    similarity = similarity_score(model, x_l, x_r)
    print(similarity)