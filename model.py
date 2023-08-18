import torch
import torch.nn as nn


class SiameseNetStereoMatching(nn.Module):
    """
    Create a Siamese Neural network for patch matching in stereo matching
    """
    def __init__(self, in_channel, out_channel=64):
        super(SiameseNetStereoMatching, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor):

        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))

        x = self.act3(self.conv3(x))
        x = self.conv4(x) # batch, channel, hight, width

        # nomalize features for each pixel
        features = torch.nn.functional.normalize(x, dim=1, p=2)

        return features


if __name__ == '__main__':
    in_channel = 3  # Example input channel
    batch_size = 3  # Example batch size
    height = 60     # Example height
    width = 60      # Example width
    model = SiameseNetStereoMatching(in_channel)
    input_data = torch.randn(batch_size, in_channel, height, width)
    output = model(input_data)
    
    print(input_data.shape)
    print(output.shape)

    print(output)