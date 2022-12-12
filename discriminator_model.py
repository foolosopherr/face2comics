import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, 4, stride, padding, bias=False, padding_mode='reflect'),
           nn.InstanceNorm2d(out_channels, affine=True),
        #    nn.BatchNorm2d(out_channels),
           nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

# x, y <- concatenate these along the channels
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features:
            layers.append(
                CNNBlock(in_channels, feature, stride=2 if feature != features[-1] else 1)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)

def test():
    x = torch.randn((1, 3, 512, 512))
    y = torch.randn((1, 3, 512, 512))
    model = Discriminator()
    preds = model(x, y)
    print('Should be (1, 1, 30, 30)')
    print(preds.shape)

if __name__ == '__main__':
    test()
