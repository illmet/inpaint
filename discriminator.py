import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d):
      super(Discriminator, self).__init__()

      use_bias = False

      self.model = nn.Sequential(
          # Layer 1
          nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(0.2, True),

          # Layer 2
          nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
          norm_layer(ndf * 2),
          nn.LeakyReLU(0.2, True),

          # Layer 3
          nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
          norm_layer(ndf * 4),
          nn.LeakyReLU(0.2, True),

          # Output layer
          nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=1)
      )

  def forward(self, input):
      return self.model(input)