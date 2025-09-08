import torch
import torch.nn as nn
import torchvision



class LinearDisc(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
    nn.Flatten(1),
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
    )

  def forward(self, x): return self.layers(x)

class LinearGen(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(256, 512),
      nn.ReLU(),
      nn.Linear(512, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, 28*28),
      nn.Tanh(),
      nn.Unflatten(1, (1, 28, 28)),
    )

  def forward(self, x): return self.layers(x)

def get_noise(size): return torch.normal(0., 1.0, (size, 256))



if __name__ == "__main__":
  D = LinearDisc()
  G = LinearGen()
  print(D(torch.normal(0., 1., (8, 1, 28, 28))))
  print(G(get_noise(1)))
