import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.datasets import MNIST as mnist
import cv2
from torchvision.utils import make_grid

dataset= mnist(".", download=True)

class LinearDisc(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
    nn.Flatten(1),
    nn.Linear(28*28, 256),
    nn.LeakyReLU(),
    nn.Linear(256, 2),
    nn.LogSoftmax(dim=1),
    )

  def forward(self, x): return self.layers(x)

class LinearGen(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(256, 100),
      nn.LeakyReLU(),
      nn.Linear(100, 28*28),
      nn.Tanh(),
      nn.Unflatten(1, (1, 28, 28)),
    )

  def forward(self, x): return self.layers(x)

def get_noise(size):
  out = []
  for _ in range(size):
    noise = torch.normal(0., 1.0, (256,))
    out.append(noise)
    #print(out.mean(1))
  return torch.stack(out).float()
def get_real_data(size):
  out = []
  for _ in range(size):
    i = torch.randint(high=len(dataset)-1, size=(1,))
    batch, label = dataset.__getitem__(i.item())
    img = torchvision.transforms.functional.pil_to_tensor(batch)
    img = img/127.5 - 1.0
    out.append(img)
  return torch.stack(out).float()
    
  return X_train[samples, :, :, :]
def train_disc(optim, model, real_x, gen_x):
  optim.zero_grad()
  real_out = model(real_x)
  loss = -1 * real_out[:, 0].sum()
  loss.backward()

  gen_out = model(gen_x)
  gen_loss =  -1 * gen_out[:, 1].sum()
  gen_loss.backward()

  optim.step()
  return (gen_loss + loss).item()

def train_gen(optim, model, disc_model, batch_size):
  optim.zero_grad()
  X = get_noise(batch_size)
  gen_out = model(X)
  disc_out = disc_model(gen_out)
  loss = -1 * disc_out[:, 0].sum()
  loss.backward()

  #disc_out = disc_model(model(X))
  #neg_loss = (1 - disc_out[:, 1]).sum()
  #neg_loss.backward()
  optim.step()
  return loss.item()



if __name__ == "__main__":
  batch_size = 64*2
  n_iter = len(dataset) // batch_size 
  D = LinearDisc()
  G = LinearGen()
  for param in D.parameters():
    if isinstance(param, nn.Linear):
      torch.nn.init.kaiming_uniform(param.weight)
      torch.nn.init.constant_(param.bias, 0)
  for param in G.parameters():
    if isinstance(param, nn.Linear):
      torch.nn.init.kaiming_uniform(param.weight)
      torch.nn.init.constant_(param.bias, 0)


  D_optim = torch.optim.Adam(D.parameters(), lr=0.0003, betas=(0.5, 0.999))
  G_optim = torch.optim.Adam(G.parameters(), lr=0.0003, betas=(0.5, 0.999))
  for epoch in range(100):
    total_D_loss = []
    D.train()
    G.eval()
    for _ in range(n_iter):
      real_x = get_real_data(batch_size)
      gen_x = G(get_noise(batch_size))
      loss = train_disc(D_optim, D, real_x, gen_x)
      total_D_loss.append(loss)
      print(loss)
    print("Avg Disc loss:", np.array(total_D_loss).mean())
    
    total_G_loss = []
    D.eval()
    G.train()
    for _ in range(n_iter):
      loss = train_gen(G_optim, G, D, batch_size)
      total_G_loss.append(loss)
      print(loss)
    print("Avg Gen loss:", np.array(total_G_loss).mean())
    
    # saving img
    #import pdb; pdb.set_trace()
    G_out = (G(get_noise(144)) + 1.0 ) * 127.5
    grid_img = make_grid(G_out, nrow=12).permute(1,2, 0).numpy().astype(np.uint8)
    cv2.imwrite(f"gan_epoch_{epoch}.png", grid_img)














