import math
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.datasets import MNIST as mnist
import cv2
from torchvision.utils import make_grid

dataset= mnist(".", download=True)


def parzen_window_estimate(x, samples, h):
  # x: tensor
  # sample: training sample
  # h: float
  n_eval, d = x.shape
  n_samples = samples.shape[0]
  diff = x.unsqueeze(1) - samples.unsqueeze(0)
  norm_squared = torch.sum(diff**2, dim=-1)
  # Compute gaussian kernel value
  kernel_vals = torch.exp(-norm_squared / (2 * h**2)) / ((2 * torch.pi * h**2)**(d/2))
  # return density estimate
  density_estimate = torch.mean(kernel_vals, dim=1)
  return density_estimate

#parzen_window_estimate(torch.randn((100, 240)).shape
#import sys.exit(0)

class CLinearGen(nn.Module):
  def __init__(self):
    super().__init__()
    self.x_l1 = nn.Linear(110, 256)  # 100 * 256 = 25600 + 256 = 26_000
    self.bn_1 = nn.BatchNorm1d(256, 0.8)
    self.x_l2 = nn.Linear(256, 1024)  # 550
    self.bn_2 = nn.BatchNorm1d(1024, 0.8)
    self.relu = nn.LeakyReLU()
    self.x_l3 = nn.Linear(1024, 784)  # 150 * 784 = 118_000
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(0.2)
    self.tanh = nn.Tanh()
    # 145_000 parameters

  def forward(self, x, y):
    x = self.x_l1(torch.cat((x, y), 1))   # B, 200
    x = self.bn_1(x)
    x = self.relu(x)
    x = self.x_l2(x)
    x = self.bn_2(x)
    x = self.relu(x)
    x = self.x_l3(x)
    out = self.tanh(x)
    out = out.unflatten(1, (1, 28, 28))  # dim idx, size;
    return out

class CLinearDisc(nn.Module):
  def __init__(self):
    super().__init__()
    self.dropout = nn.Dropout(0.2)
    self.relu = nn.LeakyReLU()
    self.x_maxout = []
    self.bn_1 = nn.BatchNorm1d(256, 0.8)
    self.bn_2 = nn.BatchNorm1d(100, 0.8)
    self.x_maxout1 = nn.Linear(784+10, 256) # 784 * 256 + 256 = 200_000
    self.x_maxout2 = nn.Linear(256, 100)
    self.x_maxout3 = nn.Linear(100, 1)
    self.x_maxout4 = nn.Linear(784, 240)
    self.x_maxout5 = nn.Linear(784, 240)
    self.x_maxout = [self.x_maxout1, self.x_maxout2, self.x_maxout3, self.x_maxout4, self.x_maxout5]  
    
    self.y_maxout1 = nn.Linear(10, 50)  # 550
    self.y_maxout2 = nn.Linear(10, 50)
    self.y_maxout3 = nn.Linear(10, 50)
    self.y_maxout4 = nn.Linear(10, 50)
    self.y_maxout5 = nn.Linear(10, 100)
    self.y_maxout = [self.y_maxout1, self.y_maxout2, self.y_maxout3, self.y_maxout4, self.y_maxout5]
    out_dim = 1
    self.xy_maxout1 = nn.Linear(200 + 50, out_dim)  # 256 * 50 + 50 = 12800  Overall: 212_000 params
    self.xy_maxout2 = nn.Linear(240+50, out_dim)
    self.xy_maxout3 = nn.Linear(240+50, out_dim)
    self.xy_maxout4 = nn.Linear(240+50, out_dim)
    self.xy_maxout = [self.xy_maxout1, self.xy_maxout2, self.xy_maxout3, self.xy_maxout4]

    self.sigmoid = nn.Sigmoid()
    self.log_softmax = nn.LogSoftmax(dim=1)
  
  def forward(self, x, y):
    return self.forward_unoff(x, y)

  def forward_unoff(self, x, y):
    x = x.flatten(1)
    x = torch.cat((x, y), 1)
    x = self.x_maxout1(x)
    x = self.bn_1(x)
    x = self.relu(x)
    x = self.x_maxout2(x)
    x = self.bn_2(x)
    x = self.relu(x)
    x = self.x_maxout3(x)
    return x


  def forward_off(self, x, y):
    x = x.flatten(1)
    hx = self.x_maxout[0](x)
    # shape of h : Bx240
    for layer in self.x_maxout[1:]:
      hx = torch.max(hx, layer(x))

    # h is out finally
    hy = self.y_maxout[0](y)
    for layer in self.y_maxout[1:]:
       hy = torch.max(hy, layer(y))

    # out from y
    xy = torch.cat((hx, hy), dim=1)  # B, 290

    hxy = self.xy_maxout[0](xy)
    for layer in self.xy_maxout[1:]:
      xy = self.dropout(xy)
      hxy = torch.max(hxy, layer(xy))
    
    out = self.sigmoid(hxy)
    # out.shape B, 240 --> B, 1
    out = self.log_softmax(out)
    return out


def get_noise(size):
  out = []
  for _ in range(size):
    noise = torch.normal(0., 1.0, (100,))
    out.append(noise)
    #print(out.mean(1))
  return torch.stack(out).float()
def get_real_data(size):
  out = []
  labels = []
  for _ in range(size):
    i = torch.randint(high=len(dataset)-1, size=(1,))
    batch, label = dataset.__getitem__(i.item())
    img = torchvision.transforms.functional.pil_to_tensor(batch)
    img = img/127.5 - 1.0
    out.append(img)
    labels.append(torch.nn.functional.one_hot(torch.tensor(label), num_classes=10))
  return torch.stack(out).float(), torch.stack(labels).float()
    
def train_disc(optim, model, real_x, y, gen_x):
  optim.zero_grad()
  real_out = model(real_x, y)  # [-0.9, 0.2] ---> [0, 1]
  #import pdb; pdb.set_trace()
  #real_out = torch.max(real_out, torch.tensor(0.00001))
  #real_out = real_out ** 2 + 0.000001
  #loss = ((real_out - 1)**2).mean()
  loss = -1 * real_out.sigmoid().log().sum()
  #gen_loss = torch.tensor(0.0)
  batch_size = y.shape[0]
  y_rand = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10).float()
  gen_out = model(gen_x, y)
  #gen_out = torch.max(gen_out, torch.tensor(0.00001))
  gen_loss = -1 * (1 - gen_out.sigmoid()).log().sum()
  #gen_loss = ((gen_out - 0)**2).mean()
  d_loss = (gen_loss + loss) / 2 
  d_loss.backward()
  optim.step()
  #print(f"How D knows real img wrt label: {loss.item()} How D detects G img wrt label: {gen_loss.item()}")
  #return (loss).item()
  return gen_loss.item(), loss.item()

def train_gen(optim, model, disc_model, batch_size):
  optim.zero_grad()
  X = get_noise(batch_size)
  y = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size,)), num_classes=10).float()
  gen_out = model(X, y)
  disc_out = disc_model(gen_out, y)
  #disc_out = torch.max(disc_out, torch.tensor(0.00001))
  loss = -1 * (disc_out.sigmoid()).log().sum()
  #loss = ((disc_out - 1)**2).mean() 
  loss.backward()

  #disc_out = disc_model(model(X))
  #neg_loss = (1 - disc_out[:, 1]).sum()
  #neg_loss.backward()
  optim.step()
  return loss.item()



if __name__ == "__main__":
  batch_size = 128 
  n_iter = len(dataset) // batch_size  # 468
  print(n_iter)
  D = CLinearDisc()
  G = CLinearGen()
  for param in D.parameters():
    if isinstance(param, nn.Linear):
      torch.nn.init.kaiming_uniform(param.weight)
      torch.nn.init.constant_(param.bias, 0)
  for param in G.parameters():
    if isinstance(param, nn.Linear):
      torch.nn.init.kaiming_uniform(param.weight)
      torch.nn.init.constant_(param.bias, 0)

  
  D_optim = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
  G_optim = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
  t_iter = 0
  for epoch in range(100):
    ######TRAIN GEN############
    total_G_loss = []
    total_D_loss = []
    D.eval()
    G.train()
    for k in range(n_iter):
      loss = train_gen(G_optim, G, D, batch_size)
      total_G_loss.append(loss)
      #if k % 10 == 0: print(f"[{k}/{n_iter}] How G fools D: {loss:.2f}")
    
    ##########TRAIN DISC##############
      real_x, real_y = get_real_data(batch_size)
      gen_x = G(get_noise(batch_size), real_y).detach()
      gen_loss, loss = train_disc(D_optim, D, real_x, real_y, gen_x)
      total_D_loss.append([loss, gen_loss])

      t_iter += 1
      if t_iter % 400 == 0:
        mock_y = torch.nn.functional.one_hot(torch.tensor([i for i in range(10)] * 10), num_classes=10).float()
        G_out = (G(get_noise(100), mock_y) + 1.0 ) * 127.5
        grid_img = make_grid(G_out, nrow=10).permute(1,2, 0).numpy().astype(np.uint8)
        cv2.imwrite(f"cgan_bce/gan_iter_{t_iter}.png", grid_img)
        if t_iter % 1600 == 0:
          torch.save(G.state_dict(), f"mnist_generator_{epoch}.pth")
 
      #if k %10 == 0: print(f"[{k}/{n_iter}] How D ispect G: {loss:.2f}")
    print("epoch: ", epoch, "Avg Disc loss:", np.array(total_D_loss).mean(0))
    print("epoch:", epoch, "Avg Gen loss:", np.array(total_G_loss).mean())
    ###########################
    # saving img
    #import pdb; pdb.set_trace()











