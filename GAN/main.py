import cv2
import numpy as np
from tinygrad.helpers import getenv, colored, trange
from tinygrad import Tensor, nn, TinyJit, GlobalCounters
from tinygrad.nn.datasets import mnist

X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))
X_train = Tensor(X_train.numpy()[Y_train.numpy()==0])
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

class Generator:
  def __init__(self):
    self.layers = [
      nn.Linear(256, 256), # distribution of the linear layer might be different
      Tensor.relu,  # in working code they use leakyrelu
      nn.Linear(256, 512),
      Tensor.relu,
      nn.Linear(512, 512),
      Tensor.relu,
      nn.Linear(512, 784),
      Tensor.tanh,
      lambda x: x.reshape(-1, 1, 28, 28,)
    ]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

class Discriminator:
  def __init__(self):
    self.layers = [
      lambda x: x.flatten(1),
      nn.Linear(784, 1024),
      Tensor.leakyrelu,
      nn.Linear(1024, 512),
      Tensor.leakyrelu,
      nn.Linear(512, 256),
      Tensor.leakyrelu,
      nn.Linear(256, 2),
      # 1) this model should output 2 values not one. model(x) --> [0.1, 0.7]
      Tensor.log_softmax,
      #Tensor.sigmoid,
    ]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

#@TinyJit
@Tensor.train()
def train_discriminator(X:Tensor, fake_X) -> Tensor:
  # Train model on teaching about real images
  D_opt.zero_grad()
  inpt = X
  y = Tensor.zeros((batch_size, 2))
  # y shape is [64, 2]
  y_ones = Tensor.ones(batch_size)
  y_zeros = Tensor.zeros(batch_size)
  y = Tensor.cat(y_ones, y_zeros)
  res = D(inpt)
  #loss = Tensor.binary_crossentropy(res, y).backward()
  import pdb; pdb.set_trace()
  loss = (res * y).mean()
  loss.backward()
  



  D_opt.step()
  return loss

@Tensor.train()
def train_generator(noise) -> Tensor:
  G_opt.zero_grad()
  fake_y = Tensor.ones(batch_size)
  #fake_y = fake_y - 2.0
  generated_images = G(noise)
  discriminator_res = D(generated_images).squeeze()
  loss = (discriminator_res * fake_y).mean().backward()
  #loss = discriminator_res.binary_crossentropy(fake_y).backward()
  G_opt.step()
  return loss

# -log-likelihood = -1 * log(y_pred) * log(y_true)
D = Discriminator()
G = Generator()
D_opt = nn.optim.Adam(nn.state.get_parameters(D), lr=0.005)
G_opt = nn.optim.Adam(nn.state.get_parameters(G), lr=0.005)
batch_size=64
n_steps = X_train.shape[0] // batch_size
n_steps = 5
print("batch size: ", batch_size, "n_steps:", n_steps)

for i in (t:=trange(getenv("STEPS", 50))):
  #GlobalCounters.reset()  # what does this DEBUG=2 timing do?
  total_loss = []
  for _ in range(n_steps): 
    samples = Tensor.randint(getenv("BS", batch_size), high=X_train.shape[0])
    X = X_train[samples]
    y = Tensor.ones(batch_size)
    noise = Tensor.randint(batch_size, 256) / 127.5 - 1.0  # convert to [-1, 1]
    fake_X = G(noise).detach()
    d_loss = train_discriminator(X, fake_X)
    total_loss.append(d_loss.item())
  
  total_g_loss = []
  for _ in range(n_steps):
    g_loss = train_generator(noise)
    total_g_loss.append(g_loss.item())
  
  print(f'd_loss: {np.array(total_loss).mean():.2f} | g_loss: {np.array(total_g_loss).mean():.2f}') 
  #t.set_description(f"loss: {loss.item():6.2f}")

  generated_img = G(Tensor.randint(1, 256))
  cv2.imwrite(f'gan_generated_{i}.png', generated_img.permute(0, 2, 3, 1).numpy()[0])
print("Completed")
