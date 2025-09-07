import cv2
import numpy as np
from tinygrad.helpers import getenv, colored, trange
from tinygrad import Tensor, nn, TinyJit, GlobalCounters
from tinygrad.nn.datasets import mnist

X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))
#X_train = Tensor(X_train.numpy()[Y_train.numpy()==2])
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
bias = True
class Generator:
  def __init__(self):
    self.layers = [
      nn.Linear(256, 256, bias=bias),
      Tensor.leakyrelu,
      nn.Linear(256, 512, bias=bias),
      Tensor.leakyrelu,
      nn.Linear(512, 512, bias=bias),
      Tensor.leakyrelu,
      nn.Linear(512, 784, bias=bias),
      Tensor.tanh,
      lambda x: x.reshape(-1, 1, 28, 28,),
    ]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

class Discriminator:
  def __init__(self):
    self.layers = [
      lambda x: x.flatten(1),
      nn.Linear(784, 1024, bias=bias),
      Tensor.leakyrelu,
      nn.Linear(1024, 512, bias=bias),
      Tensor.leakyrelu,
      nn.Linear(512, 256, bias=bias),
      Tensor.leakyrelu,
      nn.Linear(256, 2, bias=bias),
      # 1) this model should output 2 values not one. model(x) --> [0.1, 0.7]
      Tensor.log_softmax,  # k/(sum(k)
      #Tensor.sigmoid,     # (k^2)/
    ]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)


#-----------------------------------------------------------------------------#

#@TinyJit
@Tensor.train()
def train_discriminator(X:Tensor, fake_X:Tensor) -> Tensor:
  # Train model on teaching about real images
  D_opt.zero_grad()
  y = Tensor.zeros((batch_size, 2))
  # y shape is [64, 2]
  y_ones = Tensor.ones(batch_size, 1)
  y_zeros = Tensor.zeros(batch_size, 1)
  y = Tensor.cat(y_ones, y_zeros, dim=1)
  res = D(X)
  #loss = Tensor.binary_crossentropy(res, y).backward()
  loss = -1 * (res * y).mean()
  #D_opt.step()
  # gradient for discriminator
  D_opt.zero_grad()
  fake_y = Tensor.cat(y_zeros, y_ones, dim=1)
  fake_res = D(fake_X)
  fake_loss = -1 * (fake_res * fake_y).mean()
  loss.backward()
  fake_loss.backward()
  D_opt.step()
    #Concerns: If I update twice in each train loop the gradients of discriminator, it might be strong than Generator. What if we update params only once?
  print("Discriminator loss: ", ((loss + fake_loss)/2).item())
  return loss + fake_loss  # Maybe summing might be a bad idea!


@Tensor.train()
def train_generator(noise) -> Tensor:
  G_opt.zero_grad()
  y = Tensor.cat(Tensor.ones(batch_size, 1), Tensor.zeros(batch_size, 1), dim=1)
  generated_images = G(noise)
  generated_images = (generated_images + 1.0) * 127.5
  discriminator_res = D(generated_images)
  loss = -1 * (discriminator_res * y).mean()
  loss.backward()
  G_opt.step()
  print("Generator loss: ", loss.item()) 
  return loss

# -log-likelihood = -1 * log(y_pred) * log(y_true)
D = Discriminator()
G = Generator()
for m in [D, G]:
  for l in m.layers:
    if isinstance(l, nn.Linear):
      # uniform the weight
      l.weight = Tensor.scaled_uniform(l.weight.shape)
      pass
      # scaled uniform return value between -1 and 1; while uniform by itself return 0 and 1

print("weights are scaled uniform")
D_opt = nn.optim.Adam(nn.state.get_parameters(D), lr=0.003)
G_opt = nn.optim.Adam(nn.state.get_parameters(G), lr=0.003)
batch_size=128
n_steps = X_train.shape[0] // batch_size
print("batch size: ", batch_size, "n_steps:", n_steps)


for i in (t:=trange(getenv("STEPS", 50))):
  #GlobalCounters.reset()  # what does this DEBUG=2 timing do?
  # MAXIM: it all depends on your faith!
  #----Training discriminator-----
  print("Training discriminator")
  total_d_loss = []
  for _ in range(n_steps): 
    samples = Tensor.randint(getenv("BS", batch_size), high=X_train.shape[0])
    X = X_train[samples]
    y = Tensor.ones(batch_size)
    noise = Tensor.randint(batch_size, 256, low=0, high=255) / 127.5 - 1.0  # convert to [-1, 1]
    fake_X = G(noise)
    fake_X = (fake_X + 1.0) * 127.5
    d_loss = train_discriminator(X, fake_X)
    total_d_loss.append(d_loss.item())
  
  print("Training generator")
  #----Training generator------
  total_g_loss = []
  for _ in range(n_steps):
    noise = Tensor.randint(batch_size, 256, low=0, high=255) / 127.5 - 1.0  # convert to [-1, 1]  # I think this is bad!
    g_loss = train_generator(noise)
    total_g_loss.append(g_loss.item())
  
  for m in [D, G]:
    for idx, layer in enumerate(m.layers):
      if isinstance(layer, nn.Linear):
        print("Linear at ", idx, ":", layer.weight.mean().item())
 
  print(f'd_loss: {np.array(total_d_loss).mean():.2f} | g_loss: {np.array(total_g_loss).mean():.2f}') 

  noise = Tensor.randint(10, 256, low=0, high=255) / 127.5 - 1.0  # convert to [-1, 1]
  generated_img = G(noise)

  import pdb; pdb.set_trace()
  cv2.imwrite(f'gan_generated_{i}.png', (np.hstack(((generated_img[:,0,:,:] + 1.0) * 127.5).numpy()).astype(np.uint8)))
print("Completed")
