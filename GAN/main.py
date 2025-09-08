import cv2
import numpy as np
from tinygrad.helpers import getenv, colored, trange
from tinygrad import Tensor, nn, TinyJit, GlobalCounters
from tinygrad.nn.datasets import mnist
import pdb
X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))
X_train = Tensor(X_train.numpy()[Y_train.numpy()==2].astype(np.float32))
print(X_train.shape, X_train.dtype, Y_train.shape, X_test.shape, Y_test.shape)
bias = True

class Generator:
  def __init__(self):
    self.layers = [
      nn.Linear(256, 512, bias=bias),
      Tensor.leaky_relu,
      #Tensor.dropout(0.2),
      nn.Linear(512, 1024, bias=bias),
      Tensor.leaky_relu,
      #Tensor.dropout(0.2),
      nn.Linear(1024, 512, bias=bias),
      Tensor.leaky_relu,
      nn.Linear(512, 784, bias=bias),
      Tensor.tanh,
      lambda x: x.reshape(-1, 1, 28, 28,),
    ]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)
  def debug(self,):
    for l in self.layers:
      if isinstance(l, nn.Linear):
        print("Does it has a nan in weight and its grad, bias and its grad:", l.weight.isnan().any().item(), l.weight.grad.isnan().any().item(), l.bias.isnan().any().item(), l.bias.grad.isnan().any().item())
        print("Does it has an inf in weight and its grad, bias and its grad:", l.weight.isinf().any().item(), l.weight.grad.isinf().any().item(), l.bias.isinf().any().item(), l.bias.grad.isinf().any().item())
        

class Discriminator:
  def __init__(self):
    self.layers = [
      lambda x: x.flatten(1),
      nn.Linear(784, 1024, bias=bias),
      Tensor.leaky_relu,
      nn.Linear(1024, 512, bias=bias),
      Tensor.leaky_relu,
      nn.Linear(512, 256, bias=bias),
      Tensor.leaky_relu,
      nn.Linear(256, 2, bias=bias),
      # 1) this model should output 2 values not one. model(x) --> [0.1, 0.7]
      Tensor.log_softmax,  # k/(sum(k)
      #Tensor.sigmoid,     # (k^2)/
    ]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

  def debug(self,):
    for l in self.layers:
      if isinstance(l, nn.Linear):
        print("Does it has a nan in weight and its grad, bias and its grad:", l.weight.isnan().any().item(), l.weight.grad.isnan().any().item(), l.bias.isnan().any().item(), l.bias.grad.isnan().any().item())
        print("Does it has an inf in weight and its grad, bias and its grad:", l.weight.isinf().any().item(), l.weight.grad.isinf().any().item(), l.bias.isinf().any().item(), l.bias.grad.isinf().any().item())
  

#-----------------------------------------------------------------------------#

#@TinyJit
@Tensor.train()
def train_discriminator(X:Tensor, fake_X:Tensor) -> Tensor:
  #D_opt.zero_grad()
  for i, layer in enumerate(D.layers):
    if isinstance(layer, nn.Linear):
      D.layers[i].weight.requires_grad = True
      D.layers[i].weight.grad = None
      D.layers[i].bias.requires_grad = True
      D.layers[i].bias.grad = None
  # Train model on teaching about real images
  y_ones = Tensor.ones(batch_size, 1)
  y_zeros = Tensor.zeros(batch_size, 1)
  y = Tensor.cat(y_ones, y_zeros, dim=1)
  fake_y = Tensor.cat(y_zeros, y_ones, dim=1)
  res = D(X)
  loss = (-1 * res * y).sum(1).mean()
  loss.backward()
  #D_opt.step()
  # gradient for discriminator
  #D_opt.zero_grad()
  fake_res = D(fake_X)
  fake_loss = (-1 * fake_res * fake_y).sum(1).mean()
  fake_loss.backward()
  #D_opt.step()

  #----manual----
  for i, layer in enumerate(D.layers):
    if isinstance(layer, nn.Linear):
      assert layer.weight.grad is not None, pdb.set_trace()
      print(f"{layer.weight.grad.mean().item()=} | {layer.weight.mean().item()=} | {layer.bias.grad.mean().item()=} | {layer.bias.grad.mean().item()=}")
      D.layers[i].weight = layer.weight - 0.003 * layer.weight.grad
      D.layers[i].bias = layer.bias - 0.003 * layer.bias.grad
  ############
  print("Discriminator loss: ", ((loss + fake_loss)/2).item(), "D res mean:", res.mean().item(), "fake res mean:", fake_res.mean().item())
  return loss + fake_loss  # Maybe summing might be a bad idea!


@Tensor.train()
def train_generator(noise) -> Tensor:
  #G_opt.zero_grad()
  for i, layer in enumerate(G.layers):
    if isinstance(layer, nn.Linear):
      G.layers[i].weight.requires_grad = True
      G.layers[i].weight.grad = None
      G.layers[i].bias.requires_grad = True
      G.layers[i].bias.grad = None
  #
  y = Tensor.cat(Tensor.ones(batch_size, 1), Tensor.zeros(batch_size, 1), dim=1) 
  discriminator_res = D(G(noise)) # is the nan generated un the grad at the D or at the G. when opt_G.step, the G becomes nan. What does cause this?

  loss = (-1 * discriminator_res * y).sum(1).mean()
  loss.backward() 
  # do the norm of the gradients.
  #print([G.layers[i].weight.grad.mean().item() for i in [0, 2, 4, 6]])
  #G_opt.step()
  #---manual optim---
  for i, layer in enumerate(G.layers):
    if isinstance(layer, nn.Linear):
      print(f"{layer.weight.grad.mean().item()=} | {layer.weight.mean().item()=} | {layer.bias.grad.mean().item()=} | {layer.bias.grad.mean().item()=}")
      G.layers[i].weight = layer.weight - 0.03 * layer.weight.grad
      G.layers[i].bias = layer.bias - 0.03 * layer.bias.grad
  #-----------------
  print("Generator loss: ", loss.item(), "noise mean val:", noise.mean().item(), "D output mean:", discriminator_res.mean().item()) 
  return loss

# -log-likelihood = -1 * log(y_pred) * log(y_true)
D = Discriminator()
G = Generator()
for m in [D, G]:
  for l in m.layers:
    if isinstance(l, nn.Linear):
      # uniform the weight
      l.weight = Tensor.scaled_uniform(l.weight.shape)
      l.bias = Tensor.zeros_like(l.bias)
      # scaled uniform return value between -1 and 1; while uniform by itself return 0 and 1

print("weights are scaled uniform")
#D_opt = nn.optim.SGD(nn.state.get_parameters(D), lr=0.02)
##G_opt = nn.optim.SGD(nn.state.get_parameters(G), lr=0.02)
batch_size=32
n_steps = 4 #X_train.shape[0] // batch_size
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
    X = X/127.5 - 1.0
    noise = Tensor.randint(batch_size, 256, low=0, high=255) / 127.5 - 1.0  # convert to [-1, 1]
    print(noise.dtype)
    print(X.dtype)
    fake_X = G(noise)
    #print(X.mean().item(), fake_X.mean().item())
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
  image = np.hstack(((generated_img[:,0,:,:] + 1.0) * 127.5).numpy())
  image = image.astype(np.uint8)
  cv2.imwrite(f'gan_generated_{i}.png', image)
print("Completed")
