import pathlib
from PIL import Image


if __name__ == "__main__":
  images = []
  for idx in range(1, 10000//400+1):
    img_path = f"cgan_bce/gan_iter_{int(idx*400)}.png"
    img  = Image.open(img_path)
    images.append(img)
  
  images[0].save("cgan_bce.gif", save_all=True, append_images=images[1:]+images[-1:]+images[-1:], optimize=False, duration=300)

