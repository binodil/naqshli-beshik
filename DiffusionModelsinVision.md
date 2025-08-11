# source: file:///C:/Users/USER/Downloads/2209.04747.pdf


What is a diffusion model? A two stage deep generative model. First stage is forward diffusion stage. Second stage is reverse diffusion stage.

The input data is gradually perturbed over several steps by adding Gaussian noise in the foward diffusion stage.
In the reverse stage, a model task is to recovering the original input data by learning to gradually reverse the diffusion process, step by step.

What are common generative models?
1) GAN,
2) Variational auto-encoder
3) Energy-based models
4) autoregressive models
5) Normalizing flows.
6) diffusion models

Diffusion models raised gen ai to new bar. Models like Imagen and Latent Diffusion Models (LDMs).

Stable Diffusion is a child of Latent Diffusion Models. Generate images based on given text.

Generative modelling tasks:
1) Image generation
2) Image super-resolution
3) Image Inpainting
4) Image editing
5) Iamge-to-image translation
6) etc.

The latent representation space learned by diffusion models was also found to be useful in discriminative tasks, image segmentation, classification, anomaly detection.



We have a X_0 image;
in forward pass:
we apply Forward SDE, DDPM, NCSN, where to slowly add Gaussian noise to the X_0 image for T steps.

Then we do backward:
Apply reverse SDE, DDPm, NCSN, where we go backward to the original X_0 image.



AWESOME Diffusion models!!! From scratch: https://github.com/Animadversio/DiffusionFromScratch/tree/master

Diffusion backbone can be transformer or U-nets


What is a U-net. It can be a Conv layer + reversed conv layers. In classical conv, the matrix is shrinks to the laten vector. then we can apply upsample with convolutions to restore the original image size. So this is one of U net in the industry. it is good for good learning features.  Paper (https://arxiv.org/pdf/1505.04597)



OpenAi (2021) Diffusion Models better than GAN!
https://arxiv.org/pdf/2105.05233


CLIP-model. It is a Text-to-Image and Image-to-Text model. How to connect CLIP to the Diffusion models? (https://arxiv.org/pdf/2103.00020) (2021). Use Cnn or ViT? Code is open sourced.


LoRA? Low-Rank Adoptation - it is a technique used to adapt machine learning models to new contexts. (https://arxiv.org/pdf/2106.09685). Inject trainable rank decomposition matrices into each layer of the Transformer architecture, and feezing original model weights. LoRA makes the training much easier.

A neural network contains many dense layers which perform matrix multiplication. The weight matrices in these layers typically have full-rank. When adapting to a specific task, pre-trained language models have a low 'instrinsic dimension' and can still learn efficiently despite a random projection to ta smaller subspace.


