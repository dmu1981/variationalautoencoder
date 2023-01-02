# Variational Autoencoder
Variational autoencoders are a method to learn the underlying distribution of data samples and to sample from that distribution, thus generate artificial data samples which follow the same distribution as your samples.

![All losses](optimal_samples.png)

The above samples are generated from training a VAE with a  [flower dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) published on Kaggle and then drawing samples from the learned distribution.

## Classic Autoencoders
An autoencoder is a neural network that learns a compressed representation of the input data. Tghe network typically consists of two parts. The encoder part maps the input data to a latent space with lower dimensionality. The decoder part uses this latent data representation and learns a mapping back to the original space. 

Imaging an image of size 256x256 with three color channels. This represents 196.608 input dimensions (256x256x3). Further imaging a convolutional neural network that maps this input image to a 4096-dimensional vector space. This would represent a ~98% reduction in dimensionality. 
## The network architecture
### Downsampling
Downsampling is handeled by two [convolutional layers](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d) with a [GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html?highlight=gelu) activiation in between. [Batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html?highlight=batch+normalization) is employed to support better gradient flow. After the double convolutions, 2x2 [max-pooling](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool2d) is used to reduce the spatial extent of the data.

    class Down(nn.Module):
      def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.relu = nn.GELU()
        self.fw = nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3,3), padding='same'),
          nn.BatchNorm2d(num_features=mid_channels),
          nn.GELU(),
          nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(3,3), padding='same'),
          nn.BatchNorm2d(num_features=out_channels),
          nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )

      def forward(self, x):
        return self.fw(x)

NOTE that the second GELU activation after convolution is missing here as the output of this operation will also be used differently later. In the actual autoencoder architecture, the activation will be added back in, though. 

### Upsampling
Upsampling is handeled by a bilinear upsampling of the previous feature mapes plus a sequence of two convolution layers with GELU activiations and batch normalization.

    class Up(nn.Module):
      def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.relu = nn.GELU()
        self.fw = nn.Sequential(
          nn.BatchNorm2d(num_features=in_channels),
          nn.GELU(),      
          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
          nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3,3), padding='same'),
          nn.BatchNorm2d(num_features=mid_channels),
          nn.GELU(),
          nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(3,3), padding='same')
        )

      def forward(self, x):
        return self.fw(x)

### The encoder
The encoder is a sequence of down-convolutions (see above) followed by two fully-connected layers outputing the compressed embedding. A classical autoencoder would only output the MU component and learn to reconstruct the original image from that. Our variational autoencoder learns to reconstruct samples a gaussian distribution and thus the encoder outputs the parameters of said distribution, namely $\mu$ and $\sigma$.

    class Encoder(nn.Module):
      def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
          Down( 3,  8,  8), nn.GELU(),
          Down( 8, 16, 16), nn.GELU(),
          Down(16, 32, 32), nn.GELU(),
          Down(32, 64, 64), nn.GELU(),
          Down(64, 128, 128), nn.GELU(),
          Down(128, 256, 256), nn.GELU(),
          nn.Flatten(),
        )
        self.mu = nn.Linear(4096, 512)
        self.std = nn.Linear(4096, 512)

      def forward(self, x):
        x = self.encoder(x)
        return self.mu(x), torch.exp(self.std(x))
### The decoder
The decoder takes a 512-dimensional embedding in the latent space and reconstructs the original image by a sequence of up-convolutions (see above) followed by a hyperbolic tangens as the final activiation layer (too make sure outputs stay between -1 and +1).

    class Decoder(nn.Module):
      def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 4096)
        self.gelu = nn.GELU()
        self.decoder = nn.Sequential(
          Up(256, 256, 128), # 128x8x8
          Up(128, 128, 64), # 64x16x16
          Up(64, 64, 32), # 32x32x32
          Up(32, 32, 16), # 16x64x64
          Up(16, 16, 16),  # 16x128x128
          Up(16, 16, 3),  # 3x256x256
        )
        self.tanh = nn.Tanh()

      def forward(self, x):
        x = self.gelu(self.linear(x))
        return self.tanh(self.decoder(x.view(-1,256,4,4)))
### The VAE itself
The variational autoencoder itself is then straight forward to define. First, encode the target image with the encoder. Then we draw a random samples from the gaussian distribution defined by $\mu$ and $\sigma$. Finally, we decode the drawn sample using the decoder

    class VAE(nn.Module):    
      def __init__(self):
        super().__init__()
        self.dist = torch.distributions.Normal(0, 1)
        self.dist.loc = self.dist.loc.cuda()
        self.dist.scale = self.dist.scale.cuda()
        self.encoder = Encoder()
        self.decoder = Decoder()

      def forward(self, x):
        mu, std = self.encoder(x)
        z = mu + std * self.dist.sample(mu.shape)
        x = self.decoder(z)
        self.kl = (std**2 + mu**2 - torch.log(std) - 1).mean()
        return x        
## Regularizing the latent space
If we would just train the VAE with, for example, a mean-square error loss on the reconstructed image (compared to the original one), the optimal strategy for the VAE would be to zero the standard deviation $\sigma$ as that would allow to pass deterministic information from the encoder to the decoder via the $\mu$ part of the encoding. This, however, would yield a highly unstructured latent space. We can regularize the latent space by forcing the VAE to assume a certain (Gaussian in this case) distribution. The gaussian would be fully defined by $\mu$ and $\sigma$ and we calculate the Kullback-Leibler divergence between the actual and the target distribution. For a Gaussian with $\mu=0$ and $\sigma=1$, this KL-divergence is given by

    self.kl = (std**2 + mu**2 - torch.log(std) - 1).mean()

We add this term to the overall loss.
## Perceptual and variational loss
It is well [known](https://arxiv.org/pdf/2001.03444.pdf) that a pure MSE (mean-squared error) loss on the resulting reconstructed data vs the original data does yield blurry samples. This is due to  

## Interpolation in the latent space
Since the latent space is properly regularized, it is possible to decode interpolated samples within that space to smoothly morph samples into each other. The below image is an example of this interpolation in the latent space.

![Latent Space](latent_interpolation.png)

