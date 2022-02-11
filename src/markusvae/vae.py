import numpy as np
import torch
import torch.nn as nn
import os
from collections import namedtuple
from torch.functional import Tensor
from typing import List

from MarkusVAE.utils import conv_sizes
Sizes = namedtuple('Sizes', 'channel, height, width')

class Customizations:

    def __init__(self, ichannel, iheight, iwidth, hchannels, hlatent):
        self.ichannel = ichannel      # int type
        self.iheight = iheight      # int
        self.iwidth = iwidth        # int
        self.hchannels = hchannels    # []int
        self.hlatent = hlatent       # int
        
    def get_input_sizes(self):
        "return custom class Sizes"
        return Sizes(self.ichannel, self.iheight, self.iwidth)
        
    def get_hidden_channels(self):
        "return list of int"
        return self.hchannels
        

class VAE(nn.Module):
    def __init__(self, customs=None):
        """VAE architecture hardcoded for 3x32x32 input.
        If you want different input size you should update:
        1. in_channels variable
        2. """
        super(VAE, self).__init__()
        self.iSizes = Sizes(3,32,32) #customs.get_input_sizes()

        
        self.hidden_channels = [64,128,256]# customs.get_hidden_channels()

        self._depth = len(self.hidden_channels)
        self._kernel = (3, 3)
        self._stride = (2, 2)
        self._padding = (1, 1)

        self.sizes = conv_sizes(N=self._depth, H=iSizes.height, W=iSizes.width, 
                kernel=self._kernel, stride = self._stride, padding=self._padding)

        self.latent_dim = latent_dim
        self.h_dim = np.prod(self.sizes[-1]) * self.hidden_channels[-1]
        self.batch_size = None  # will be defined by input data 


        

        # Encoder
        emodules = []
        tmp_channel = iSizes.channel
        for hchannel in self.hidden_channels:
          emodules.append(
            nn.Sequential(
              nn.Conv2d(tmp_channel, hchannel, kernel_size=self._kernel, 
                        stride=self._stride, padding=self._padding),
              nn.ReLU(),PrintShape(),
              )
          )
          tmp_channel = hchannel
        
        emodules.extend([Flatten(),PrintShape(),])
        self.encoder = nn.Sequential(*emodules)
        
        self.mu_fc = nn.Linear(self.h_dim, latent_dim)
        self.logvar_fc = nn.Linear(self.h_dim, latent_dim)

        # Decoder
        #  latent_dim Linear -> h_dimp Linear -> CxHxW Tensor
        dmodules =[nn.Linear(latent_dim, self.h_dim)]
        dmodules.extend([Review(channel=self.hidden_channels[-1], sizes=self.sizes[-1]),PrintShape()])
         
        tmp_channel = self.hidden_channels[-1]
        for hchannel in list(reversed(self.hidden_channels))[1:]:
            dmodules.append(
              nn.Sequential(
                nn.ConvTranspose2d(tmp_channel, hchannel, kernel_size=self._kernel,  
                stride=self._stride, padding=self._padding, output_padding=self._padding),
                PrintExample(),nn.ReLU(),PrintShape(),
                )
            )
            tmp_channel = hchannel

        # Final decoder layer
        dmodules.append(
              nn.Sequential(
                nn.ConvTranspose2d(tmp_channel, iSizes.channel, kernel_size=self._kernel,  
                stride=self._stride, padding=self._padding, output_padding=self._padding),
                nn.Sigmoid(), PrintShape(),
                )
            )

        self.decoder = nn.Sequential(*dmodules)
        
    def encode(self, input: Tensor) -> List[Tensor]:
        features = self.encoder(input)
        mu, logvar = self.mu_fc(features), self.logvar_fc(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=DEVICE)
        z = mu + std * esp
        return z
    
    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)


    def forward(self, input: Tensor) -> List[Tensor]:
        assert input.shape[-3:] == torch.Size([self.iSizes.channel, self.iSizes.height, self.iSizes.width])
        mu, logvar = self.encode(input)
        sample = self.reparameterize(mu, logvar)
        restore = self.decode(sample)
        return restore

    def fit(self, train_loader, optimizer, criterion, epochs, batch_size, resume=False):

      if os.path.exists(MODEL_FILE) and not resume:
        print(f'Loading model from {MODEL_FILE}...')
        self.load_state_dict(torch.load(MODEL_FILE))
      else:
        if os.path.exists(MODEL_FILE) and resume:
          print(f'Loading model from {MODEL_FILE} and ...')
          self.load_state_dict(torch.load(MODEL_FILE))
        print('Training the model...')
        for epoch in range(epochs):
    
          for batch in train_loader:
            X, y = batch[0].to(DEVICE), batch[1].to(DEVICE)   # B-3-32-32, B
            optimizer.zero_grad()

            assert X.shape == (batch_size, self.iSizes.channel, self.iSizes.height, self.iSizes.width), f'wrong input shape x={X.shape}'
            answer = self.forward(X) 
            assert answer.shape == (batch_size, self.iSizes.channel, self.iSizes.height, self.iSizes.width), f'wrong answer shape={answer.shape}'
            loss = criterion(X, answer)
            loss.backward()

            optimizer.step()

          print(f'epoch={epoch}. loss={loss}')

        torch.save(self.state_dict(), f=MODEL_FILE)

