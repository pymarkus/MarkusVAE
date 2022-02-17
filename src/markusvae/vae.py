import numpy as np
import torch
import torch.nn as nn
import os
from collections import namedtuple
from torch.functional import Tensor
from typing import List

from utils import conv_sizes, PrintShape, Flatten, Review
Sizes = namedtuple('Sizes', 'channel, height, width')

class Customs:

    def __init__(self, input_channel, input_height, input_width, hidden_channels, latent_dim, device, model_file=None):
        self.input_channel = input_channel      # int type
        self.input_height = input_height      # int
        self.input_width = input_width        # int
        self.hidden_channels = hidden_channels    # []int
        self.latent_dim = latent_dim       # int
        self.device = device
        self.model_file = model_file      # string
        
    def get_input_sizes(self):
        "return custom class Sizes"
        return Sizes(self.input_channel, self.input_height, self.input_width)
        
    def get_hidden_channels(self):
        "return list of int"
        return self.hidden_channels
    
    def get_latent_dim(self):
        return self.latent_dim
    
    def get_device(self):
        return self.device
        

class VAE(nn.Module):
    def __init__(self, customs):
        """VAE architecture hardcoded for 3x32x32 input.
        If you want different input size you should update:
        1. in_channels variable
        2. """
        super(VAE, self).__init__()
        self.input_sizes = customs.get_input_sizes()
        self.hidden_channels = customs.get_hidden_channels()
        self.latent_dim = customs.get_latent_dim()
        self.device = customs.get_device()
        self.customs = customs

        self._depth = len(self.hidden_channels)
        self._kernel = (3, 3)
        self._stride = (2, 2)
        self._padding = (1, 1)

        self.hidden_sizes = conv_sizes(N=self._depth, H=self.input_sizes.height, W=self.input_sizes.width,
                kernel=self._kernel, stride = self._stride, padding=self._padding)


        self.h_dim = np.prod(self.hidden_sizes[-1]) * self.hidden_channels[-1]
        self.batch_size = None  # will be defined by input data 


        

        # Encoder
        emodules = []
        tmp_channel = self.input_sizes.channel
        for hchannel in self.hidden_channels:
          emodules.append(
            nn.Sequential(
              nn.Conv2d(tmp_channel, hchannel, kernel_size=self._kernel, 
                        stride=self._stride, padding=self._padding),
              nn.ReLU(), PrintShape(),
              )
          )
          tmp_channel = hchannel
        
        emodules.extend([Flatten(),PrintShape(),])
        self.encoder = nn.Sequential(*emodules)
        
        self.mu_fc = nn.Linear(self.h_dim, self.latent_dim)
        self.logvar_fc = nn.Linear(self.h_dim, self.latent_dim)

        # Decoder
        #  latent_dim Linear -> h_dimp Linear -> CxHxW Tensor
        dmodules =[nn.Linear(self.latent_dim, self.h_dim)]
        dmodules.extend([Review(channel=self.hidden_channels[-1], sizes=self.hidden_sizes[-1]),PrintShape()])
         
        tmp_channel = self.hidden_channels[-1]
        for hchannel in list(reversed(self.hidden_channels))[1:]:
            dmodules.append(
              nn.Sequential(
                nn.ConvTranspose2d(tmp_channel, hchannel, kernel_size=self._kernel,  
                stride=self._stride, padding=self._padding, output_padding=self._padding),
                nn.ReLU(),PrintShape(),
                )
            )
            tmp_channel = hchannel

        # Final decoder layer
        dmodules.append(
              nn.Sequential(
                nn.ConvTranspose2d(tmp_channel, self.input_sizes.channel, kernel_size=self._kernel,
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
        esp = torch.randn(*mu.size(), device=self.device)
        z = mu + std * esp
        return z
    
    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)


    def forward(self, input: Tensor) -> List[Tensor]:
        assert input.shape[-3:] == torch.Size([self.input_sizes.channel, self.input_sizes.height, self.input_sizes.width])
        mu, logvar = self.encode(input)
        sample = self.reparameterize(mu, logvar)
        restore = self.decode(sample)
        return restore

    def fit(self, train_loader, optimizer, criterion, epochs, batch_size):
      model_file = self.customs.model_file
      if os.path.exists(model_file):
        print('Loading the model...')
        self.load_state_dict(torch.load(model_file))
      print('Training the model...')
      for epoch in range(epochs):
    
        for batch in train_loader:
          X, y = batch[0].to(self.device), batch[1].to(self.device)   # B-3-32-32, B
          optimizer.zero_grad()

          assert X.shape == (batch_size, 3, 32, 32), f'wrong input shape x={X.shape}'
          answer = self.forward(X) 
          assert answer.shape == (batch_size, 3, 32, 32), f'wrong answer shape={answer.shape}'
          loss = criterion(X, answer)
          loss.backward()

          optimizer.step()

        print(f'epoch={epoch}. loss={loss}')

      torch.save(self.state_dict(), f=MODEL_FILE)

