import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor as Tensor
from typing import List, Any

from abc import abstractmethod


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):
    def __init__(self,
                 input_size: int,
                 latent_size: int,
                 hidden_sizes: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_size

        modules = []
        if hidden_sizes is None:
            hidden_sizes = [32, 64, 128, 256, 512]

        # Build Encoder
        in_features = input_size
        for out_features in hidden_sizes:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.LeakyReLU())
            )
            in_features = out_features

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_sizes[-1], latent_size)
        self.fc_var = nn.Linear(hidden_sizes[-1], latent_size)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_size, hidden_sizes[-1])

        hidden_sizes.reverse()

        for i in range(len(hidden_sizes) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.BatchNorm1d(hidden_sizes[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
            nn.BatchNorm1d(hidden_sizes[-1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[-1], input_size),
            nn.Sigmoid(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the input data space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input sample x, returns the reconstructed sample
        :param x: (Tensor) [B x C]
        :return: (Tensor) [B x C]
        """
        return self.forward(x)[0]


class ConditionalVAE(BaseVAE):

    def __init__(self,
                 input_size: int,
                 latent_size: int,
                 hidden_sizes: List = None,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_size

        modules = []
        if hidden_sizes is None:
            hidden_sizes = [32, 64, 128, 256, 512]

        in_features = input_size + 1 # To account for the extra label channel
        
        # Build Encoder
        for h_dim in hidden_sizes:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(inplace=True))
            )
            in_features = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_sizes[-1], latent_size)
        self.fc_var = nn.Linear(hidden_sizes[-1], latent_size)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_size + 1, hidden_sizes[-1])

        hidden_sizes.reverse()

        for i in range(len(hidden_sizes) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.BatchNorm1d(hidden_sizes[i + 1]),
                    nn.LeakyReLU(inplace=True))
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_sizes[-1], hidden_sizes[-1]),
                            nn.BatchNorm1d(hidden_sizes[-1]),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(hidden_sizes[-1], input_size),
                            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels'].float()
        x = torch.cat([input, y], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, y], dim = 1)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = torch.cat([z, y], dim=1)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]