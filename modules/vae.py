import os
import pickle
import torch
import torch.nn as nn
from building_modules import Encoder, Decoder


class VAE(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels,
                 channel_multipliers,
                 num_res_blocks,
                 latent_dim,
                 device,
                 img_size=256):
        """Variational Autoencoder.

        Parameters:
        -----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        base_channels : int
            Number of base channels.
        channel_multipliers : list of ints
            Channel multipliers.
        num_res_blocks : int
            Number of residual blocks.
        latent_dim : int
            Latent dimension.
        img_size : int
            Image size.
        """
        
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.device = device

        self.encoder = Encoder(in_channels=in_channels,
                               base_channels=base_channels,
                               channel_multipliers=channel_multipliers,
                               num_res_blocks=num_res_blocks)
        
        self.decoder = Decoder(out_channels=out_channels,
                               base_channels=base_channels,
                               channel_multipliers=channel_multipliers,
                               num_res_blocks=num_res_blocks)
        """
        self.latent_conv = nn.Conv2d(in_channels=base_channels * channel_multipliers[-1],
                                     out_channels=4,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        
        self.decoder_input = nn.Conv2d(in_channels=2,
                                       out_channels=base_channels * channel_multipliers[-1],
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
        """
        # Calculate the number of features in the linear layer
        self.shape_before_flatten = (base_channels * channel_multipliers[-1], 
                                     img_size // 2 ** len(channel_multipliers), 
                                     img_size // 2 ** len(channel_multipliers))
        self.num_dense_features = int(torch.prod(torch.Tensor([self.shape_before_flatten])).item())

        self.mean = nn.Linear(in_features=self.num_dense_features, 
                              out_features=latent_dim)
        
        self.log_var = nn.Linear(in_features=self.num_dense_features, 
                                 out_features=latent_dim)
        
        self.decoder_input = nn.Linear(in_features=latent_dim, 
                                       out_features=self.num_dense_features)
        
        self.num_parameters = self._calculate_num_parameters()
        
    
    def reparameterize(self, mean, log_var):
        """Reparameterization trick.
        
        Parameters:
        -----------
        mean : torch.Tensor
            Mean tensor.
        log_var : torch.Tensor
            Log variance tensor.
        
        Returns:
        --------
        z : torch.Tensor
            Sampled latent tensor.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(self.device)
        z = mean + eps * std
        return z
    


    def forward(self, x):

        y = self.encoder(x) # The output still have convolutional structure

        y = y.view(-1, self.num_dense_features) # Flatten the output

        mean = self.mean(y)
        log_var = self.log_var(y)
        
        z = self.reparameterize(mean, log_var)

        z = self.decoder_input(z)

        z = z.view(-1, *self.shape_before_flatten) # Reshape the tensor

        x_pred = self.decoder(z)

        return x_pred, mean, log_var
    

    def sample(self, num_samples, binary=True):
        """Sample from the latent space.
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to generate.
        binary : bool
            If True, the output is binarized.
        
        Returns:
        --------
        samples : torch.Tensor
            Samples from the latent space.
        """
        samples = []
        for i in range(num_samples):
            with torch.no_grad():
                condition = True
                while condition:
                    z = torch.randn(1, self.latent_dim).to(self.device)
                    z = self.decoder_input(z)
                    z = z.view(-1, *self.shape_before_flatten)
                    sample = self.decoder(z).squeeze(0)
                    if binary:
                        sample = (sample > 0.5).float()
                    if torch.sum(sample) > 100:
                        condition = False
                samples.append(sample)
        samples = torch.stack(samples)
        return samples



    def _calculate_num_parameters(self):
        """Calculate the number of model parameters."""
        num_parameters = 0
        for param in self.parameters():
            num_parameters += param.numel()
        return num_parameters
    

    def save_model(self, save_folder):
        """Save the parameters and the model state_dict
        
        Parameters:
        ----------
        save_folder: str
            Folder to save the model
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        param_file = os.path.join(save_folder, 'VAEParameters.pkl')
        parameters = [self.in_channels,
                      self.out_channels,
                      self.base_channels,
                      self.channel_multipliers,
                      self.num_res_blocks,
                      self.latent_dim,
                      self.device,
                      self.img_size]
        with open(param_file, 'wb') as f:
            pickle.dump(parameters, f)
    
        model_file = os.path.join(save_folder, 'VAEModel.pt')
        torch.save(self.state_dict(), model_file)


    
    @staticmethod
    def save_history(history, save_folder):
        """Save the training history
        
        Parameters
        ----------
        history : dict
            Training history.
        save_folder : str
            Path to the folder where to save the training and validation history.
            
        Returns
        -------
        None."""
        filename = os.path.join(save_folder, 'vae_history.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(history, f)
    


    @classmethod
    def load_model(cls, save_folder, swa_version=False):
        """Load the parameters and the model state_dict
        
        Parameters:
        ----------
        save_folder: str
            Folder to load the model from
        """
        param_file = os.path.join(save_folder, 'VAEParameters.pkl') 
        with open(param_file, 'rb') as f:
            parameters = pickle.load(f)
        
        model = cls(*parameters)

        if swa_version:
            model_file = os.path.join(save_folder, 'SWAModel.pt')
        else:
            model_file = os.path.join(save_folder, 'VAEModel.pt')
        
        model.load_state_dict(torch.load(model_file, map_location='cuda:0'))
    
        return model
    


    @staticmethod
    def load_history(save_folder):
        """Load the training history
            
        Parameters
        ----------
        save_folder : str
            Path to the folder where the training history is saved.
        
        Returns
        -------
        history : dict
            Training and validation history.
        """
        history_file = os.path.join(save_folder, 'diffusion_history.pkl')
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        return history
    







class AE(nn.Module):

    def __init__(self, device):

        super(AE, self).__init__()

        self.l1 = nn.Linear(256*256, 2048)
        self.l2 = nn.Linear(2048, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, 256)

        self.l5 = nn.Linear(256, 512)
        self.l6 = nn.Linear(512, 1024)
        self.l7 = nn.Linear(1024, 2048)
        self.l8 = nn.Linear(2048, 256*256)

        self.num_parameters = self._calculate_num_parameters()
        self.device = device

    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)

        return nn.Sigmoid()(x)


    def _calculate_num_parameters(self):
        """Calculate the number of model parameters."""
        num_parameters = 0
        for param in self.parameters():
            num_parameters += param.numel()
        return num_parameters
    


        

        
    

