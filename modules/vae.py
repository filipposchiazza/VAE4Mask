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

        self.encoder = Encoder(in_channels=in_channels,
                               base_channels=base_channels,
                               channel_multipliers=channel_multipliers,
                               num_res_blocks=num_res_blocks)
        
        self.decoder = Decoder(output_channels=out_channels,
                               base_channels=base_channels,
                               channel_multipliers=channel_multipliers,
                               num_res_blocks=num_res_blocks)
        
        # Calculate the number of features in the linear layer
        self.shape_before_flatten = (base_channels * channel_multipliers[-1], 
                                     img_size // 2 ** len(channel_multipliers), 
                                     img_size // 2 ** len(channel_multipliers))

        self.mean = nn.Linear(in_features=torch.mul(*self.shape_before_flatten),
                              out_features=latent_dim)
        
        self.log_var = nn.Linear(in_features=torch.mul(*self.shape_before_flatten),
                                 out_features=latent_dim)
        
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
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    


    def forward(self, x):

        y = self.encoder(x) # The output still have convolutional structure

        y = y.view(-1, self.linear_in_features) # Flatten the output

        mean = self.mean(y)
        log_var = self.log_var(y)
        
        z = self.reparameterize(mean, log_var)

        z = z.view(-1, *self.shape_before_flatten) # Reshape the tensor

        x_pred = self.decoder(z)

        return x_pred, mean, log_var



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
    def load_model(cls, save_folder):
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
        

        
    

