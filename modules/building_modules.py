import torch
import torch.nn as nn  
import torch.nn.functional as F



class ResidualBlock(nn.Module):

    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_groups=8,
                 activation_fn=F.silu):
        """Residual block with group normalization.
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        num_groups : int
            Number of groups for group normalization. Default is 8.
        activation_fn : torch activation function
            Activation function to use. Default is F.silu.
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.activation_fn = activation_fn
        
        super(ResidualBlock, self).__init__()

        self.group_norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, 
                               out_channels, 
                               kernel_size=3, 
                               padding=1)
        
        self.group_norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, 
                               out_channels, 
                               kernel_size=3, 
                               padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
            
        else:
            self.residual_layer = nn.Conv2d(in_channels, 
                                            out_channels, 
                                            kernel_size=1, 
                                            padding = 0)
            
    
    def forward(self,x):
        
        res = self.residual_layer(x)
        
        x = self.group_norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        
        x = self.group_norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        
        return x + res
    

class DownSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """Downsampling layer.
        
        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        """
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=3,
                              stride=2,
                              padding=1)
        
    def forward(self, inputs):
        return self.conv(inputs)


class UpSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """Upsampling layer.
        
        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        """
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1)
        
    def forward(self, x):
        return self.deconv(x)
    


class Encoder(nn.Module):

    def __init__(self,
                 input_channels,
                 base_channels,
                 channel_multipliers,
                 num_res_blocks):
        """Encoder module.
        
        Parameters:
        -----------
        input_channels : int
            Number of input channels.
        base_channels : int
            Number of base channels.
        channel_multipliers : list
            List of channel multipliers.
        num_res_blocks : list   
            List of number of residual blocks.
        """

        super(Encoder, self).__init__()

        self.input_channels = input_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks

        # First convolution to reduce the image dimensionality
        self.conv0 = nn.Conv2d(in_channels=input_channels,
                               out_channels=base_channels,
                               kernel_size=2,
                               stride=2,
                               padding=0)
        
        # DownBlock
        self.downblock = nn.ModuleList()
        for i in range(len(self.channel_multipliers)):
            ch = self.base_channels * self.channel_multipliers[i]

            for _ in range(self.num_res_blocks[i]):
                self.downblock.append(ResidualBlock(in_channels=ch,
                                                    out_channels=ch,
                                                    num_groups=16))
            
            if i < len(self.channel_multipliers) - 1:
                ch_next = self.base_channels * self.channel_multipliers[i+1]
                self.downblock.append(DownSample(in_channels=ch,
                                                 out_channels=ch_next))
                

    def forward(self, x):
        
        x = self.conv0(x)
        
        for layer in self.downblock:
            x = layer(x)
        
        return x
    


class Decoder(nn.Module):
    
        def __init__(self,
                     output_channels,
                     base_channels,
                     channel_multipliers,
                     num_res_blocks):
            """Decoder module.

            Parameters:
            -----------
            output_channels : int
                Number of output channels.
            base_channels : int
                Number of base channels.
            channel_multipliers : list  
                List of channel multipliers.
            num_res_blocks : list
                List of number of residual blocks.
            """
            
            super(Decoder, self).__init__()
    
            self.output_channels = output_channels
            self.base_channels = base_channels
            self.channel_multipliers = channel_multipliers
            self.num_res_blocks = num_res_blocks
    
            # UpBlock
            self.upblock = nn.ModuleList()
            for i in range(len(self.channel_multipliers)):
                ch = self.base_channels * list(reversed(self.channel_multipliers))[i]
                
                for _ in range(list(reversed(self.num_res_blocks))[i]):
                    self.upblock.append(ResidualBlock(in_channels=ch,
                                                      out_channels=ch,
                                                      num_groups=16))
                
                if i < len(self.channel_multipliers) - 1:
                    ch_next = self.base_channels * list(reversed(self.channel_multipliers))[i+1]
                    self.upblock.append(UpSample(in_channels=ch,
                                                 out_channels=ch_next))
            
            # Last convolution to increase the image dimensionality
            self.conv_out = nn.ConvTranspose2d(in_channels=self.base_channels,
                                               out_channels=self.output_channels,
                                               kernel_size=2,
                                               stride=2,
                                               padding=0)
            
            self.sigmoid = nn.Sigmoid()
    
        def forward(self, x):
            
            for layer in self.upblock:
                x = layer(x)
            
            x = self.conv_out(x)
            x = self.sigmoid(x)
            
            return x
    