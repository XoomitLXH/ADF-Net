import torch
import torch.nn as nn
import torch.nn.functional as F

                      
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

                                
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

                                      
class AttentionFusionBlock(nn.Module):
    def __init__(self, channel_d, channel_s, channel_h, out_channels):
        super(AttentionFusionBlock, self).__init__()
                                       
                            
        self.conv_cat = nn.Conv2d(channel_d + channel_s + channel_h, out_channels, kernel_size=1, bias=False)
        self.se_layer = SELayer(out_channels)
        self.residual_block = ResidualBlock(out_channels)

    def forward(self, D, S, H):
                            
        f_concat = torch.cat([D, S, H], dim=1)
        f_concat = self.conv_cat(f_concat)
                                   
        f_weighted = self.se_layer(f_concat)
              
        out = self.residual_block(f_weighted)
        return out

                 
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

                               
                      
class HFEM_ADF(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, color_high_freq=True):
        super(HFEM_ADF, self).__init__()
        self.color_high_freq = color_high_freq              
        
                                   
        self.hfem_init = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
                               
        if self.color_high_freq:
            self.color_aware_conv = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(base_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
                                     
        self.hfem_down_blocks = nn.ModuleList()
        current_channels = base_channels
        for i in range(4):        
            self.hfem_down_blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(current_channels * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            current_channels *= 2

    def forward(self, x_high):
        h_features = []
        
                
        h = self.hfem_init(x_high)
        
                           
        if self.color_high_freq:
            h = self.color_aware_conv(h)
        
        h_features.append(h)     
        
                
        for i, block in enumerate(self.hfem_down_blocks):
            h = block(h)
            h_features.append(h)                    
        
                          
        f_bottle_h = h_features.pop()                                            
        return h_features                     
