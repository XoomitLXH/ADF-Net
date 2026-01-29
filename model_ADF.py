import torch
import torch.nn as nn
import torch.nn.functional as F

from modules_ADF import ResidualBlock, SELayer, AttentionFusionBlock, UpsampleBlock, HFEM_ADF

class ADF_Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super(ADF_Net, self).__init__()
        
                                                    
        self.encoder_init = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_blocks = nn.ModuleList()
        current_channels = base_channels
        for i in range(4):
            self.encoder_blocks.append(nn.Sequential(
                ResidualBlock(current_channels),
                nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(current_channels * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            current_channels *= 2
        
        self.bottleneck = ResidualBlock(current_channels)           

                                                                
                         
        self.head_a_gap = nn.AdaptiveAvgPool2d(1)
        self.head_a_fc1 = nn.Linear(current_channels, current_channels // 2)
        self.head_a_fc2 = nn.Linear(current_channels // 2, 3)            

                          
        self.head_t_decoder = nn.Sequential(
            ResidualBlock(current_channels),
            UpsampleBlock(current_channels, current_channels // 2),           
            ResidualBlock(current_channels // 2),
            UpsampleBlock(current_channels // 2, current_channels // 4),           
            nn.Conv2d(current_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()                
        )

                                                
        self.decoder_blocks = nn.ModuleList()
        decoder_channels = [base_channels * 8, base_channels * 4, base_channels * 2, base_channels]                      

        for i in range(4):
            if i == 0:                      
                D_input_channels = current_channels 
                S_H_channels = base_channels * (2**(3-i))                         
            else:
                D_input_channels = decoder_channels[i-1]
                S_H_channels = decoder_channels[i]              
            
            self.decoder_blocks.append(nn.Sequential(
                UpsampleBlock(D_input_channels, S_H_channels),          
                AttentionFusionBlock(S_H_channels, S_H_channels, S_H_channels, S_H_channels)                           
            ))

        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

                                           
                              
        self.hfem = HFEM_ADF(in_channels=in_channels, base_channels=base_channels, color_high_freq=True) 

    def forward(self, x, x_high):
             
        s_features = []
        s = self.encoder_init(x)
        s_features.append(s)     
        for i, block in enumerate(self.encoder_blocks):
            s = block(s)
            s_features.append(s)                       
        
        f_bottle = s_features.pop()                

              
        h_features = self.hfem(x_high)                    
        
                       
               
        a_feature = self.head_a_gap(f_bottle).squeeze(-1).squeeze(-1)
        a_feature = F.leaky_relu(self.head_a_fc1(a_feature), 0.2)
        A = self.head_a_fc2(a_feature)
                             
        A = torch.sigmoid(A)

                
        T = self.head_t_decoder(f_bottle)
        
             
        d = self.bottleneck(f_bottle)          
        
                                         
        for i in range(4):                            
            s_i = s_features[-(i+1)]                 
            h_i = h_features[-(i+1)]                 
            
            upsampled_d = self.decoder_blocks[i][0](d)             
            d = self.decoder_blocks[i][1](upsampled_d, s_i, h_i)        

        J_E = self.output_conv(d)

        return J_E, A, T
