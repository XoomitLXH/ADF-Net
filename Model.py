import torch
from torch import nn
import torch.nn.functional as F
import functools

from Model_util import PALayer, ConvGroups, FE_Block, Fusion_Block, ResnetBlock, ConvBlock, CALayer, SKConv
from Parameter_test import atp_cal, Dense


class fusion_h(nn.Module):
    def __init__(self, dim=3, block_depth=3):
        super(fusion_h, self).__init__()
        self.sig = nn.Sigmoid()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv_fuse = nn.Conv2d(dim, dim, 1)

    def forward(self, x, y):
        x = self.sig(x)
        y = self.conv1(y)
        return self.conv_fuse(x * y)


class Conv1x1(nn.Module):
    def __init__(self, inc=3, outc=3):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inc, outc, 1)

    def forward(self, x):
        return self.conv(x)


class Base_Model(nn.Module):
    def __init__(self, ngf=64, bn=False):
        super(Base_Model, self).__init__()
             
        self.down1 = ResnetBlock(3, first=True)

        self.down2 = ResnetBlock(ngf, levels=2)

        self.down3 = ResnetBlock(ngf * 2, levels=2, bn=bn)

        self.down1_high = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(3, ngf, kernel_size=7, padding=0),
                                        nn.InstanceNorm2d(ngf),
                                        nn.ReLU(True))

        self.down2_high = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf * 2),
                                        nn.ReLU(True))

        self.down3_high = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf * 4),
                                        nn.ReLU(True))

        self.res = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True),
            CALayer(ngf * 4),
            PALayer(ngf * 4)
        )

        self.res_atp = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True),
            CALayer(ngf * 4),
            PALayer(ngf * 4)
        )

        self.res_tran = nn.Sequential(
            ResnetBlock(ngf * 4, levels=6, down=False, bn=True),
            CALayer(ngf * 4),
            PALayer(ngf * 4)
        )

        self.fusion_layer = nn.ModuleList([fusion_h(dim=2 ** i * ngf) for i in range(0, 3)])
        self.skfusion = SKConv(features=2 ** 3 * ngf)
        self.conv1 = Conv1x1(inc=2 ** 3 * ngf, outc=2 ** (3 - 1) * ngf)
             

        self.up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(ngf * 2) if not bn else nn.BatchNorm2d(ngf * 2),
            CALayer(ngf * 2),
            PALayer(ngf * 2))

        self.up2 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            CALayer(ngf),
            PALayer(ngf))

        self.up3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
            nn.Tanh())

        self.info_up1 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(ngf * 2) if not bn else nn.BatchNorm2d(ngf * 2, eps=1e-5),
        )

        self.info_up2 = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf)                                                 
        )

        self.fam1 = FE_Block(ngf, ngf)
        self.fam2 = FE_Block(ngf, ngf * 2)
        self.fam3 = FE_Block(ngf * 2, ngf * 4)

        self.att1 = Fusion_Block(ngf)
        self.att2 = Fusion_Block(ngf * 2)
        self.att3 = Fusion_Block(ngf * 4, bn=bn)

        self.merge2 = nn.Sequential(
            ConvBlock(ngf * 2, ngf * 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        self.merge3 = nn.Sequential(
            ConvBlock(ngf, ngf, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=False)
        )

        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.atp = atp_cal()
        self.tran = Dense()

    def forward(self, hazy, high):
                                                                     
                               

        x_down1_high = self.down1_high(high)                      

        x_down2_high = self.down2_high(x_down1_high)                       

        x_down3_high = self.down3_high(x_down2_high)                     

                                                           

        x_down1 = self.down1(hazy)                               

        att1 = self.att1(x_down1_high, x_down1)

        x_down2 = self.down2(x_down1)                             
        att2 = self.att2(x_down2_high, x_down2)
        fuse2 = self.fam2(att1, att2)

        x_down3 = self.down3(x_down2)                           
        att3 = self.att3(x_down3_high, x_down3)
        fuse3 = self.fam3(fuse2, att3)

                                                      

        x6 = self.res(x_down3)

        fuse_up2 = self.info_up1(fuse3)
        fuse_up2 = self.merge2(fuse_up2 + x_down2)

        fuse_up3 = self.info_up2(fuse_up2)
        fuse_up3 = self.merge3(fuse_up3 + x_down1)

        x_up2 = self.up1(x6 + fuse3)

        x_up3 = self.up2(x_up2 + fuse_up2)

        x_up4 = self.up3(x_up3 + fuse_up3)

                                                   

        x6_atp = self.res_atp(x_down3)
        atp = self.atp(x6_atp)

                                                    

        x6_tran = self.res_tran(x_down3)
        tran = self.tran(x6_tran, x_up4)

                                                                       

        zz = torch.abs((tran)) + (10 ** -10)     
        shape_out1 = atp.data.size()

        shape_out = shape_out1[2:4]
        if shape_out1[2] >= shape_out1[3]:
            atp = F.avg_pool2d(atp, shape_out1[3])
        else:
            atp = F.avg_pool2d(atp, shape_out1[2])
        atp = self.upsample(self.relu(atp), size=shape_out)

        haze = (x_up4 * zz) + atp * (1 - zz)
        dehaze = (hazy - atp) / zz + atp        

        return haze, dehaze, x_up4, tran, atp


if __name__ == '__main__':
    G = Base_Model()
    a = torch.randn(1, 3, 512, 768)
    b = torch.randn(1, 3, 512, 768)
    G(a, b)


class Discriminator(nn.Module):

    def __init__(self, inp=3, out=1):

        super(Discriminator, self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        model = [
            nn.Conv2d(inp, 64, kernel_size=4, stride=2, padding=1),                    
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, out, kernel_size=4, stride=1, padding=1)                                          
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        result = self.model(input)

        return result

                            
                                                    
                                          
                                                                                                    
                                                                                                           
 
                           
                        
                  
