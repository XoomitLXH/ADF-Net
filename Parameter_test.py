import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Model_util import CALayer, PALayer


def conv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
                         nn.AvgPool2d(kernel_size=2, stride=2))


def deconv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.UpsamplingNearest2d(scale_factor=2))


def blockUNet1(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class atp_cal(nn.Module):
    def __init__(self, output_nc=3, nf=256):
        super(atp_cal, self).__init__()
                            
        layer_idx = 0
        name = 'layer%d' % layer_idx
                                  
                                                                               
                            
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf, name, transposed=False, bn=True, relu=False, dropout=False)
                          
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf, nf, name, transposed=False, bn=True, relu=False, dropout=False)
                     
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf, nf, name, transposed=False, bn=True, relu=False, dropout=False)
                     
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf, nf, name, transposed=False, bn=True, relu=False, dropout=False)
                    
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf, nf, name, transposed=False, bn=True, relu=False, dropout=False)
                    
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer7 = blockUNet(nf, nf, name, transposed=False, bn=True, relu=False, dropout=False)
                         
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer8 = blockUNet(nf, nf, name, transposed=False, bn=True, relu=False, dropout=False)

                        
                    
        name = 'dlayer%d' % layer_idx
        d_inc = nf
        dlayer8 = blockUNet(nf, nf, name, transposed=True, bn=False, relu=True, dropout=True)

                                     
                    
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf
        dlayer7 = blockUNet(nf, nf, name, transposed=True, bn=True, relu=True, dropout=True)
                    
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer6 = blockUNet(nf, nf, name, transposed=True, bn=True, relu=True, dropout=True)
                    
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer5 = blockUNet(nf, nf, name, transposed=True, bn=True, relu=True, dropout=False)
                     
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer4 = blockUNet(nf, nf, name, transposed=True, bn=True, relu=True, dropout=False)
                     
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 4 * 2
        dlayer3 = blockUNet(nf, nf, name, transposed=True, bn=True, relu=True, dropout=False)
                     
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 2 * 2
        dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
                      
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        d_inc = nf * 2
        dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
        dlayer1.add_module('%s_tanh' % name, nn.LeakyReLU(0.2, inplace=True))

                              
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8
        self.ca1 = nn.Sequential(CALayer(nf), PALayer(nf))
        self.dlayer8 = dlayer8
        self.ca2 = nn.Sequential(CALayer(nf), PALayer(nf))
        self.dlayer7 = dlayer7
        self.ca3 = nn.Sequential(CALayer(nf), PALayer(nf))
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

        self.up1_atp = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(nf, nf // 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(nf // 2),
        )

        self.up2_atp = nn.Sequential(
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(nf // 2, nf // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(nf // 4),
        )

        self.up3_atp = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf // 4, 3, kernel_size=7, padding=0),
            nn.Tanh())

    def forward(self, x):
                          
                        

                                                
                                               
        out3 = self.layer3(x)                                
        out4 = self.layer4(out3)                             
        out5 = self.layer5(out4)                             
        out6 = self.layer6(out5)                             
        out7 = self.layer7(out6)                             
        out8 = self.layer8(out7)                             

        out8 = self.ca1(out8)
        dout8 = self.dlayer8(out8)                             
        dout8_out7 = dout8 + out7               
        dout8_out7 = self.ca2(dout8_out7)

        dout7 = self.dlayer7(dout8_out7)                        
        dout7_out6 = dout7 + out6               
        dout7_out6 = self.ca3(dout7_out6)

        dout6 = self.dlayer6(dout7_out6)                        
        dout6_out5 = dout6 + out5               
        dout5 = self.dlayer5(dout6_out5)                        
        dout5_out4 = dout5 + out4                 
        dout4 = self.dlayer4(dout5_out4)                         
        dout4_out3 = dout4 + out3                

        dout3 = self.dlayer3(dout4_out3)                         
        dout2 = self.up1_atp(dout3)                              
        dout1 = self.up2_atp(dout2)                             
        dout0 = self.up3_atp(dout1)                            
        return dout0


if __name__ == '__main__':
    x = torch.rand((3, 3, 256, 256))
    y = torch.rand((3, 3, 256, 256))
    c = torch.randn(3, 256, 64, 64)
    g = atp_cal()
    g(c)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)


class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()

                                             
        haze_class = models.densenet121(pretrained=True)

        self.conv0 = haze_class.features.conv0
        self.norm0 = haze_class.features.norm0
        self.relu0 = haze_class.features.relu0
        self.pool0 = haze_class.features.pool0

                                                       
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

                                                       
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

                                                       
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

                                                    

        self.dense_block4 = BottleneckBlock(512, 256)
        self.trans_block4 = TransitionBlock(768, 128)

                                                     
        self.pa1 = nn.Sequential(CALayer(384), PALayer(384))
        self.dense_block5 = BottleneckBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)

                                                      
        self.pa2 = nn.Sequential(CALayer(256), PALayer(256))
        self.dense_block6 = BottleneckBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)

                                                      
        self.pa3 = nn.Sequential(CALayer(64), PALayer(64))
        self.dense_block7 = BottleneckBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)

                     
                                                 
        self.pa4 = nn.Sequential(CALayer(32), PALayer(32))
        self.dense_block8 = BottleneckBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)

        self.conv_refin = nn.Conv2d(19, 20, 3, 1, 1)
        self.tanh = nn.Tanh()

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)       
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)       
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)       
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)       

        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
                                                                            

        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, haze):
                                 
                                                                

                                  
                                    
                                         
        x1 = self.trans_block1(x)

                                  
        x2 = self.trans_block2(self.dense_block2(x1))
                          

                                 
        x3 = self.trans_block3(self.dense_block3(x2))

                                                 

                                
        x4 = self.trans_block4(self.dense_block4(x3))
                           
        x42 = torch.cat([x4, x2], 1)

        x42 = self.pa1(x42)
                                  
        x5 = self.trans_block5(self.dense_block5(x42))
                          
        x52 = torch.cat([x5, x1], 1)
                                  

        x52 = self.pa2(x52)
        x6 = self.trans_block6(self.dense_block6(x52))
        x6 = self.pa3(x6)
                                    
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.pa4(x7)
                                      
        x8 = self.trans_block8(self.dense_block8(x7))

                         
                        
                          
        x8 = torch.cat([x8, haze], 1)

                         
                          
        x9 = self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
                          
        shape_out = shape_out[2:4]
                         
                           
                           
                           
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
                             
                             
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze


class Parameter(nn.Module):
    def __init__(self):
        super(Parameter, self).__init__()

        self.atp_est = G2(output_nc=3, nf=64)

        self.tran_dense = Dense()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.tanh = nn.Tanh()

        self.refine1 = nn.Conv2d(6, 20, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.threshold = nn.Threshold(0.1, 0.1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)       
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)       
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)       
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)       

        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)

        self.upsample = F.upsample_nearest

        self.batch1 = nn.BatchNorm2d(20)

    def forward(self, x, y):
        tran = self.tran_dense(x)
        atp = self.atp_est(x)

        zz = torch.abs((tran)) + (10 ** -10)     
        shape_out1 = atp.data.size()

        shape_out = shape_out1[2:4]
        atp = F.avg_pool2d(atp, shape_out1[2])
        atp = self.upsample(self.relu(atp), size=shape_out)

        haze = (y * zz) + atp * (1 - zz)
        dehaze = (x - atp) / zz + atp        

        return haze, dehaze


