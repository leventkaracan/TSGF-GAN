import torch
import torch.nn as nn
from core.spectral_norm import spectral_norm as _spectral_norm
from torch.nn import functional as F
import math
import numpy as np
from guided_filter_pytorch.guided_filter import GuidedFilter
from torchvision import transforms as T
from torchvision.utils import save_image
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

############################## Discriminator  ##################################

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module



class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=True):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=spectral_norm, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.ReLU(True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                #nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                #norm_layer(nf), nn.ReLU(True)
                spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, bias=False), True),
                #nn.ReLU(True)
                nn.LeakyReLU(0.1, True)
                
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            #nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            #norm_layer(nf),
            #nn.ReLU(True)
            spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), True),
            #nn.ReLU(True)
            nn.LeakyReLU(0.1, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
            
############################## TSGF-GAN ##################################

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, eps = 1e-5, momentum = 0.1 ),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, eps = 1e-5, momentum = 0.1))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return x + self.main(x)


class TSGFGAN(BaseNetwork):
    def __init__(self, init_weights=True, dgf_r=10, dgf_eps=0.1):
        super(TSGFGAN, self).__init__()
        channel = 256
        stack_num = 8
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True)
            )
        self.res_blocks = nn.Sequential(ResidualBlock(256,256), ResidualBlock(256,256), ResidualBlock(256,256), ResidualBlock(256,256), ResidualBlock(256,256), ResidualBlock(256,256), ResidualBlock(256,256), ResidualBlock(256,256), ResidualBlock(256,256))
            
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channel*2, 128, kernel_size=4, stride=2, padding=1,  bias=False),
            nn.BatchNorm2d(128, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
            #nn.Tanh()
            )
            

        self.dgf_eps2 = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)

            
        self.guided_map_conv1 = nn.Conv2d(1,  64, 1)
        self.guided_map_relu1 = nn.ReLU(inplace=True)
        self.guided_map_conv2 = nn.Conv2d(64,  1, 1)

        self.guided_filter = GuidedFilter(dgf_r, self.dgf_eps2)
            

        if init_weights:
            self.init_weights()
            

    def forward(self, imgA, imgB):
        # extracting features
        t = 2
        img_AB = torch.cat((imgA.unsqueeze(0), imgB.unsqueeze(0)),dim=0)
        img_AB = img_AB.permute(1, 0, 2, 3, 4)
        b, t, c, h, w = img_AB.size()
        #masks = masks.view(b*t, 1, h, w)
        masks = torch.ones((b*t , 1, h,w)).to(device)
        enc_feat = self.encoder(img_AB.contiguous().view(b*t, c, h, w))
        enc_feat =self.res_blocks(enc_feat)
        _, c, h, w = enc_feat.size()
        enc_feat = enc_feat.view(b,t,c,h,w)
        enc_feat = torch.cat((enc_feat[:,0], enc_feat[:,1]), 1)
        output1 = self.decoder(enc_feat)
        imgGuide = output1


        #save_image(imgGuide, "/content/gdrive/MyDrive/MFIF_Projects/mfif2/guide.png")
        g = self.guided_map_relu1(self.guided_map_conv1(imgGuide))
        g = self.guided_map_conv2(g)
        print(self.dgf_eps2.item())
        output2 = self.guided_filter(g, output1)
        output2 = output2.clamp(0, 1)
        #save_image(output2, "/content/gdrive/MyDrive/MFIF_Projects/mfif2/output.png")
        #output2 = (output2 * torch.sigmoid (1000 * (output2 - 0.5)))
        #output2 = (output2>0.5).float()
        #output2 = F.interpolate(output2, (imgA.size(2),imgA.size(3)), mode='nearest')
        #print(output2.size())
        output3 = output2 * imgA + (1-output2) * imgB
        #output = torch.tanh(output)
        return output1, output2, output3
