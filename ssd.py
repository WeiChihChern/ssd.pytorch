import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes, batchNorm=False):
        super(SSD, self).__init__()
        self.phase     = phase
        self.num_classes = num_classes
        self.cfg       = voc
        print("Training Config (config.py):\n", voc)
        self.priorbox  = PriorBox(self.cfg)
        self.priors    = Variable(self.priorbox.forward(), requires_grad=False)
        self.size      = size

        # SSD network
        # self.vgg     = nn.ModuleList(base)
        self.jacinto = nn.ModuleList(base)
        self.extras  = nn.ModuleList(extras)
        self.loc  = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            # self.detect = Detect(num_classes, 0, 200, 0.01, 0.45) #old pytorch
            self.detect  = Detect() # new pytorch

    
    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """

        # jacinto -> extra (1x1 conv) -> (loc, conf) 1x1 conv 
        sources = list()
        extra   = list()
        loc     = list()
        conf    = list()

        ### Base network
        for k in range(19):
            x = self.jacinto[k](x)
        sources.append(x)
        
        for k in range(19, 33):
            x = self.jacinto[k](x)
        sources.append(x) 
        
        for k in range(33, 38): 
            x = self.jacinto[k](x)
            if k != 33:
                sources.append(x)

        ### Feeding 6 of basenet's output to extra layers (1x1 conv layers)
        for k in range(len(self.extras)):
            x = F.relu(self.extras[k](sources[k]))
            extra.append(x)

        ### Apply multibox head to source layers
        counter = 0
        for (x, l, c) in zip(extra, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc  = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45, # num_classes, bkg_label, top_k, conf_thresh, nms_thresh
            # PyTorch1.5.0 support new-style autograd function
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



def jacintoNet(printmodel=False):
    layers = []
    layers.append(nn.Conv2d(3, 32, kernel_size=5, padding=2, groups=1, stride=2))
    layers.append(nn.BatchNorm2d(32))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=4, stride=1))
    layers.append(nn.BatchNorm2d(32))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) #6

    layers.append(nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=1, stride=1))
    layers.append(nn.BatchNorm2d(64))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=4, stride=1))
    layers.append(nn.BatchNorm2d(64))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) #13
    
    layers.append(nn.Conv2d(64,  128, kernel_size=3, padding=1, groups=1, stride=1))
    layers.append(nn.BatchNorm2d(128))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=4, stride=1)) 
    layers.append(nn.BatchNorm2d(128))                                                 # feed to ctx_output1
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) #20
 
    layers.append(nn.Conv2d(128,  256, kernel_size=3, padding=1, groups=1, stride=1))
    layers.append(nn.BatchNorm2d(256))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(256,  256, kernel_size=3, padding=1, groups=4, stride=1)) 
    layers.append(nn.BatchNorm2d(256))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) #27

    layers.append(nn.Conv2d(256,  512, kernel_size=3, padding=1, groups=1, stride=1))
    layers.append(nn.BatchNorm2d(512))
    layers.append(nn.ReLU(inplace=True)) #30
    layers.append(nn.Conv2d(512,  512, kernel_size=3, padding=1, groups=4, stride=1)) 
    layers.append(nn.BatchNorm2d(512))   #32                                          # feed to ctx_output2
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))   # 34                       # feed to ctx_output3
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))                              # feed to ctx_output4
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))                              # feed to ctx_output5
    layers.append(nn.MaxPool2d(kernel_size=(1,3), stride=2))                          # feed to ctx_output6
    
    if printmodel == True:
        counter = 0
        for i in layers:
            print(counter, " | ", end="")
            print(i)
            counter += 1

    return layers


def jacinto_extras():
    layers = []
    out_channel = 256
    # 1x1 conv layers
    layers.append(nn.Conv2d(128, out_channel, kernel_size=1, stride=1, padding=0))
    layers.append(nn.Conv2d(512, out_channel, kernel_size=1, stride=1, padding=0))
    layers.append(nn.Conv2d(512, out_channel, kernel_size=1, stride=1, padding=0))
    layers.append(nn.Conv2d(512, out_channel, kernel_size=1, stride=1, padding=0))
    layers.append(nn.Conv2d(512, out_channel, kernel_size=1, stride=1, padding=0))
    layers.append(nn.Conv2d(512, out_channel, kernel_size=1, stride=1, padding=0))
    return layers



# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers





def multibox(jacinto, extra_layers, cfg, num_classes, use_batchNorm=False):
    """cfg is number of prior boxes' configuration, because output channel depends on this
        mbox = {
            '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
            '512': [],
        }
    """
    # # # For Debug
    # print("Base network: ")
    # counter = 0
    # for i in vgg:
    #     print('('+str(counter)+')', end=' | ')
    #     print(i)
    #     counter+=1
    # counter = 0
    # print("Extra layers: ")
    # for i in extra_layers:
    #     print('('+str(counter)+')', end=' | ')
    #     print(i)
    #     counter+=1

    loc_layers  = []
    conf_layers = []
    
    
    #loc layers
    for i in range(6):
        loc_layers += [nn.Conv2d(256, cfg[i] * 4, kernel_size=3, padding=1)]
    #conf layers
    for i in range(6):
        conf_layers+= [nn.Conv2d(256, cfg[i] * num_classes, kernel_size=3, padding=1)]

    return jacinto, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300' : [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512' : [],
    '320' : [4, 6, 6, 6, 4, 4],  # 768x320
}


def build_ssd(phase, size=300, num_classes=21, use_batchNorm=False):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # if size != 300:
    #     print("ERROR: You specified size " + repr(size) + ". However, " +
    #           "currently only SSD300 (size=300) is supported!")
    #     return
    
    ## VGG
    # base_, extras_, head_ = multibox(vgg(base[str(size)], 3, use_batchNorm),
    #                                  add_extras(extras[str(size)], 1024, use_batchNorm),
    #                                  mbox[str(size)], num_classes,
    #                                  use_batchNorm)

    ## Jacinto
    base_, extras_, head_ = multibox(jacintoNet(),
                                     jacinto_extras(),
                                     mbox[str(size)], num_classes,
                                     use_batchNorm)       

    return SSD(phase, size, base_, extras_, head_, num_classes, use_batchNorm)
