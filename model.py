import torch
from torch import nn
import torch.nn.functional as F

class DarkNet19(nn.Module):
    def __init__(self,):
        super(DarkNet19,self).__init__()
        self.block1 = nn.Sequential(
                nn.Conv2d(3,32,3,1,1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(32,64,3,1,1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(64,128,3,1,1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128,64,1,1,0),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(64,128,3,1,1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(128,256,3,1,1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256,128,1,1,0),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128,256,3,1,1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(256,512,3,1,1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.Conv2d(512,256,1,1,0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256,512,3,1,1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.Conv2d(512,256,1,1,0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256,512,3,1,1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                )
        self.block2 = nn.Sequential(
                nn.MaxPool2d(2,2),
                nn.Conv2d(512,1024,3,1,1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(),
                nn.Conv2d(1024,512,1,1,0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.Conv2d(512,1024,3,1,1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(),
                nn.Conv2d(1024,512,1,1,0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.Conv2d(512,1024,3,1,1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(),
                )
        self.yolo = nn.Conv2d(2048+1024,1000,1,1,0)
    def reorg(self, x, stride_h=2, stride_w=2):
        b, c, h, w = x.size()
        _h, _w, = h//stride_h, w//stride_h
        x = x.view(b, c, _h, stride_h, _w, stride_w).transpose(3,4).contiguous()
        x = x.view(b, c, _h*_w, stride_h*stride_w).transpose(2,3).contiguous()
        x = x.view(b, c, stride_h* stride_w, _h, _w).transpose(1,2).contiguous()
        x = x.view(b, -1, _h, _w)
        return x

    def forward(self, x):
        img_size = [ x.size(2), x.size(3)]
        x = self.block1(x)
        x1 = x
        x = self.block2(x)
        x2 = x
        x = torch.cat([x2, self.reorg(x1)], dim=1)
        x = self.yolo(x)
        x = F.avg_pool2d(x, (img_size[0]/32, img_size[1]/32))
        return x


if __name__ == "__main__":
    net = DarkNet19()
    print net
    a = torch.rand(4,3,416,416)
    print net(a).shape

