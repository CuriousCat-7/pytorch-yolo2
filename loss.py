import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import bbox_iou_new
from region_loss import build_targets
import pdb

def yolo_loss(pre, gt, anchors, cuda):
    # pre B, C, H, W
    # C = A * (4+1+num_classes)
    B, num_Channls, H, W = pre.size()
    anchors_list = anchors
    A = len(anchors)/2
    C = int(num_Channls/A - 5) # C = num_classes
    anchors = torch.Tensor(anchors).view(2,-1)
    if cuda:
        anchors = anchors.cuda()
    w_anchors = anchors[0]
    h_anchors = anchors[1]

    pre = pre.view(B,A ,-1, H, W)
    prebboxes = pre[:,:,:4,:,:] # bx5x4xHxW tx, ty, th, tw
    preconf = pre[:,:,4,:,:]
    preclasses = pre[:,:,5:,:,:]

    # get bx, by
    grid = torch.range(0,W-1).div(W)
    if cuda:
        grid = grid.cuda()
    prebboxes[:,:,:2,:,:] = F.sigmoid(prebboxes[:,:,:2,:,:].clone())
    x = prebboxes[:,:,0,:,:]
    y = prebboxes[:,:,1,:,:]
    w = prebboxes[:,:,2,:,:]
    h = prebboxes[:,:,3,:,:]
    conf = F.sigmoid(preconf)
    prebboxes[:,:,0:1,:,:] = prebboxes[:,:,0:1,:,:].clone() + grid.view(1,1,1,-1,1) # 1x1x1x13x1
    prebboxes[:,:,1:2,:,:] = prebboxes[:,:,1:2,:,:].clone() + grid.view(1,1,1,1,-1) # 1x1x1x1x13
    # get bh, bw
    prebboxes[:,:,2:,:,:] = torch.exp(prebboxes[:,:,2:,:,:].clone())
    prebboxes[:,:,2:3,:,:] = prebboxes[:,:,2:3,:,:].clone().mul(h_anchors.view(1,A,1,1,1))
    prebboxes[:,:,3:4,:,:] = prebboxes[:,:,3:4,:,:].clone().mul(w_anchors.view(1,A,1,1,1))
    # from xiao
    prebboxes = prebboxes.transpose(0,1).contiguous().view(4,-1).transpose(0,1).contiguous()
    prebboxes = prebboxes.cpu()
    oobject_scale = 1
    object_scale = 5
    thresh = 0.6
    seen = 0
    coord_scale = 1
    class_scale = 1

    nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = \
    build_targets(prebboxes, gt.data, anchors_list, A, C, H, W,\
    oobject_scale, object_scale, thresh, seen)


    tx, ty, tw, th, tconf, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), tconf.cuda(), tcls.cuda()
    cls_mask = cls_mask.byte()
    tcls = tcls.view(-1)[cls_mask.view(-1)].long()

    coord_mask = coord_mask.cuda()
    conf_mask  = conf_mask.cuda().sqrt()
    cls_mask   = cls_mask.view(-1, 1).repeat(1,C).cuda()

    preclasses = preclasses.contiguous().view(-1)[cls_mask.view(-1)].view(-1,C)

    #loss_x = coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
    #loss_y = coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
    #loss_w = coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
    #loss_h = coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
    loss_x = coord_scale * F.mse_loss(x*coord_mask, tx*coord_mask, size_average=False)/2.0
    loss_y = coord_scale * F.mse_loss(y*coord_mask, ty*coord_mask, size_average=False)/2.0
    loss_w = coord_scale * F.mse_loss(w*coord_mask, tw*coord_mask, size_average=False)/2.0
    loss_h = coord_scale * F.mse_loss(h*coord_mask, th*coord_mask, size_average=False)/2.0
    loss_conf = F.mse_loss(conf*conf_mask, tconf*conf_mask, size_average=False)/2.0
    loss_cls = class_scale * F.cross_entropy(preclasses, tcls,size_average=False)
    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
    return loss

if __name__ == "__main__":
    from model import DarkNet19
    net = DarkNet19()
    print net
    from dataset import listDataset
    from torchvision import transforms
    filepath = '/data/limingyao/data/VOC/voc_train.txt'
    dataset = listDataset(filepath, shape=(416,416), transform=transforms.ToTensor(), train=True, batch_size=8, num_workers=8)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 8, shuffle=False, num_workers=8, pin_memory=True)
    idx, (img, label) = enumerate(train_loader).next()
    label = label.float()
    img = img.cuda()
    net.cuda()
    img = Variable(img)
    label= Variable(label)
    out = net(img)
    loss = yolo_loss(out, label, net.anchors, True)
    loss.backward()
    print loss
