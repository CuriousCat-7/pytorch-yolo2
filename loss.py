import torch
import torch.nn.functional as F
from utils import bbox_iou_new

def yolo_loss(pre, gt, num_anchors, w_anchors, h_anchors):
    B, C, W, H = pre.size()
    
    pre.view(B, num_anchors,-1, W, H).contiguous()
    prebboxes = pre[:,:,:4,:,:] # bx5x4xWxH tx, ty, tw, th
    preconf = pre[:,:,4:5,:,:]
    preclasses = pre[:,:,5:,:,:]
    # get bx, by
    grid = torch.range(0,W-1).div(W)
    prebboxes[:,:,:2,:,:] = F.sigmoid(prebboxes[:,:,:2,:,:])
    prebboxes[:,:,0:1,:,:] = prebboxes[:,:,0:1,:,:] + grid.view(1,1,1,-1,1) # 1x1x1x13x1
    prebboxes[:,:,1:2,:,:] = prebboxes[:,:,1:2,:,:] + grid.view(1,1,1,1,-1) # 1x1x1x1x13
    # get bw, bh
    prebboxes[:,:,2:,:,:] = torch.exp(prebboxes[:,:,2:,:,:])
    prebboxes[:,:,2:3,:,:] = prebboxes[:,:,2:3,:,:].mul(w_anchors.view(1,num_anchors,1,1,1))
    prebboxes[:,:,3:4,:,:] = prebboxes[:,:,3:4,:,:].mul(h_anchors.view(1,num_anchors,1,1,1))
    
    # loss prepare
    conf_mask  = torch.ones(B, A, H, W) * noobject_scale
    coord_mask = torch.zeros(B, A, H, W)
    cls_mask   = torch.zeros(B, A, H, W)
    tx         = torch.zeros(B, A, H, W) 
    ty         = torch.zeros(B, A, H, W) 
    tw         = torch.zeros(B, A, H, W) 
    th         = torch.zeros(B, A, H, W) 
    tconf      = torch.zeros(B, A, H, W)
    tcls       = torch.zeros(B, A, H, W) 

    for b in xrange(B):
        for t in xrange(50):
            if gt[b, t*5+1] == 0
                break

            gx = gt[b, t*5 + 1]
            gy = gt[b, t*5 + 2]
            gi = int(gx)
            gj = int(gy)
            bboxs = prebboxes[b,:,:,gi,gj] # num_anchors x 4
            gtbbox = gt[b, t*5+1: (t+1)*5] # 4
            ious = torch.zeros(num_anchors)
            if pre.is_cuda:
                ious = ious.cuda()
            for a in xrange(num_anchors):
                bbox = bboxs[a]
                ious[a] = bbox_iou_new(bbox, gtbbox)
            val, idx = ious.topk(k=1)




