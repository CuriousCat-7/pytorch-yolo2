import torch
from model import DarkNet19
from region_loss import RegionLoss
from dataset import listDataset
from torchvision import transforms
import sys
# todo:
# acc
# load weigth from darknet
# understand region_loss
# multi-gpu

filepath = '/data/limingyao/data/VOC/voc_train.txt'
save_name = '/data/limingyao/data/models/yolo.t7'
dataset = listDataset(filepath, shape=(416,416), transform=transforms.ToTensor(), train=True, batch_size=8, num_workers=8)
train_loader = torch.utils.data.DataLoader(dataset, batch_size = 8, shuffle=False, num_workers=8, pin_memory=True)

net = DarkNet19()
net.cuda()
if sys.argv[1] == 'load':
    net.load_state_dict(torch.load(save_name))
    print "net loaded"
print net

reginLoss = RegionLoss(num_classes = 20, anchors=net.anchors, num_anchors=5)
optim = torch.optim.Adam(net.parameters())
itx = 0
# train
for epoch in xrange(15):
    for idx, (img, label) in enumerate(train_loader):
        itx += 1
        net.train()
        label = label.float()
        img = img.cuda()
        out = net(img)
        loss = reginLoss(out, label)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if itx%100==0:
            print 'loss {}, in ite {}'.format(loss, itx)
    ## eval in train
    pass
    # - save model
    torch.save(net.state_dict(), save_name)
    print "model saved"
    # - print loss
    print 'loss {}, in ite {}'.format(loss, itx)
