import torch
torch.backends.cudnn.benchmark = True
import torch.optim as optim
import os
from dataset.imgdata import getloader
from utils.model_utils import get_arch
from utils.common import print_network
from torch.optim.lr_scheduler import MultiStepLR
import time
import argparse
from loss import SSIM

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default="IDT")
parser.add_argument('--in_chans', type=int, default=3)
parser.add_argument('--embed_dim', type=int, default=32)
parser.add_argument('--win_size', type=int, default=8)
parser.add_argument('--depths', type=int, nargs='+', default=[3, 3, 2, 2, 1, 1, 2, 2, 3])
parser.add_argument('--num_heads', type=int, nargs='+', default=[1, 2, 4, 8, 16, 16, 8, 4, 2])
parser.add_argument('--mlp_ratio', type=float, default=4.0)
parser.add_argument('--qkv_bias', type=bool, default=True)
parser.add_argument('--downtype', type=str, default='Downsample', help="Downsample|Shufflesample")
parser.add_argument('--uptype', type=str, default='Upsample', help="Upsample|Unshufflesample")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--shuffle', action='store_true', help='shuffle for dataloader')
parser.add_argument('--crop_size', type=int, default=128, help='crop size for network')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. -1 for CPU')
parser.add_argument('--nepochs', type=int, default=400)
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--channel', type=int, default=32)
parser.add_argument('--data_path', type=str, default="E:\\rain_data_train_Heavy")
parser.add_argument('--save_path', type=str, default="E:\\transformer_model")
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--dis_freq', type=int, default=40)
parser.add_argument('--milestone', type=int, nargs='+', default=[100, 250, 350], help="When to decay learning rate")
parser.add_argument('--embed', type=int, default=32)
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

if __name__ == '__main__':
    savepath = opt.save_path
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    trainloader = getloader(opt)
    model = get_arch(opt)

    print_network(model)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.1)

    ######### DataParallel ###########
    model = torch.nn.DataParallel(model)
    model.cuda()
    ssimloss = SSIM().cuda()
    run_loss = 0.0
    global_step = 0

    print("Start training!")
    batch_num = 0
    for epoch in range(opt.nepochs):
        print("####### Epoch:%d starts #######" %(epoch + 1))
        Dis_start = time.time()
        for train_x, train_y in trainloader:
            global_step += 1
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            optimizer.zero_grad()
            out = model(train_x)
            loss = -1 * ssimloss(out, train_y)
            loss.backward()
            optimizer.step()
            batch_num += 1
            run_loss += loss.item()
            if global_step % opt.dis_freq == 0:
                Dis_end = time.time()
                print("|Epoch:%d, global step:%d, loss:%.3f time:%.3f|" %(epoch+1, global_step, run_loss/opt.dis_freq,
                                                                          Dis_end - Dis_start))
                run_loss = 0.0
                Dis_start = Dis_end

        scheduler.step()
        torch.save(model.state_dict(), os.path.join(savepath, 'net_latest.pth'))
        if (epoch + 1) % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(savepath,
                                                        'net_epoch_%d.pth' % (epoch + 1)))
