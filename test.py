import torch
import os
from utils.common import print_network
from skimage import img_as_ubyte
from dataset.imgdata import getevalloader
import utils
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--preprocess', type=str, default='crop')
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
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. -1 for CPU')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--eval_workers', type=int, default=1)
parser.add_argument('--weights', type=str, default='')
opt = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

if __name__ == '__main__':

    model_restoration= utils.get_arch(opt)
    print_network(model_restoration)
    model_restoration = torch.nn.DataParallel(model_restoration)

    model_restoration.load_state_dict(torch.load(opt.weights))
    print("===>Testing using weights: ", opt.weights)

    model_restoration.cuda()
    model_restoration.eval()

    inp_dir = opt.data_path

    eval_loader = getevalloader(opt)

    result_dir = opt.save_path
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for input_, file_ in tqdm(eval_loader):
            B = input_.shape[0]
            input_ = input_.cuda()
            restored = model_restoration(input_)
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1)

            for i in range(B):
                restored_ = restored[i].numpy()
                save_file = os.path.join(result_dir, file_[i])
                utils.save_img(save_file, img_as_ubyte(restored_))