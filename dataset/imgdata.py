import torch.utils.data as data
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import os
from glob import glob
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import random
augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


def prepare_Rain200H(datapath):
    print("process Rain200H!")
    input_path = os.path.join(datapath, 'input')
    target_path = os.path.join(datapath, 'groundtruth')
    imgs = []
    gts = []
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "rain-%d.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    return imgs, gts


def prepare_Rain200L(datapath):
    print("process Rain200L!")
    input_path = os.path.join(datapath, 'input')
    target_path = os.path.join(datapath, 'groundtruth')
    imgs = []
    gts = []
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "rain-%d.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    return imgs, gts


def prepare_Rain12600(datapath):
    print("process DDN!")
    input_path = os.path.join(datapath, 'input')
    target_path = os.path.join(datapath, 'groundtruth')
    imgs = []
    gts = []
    for i in range(900):
        target_file = "%d.jpg" % (i + 1)
        for j in range(14):
            input_file = "%d_%d.jpg" % (i + 1, j + 1)
            imgs.append(os.path.join(input_path, input_file))
            gts.append(os.path.join(target_path, target_file))
    return imgs, gts


def prepare_DID(datapath):
    print("process DID!")
    imgs = []
    gts = []
    inputpath = os.path.join(datapath, 'input')
    gtpath = os.path.join(datapath, 'groundtruth')
    for i in range(12000):
        target_file = "%d.jpg" % (i + 1)
        input_file = "%d.jpg" % (i + 1)
        imgs.append(os.path.join(inputpath, input_file))
        gts.append(os.path.join(gtpath, target_file))
    return imgs, gts


def prepare_SPA(datapath):
    print("process SPA!")
    imgs = []
    gts = []
    inputpath = os.path.join(datapath, 'input')
    gtpath = os.path.join(datapath, 'groundtruth')
    for i in range(638464):
        target_file = "%d.png" % (i + 1)
        input_file = "%d.png" % (i + 1)
        imgs.append(os.path.join(inputpath, input_file))
        gts.append(os.path.join(gtpath, target_file))
    return imgs, gts


def prepare_raindrop(datapath):
    print("process Raindrop!")
    inputpath = os.path.join(datapath, 'input')
    gtpath = os.path.join(datapath, 'groundtruth')
    clean_filenames = []
    noisy_filenames = []
    for i in range(861):
        target_file = "%d_clean.png" % (i)
        input_file = "%d_rain.png" % (i)
        noisy_filenames.append(os.path.join(inputpath, input_file))
        clean_filenames.append(os.path.join(gtpath, target_file))
    return noisy_filenames, clean_filenames


def prepare_RainDS_syn(datapath):
    print("process RainDS_syn!")
    clean_filenames = []
    noisy_filenames = []
    noisy_filenames.extend(glob(os.path.join(datapath, 'rainstreak', '*.png')))
    noisy_filenames.extend(glob(os.path.join(datapath, 'raindrop', '*.png')))
    noisy_filenames.extend(glob(os.path.join(datapath, 'rainstreak_raindrop', '*.png')))
    gtpath = os.path.join(datapath, 'gt')
    for f in noisy_filenames:
        filename = os.path.basename(f)
        if filename.startswith('pie'):
            f, l = filename.split('-')[0], filename.split('-')[-1]
            cleanname = f + '-norain-' + l
        else:
            f, l = filename.split('-')[0], filename.split('-')[-1]
            cleanname = 'norain-' + l
        clean_filenames.append(os.path.join(gtpath, cleanname))
    return noisy_filenames, clean_filenames


class DataLoaderTrain(data.Dataset):
    def __init__(self, opt):
        super(DataLoaderTrain, self).__init__()
        self.opt = opt
        if opt.data_path.find('Heavy') != -1:
            imgs, gts = prepare_Rain200H(opt.data_path)
        elif opt.data_path.find('Light') != -1:
            imgs, gts = prepare_Rain200L(opt.data_path)
        elif opt.data_path.find('Rain12600') != -1:
            imgs, gts = prepare_Rain12600(opt.data_path)
        elif opt.data_path.find('DID-MDN') != -1:
            imgs, gts = prepare_DID(opt.data_path)
        elif opt.data_path.find('sparain') != -1:
            imgs, gts = prepare_SPA(opt.data_path)
        elif opt.data_path.find('raindrop') != -1:
            imgs, gts = prepare_raindrop(opt.data_path)
        elif opt.data_path.find('RainDS_syn') != -1:
            imgs, gts = prepare_RainDS_syn(opt.data_path)
        else:
            raise (RuntimeError('Cannot find dataset!'))

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + opt.base_dir + "\n"))
        self.imgs = imgs
        self.gts = gts
        self.sizex = len(self.imgs)
        self.count = 0
        self.crop_size = opt.crop_size

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        tar_path = self.gts[index_]
        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        hh, ww = tar_img.shape[1], tar_img.shape[2]
        rr = random.randint(0, hh - self.crop_size)
        cc = random.randint(0, ww - self.crop_size)
        # Crop patch
        inp_img = inp_img[:, rr:rr + self.crop_size, cc:cc + self.crop_size]
        tar_img = tar_img[:, rr:rr + self.crop_size, cc:cc + self.crop_size]
        # Data Augmentations
        aug = random.randint(0, 3)
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        return inp_img, tar_img


class DataLoaderEval(data.Dataset):
    def __init__(self, opt):
        super(DataLoaderEval, self).__init__()
        self.opt = opt
        imgs = glob(os.path.join(opt.data_path, '*.png')) + glob(os.path.join(opt.data_path, '*.jpg'))

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + opt.base_dir + "\n"))
        self.imgs = imgs
        self.sizex = len(self.imgs)
        self.count = 0

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        inp_img = Image.open(inp_path).convert('RGB')
        inp_img = TF.to_tensor(inp_img)
        return inp_img, os.path.basename(inp_path)


class DataLoaderTrainNoise(Dataset):
    def __init__(self, rgb_dir, crop_size):
        super(DataLoaderTrainNoise, self).__init__()

        gt_dir = 'groundtruth'
        input_dir = 'input'
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.crop_size = crop_size

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        ps = self.crop_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)
        # Crop Input and Target
        H = clean.shape[1]
        W = clean.shape[2]
        if H - ps == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]
        apply_trans = transforms_aug[random.getrandbits(3)]
        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)
        return clean, noisy


def getloader(opt):
    dataset = DataLoaderTrain(opt)
    print("Dataset Size:%d" %(len(dataset)))
    trainloader = data.DataLoader(dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True)
    return trainloader


def getnoiseloader(data_path, crop_size=128, batch_size=16, num_workers=4, shuffle=True):
    dataset = DataLoaderTrainNoise(data_path, crop_size)
    print("Dataset Size:%d" %(len(dataset)))
    trainloader = data.DataLoader(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True)
    return trainloader, len(dataset)


def getevalloader(opt):
    dataset = DataLoaderEval(opt)
    print("Dataset Size:%d" %(len(dataset)))
    evalloader = data.DataLoader(dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.eval_workers,
            pin_memory=True)
    return evalloader



