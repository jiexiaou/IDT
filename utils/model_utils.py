import torch
import os
from collections import OrderedDict


def freeze(model):
    for p in model.parameters():
        p.requires_grad=False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True


def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)



def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch


def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr


def get_arch(opt):
    from model import IDT
    arch = opt.arch
    print('You choose '+arch+'...')
    if arch.lower() == 'idt':
        model = IDT(in_chans=opt.in_chans, embed_dim=opt.embed_dim, depths=opt.depths,
                                num_heads=opt.num_heads, win_size=opt.win_size, mlp_ratio=opt.mlp_ratio,
                                qkv_bias=opt.qkv_bias)
    else:
        raise Exception("Arch error!")

    return model