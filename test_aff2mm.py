import warnings
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler
import numpy as np
from models import *
from dataloader import SubsetSequentialSampler, SubsetRandomSampler, Prefetcher, Aff2TestDataset, Aff2CompDataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import os
import time
from collections import defaultdict
import opts
import random
import logging
import matplotlib.pyplot as plt
from metrics.accf1 import MultiLabelF1_test

model_path = "/home/qitam/sdb2/home/qiteam_project/roukai/Action-Unit-Detection/hrformer_AU_MT/pretrain/best.pth"  # path to the model
result_path = 'results'  # path where the result .txt files should be stored
# should be the same path

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices, shuffle=True):
        self.indices = indices
        if shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)


    def __len__(self):
        return len(self.indices)

def au_to_str(arr):
    str = "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9], arr[10], arr[11], arr[12], arr[13])
    return str

def ex_to_str(arr):
    str = "{:d}".format(arr)
    return str

def va_to_str(v,a):
    str = "{:.3f},{:.3f}".format(v, a)
    return str


def _strip_module_prefix(state_dict):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

def _adapt_pos_embedding_if_needed(state_dict, model):
    key = "video_model.t_former.pos_embedding"
    if key not in state_dict:
        return state_dict
    model_state = model.state_dict()
    if key not in model_state:
        return state_dict

    src = state_dict[key]
    dst = model_state[key]
    if src.shape == dst.shape:
        return state_dict

    # Expect [1, 1+N, D] -> [1, 2, D]
    if src.dim() == 3 and dst.dim() == 3 and dst.shape[1] == 2 and src.shape[0] == dst.shape[0] and src.shape[2] == dst.shape[2]:
        cls_tok = src[:, :1, :]          # [1,1,D]
        patch_tok = src[:, 1:, :].mean(dim=1, keepdim=True)  # [1,1,D] average all patches
        state_dict[key] = torch.cat([cls_tok, patch_tok], dim=1)  # [1,2,D]
        print(f"[ckpt] adapted {key}: {tuple(src.shape)} -> {tuple(state_dict[key].shape)}")
        return state_dict

    print(f"[ckpt] cannot adapt {key}: ckpt={tuple(src.shape)} model={tuple(dst.shape)} (will likely fail strict load)")
    return state_dict


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
        print('cpu selected!')
    
    # model
    model = SpatialTemporalHrFormer(modality='A;V', task='AU', video_pretrained=False, num_patches=1)
    modes = model.modes
    model = model.to(device)

    print('Loading weight from:{}'.format(model_path))
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    ckpt = _strip_module_prefix(ckpt)
    ckpt = _adapt_pos_embedding_if_needed(ckpt, model)

    model.load_state_dict(ckpt, strict=True)
    model.eval()
    # disable grad, set to eval
    for p in model.parameters():
        p.requires_grad = False
    for p in model.children():
        p.train(False)

    # load dataset (first time this takes longer)
    # dataset = Aff2TestDataset(opt)
    dataset = Aff2CompDataset(opt)
    dataset.set_modes(modes)
    dataset.set_aug(False)

    metric_au = MultiLabelF1_test(ignore_index=-1)

    # select the frames we want to process (we choose VAL and TEST)

    # 选择下采样率
    downsample_rate = 1000
    downsample = np.zeros(len(dataset), dtype=int)
    downsample[np.arange(0, len(dataset) - 1, downsample_rate)] = 1
    val_sampler = SubsetSequentialSampler(np.nonzero(dataset.val_ids * downsample)[0], shuffle=False)
    print('Test set length: ' + str(sum(dataset.val_ids * downsample)))
    loader = DataLoader(dataset, batch_size=256, sampler=val_sampler, num_workers=0, pin_memory=False, drop_last=False)

    output = torch.zeros((len(dataset), 21), dtype=torch.float32)
    #labels = torch.zeros((len(dataset), 17), dtype=torch.float32)
    # run inference
    os.makedirs(result_path, exist_ok=True)
    au_result_folder = os.path.join(result_path, 'au_HRformer_F1.txt')

    header = {"AU": "AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26,Micro-Average,Macro-Average", # 0,0,0,0,1,0,0,0
              "VA": "valence,arousal", # 0.602,0.389 or -0.024,0.279
              "EX": "Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise" # 4
             }

    au_writer = open(au_result_folder, "w")
    au_writer.write(header['AU'])
    au_writer.write('\n')

    bar = tqdm(loader)
    for data in bar:
        ids = data['Index'].long()
        x = {}
        for mode in modes:
            x[mode] = data[mode].to(device)

        label_ex = data['EX']
        label_ex[label_ex == -1] = 7
        labels = {
            'VA': data['VA'],
            'AU': data['AU'],
            'EX': label_ex,
        }
        result = model(x)

        predict = result.detach().cpu()
        pred_au = torch.sigmoid(predict[:, :12]).detach().cpu().squeeze().numpy()
        round_au = np.round(pred_au)

        metric_au.update(y_pred=round_au, y_true=labels['AU'].numpy())

    f1, micro_f1, macro_f1 = metric_au.get()
    f1.append(micro_f1)
    f1.append(macro_f1)

    au_writer.write(au_to_str(f1))
    au_writer.close()