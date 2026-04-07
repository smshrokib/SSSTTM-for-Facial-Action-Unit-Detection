import warnings
import torch
import torch.cuda
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
import opts2
import random
import logging
import matplotlib.pyplot as plt
from metrics.accf1 import MultiLabelF1_test
from torchvision.transforms import ToPILImage
import cv2

model_path = "/home/qitam/sdb2/home/qiteam_project/roukai/Action-Unit-Detection/hrformer_AU_MT/pretrain/best.pth"  # path to the model
result_path = "results/pic"  # path where the result images should be stored


# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)


# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, save_name):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)  # avoid division by zero
    cam = cv2.resize(cam, (W, H))

    # Convert to proper uint8 ndarray for OpenCV
    cam_uint8 = (cam * 255).round().astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    # Ensure img is uint8 before blending
    if img.dtype != np.uint8:
        img_to_blend = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img_to_blend = img
    cam_img = (0.3 * heatmap + 0.7 * img_to_blend).astype(np.uint8)

    cv2.imwrite(save_name, cam_img)


def load_compatible_state_dict(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt
    # handle common checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]

    model_state = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state.items():
        # strip possible "module." prefix
        kk = k[7:] if k.startswith("module.") else k
        if kk in model_state and model_state[kk].shape == v.shape:
            filtered[kk] = v
        else:
            skipped.append((kk, tuple(v.shape) if hasattr(v, "shape") else None,
                            tuple(model_state[kk].shape) if kk in model_state else None))

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print(f"[ckpt] loaded params: {len(filtered)}/{len(model_state)}")
    if skipped:
        print("[ckpt] skipped (shape mismatch / not found):")
        for kk, s_ckpt, s_model in skipped[:30]:
            print(f"  - {kk}: ckpt={s_ckpt}, model={s_model}")
        if len(skipped) > 30:
            print(f"  ... and {len(skipped)-30} more")
    if missing:
        print(f"[ckpt] missing keys (initialized randomly): {len(missing)}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)}")


if __name__ == '__main__':
    opt = opts2.parse_opt()
    opt = vars(opt)
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print('cpu selected!')

    # model
    # model = VGGVisualFormer(modality='A;V', task='AU', video_pretrained=False, num_patches=1)
    model = SpatialTemporalHrFormer(modality='A;V', task='AU', video_pretrained=False, num_patches=1)
    # model = SpatialTemporalFormer(modality='A;V', task='AU', video_pretrained=False, num_patches=1)
    # model = TwoStreamAuralVisualFormer(modality='A;V', task='AU')
    modes = model.modes
    model = model.to(device)
    print('Loading weight from:{}'.format(model_path))
    load_compatible_state_dict(model, model_path, device)
    model.eval()
    # disable grad, set to eval
    # for p in model.parameters():
    #     p.requires_grad = False
    # for p in model.children():
    #     p.train(False)

    categories = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]
    CAM_AU_id = 0
    istransformer = False

    fmap_block = list()
    grad_block = list()

    if istransformer:
        model.video_model.s_former.hr_emb.final_layer.register_forward_hook(farward_hook)  # 9
        model.video_model.s_former.hr_emb.final_layer.register_backward_hook(backward_hook)
    else:
        model.video_model.s_former.hr_emb.register_forward_hook(farward_hook)  # 9
        model.video_model.s_former.hr_emb.register_backward_hook(backward_hook)
    print(f'CAM AU id: ' + categories[CAM_AU_id])
    print(f'is transformer? {istransformer}')

    # load dataset (first time this takes longer)
    # dataset = Aff2TestDataset(opt)
    dataset = Aff2CompDataset(opt)
    dataset.set_modes(modes)
    dataset.set_aug(False)

    metric_au = MultiLabelF1_test(ignore_index=-1)

    # select the frames we want to process (we choose VAL and TEST)

    downsample_rate = 10000
    downsample = np.zeros(len(dataset), dtype=int)
    downsample[np.arange(0, len(dataset) - 1, downsample_rate)] = 1
    val_sampler = SubsetSequentialSampler(np.nonzero(dataset.val_ids * downsample)[0], shuffle=False)
    print('Test set length: ' + str(sum(dataset.val_ids * downsample)))
    loader = DataLoader(dataset, batch_size=1, sampler=val_sampler, num_workers=0, pin_memory=False, drop_last=False)

    #labels = torch.zeros((len(dataset), 17), dtype=torch.float32)
    # run inference
    os.makedirs(result_path, exist_ok=True)

    to_pil_image = ToPILImage()

    bar = tqdm(loader)
    for data in bar:
        # avoid storing hooks forever
        fmap_block.clear()
        grad_block.clear()

        ids = data['Index'].long()
        id = ids.item()
        result_dir = os.path.join(result_path, str(id))
        os.makedirs(result_dir, exist_ok=True)

        x = {}
        origin_img = data['origin'][0, 0]
        img = origin_img.numpy()

        for mode in modes:
            x[mode] = data[mode].to(device)

        result = model(x)

        # Get feature map ONCE from forward hook
        fmap = fmap_block[0].detach().cpu().numpy().squeeze()

        # Generate CAM for every AU
        for cam_id, au_name in enumerate(categories):
            grad_block.clear()
            model.zero_grad()

            loss = result[0, cam_id]
            loss.backward(retain_graph=(cam_id != len(categories) - 1))

            grads_val = grad_block[0].detach().cpu().numpy().squeeze()

            if istransformer:
                out_name = os.path.join(result_dir, f"{au_name} CAM Trans.png")
            else:
                out_name = os.path.join(result_dir, f"{au_name} CAM.png")

            cam_show_img(img, fmap, grads_val, out_name)