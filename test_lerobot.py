# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from tqdm import tqdm
import imageio.v3 as iio

from core.utils import to_tensors

parser = argparse.ArgumentParser(description="E2FGVI")
parser.add_argument("-c", "--ckpt", type=str, default="release_model/E2FGVI-CVPR22.pth")
parser.add_argument("--lerobot_dir", type=str, default="examples/mug_on_platform_20250830_human")
parser.add_argument("--model", type=str, choices=['e2fgvi', 'e2fgvi_hq'], default="e2fgvi")
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--savefps", type=int, default=24)

# args for e2fgvi_hq (which can handle videos with arbitrary resolution)
parser.add_argument("--set_size", action='store_true', default=False)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)

args = parser.parse_args()

ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps

lerobot_dir = args.lerobot_dir

# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks
def read_mask(mpath, size):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks


# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size

def process_single_chunk(chunk_imgs, chunk_masks, model, size):
    device = chunk_imgs.device
    h, w = size[1], size[0]
    video_length = chunk_imgs.shape[1]

    # Convert back to frames for binary mask creation
    frames = ((chunk_imgs[0] + 1) / 2 * 255).cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

    # Create binary masks
    binary_masks = []
    for i in range(video_length):
        mask = chunk_masks[0, i, 0].cpu().numpy()  # Get single channel mask
        binary_mask = np.expand_dims((mask != 0).astype(np.uint8), 2)
        binary_masks.append(binary_mask)

    comp_frames = [None] * video_length

    # Process frames in neighbor_stride steps
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                             min(video_length, f + neighbor_stride + 1))
        ]

        if len(neighbor_ids) < 2: print("CHANGE CHUNK SIZE") # dumb hack to remind me to handle this edge case better

        ref_ids = get_ref_index(f, neighbor_ids, video_length)
        selected_imgs = chunk_imgs[:1, neighbor_ids + ref_ids, :, :, :]
        selected_masks = chunk_masks[:1, neighbor_ids + ref_ids, :, :, :]

        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks)
            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :h + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4)[:, :, :, :, :w + w_pad]
            pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
            pred_imgs = pred_imgs[:, :, :h, :w]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255

            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_imgs[i]).astype(np.uint8) * binary_masks[idx] + \
                      frames[idx] * (1 - binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + \
                                     img.astype(np.float32) * 0.5

        del selected_imgs, selected_masks, masked_imgs, pred_imgs
        torch.cuda.empty_cache()

    return comp_frames


def main_worker():
    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "e2fgvi":
        size = (432, 240)
    elif args.set_size:
        size = (args.width, args.height)
    else:
        size = None

    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()

    chunk_size = 100

    # prepare datset
    args.use_mp4 = True
    print(
        f'Loading videos and masks from: {args.lerobot_dir} | INPUT MP4 format: {args.use_mp4}'
    )

    videos = sorted(os.listdir(f"{lerobot_dir}/observation.images.cam_azure_kinect.color/"))
    output_dir = f"{lerobot_dir}/e2fgvi_vid/"
    os.makedirs(output_dir, exist_ok=True)

    for video_fname in tqdm(videos):
        vid_path = f"{lerobot_dir}/observation.images.cam_azure_kinect.color/{video_fname}"
        mask_path = f"{lerobot_dir}/gsam2_masks/{video_fname}_masks"
        save_path = os.path.join(output_dir, video_fname)
        if not os.path.exists(mask_path): continue
        if os.path.exists(save_path): continue

        all_frames = iio.imread(vid_path)
        total_frames = len(all_frames)
        all_comp_frames = []

        for chunk_start in tqdm(range(0, total_frames, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            
            # Load chunk
            chunk_frames_data = all_frames[chunk_start:chunk_end]
            chunk_frames = [Image.fromarray(i) for i in chunk_frames_data]
            chunk_frames, _ = resize_frames(chunk_frames, size)
            
            chunk_imgs = to_tensors()(chunk_frames).unsqueeze(0) * 2 - 1
            chunk_masks = read_mask(mask_path, size)[chunk_start:chunk_end]
            chunk_masks = to_tensors()(chunk_masks).unsqueeze(0)
            
            chunk_imgs, chunk_masks = chunk_imgs.to(device), chunk_masks.to(device)
            
            # Process chunk
            chunk_comp_frames = process_single_chunk(chunk_imgs, chunk_masks, model, size)
            all_comp_frames.extend(chunk_comp_frames)
            
            # Clear GPU memory
            del chunk_imgs, chunk_masks
            torch.cuda.empty_cache()

        size = (1280, 720)
        all_comp_frames = [cv2.resize(i, size) for i in all_comp_frames]
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 default_fps, size)
        for f in range(total_frames):
            comp = all_comp_frames[f].astype(np.uint8)
            writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
        writer.release()

if __name__ == '__main__':
    main_worker()
