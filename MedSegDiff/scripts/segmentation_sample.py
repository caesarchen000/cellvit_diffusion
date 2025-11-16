

import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
from pathlib import Path

# Add parent directories to path for imports
script_dir = Path(__file__).parent
medsegdiff_dir = script_dir.parent
seg_dir = medsegdiff_dir.parent
sys.path.insert(0, str(medsegdiff_dir))
sys.path.insert(0, str(seg_dir))

sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
import cv2
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def post_process_mask(mask, threshold=0.5, smooth=True, kernel_size=5):
    """
    Post-process mask to reduce noise:
    1. Convert to numpy if tensor
    2. Apply Gaussian smoothing
    3. Threshold to binary
    4. Morphological operations to clean up
    """
    # Convert to numpy if tensor
    if isinstance(mask, th.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.array(mask)
    
    # Handle different shapes
    if len(mask_np.shape) == 4:  # (B, C, H, W)
        mask_np = mask_np[0, 0] if mask_np.shape[1] > 0 else mask_np[0]
    elif len(mask_np.shape) == 3:  # (C, H, W) or (B, H, W)
        mask_np = mask_np[0] if mask_np.shape[0] < mask_np.shape[-1] else mask_np
    elif len(mask_np.shape) == 2:  # (H, W)
        pass
    else:
        mask_np = mask_np.squeeze()
    
    # Normalize to [0, 1]
    if mask_np.max() > 1.0:
        mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
    
    # Apply Gaussian smoothing
    if smooth:
        mask_np = cv2.GaussianBlur(mask_np.astype(np.float32), (kernel_size, kernel_size), 0)
    
    # Threshold
    binary_mask = (mask_np > threshold).astype(np.float32)
    
    # Morphological operations to clean up small noise
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Convert back to tensor if input was tensor
    if isinstance(mask, th.Tensor):
        return th.from_numpy(binary_mask).float().to(mask.device)
    return binary_mask


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    logger.configure(dir = args.out_dir)

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir,transform_test)
        args.in_ch = 5
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)

        ds = CustomDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    for _ in range(len(data)):
        b, m, path = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        if args.data_name == 'ISIC':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
            slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
        else:  # CUSTOM dataset
            # Extract ID from path (handle both flat and nested structures)
            if isinstance(path, (list, tuple)) and len(path) > 0:
                path_str = path[0] if isinstance(path[0], str) else str(path[0])
            else:
                path_str = str(path)
            # Get filename without extension as ID
            slice_ID = os.path.splitext(os.path.basename(path_str))[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            if getattr(args, "ensemble_source", "mask") == "cal":
                mask_tensor = cal_out
            else:
                mask_tensor = sample[:, -1:, :, :]
            mask_tensor = mask_tensor.detach()
            
            # Post-process to reduce noise (optional, controlled by args)
            if hasattr(args, 'post_process') and args.post_process:
                processed = post_process_mask(
                    mask_tensor,
                    threshold=getattr(args, 'threshold', 0.5),
                    smooth=True,
                )
                mask_for_ensemble = processed
            else:
                mask_for_ensemble = mask_tensor

            mask_for_ensemble = mask_for_ensemble.squeeze(0)
            if mask_for_ensemble.ndim == 4:
                mask_for_ensemble = mask_for_ensemble[0]
            if mask_for_ensemble.ndim == 2:
                mask_for_ensemble = mask_for_ensemble.unsqueeze(0)

            enslist.append(mask_for_ensemble)

            if args.debug:
                # print('sample size is',sample.size())
                # print('org size is',org.size())
                # print('cal size is',cal.size())
                if args.data_name == 'ISIC':
                    # s = th.tensor(sample)[:,-1,:,:].unsqueeze(1).repeat(1, 3, 1, 1)
                    o = th.tensor(org)[:,:-1,:,:]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)
                    # co = co.repeat(1, 3, 1, 1)

                    s = sample[:,-1,:,:]
                    b,h,w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)

                    tup = (ss,o,c)
                elif args.data_name == 'BRATS':
                    s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                    m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                    o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                    o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                    o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                    o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

                compose = th.cat(tup,0)
                vutils.save_image(compose, fp = os.path.join(args.out_dir, str(slice_ID)+'_output'+str(i)+".jpg"), nrow = 1, padding = 10)
        ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        
        # Optionally report ensemble statistics before further processing
        if isinstance(ensres, th.Tensor) and getattr(args, "dump_stats", False):
            logger.log(
                f"[stats] {slice_ID}: raw min={ensres.min().item():.4f}, "
                f"max={ensres.max().item():.4f}, mean={ensres.mean().item():.4f}"
            )

        # Normalize ensemble result to [0, 1] before post-processing
        if isinstance(ensres, th.Tensor):
            # Handle different shapes
            if len(ensres.shape) == 2:  # (H, W)
                ensres = ensres.unsqueeze(0)  # (1, H, W)
            elif len(ensres.shape) == 3 and ensres.shape[0] == 1:  # (1, H, W)
                pass
            elif len(ensres.shape) == 3:  # (C, H, W) or (H, W, C)
                if ensres.shape[0] < ensres.shape[-1]:
                    ensres = ensres[0:1]  # Take first channel
                else:
                    ensres = ensres.unsqueeze(0)
            elif len(ensres.shape) == 4:  # (B, C, H, W)
                ensres = ensres[0, 0:1]  # Take first batch, first channel
            
            # Normalize to [0, 1] range
            ensres_min = ensres.min()
            ensres_max = ensres.max()
            if (ensres_max - ensres_min) > 1e-8:
                ensres = (ensres - ensres_min) / (ensres_max - ensres_min)
            else:
                ensres = ensres - ensres_min  # If all same value, just subtract min
        
            if getattr(args, "dump_stats", False):
                logger.log(
                    f"[stats] {slice_ID}: normalized min={ensres.min().item():.4f}, "
                    f"max={ensres.max().item():.4f}, mean={ensres.mean().item():.4f}"
                )

        # Final post-processing on ensemble result
        if hasattr(args, 'post_process') and args.post_process:
            ensres = post_process_mask(ensres, threshold=getattr(args, 'threshold', 0.5), smooth=True)
            # Ensure it's a tensor with correct shape
            if not isinstance(ensres, th.Tensor):
                ensres = th.from_numpy(ensres).float()
            if len(ensres.shape) == 2:
                ensres = ensres.unsqueeze(0)
        
        # Ensure proper shape for save_image: (C, H, W) or (1, H, W)
        if len(ensres.shape) == 2:
            ensres = ensres.unsqueeze(0)
        elif len(ensres.shape) == 4:
            ensres = ensres.squeeze(0)
        
        # If single channel, repeat to 3 channels for better visualization
        if ensres.shape[0] == 1:
            ensres = ensres.repeat(3, 1, 1)
        
        # Ensure values are in [0, 1] range
        ensres = th.clamp(ensres, 0.0, 1.0)
        
        vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)

def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",         #path to pretrain model
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results/',
        multi_gpu = None, #"0,1,2"
        debug = False,
        post_process = True,  # Enable post-processing to reduce noise
        threshold = 0.5,      # Threshold for binarization
        ensemble_source = "mask",  # Use diffusion mask channel by default
        dump_stats = False,   # Print per-sample stats to diagnose saturation
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
