import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import pandas as pd
from skimage.transform import rotate
from glob import glob
from sklearn.model_selection import train_test_split
import nibabel


class CustomDataset(Dataset):
    def __init__(self, args, data_path, transform=None, mode="Training", plane=False):

        print("loading data from the directory :", data_path)
        path = data_path
        
        # Try flat structure first: data_dir/images/*.png and data_dir/masks/*.png
        images = sorted(glob(os.path.join(path, "images/*.png")))
        masks = sorted(glob(os.path.join(path, "masks/*.png")))

        self.name_list = []
        self.label_list = []  # Each entry will hold a list of instance mask paths

        if len(images) > 0:
            if len(images) != len(masks):
                raise ValueError(
                    f"Flat structure expects 1:1 image-mask pairs but found {len(images)} images and {len(masks)} masks"
                )
            for img_path, mask_path in zip(images, masks):
                self.name_list.append(img_path)
                self.label_list.append([mask_path])

        # If no images found, try nested structure: data_dir/ID/images/ID.png and data_dir/ID/masks/ID.png
        if len(self.name_list) == 0:
            print("Flat structure not found, trying nested structure...")
            images = sorted(glob(os.path.join(path, "*", "images", "*.png")))
            masks = sorted(glob(os.path.join(path, "*", "masks", "*.png")))
            
            # For nested structure, we need to pair images with masks by ID
            if len(images) > 0 and len(masks) > 0:
                # Create a dictionary to match images with masks
                # Structure: {img_id: img_path}
                img_dict = {}
                for img_path in images:
                    # Extract ID from path: data_dir/ID/images/ID.png -> ID
                    img_id = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                    img_dict[img_id] = img_path
                
                # Structure: {mask_id: [mask_path1, mask_path2, ...]}
                mask_dict = {}
                for mask_path in masks:
                    mask_id = os.path.basename(os.path.dirname(os.path.dirname(mask_path)))
                    if mask_id not in mask_dict:
                        mask_dict[mask_id] = []
                    mask_dict[mask_id].append(mask_path)
                
                # Match images with mask groups
                matched_images = []
                matched_mask_groups = []
                for img_id in sorted(img_dict.keys()):
                    if img_id in mask_dict:
                        matched_images.append(img_dict[img_id])
                        matched_mask_groups.append(sorted(mask_dict[img_id]))
                    else:
                        print(f"Warning: Image {img_id} has no corresponding masks")
                
                self.name_list = matched_images
                self.label_list = matched_mask_groups
                if len(self.name_list) > 0:
                    total_masks = sum(len(group) for group in self.label_list)
                    print(f"Found {len(self.name_list)} images with {total_masks} instance masks in nested structure")

        if len(self.name_list) == 0:
            raise ValueError(f"No images found in {path}. Expected either:\n"
                           f"  - Flat: {path}/images/*.png and {path}/masks/*.png\n"
                           f"  - Nested: {path}/ID/images/ID.png and {path}/ID/masks/ID.png")
        
        assert len(self.name_list) == len(self.label_list), "Image and mask group counts must match"
        self.data_path = path
        self.mode = mode

        self.img_transform = transform
        if args is not None and hasattr(args, "image_size"):
            mask_resize = transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=InterpolationMode.NEAREST,
            )
        else:
            mask_resize = transforms.Resize(
                (256, 256), interpolation=InterpolationMode.NEAREST
            )
        self.mask_transform = transforms.Compose(
            [
                mask_resize,
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda t: (t > 0).float()),
            ]
        )

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(name)

        img = Image.open(img_path).convert("RGB")
        mask_paths = self.label_list[index]
        if len(mask_paths) == 0:
            raise ValueError(f"No masks found for image {img_path}")

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        if self.img_transform:
            state = torch.get_rng_state()
            img = self.img_transform(img)
            torch.set_rng_state(state)
        else:
            img = transforms.ToTensor()(img)

        merged_mask = None
        for msk_path in mask_paths:
            inst_mask = Image.open(msk_path).convert("L")
            if self.mask_transform:
                inst_mask = self.mask_transform(inst_mask)
            else:
                inst_mask = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Lambda(lambda t: (t > 0).float()),
                    ]
                )(inst_mask)
            merged_mask = inst_mask if merged_mask is None else torch.maximum(merged_mask, inst_mask)

        mask = merged_mask

        return (img, mask, name)
        # if self.mode == 'Training':
        #     return (img, mask, name)
        # else:
        #     return (img, mask, name)


class CustomDataset3D(torch.utils.data.Dataset):
    def __init__(self, data_path, transform):
        super().__init__()
        
        print("loading data from the directory :", data_path)
        path = data_path
        images = sorted(glob(os.path.join(path, "images/*.nii.gz")))
        masks = sorted(glob(os.path.join(path, "masks/*.nii.gz")))

        assert len(images) == len(masks), "Number of images and masks must be the same"
        
        self.valid_cases = [(img_path, seg_path) for img_path, seg_path in zip(images, masks)]

        self.all_slices = []
        for case_idx, (img_path, seg_path) in enumerate(self.valid_cases):
            seg_vol = nibabel.load(seg_path)
            img = nibabel.load(img_path)
            assert (
                img.shape == seg_vol.shape
            ), f"Image and segmentation shape mismatch: {img.shape} vs {seg_vol.shape}, Flies: {img_path}, {seg_path}"
            num_slices = img.shape[-1]
            self.all_slices.extend(
                [(case_idx, slice_idx) for slice_idx in range(num_slices)]
            )
            
        self.data_path = path
        
        self.transform = transform

    def __len__(self):
        return len(self.all_slices)

    def __getitem__(self, x):
        case_idx, slice_idx = self.all_slices[x]
        img_path, seg_path = self.valid_cases[case_idx]

        nib_img = nibabel.load(img_path)
        nib_seg = nibabel.load(seg_path)

        image = torch.tensor(nib_img.get_fdata(),dtype=torch.float32)[:, :, slice_idx].unsqueeze(0).unsqueeze(0)
        label = torch.tensor(nib_seg.get_fdata(),dtype=torch.float32)[:, :, slice_idx].unsqueeze(0).unsqueeze(0)
        label = torch.where(
            label > 0, 1, 0
        ).float()  # merge all tumor classes into one

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)
        return (
            image,
            label,
            img_path.split(".nii")[0] + "_slice" + str(slice_idx) + ".nii",
        )  # virtual path