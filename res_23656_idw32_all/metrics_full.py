
import os
import sys
import glob
import json

import torch
import torchvision
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional import accuracy
from einops import rearrange
from torchvision.models import vit_h_14, ViT_H_14_Weights
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda:0")


def larger_the_better(gt, comp):
    return gt > comp

def smaller_the_better(gt, comp):
    return gt < comp

def mse_metric(img1, img2):
    return (np.square(img1 - img2)).mean()

def pcc_metric(img1, img2):
    return np.corrcoef(img1.reshape(-1), img2.reshape(-1))[0, 1]

def ssim_metric(img1, img2):
    return ssim(img1, img2, data_range=img2.max()-img2.min(), channel_axis=-1)

def identity(x):
    return x

def crop_grid_image(img_file, expect_cols, single_image_size=256, border_size=2):
    grid_image = Image.open(img_file)
    grid_width, grid_height = grid_image.size
    num_cols = (grid_width + border_size) // (single_image_size + border_size)
    num_rows = (grid_height + border_size) // (single_image_size + border_size)
    assert num_cols == expect_cols, \
        f"Expected {expect_cols} columns, but got {num_cols} columns."

    # Calculate the effective width and height of each image including the border
    effective_image_size = single_image_size + border_size

    # Initialize a 2D list to store the cropped images
    cropped_images = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    # Iterate over each position in the grid and crop the individual images
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * effective_image_size + border_size
            upper = row * effective_image_size + border_size
            right = left + single_image_size
            lower = upper + single_image_size
            cropped_image = grid_image.crop((left, upper, right, lower))
            
            # Store the cropped image in the 2D list
            cropped_images[row][col] = cropped_image

    return cropped_images

def pair_wise_score(pred_imgs, gt_imgs, metric, is_sucess):
    # pred_imgs: n, w, h, 3
    # gt_imgs: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    assert len(pred_imgs) == len(gt_imgs)
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0
    corrects = []
    for idx, pred in enumerate(pred_imgs):
        gt = gt_imgs[idx]
        gt_score = metric(pred, gt)
        rest = [img for i, img in enumerate(gt_imgs) if i != idx]
        count = 0
        for comp in rest:
            comp_score = metric(pred, comp)
            if is_sucess(gt_score, comp_score):
                count += 1
        corrects.append(count / len(rest))
    return corrects

def metrics_only(pred_imgs, gt_imgs, metric, *args, **kwargs):
    print(pred_imgs.shape, gt_imgs.shape)
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0
    return metric(pred_imgs, gt_imgs)

def metrics_image_level(pred_imgs, gt_imgs, metric, *args, **kwargs):
    # pred_imgs: n, w, h, 3
    # gt_imgs: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0
    scores = []
    for pred, gt in zip(pred_imgs, gt_imgs):
        score = metric(pred, gt)
        scores.append(score)
    return scores

class psm_wrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)

    @torch.no_grad()
    def __call__(self, img1, img2):
        if img1.shape[-1] == 3:
            img1 = rearrange(img1, 'w h c -> c w h')
            img2 = rearrange(img2, 'w h c -> c w h')
        img1 = img1 * 2 - 1.0
        img2 = img2 * 2 - 1.0
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        return self.lpips(torch.FloatTensor(img1).to(self.device), torch.FloatTensor(img2).to(self.device)).item()

def run_psm_metric_only(gt_images, pred_images, avg_images=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    lpips_vals = []
    for gt_image, pred_image in zip(gt_images, pred_images):
        gt_image = torch.FloatTensor(gt_image).to(device)
        pred_image = torch.FloatTensor(pred_image).to(device)
        lpips_val = lpips(
            (gt_image.unsqueeze(0) * 2 - 1).to(device),
            (pred_image.unsqueeze(0) * 2 - 1).to(device)
        ).item()
        lpips_vals.append(lpips_val)
    return np.mean(lpips_vals) if avg_images else lpips_vals

def get_similarity_metric(img1, img2, method='pair-wise', metric_name='mse', **kwargs):
    # img1: n, w, h, 3
    # img2: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    if img1.shape[-1] != 3:
        img1 = rearrange(img1, 'n c w h -> n w h c')
    if img2.shape[-1] != 3:
        img2 = rearrange(img2, 'n c w h -> n w h c')

    if method == 'pair-wise':
        eval_procedure_func = pair_wise_score 
    elif method == 'metrics-only':
        eval_procedure_func = metrics_only
    elif method == 'image-level':
        eval_procedure_func = metrics_image_level
    else:
        raise NotImplementedError

    if metric_name == 'mse':
        metric_func = mse_metric
        decision_func = smaller_the_better
    elif metric_name == 'pcc':
        metric_func = pcc_metric
        decision_func = larger_the_better
    elif metric_name == 'ssim':
        metric_func = ssim_metric
        decision_func = larger_the_better
    elif metric_name == 'psm':
        metric_func = psm_wrapper()
        decision_func = smaller_the_better
    else:
        raise NotImplementedError
    
    return eval_procedure_func(img1, img2, metric_func, decision_func, **kwargs)


# main
img_file = str(sys.argv[1])
transform = torchvision.transforms.ToTensor()

separate_imgs = crop_grid_image(img_file, expect_cols=5)
nrow = len(separate_imgs)
ncol = len(separate_imgs[0])

metrics = {
    "LPIPS": [],
    # "LPIPS baseline": [],
    "LPIPS (pair)": [],
    # "LPIPS baseline (pair)": [],
    "Pixel-wise MSE": [],
    # "Pixel-wise MSE baseline": [],
    "Pixel-wise MSE (pair)": [],
    # "Pixel-wise MSE baseline (pair)": [],
    "Pixel-wise Corr": [],
    # "Pixel-wise Corr baseline": [],
    "Pixel-wise Corr (pair)": [],
    # "Pixel-wise Corr baseline (pair)": [],
    "SSIM": [],
    # "SSIM baseline": [],
    "SSIM (pair)": [],
    # "SSIM baseline (pair)": []
}

lpips_scores_image = []
mse_scores_image = []
pcc_scores_image = []
ssim_scores_image = []

lpips_pairwise = []
mse_pairwise = []
pcc_pairwise = []
ssim_pairwise = []

gt_images = torch.stack([transform(separate_imgs[i][0]) for i in range(nrow)]).detach().cpu().numpy()
for s in range(1, ncol):
    pred_images = torch.stack([transform(separate_imgs[i][s]) for i in range(nrow)]).detach().cpu().numpy()
    pred_images_random = np.random.random(pred_images.shape)
    
    lpips_score_image = run_psm_metric_only(gt_images, pred_images, avg_images=False)
    lpips_scores_image.append(lpips_score_image)
    mse_score_image = get_similarity_metric(pred_images, gt_images, "image-level", "mse")
    mse_scores_image.append(mse_score_image)
    pcc_score_image = get_similarity_metric(pred_images, gt_images, "image-level", "pcc")
    pcc_scores_image.append(pcc_score_image)
    ssim_score_image = get_similarity_metric(pred_images, gt_images, "image-level", "ssim")
    ssim_scores_image.append(ssim_score_image)
    
    lpips_score = run_psm_metric_only(gt_images, pred_images, avg_images=True)
    mse_score = get_similarity_metric(pred_images, gt_images, "metrics-only", "mse")
    pcc_score = get_similarity_metric(pred_images, gt_images, "metrics-only", "pcc")
    ssim_score = get_similarity_metric(pred_images, gt_images, "metrics-only", "ssim")
    metrics["LPIPS"].append(float(lpips_score))
    metrics["Pixel-wise MSE"].append(float(mse_score))
    metrics["Pixel-wise Corr"].append(float(pcc_score))
    metrics["SSIM"].append(float(ssim_score))

    lpips_pair_score = get_similarity_metric(pred_images, gt_images, "pair-wise", "psm")
    mse_pair_score = get_similarity_metric(pred_images, gt_images, "pair-wise", "mse")
    pcc_pair_score = get_similarity_metric(pred_images, gt_images, "pair-wise", "pcc")
    ssim_pair_score = get_similarity_metric(pred_images, gt_images, "pair-wise", "ssim")
    
    lpips_pairwise.append(lpips_pair_score)
    metrics["LPIPS (pair)"].append(float(np.mean(lpips_pair_score)))
    mse_pairwise.append(mse_pair_score)
    metrics["Pixel-wise MSE (pair)"].append(float(np.mean(mse_pair_score)))
    pcc_pairwise.append(pcc_pair_score)
    metrics["Pixel-wise Corr (pair)"].append(float(np.mean(pcc_pair_score)))
    ssim_pairwise.append(ssim_pair_score)
    metrics["SSIM (pair)"].append(float(np.mean(ssim_pair_score)))

metrics["LPIPS (avg)"] = float(np.mean(metrics["LPIPS"]))
metrics["Pixel-wise MSE (avg)"] = float(np.mean(metrics["Pixel-wise MSE"]))
metrics["Pixel-wise Corr (avg)"] = float(np.mean(metrics["Pixel-wise Corr"]))
metrics["SSIM (avg)"] = float(np.mean(metrics["SSIM"]))
metrics["LPIPS (pair) (avg)"] = float(np.mean(metrics["LPIPS (pair)"]))
metrics["Pixel-wise MSE (pair) (avg)"] = float(np.mean(metrics["Pixel-wise MSE (pair)"]))
metrics["Pixel-wise Corr (pair) (avg)"] = float(np.mean(metrics["Pixel-wise Corr (pair)"]))
metrics["SSIM (pair) (avg)"] = float(np.mean(metrics["SSIM (pair)"]))


print(metrics)
with open(os.path.splitext(img_file)[0] + ".json", "w") as f:
    json.dump(metrics, f, indent=4)
    
lpips_scores_image = np.column_stack(lpips_scores_image)
np.savetxt(os.path.splitext(img_file)[0] + "_lpips.txt", lpips_scores_image, delimiter="\t")

mse_scores_image = np.column_stack(mse_scores_image)
np.savetxt(os.path.splitext(img_file)[0] + "_mse.txt", mse_scores_image, delimiter="\t")

pcc_scores_image = np.column_stack(pcc_scores_image)
np.savetxt(os.path.splitext(img_file)[0] + "_pcc.txt", pcc_scores_image, delimiter="\t")

ssim_scores_image = np.column_stack(ssim_scores_image)
np.savetxt(os.path.splitext(img_file)[0] + "_ssim.txt", ssim_scores_image, delimiter="\t")

lpips_pairwise = np.column_stack(lpips_pairwise)
np.savetxt(os.path.splitext(img_file)[0] + "_lpips_pw.txt", lpips_pairwise, delimiter="\t")

mse_pairwise = np.column_stack(mse_pairwise)
np.savetxt(os.path.splitext(img_file)[0] + "_mse_pw.txt", mse_pairwise, delimiter="\t")

pcc_pairwise = np.column_stack(pcc_pairwise)
np.savetxt(os.path.splitext(img_file)[0] + "_pcc_pw.txt", pcc_pairwise, delimiter="\t")

ssim_pairwise = np.column_stack(ssim_pairwise)
np.savetxt(os.path.splitext(img_file)[0] + "_ssim_pw.txt", ssim_pairwise, delimiter="\t")
