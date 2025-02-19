import os, sys
import numpy as np
import torch
from dataloaders.sensorium_test_agg import SensoriumDatasetTestAgg
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.utils import make_grid
from einops import rearrange
import argparse
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2
import json
import datetime


def main(args, split):
    logs = {}
    logs["seed"] = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logs["addition_info"] = []
    
    image_size = 256 #@param [256, 512]
    vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    latent_size = int(image_size) // 8

    # Load model:
    num_channel = 3
    logs["num_channel"] = num_channel
    model = DiT_XL_2(input_size=latent_size, response_grid=32, 
                     response_grid_channels=num_channel,).to(device)
    
    # res_3mice_idw32_all/001-DiT-XL-2/checkpoints/0057000.pt
    logs["ckpt"] = args.model_ckpt
    state_dict = find_model(args.model_ckpt)
    model.load_state_dict(state_dict)
    model.eval() # important!
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)

    num_sampling_steps = 250
    cfg_scale = args.cfg
    logs["cfg_scale"] = cfg_scale
    n_samples = 4
    batch_size = 1

    logs["split"] = split
    logs["datasets"] = args.mouse_ids
    im_dataset = SensoriumDatasetTestAgg(args.mouse_ids, embed_grids=32, norm_coord=args.norm_coord)
    im_loader = DataLoader(im_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    res_dir = args.result_dir

    diffusion = create_diffusion(str(num_sampling_steps))
    sampler = diffusion.p_sample_loop if args.sampler == "ddpm" else diffusion.ddim_sample_loop

    all_samples = []
    # limit = 100
    for count, (gt_image, cond) in enumerate(im_loader):
        gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)

        z = torch.randn(n_samples*batch_size, 4, latent_size, latent_size, device=device)
        y = torch.flatten(
            cond["response_emb"].unsqueeze(1).repeat(1, n_samples, 1, 1, 1),
            start_dim=0, end_dim=1
        ).float().to(device)

        if cfg_scale > 1.0:
            z = torch.cat([z, z], 0)
            y_null = torch.cat([
                torch.zeros_like(y[:, :1, :, :]),  # the grid response
                y[:, 1:3, :, :].detach(),  # the coordiantes
                torch.zeros_like(y[:, 3:, :, :]),  # eyes-related info
            ], dim=1)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        samples = sampler(
            sample_fn, z.shape, z, clip_denoised=False, 
            model_kwargs=model_kwargs, progress=True, device=device
        )
        if cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp((samples+1.0)/2.0, min=0.0, max=1.0)

        samples = torch.flatten(
            torch.cat([
                gt_image.view(batch_size, 1, 3, 256, 256), 
                samples.detach().cpu().view(batch_size, n_samples, 3, 256, 256)], 
                dim=1), 
            start_dim=0, end_dim=1)
        all_samples.append(samples)

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples+1)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    model = model.to('cpu')
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(f'{res_dir}/samples_{split}_{args.exp_name}_{args.sampler}.png')

    with open(f"{res_dir}/evaluate_logs.json", "w") as f:
        json.dump(logs, f, indent=4)
    

if __name__ == "__main__":
    import argparse
    import datetime
    now = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddim", "ddpm"])
    parser.add_argument("--mouse-ids", type=str, nargs="+", required=True)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--model-ckpt", type=str, required=True)
    parser.add_argument("--norm-coord", type=bool, default=True)
    args = parser.parse_args()
    main(args, "test")
