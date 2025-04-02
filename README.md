
# Learnable Diffusion Framework for Mouse V1 Neural Decoding

This repository includes the codes for Sensorium-Viz, a neural decoding tool utilize the Diffusion Transformer (DiT) to reconstruct the visual stimuli with the neuron responses retrieved from the calcium image data of mouse right primary visual cortex layer 2/3. The datasets are from the SENSORIUM 2022 Challenge.

Please contact ([dengkw@umich.edu](mailto:dengkw@umich.edu) or [gyuanfan@umich.edu](mailto:gyuanfan@umich.edu)) if you have any questions or suggestions.

---

## Installation

### Environment

- With environment `.yml` file

    ```bash
    conda env create -f environment.yml
    ```

- Directly use `pip` or `conda` failed

    ```bash
    # Python >= 3.10.14
    pip install torch torchvision torchaudio
    pip install timm diffusers accelerate scikit-learn matplotlib ipykernel torchmetrics einops tqdm seaborn scikit-image gdown
    ```

### Datasets

- SENSORIUM datasets

    Download from [https://gin.g-node.org/cajal/Sensorium2022](https://gin.g-node.org/cajal/Sensorium2022). For more details about downloading and pre-processing, see `README` in `data/`

  
The following datasets and weights are shared on this [Google Drive](https://drive.google.com/drive/folders/1GbJ7V2AzVezKW3U0lwhrKYaeQni2ntef), and can be accessed via `gdown`. They should be placed outside the `data/` folder

- COCO processed images

    ```bash
    # https://drive.google.com/file/d/1G1hxuV8JxUrCZCuTKfWy5NkIRThx199r
    gdown 1G1hxuV8JxUrCZCuTKfWy5NkIRThx199r
    unzip coco_images.zip
    # rm coco_images.zip
    ```

- Synthetic responses of COCO images

    ```bash
    # https://drive.google.com/file/d/1UA2Ys493eKdwuZg9jpHFx72-5xEqi7Ex
    gdown 1UA2Ys493eKdwuZg9jpHFx72-5xEqi7Ex
    unzip coco_synthetic_response.zip
    # rm coco_synthetic_response.zip
    ```

- Model weights (optional, for repeating the results in our paper)

    The weights trained with additional 80000 synthetic responses can be downloaded via the following IDs. 

    ```bash
    gdown 195EFAEGbqe7ZGET4F2pKg8ohDcI7PeRl  # 21067
    gdown 1p4U6pQDlqDe8p_AVWcChHBHYdeVnf2ma  # 22846
    gdown 17cvbw8M-JD0-okKPvqqVK0J-2_HPMlYO  # 23343
    gdown 1V7dXfzM_PhoO8Z0LWNYNXBP3jzn_08gW  # 23656
    gdown 1h5D7sj3vnT5i8NFGRI4v2Yppvpp6VJmy  # 23964
    ```

    Unzip these files will get a folder like `res_{mouse_id}_idw32_all`. The weights are saved in a `.pt` file under `res_{mouse_id}_idw32_all/80000_synthetic/checkpoints`

## Train

Here we show an example of training a model for a mouse. It will save the checkpoints every 10000 training steps. The first checkpoint will be saved at 80000 steps.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 train.py \
    --mice "23656-14-22" \
    --results-dir res_23656_idw32_all \
    --exp-dir "80000_synthetic" \
    --model DiT-XL/2 \
    --response-dropout-prob 0.1 \
    --num-synthetic 80000 \
    --epochs 35 \
    --lr 5e-5 \
    --global-batch-size 32 \
    --global-seed 40 \
    --num-workers 8 \
    --log-every 100 \
    --ckpt-every 10000
```

To use multiple GPUs for data parallel:
`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=29501 train.py ...`

Fine-tune an existed model with only the synthetic data for cross-mice inference. For example, we have trained a model for mouse 21067 and want to fine-tune for mouse 23343:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 train.py \
    --mice "23343-5-17" \
    --results-dir res_21067to23343_idw32_all \
    --exp-dir "80000_synthetic" \
    --model DiT-XL/2 \
    --weight res_21067_idw32_all/80000_synthetic/checkpoints/0080000.pt \
    --use-real-data False \
    --response-dropout-prob 0.1 \
    --num-synthetic 80000 \
    --lr 5e-5 \
    --epochs 35 \
    --global-batch-size 32 \
    --global-seed 40 \
    --num-workers 8 \
    --log-every 100 \
    --ckpt-every 10000
```

## Inference and evaluation

Reconstruct the test images from the merged responses.

```bash
CUDA_VISIBLE_DEVICES=0 python gen_eval_aggTest.py \
    --result-dir res_23656_idw32_all/80000_synthetic \
    --exp-name 23656_at0080000_aggTest \
    --mouse-ids "23656-14-22" \
    --cfg 4.0 \
    --model-ckpt res_23656_idw32_all/80000_synthetic/checkpoints/0080000.pt
```

Then, go to the result folder to evaluate the reconstruction and summarize the metrics. You may copy the `metrics_full.py` and `summary.py` to your result folders to generate them.

```bash
cd  res_23656_idw32_all
CUDA_VISIBLE_DEVICES=0 python metrics_full.py
python summary.py
```

## Reference
The implementation of our models are mainly based on the official DiT codes: [https://github.com/facebookresearch/DiT/](https://github.com/facebookresearch/DiT/tree/main)