import glob
import os
import random
import cv2
import torch
import torchvision
import pyinterp
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


bugdata = []
with open("./data/bugdata.csv", "r") as f:
    for line in f:
        bugdata.append(line.strip())

def embed_response_idw(coord, value, num_grids=64):
    mesh = pyinterp.RTree()
    value = value.reshape(-1)  # (size, )
    mesh.packing(coord, value)

    X0, X1 = -1, 1
    Y0, Y1 = -1, 1
    STEP = 1 / (num_grids // 2 - 0.5)

    mx, my = np.meshgrid(np.arange(X0, X1 + STEP - 1e-8, STEP),
                         np.arange(Y0, Y1 + STEP - 1e-8, STEP),
                         indexing='ij')
    idw, neighbors = mesh.inverse_distance_weighting(
        np.vstack((mx.ravel(), my.ravel())).T,
        within=False,
        k=10,
        num_threads=0)
    idw = idw.reshape(mx.shape)

    # (3, num_grids, num_grids)
    return np.stack([idw, mx, my], axis=0)


class SensoriumDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """
    
    def __init__(self, 
                 mouse_ids: list[str], 
                 im_size: int = 256, 
                 im_channels: int = 3, 
                 embed_grids: int = 32,
                 use_real_data: bool = True, 
                 use_custom_split: bool = False,
                 use_custom_resp_norm: bool = False,
                 n_synthetic: int = 40000):
        
        # self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.mouse_ids = mouse_ids
        self.embed_grids = embed_grids
        self.use_real_data = use_real_data

        self.support_info = {mouse_id: {} for mouse_id in mouse_ids}
        for mouse_id in mouse_ids:
            # information for train-test split
            if use_custom_split:
                train_ids = np.load(os.path.join("data", mouse_id, "meta/trials/train.npy"))
                self.support_info[mouse_id]["train_ids"] = train_ids
            else:
                tiers = np.load(os.path.join("data", mouse_id, "meta/trials/tiers.npy"))
                self.support_info[mouse_id]["train_ids"] = np.where((tiers == "train") | (tiers == "validation"))[0]

            coco_images = os.listdir("coco_images")
            coco_images.remove("stats")  # exclude the stats directory
            
            random.seed(1000)
            random.shuffle(coco_images)
            coco_images = coco_images[:n_synthetic]
            self.support_info[mouse_id]["train_synthetic_ids"] = [fname.replace(".npy", "") for fname in coco_images]
            
            # information for data normalization based on SENSORIUM 2022
            if use_custom_resp_norm:
                train_responses = np.vstack(
                    [np.load(os.path.join("data", mouse_id, "data/responses", f"{train_id}.npy")) for train_id in train_ids]
                )
                s = train_responses.std(axis=0)
                print("Factor for response normalization based on input data:", s)
            else:
                s = np.load(os.path.join("data", mouse_id, "meta/statistics/responses/all/std.npy"))
                print("Factor for response normalization SENSORIUM provided:", s)
            threshold = 0.01 * s.mean()
            idx = s > threshold
            response_precision = np.ones_like(s) / threshold
            response_precision[idx] = 1 / s[idx]
            self.support_info[mouse_id]["response_precision"] = response_precision
            
            # coordinate information
            coord = np.load(os.path.join("data", mouse_id, "meta/neurons/cell_motor_coordinates.npy"))
            coord = coord - coord.mean(axis=0, keepdims=True)
            self.support_info[mouse_id]["coord"] = coord / np.abs(coord).max()

        self.images, self.responses = self.get_data_files()
    
    def get_data_files(self):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        ims, responses = [], []

        for mouse_id in self.mouse_ids:
            im_path = f"data/{mouse_id}/"
            assert os.path.exists(im_path), "images path {} does not exist".format(im_path)

            train_ids = self.support_info[mouse_id]["train_ids"]
            for train_id in train_ids:
                im = os.path.join(im_path, f"data/images/{train_id}.npy")
                response = os.path.join(im_path, f"data/responses/{train_id}.npy")
                if im not in bugdata and self.use_real_data:
                    ims.append(im)
                    responses.append(response)
            
            train_synthetic_ids = self.support_info[mouse_id]["train_synthetic_ids"]
            for train_synthetic_id in train_synthetic_ids:
                im = f"coco_images/{train_synthetic_id}.npy"
                response = f"coco_synthetic_response/{mouse_id}/{train_synthetic_id}.npy"
                if im not in bugdata:
                    ims.append(im)
                    responses.append(response)
    
        assert len(responses) == len(ims), "Condition Type Response but could not find captions for all images"
        print('Found {} images'.format(len(ims)))
        print('Found {} responses'.format(len(responses)))

        return ims, responses
    
    def get_image(self, the_file):
        im = np.load(the_file).astype(np.uint8)
        
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=-1)
        elif len(im.shape) == 3 and im.shape[0] == 1:
            im = np.transpose(im, (1, 2, 0))
        else:
            raise ValueError("Please check the im shape of {}: {}" % (the_file, im.shape))
        
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im = Image.fromarray(np.uint8(im), mode="RGB")
        return im

    def get_responses(self, the_file, norm=True):
        is_synthetic = (the_file.split("/")[0] == "coco_synthetic_response")
        mouse_id = the_file.split("/")[1]
        response = np.load(the_file).reshape(1, -1)
        
        if norm and not is_synthetic:
            # print("conduct norm")
            response_precision = self.support_info[mouse_id]["response_precision"]
            response = response * response_precision
        
        return response

    def _check_match(self, *files):
        res = True
        file_id = os.path.basename(files[0]).split(".")[0]
        for the_file in files[1:]:
            res = os.path.basename(the_file).split(".")[0] == file_id
            if res == False:
                return res
        return res
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im_file = self.images[index]
        response_file = self.responses[index]
        assert self._check_match(im_file, response_file)
        
        mouse_id = response_file.split("/")[1]
        coord = self.support_info[mouse_id]["coord"]
        # print(im_file, response_file, mouse_id)

        ######## Set Conditioning Info ########
        cond_inputs = {}
        cond_inputs["im_file"] = im_file
        cond_inputs["response_file"] = response_file
        
        response = self.get_responses(response_file, norm=True)
        # cond_inputs["response"] = response
        cond_inputs["response_emb"] = embed_response_idw(
            coord, 
            response, 
            num_grids=self.embed_grids)
        #######################################
        
        im = self.get_image(im_file)
        im_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.im_size),
            torchvision.transforms.CenterCrop(self.im_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5],
            )
        ])(im)
    
        # Convert input to -1 to 1 range.
        # im_tensor = (2 * im_tensor) - 1
        return im_tensor, cond_inputs