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


def embed_response_idw_rawcoord(coord, value, num_grids=64):
    mesh = pyinterp.RTree()
    value = value.reshape(-1)  # (size, )
    mesh.packing(coord, value)

    X0, X1 = coord[:, 0].min(), coord[:, 0].max()
    Y0, Y1 = coord[:, 1].min(), coord[:, 1].max()

    mx, my = np.meshgrid(np.linspace(X0, X1, num_grids),
                         np.linspace(Y0, Y1, num_grids),
                         indexing='ij')
    idw, neighbors = mesh.inverse_distance_weighting(
        np.vstack((mx.ravel(), my.ravel())).T,
        within=False,
        k=10,
        num_threads=0)
    idw = idw.reshape(mx.shape)

    # (3, num_grids, num_grids)
    mx = (mx - mx.mean()) / mx.std()
    my = (my - my.mean()) / my.std()
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
                 add_behavior_as_channel: bool = False,
                 add_pupil_center_as_channel: bool = False,
                 norm_response: bool = True,
                 norm_coord: bool = True,
                 custom_test_split: str = None):
        
        self.im_size = im_size
        self.im_channels = im_channels
        self.mouse_ids = mouse_ids
        self.embed_grids = embed_grids
        
        self.add_behavior_as_channel = add_behavior_as_channel
        self.add_pupil_center_as_channel = add_pupil_center_as_channel

        self.norm_response = norm_response
        self.norm_coord = norm_coord

        self.support_info = {mouse_id: {} for mouse_id in mouse_ids}
        for mouse_id in mouse_ids:
            # information for train-test split
            if custom_test_split is None:
                tiers = np.load(os.path.join("data", mouse_id, "meta/trials/tiers.npy"))
                self.support_info[mouse_id]["test_ids"] = np.where(np.char.startswith(tiers, "test"))[0]
            else:
                test_ids = np.load(os.path.join("data", mouse_id, f"meta/trials/{custom_test_split}.npy"))
                self.support_info[mouse_id]["test_ids"] = test_ids

            # information for data normalization based on SENSORIUM 2022
            s = np.load(os.path.join("data", mouse_id, "meta/statistics/responses/all/std.npy"))
            threshold = 0.01 * s.mean()
            idx = s > threshold
            response_precision = np.ones_like(s) / threshold
            response_precision[idx] = 1 / s[idx]
            self.support_info[mouse_id]["response_precision"] = response_precision
            
            self.support_info[mouse_id]["eye_mean"] = np.load(os.path.join("data", mouse_id, "meta/statistics/pupil_center/all/mean.npy"))
            self.support_info[mouse_id]["eye_std"] = np.load(os.path.join("data", mouse_id, "meta/statistics/pupil_center/all/std.npy"))
            self.support_info[mouse_id]["behavior_std"] = np.load(os.path.join("data", mouse_id, "meta/statistics/behavior/all/std.npy"))
            
            # coordinate information
            coord = np.load(os.path.join("data", mouse_id, "meta/neurons/cell_motor_coordinates.npy"))
            if norm_coord:
                coord = coord - coord.mean(axis=0, keepdims=True)
                coord = coord / np.abs(coord).max()
            self.support_info[mouse_id]["coord"] = coord

        self.images, self.responses, self.behaviors, self.pupil_centers = self.get_data_files()
    
    def get_data_files(self):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        ims, responses, behaviors, pupil_centers = [], [], [], []

        for mouse_id in self.mouse_ids:
            im_path = f"data/{mouse_id}/"
            assert os.path.exists(im_path), "images path {} does not exist".format(im_path)

            test_ids = self.support_info[mouse_id]["test_ids"]
            ims = [os.path.join(im_path, f"data/images/{test_id}.npy") for test_id in test_ids]
            responses = [os.path.join(im_path, f"data/responses/{test_id}.npy") for test_id in test_ids]
            behaviors = [os.path.join(im_path, f"data/behavior/{test_id}.npy") for test_id in test_ids]
            pupil_centers = [os.path.join(im_path, f"data/pupil_center/{test_id}.npy") for test_id in test_ids]

        assert len(responses) == len(ims), "Condition Type Response but could not find captions for all images"
        print('Found {} images'.format(len(ims)))
        print('Found {} responses'.format(len(responses)))
        print('Found {} behaviors'.format(len(behaviors)))
        print('Found {} pupil_centers'.format(len(pupil_centers)))

        return ims, responses, behaviors, pupil_centers
    
    def get_image(self, the_file):
        im = np.transpose(np.load(the_file), (1, 2, 0))
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im = Image.fromarray(np.uint8(im), mode="RGB")
        return im

    def get_responses(self, the_file, norm=True):
        mouse_id = the_file.split("/")[1]
        response = np.load(the_file).reshape(1, -1)
        if norm:
            response_precision = self.support_info[mouse_id]["response_precision"]
            response = response * response_precision
        return response
    
    def get_behaviors(self, the_file, norm=True):
        mouse_id = the_file.split("/")[1]
        behavior = np.load(the_file).reshape(1, -1)
        if norm:
            behavior_std = self.support_info[mouse_id]["behavior_std"]
            behavior = behavior / behavior_std
        return behavior
    
    def get_pupil_centers(self, the_file, norm=True):
        mouse_id = the_file.split("/")[1]
        pupil_center = np.load(the_file).reshape(1, -1)
        if norm:
            eye_mean = self.support_info[mouse_id]["eye_mean"]
            eye_std = self.support_info[mouse_id]["eye_std"]
            pupil_center = (pupil_center - eye_mean) / eye_std
        return pupil_center

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
        behavior_file = self.behaviors[index]
        pupil_center_file = self.pupil_centers[index]
        assert self._check_match(im_file, response_file)

        mouse_id = im_file.split("/")[1]
        coord = self.support_info[mouse_id]["coord"]

        ######## Set Conditioning Info ########
        cond_inputs = {}
        response = self.get_responses(response_file, norm=self.norm_response)
        # cond_inputs["response"] = response
        cond_inputs["behavior"] = self.get_behaviors(behavior_file, norm=True)
        cond_inputs["pupil_center"] = self.get_pupil_centers(pupil_center_file, norm=True)
        if self.norm_coord:
            cond_inputs["response_emb"] = embed_response_idw(
                coord, 
                response, 
                num_grids=self.embed_grids)
        else:
            cond_inputs["response_emb"] = embed_response_idw_rawcoord(
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