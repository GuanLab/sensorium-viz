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


class SensoriumDatasetTestAgg(Dataset):
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
                 norm_coord: bool = True):
        
        self.im_size = im_size
        self.im_channels = im_channels
        self.mouse_ids = mouse_ids
        self.embed_grids = embed_grids
        
        self.norm_coord = norm_coord
        
        self.support_info = {mouse_id: {} for mouse_id in mouse_ids}
        for mouse_id in mouse_ids:
            # coordinate information
            coord = np.load(os.path.join("data", mouse_id, "meta/neurons/cell_motor_coordinates.npy"))
            if self.norm_coord:
                coord = coord - coord.mean(axis=0, keepdims=True)
                coord = coord / np.abs(coord).max()
            self.support_info[mouse_id]["coord"] = coord

        self.images, self.responses = self.get_data_files()
    
    def get_data_files(self):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        ims, responses = [], []

        for mouse_id in self.mouse_ids:
            assert os.path.exists(f"data/{mouse_id}"), "the mouse {} does not exist".format(mouse_id)
            ims += glob.glob(f"data/{mouse_id}/data/images_test/*.npy")
            responses += glob.glob(f"data/{mouse_id}/data/responses_test/*.npy")

        assert len(responses) == len(ims), "Condition Type Response but could not find captions for all images"
        print('Found {} images'.format(len(ims)))
        print('Found {} responses'.format(len(responses)))

        return ims, responses
    
    def get_image(self, the_file):
        im = np.transpose(np.load(the_file), (1, 2, 0))
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im = Image.fromarray(np.uint8(im), mode="RGB")
        return im

    def get_responses(self, the_file):
        response = np.load(the_file).reshape(1, -1)
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

        mouse_id = im_file.split("/")[1]
        coord = self.support_info[mouse_id]["coord"]

        ######## Set Conditioning Info ########
        cond_inputs = {}
        response = self.get_responses(response_file)
        if self.norm_coord:
            cond_inputs["response_emb"] = embed_response_idw(
                coord, 
                response, 
                num_grids=self.embed_grids,
            )
        else:
            cond_inputs["response_emb"] = embed_response_idw_rawcoord(
                coord, 
                response, 
                num_grids=self.embed_grids,
            )
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