import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from skimage import io

class remoteDataSet(data.Dataset):
    def __init__(
        self,
        data_root,
        data_list,
        max_iters=None,
        num_classes=6, 
        split="train",
        transform=None,
        ignore_label=255,
        cfg=None,
        pseudo=False,
        debug=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        self.pseudo = pseudo
        with open(data_list, "r") as handle:
            content = handle.readlines()

        for fname in content:
            name = fname.strip()
            if pseudo:
                self.data_list.append(
                    {
                        "img": os.path.join(
                            self.data_root, "train_images/%s" % (name)
                        ),
                        "label": os.path.join(
                            cfg.OUTPUT_DIR, "CTR_O/%s"
                            % (
                                name
                            ),
                        ),
                        "name": name,
                    }
                    )
            else:
                self.data_list.append(
                    {
                        "img": os.path.join(
                            self.data_root, "train_images/%s" % (name)
                        ),
                        "label": os.path.join(
                            self.data_root,
                            "train_gt/%s"
                            % (name),
                        ),
                        "name": name,
                    }
                )
        
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            
        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]
        
        image = io.imread(datafiles["img"])
        label = np.array(io.imread(datafiles["label"]),dtype=np.float32)  
        #image = Image.open(datafiles["img"]).convert('RGB')
        #label = np.array(Image.open(datafiles["label"]),dtype=np.float32)  

        image = Image.fromarray(image)
        name = datafiles["name"]
        
        if self.transform is not None:
            image, label, _ = self.transform(image, label)
        return image, label, name

