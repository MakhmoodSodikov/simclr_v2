from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
import pandas as pd
import torch
from skimage import io
import os
from PIL import Image

class TezroDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index]['image_id_root'])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index]['category_id']))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
