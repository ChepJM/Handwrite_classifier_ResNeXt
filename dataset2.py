from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class InferDataset(Dataset):
    def __init__(self, annotations_list, mode='train',  transform=None):
        self.annotations_list = annotations_list
        self.transform = transform
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val':
            label = self.annotations_list[idx]['label']
            filepath = self.annotations_list[idx]['filepath']

            # Read an image with OpenCV
            # print('filepath', filepath)
            image = cv2.imread(filepath)

            # By default OpenCV uses BGR color space for color images,
            # so we need to convert the image to RGB color space.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            # return image, torch.FloatTensor(label)
            return image, label
        elif self.mode == 'test':
            filepath = self.annotations_list[idx]['filepath']

            f=open(filepath, 'rb')
            chunk = f.read()
            chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            image = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
            f.close()

            # By default OpenCV uses BGR color space for color images,
            # so we need to convert the image to RGB color space.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, filepath


    def __len__(self):
        return len(self.annotations_list)
