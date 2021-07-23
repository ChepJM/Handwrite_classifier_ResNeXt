import random
import sys
import argparse
import os
import pickle
import time
import numpy as np
import pandas as pd
import shutil
import os
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as utils
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset2 import InferDataset
from utils import get_val_augmentations, preprocess_data


def main():
    BATCH_SIZE = 4
    NUM_WORKERS = 8
    IMAGE_SIZE = 1024
    classes = 2

    device = 'cuda:0'
    albumentations_transform_validate = get_val_augmentations(IMAGE_SIZE)

    annotations_list = []
    for root, dirs, files in os.walk(r'/home/datalab/pred/'):
        if len(files)==0:
            continue
        for filename in files:
            filepath = os.path.join(root, filename)
            label = 0
            annotations_list.append({
                'label': label,
                'filepath': filepath,
            })

    annotations_list_new = []
    for annotation in annotations_list:
        annotations_list_new.append({
            'label': annotation['label'],
            'filepath': annotation['filepath'].replace('\\','/'),
        })

    validate_data = InferDataset(annotations_list=annotations_list,
                             transform=albumentations_transform_validate,
                             mode='test')
    validate_loader = DataLoader(dataset=validate_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False,
                                 drop_last=False)

    print(f'test: {len(annotations_list)}')

    model = models.resnext50_32x4d(pretrained=False)
    model.fc = nn.Linear(2048, 2)
    
    checkpoint = torch.load('/home/datalab/input/model_saved/weight_best.pth')
    model.load_state_dict(checkpoint)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    classifier = torch.nn.Softmax()
    # val
    os.makedirs(r'/home/datalab/print', exist_ok=True)
    os.makedirs(r'/home/datalab/hand', exist_ok=True)
    model.eval()
    preds_array = []
    preds_paths = []
    hand = []
    val_len = len(validate_loader)
    for idx, (imgs, filepaths) in tqdm(enumerate(validate_loader), total=len(validate_loader)):
        with torch.no_grad():
            imgs= imgs.to(device)
            output_test = model(imgs)
            preds = torch.argmax(classifier(output_test), dim=1)
            preds_array.append(preds)
            preds_paths.append(filepaths)
            for index, predict in enumerate(preds):
                if predict:
                    hand.append(filepaths[index])
                    shutil.copy(filepaths[index], r'/home/datalab/hand')
                else:
                    shutil.copy(filepaths[index], r'/home/datalab/print')
    pd.DataFrame(hand, columns=['Path']).to_csv(r"/home/datalab/hand_paths.csv", sep=';')


if __name__ == '__main__':
    main()
