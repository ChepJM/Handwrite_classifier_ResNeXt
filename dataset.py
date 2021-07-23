import cv2
import os
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, dataframe, labels, path, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = labels
        self.path = path

    def __getitem__(self, idx):
        filepath = self.dataframe.iloc[idx]['path']
        label = self.labels[idx]

        image = cv2.imread(os.path.join(self.path, filepath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

    def __len__(self):
        return self.dataframe.shape[0]
