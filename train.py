import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset import TextDataset
from utils import get_val_augmentations, get_train_augmentations, preprocess_data


def main():
    BATCH_SIZE = 4
    NUM_WORKERS = 8
    IMAGE_SIZE = 1024
    N_EPOCHS = 50
    device = torch.device("cuda:0")
    #device_ids = [0, 1]

    albumentations_transform = get_train_augmentations(IMAGE_SIZE)
    albumentations_transform_validate = get_val_augmentations(IMAGE_SIZE)
    train_df, val_df, train_labels, val_labels = preprocess_data('/home/datalab/input/noisy_imagewoof.csv')

    train_data = TextDataset(dataframe=train_df,
                           labels=train_labels,
                           path='/home/datalab/input',
                           transform=albumentations_transform)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True,
                              drop_last=False)

    validate_data = TextDataset(dataframe=val_df,
                              labels=val_labels,
                              path='/home/datalab/input',
                              transform=albumentations_transform_validate)
    validate_loader = DataLoader(dataset=validate_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False,
                                 drop_last=False)

    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(2048, 2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=200)

    best_acc_val = 0
    train_len = len(train_loader)
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0 

        for i, (imgs, labels) in tqdm(enumerate(train_loader), total=train_len):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = torch.argmax(torch.softmax(output, 1), 1).cpu().detach().numpy()
            true = labels.cpu().numpy()
            train_acc += accuracy_score(true, pred)
            scheduler.step(epoch + i / train_len)

        model.eval()
        val_loss = 0
        acc_val = 0
        val_len = len(validate_loader)
        for i, (imgs, labels) in tqdm(enumerate(validate_loader), total=val_len):
            with torch.no_grad():
                imgs_vaild, labels_vaild = imgs.to(device), labels.to(device)
                output_test = model(imgs_vaild)
                val_loss += criterion(output_test, labels_vaild).item()
                pred = torch.argmax(torch.softmax(output_test, 1), 1).cpu().detach().numpy()
                true = labels.cpu().numpy()
                acc_val += accuracy_score(true, pred)

        avg_val_acc = acc_val / val_len

        print(
            f'Epoch {epoch}/{N_EPOCHS}  train_loss {train_loss / train_len} train_acc {train_acc / train_len}  val_loss {val_loss / val_len}  val_acc {avg_val_acc}')

        if avg_val_acc > best_acc_val:
            best_acc_val = avg_val_acc
            torch.save(model.state_dict(), f'/home/datalab/input/model_saved/weight_best.pth')


if __name__ == '__main__':
    main()
