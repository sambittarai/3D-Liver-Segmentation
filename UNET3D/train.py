import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
from unet3d import UNet
import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
#from datasets import LiverDatasetRandom, LiverDatasetFixed
import progressbar

from torch.utils.data import Dataset, DataLoader
class LiverDataset(Dataset):
    def __init__(self, filepairlist):
        self.filepairlist = filepairlist
    def __len__(self):
        return len(self.filepairlist)

    def __getitem__(self, idx):
        imgpath, lblpath = self.filepairlist[idx]
        img = np.load(imgpath)
        img = np.float32(img)/128 - 1
        img = img[None,...]
        lbl = np.load(lblpath)
        return {'image' : torch.from_numpy(img), 'label' : torch.from_numpy(lbl).long()}

litsids = [1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 19, 20, 22, 25, 26, 27, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 48, 49, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 83, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 102, 103, 104, 105, 107, 108, 111, 113, 116, 117, 119, 120, 122, 124, 126, 128, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]
#litsids = [1]
data_root = './data/train'
data_files = [os.path.join(data_root, f) for f in os.listdir(data_root)]
img_files = [f for f in data_files if 'input' in f]
img_lbl_pairs = [(f, f.replace('input', 'label')) for f in img_files]
dset = LiverDataset(img_lbl_pairs)
dsetlen = len(dset)
trndset, valdset = random_split(dset, [int(0.9*dsetlen), int(0.1*dsetlen)])
trnloader = DataLoader(trndset, batch_size = 1, shuffle = True, num_workers = 2)
print(len(trnloader))
valloader = DataLoader(valdset, batch_size = 1, shuffle = False, num_workers = 2)

device = torch.device('cuda')
net = UNet(n_class = 2);
n_gpu = torch.cuda.device_count()
net = nn.DataParallel(net, device_ids = list(range(n_gpu)))
net.to(device);

optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=5e-5)
criterion = nn.CrossEntropyLoss()

lentrnloader, lenvalloader = len(trnloader), len(valloader)
print('Total : ', lentrnloader + lenvalloader)

loss1 = [] #Training loss
loss2 = [] #Validation loss

for epoch in range(101):
    print('Epoch : ', epoch)
    tr_loss = 0
    net.train()
    for param in net.parameters():
        param.requires_grad = True    
    trniter = iter(trnloader)
    for step in range(lentrnloader):
        batch = next(trniter)
        images, labels  = batch['image'], batch['label']
        inputs = images.to(device, dtype = torch.float)
        labels = labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if step % 10 == 0:
            print('Step {} loss'.format(step) + ': {}'.format(loss.item()))
        tr_loss += loss.item()
        optimizer.step()
        del inputs, labels, outputs
    epoch_loss = tr_loss/lentrnloader
    print('Training loss: {:.4f}'.format(epoch_loss))

    #training loss
    loss1 = np.append(loss1, epoch_loss)

    for param in net.parameters():
        param.requires_grad = False
    #Saves the model after every 5 epochs
    if epoch % 5 == 0:
        model_save_file = os.path.join('./models', 'model_epoch{}.bin'.format(epoch))
        torch.save(net.module.state_dict(), model_save_file)

    #saves the losses after every 2 epochs
    if epoch % 2 == 0:
        loss_file1 = os.path.join('./loss', 'loss1_{}.npy'.format(epoch))
        np.save(loss_file1, loss1)
        loss_file2 = os.path.join('./loss', 'loss2_{}.npy'.format(epoch))
        np.save(loss_file2, loss2)

    #saves the model if validation loss decreases
    if epoch != 0:
        temp1 = np.argmin(loss2)
        if temp1 == epoch:
            model_save_file = os.path.join('./models_new', 'model_epoch{}.bin'.format(epoch))
            torch.save(net.module.state_dict(), model_save_file)

    valiter = iter(valloader)
    val_loss = 0
    net.eval()
    for step in range(lenvalloader):
        batch = next(valiter)
        images, labels = batch['image'], batch['label']
        inputs = images.to(device, dtype = torch.float)
        labels = labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        del inputs, labels, outputs
    epoch_loss = val_loss/lenvalloader
    print('Validation loss: {:.4f}'.format(epoch_loss))

    #validation loss
    loss2 = np.append(loss2, epoch_loss)

