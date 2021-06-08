from torch.utils.data import Dataset,DataLoader
import torch
# class Displace
class MNISTDataset(Dataset):
    """ MNIST dataset."""

    def __init__(self, y,x,transform=None):
        self.x=(x/255)
        self.y=y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, item):
        return  (self.x[item],self.y[item])

    # def

