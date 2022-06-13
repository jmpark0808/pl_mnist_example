import pytorch_lightning as pl
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.data_dir = kwargs.get('data_dir')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers', 0)
        self.pin_memory = kwargs.get('pin_memory')

        # Data: data transformation strategy
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset_train = datasets.MNIST(root=self.data_dir, train=True, transform=self.transform, download=True)
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(self.dataset_train, [50000, 10000])
        self.dataset_test = datasets.MNIST(root=self.data_dir, train=False, transform=self.transform, download=True)


    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
