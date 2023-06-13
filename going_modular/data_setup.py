from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os


NUM_WORKERS = os.cpu_count()

"""
creates training and testing DataLoaders

takes in a training directory and testing directory path and turns them into
PyTorch datasets and then into PyTorch DataLoaders

"""

def create_dataloaders(train_dir:str,
                      test_dir : str,
                      transform : transforms.Compose,
                      batch_size : int = 1,
                      num_workers : int = NUM_WORKERS):



    train_data = datasets.ImageFolder(train_dir,transform=transform)
    test_data = datasets.ImageFolder(test_dir,transform=transform)
    
    class_names = train_data.classes
    
    train_dataloader = DataLoader(train_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    
    return train_dataloader,test_dataloader,class_names
