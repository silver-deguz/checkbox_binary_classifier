from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip
from pathlib import Path 
from PIL import Image
import numpy as np 


class CheckboxDataset(Dataset):
    def __init__(self, root, mode='train', extra_augment=False):
        super(CheckboxDataset, self).__init__()
        # add random flips for augmentation. should be used in training set
        if extra_augment:
            self.transforms = Compose([
                        Resize((28, 28)),
                        RandomHorizontalFlip(),
                        RandomVerticalFlip(),
                        ToTensor(), 
                        Normalize((0.5), (0.5)),
                        ])
        else:
            self.transforms = Compose([
                        Resize((28, 28)),
                        ToTensor(), 
                        Normalize((0.5), (0.5)),
                        ])

        self.classes = ['unchecked', 'checked']
        if not Path(f'{root}').exists():
            raise RuntimeError('Dataset not found at given root dir')
        class0 = list(Path(f"{root}/{mode}/class_0").glob('**/*.png')) 
        class1 = list(Path(f"{root}/{mode}/class_1").glob('**/*.png')) 
    
        self.images = class0 + class1
        self.labels = [0] * len(class0) + [1] * len(class1)
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.labels[idx]
        return img, label
    

def create_train_val_loaders(root, batch_size=8):
    dataset = CheckboxDataset(root=root, extra_augment=True)
    dataset_size = dataset.__len__()
    all_indices = list(range(dataset_size))

    np.random.seed(124)
    np.random.shuffle(all_indices)

    # create the validation split from the full dataset
    val_split = int(np.floor(0.2 * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]

    # use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_val = SubsetRandomSampler(val_ind)

    # define the training & validation dataloaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_train)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_val)
    return train_loader, val_loader


def create_val_loader(root):
    dataset = CheckboxDataset(root=root)
    dataset_size = dataset.__len__()
    all_indices = list(range(dataset_size))

    np.random.seed(124)
    np.random.shuffle(all_indices)

    val_split = int(np.floor(0.2 * dataset_size))
    val_ind = all_indices[: val_split]

    sample_val = SubsetRandomSampler(val_ind)
    val_loader = DataLoader(dataset, batch_size=1, sampler=sample_val)
    return val_loader


def create_test_loader(root):
    dataset = CheckboxDataset(root=root, mode='test')
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return test_loader
