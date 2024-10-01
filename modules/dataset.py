import os
import random
import torch
import torch.utils.data as data
from torchvision.io import read_image


class MaskDataset(data.Dataset):
    def __init__(self, dir1, dir2, dir3, transform=None, fraction=1.0):
        """Mask dataset for the three classes contained in the three directories.

        Parameters
        ----------
        dir1 : str
            Path to the first directory containing images (label '0').
        dir2 : str
            Path to the second directory containing images (label '1').
        dir3 : str
            Path to the third directory containing images (label '2').
        transform : torchvision.transforms
            Transformation to apply to the images.
        fraction : float
            Fraction of the dataset to keep.
        """

        self.transform = transform
        self.fraction = fraction
        
        # List all images in the directories
        self.images1 = sorted(os.listdir(dir1))
        self.images2 = sorted(os.listdir(dir2))
        self.images3 = sorted(os.listdir(dir3))
        
        # Store the directories
        self.dirs = [dir1, dir2, dir3]
        
        # Ensure all directories have images
        assert len(self.images1) > 0 and len(self.images2) > 0 and len(self.images3) > 0, "All directories must contain images"
        
        # Create a list of all images with their corresponding class labels
        self.images = [(os.path.join(dir1, img), 0) for img in self.images1] + \
                      [(os.path.join(dir2, img), 1) for img in self.images2] + \
                      [(os.path.join(dir3, img), 2) for img in self.images3]

        # shuffle the list, after setting the seed
        random.seed(42)
        random.shuffle(self.images)

        # Keep only a fraction of the dataset
        self.images = self.images[:int(len(self.images) * self.fraction)]

        
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        img_name, label = self.images[idx]
        mask = read_image(img_name)[0, :, :] / 255.0
        return mask.unsqueeze(0), label
    



def prepare_MaskDataset(img_dirs, 
                        batch_size,
                        validation_split,
                        fraction=1.0,
                        transform=None,
                        seed=123):
    dataset = MaskDataset(*img_dirs, transform=transform, fraction=fraction)
    val_len = int(len(dataset) * validation_split)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = data.random_split(dataset, 
                                                   lengths=[train_len, val_len], 
                                                   generator=generator)
    train_dataloader = data.DataLoader(train_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=False, 
                                       num_workers=4)
    val_dataloader = data.DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4)
    
    return train_dataset, val_dataset, train_dataloader, val_dataloader