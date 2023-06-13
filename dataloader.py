import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if f.endswith('.tif') and not '_mask' in f]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.file_list[idx]
        mask_name = os.path.splitext(img_name)[0] + '_mask.tif'

        image = Image.open(img_name)
        mask = Image.open(mask_name)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.ToTensor()
])

root_dir = '/Users/julien/Downloads/archive-3/lgg-mri-segmentation/kaggle_3m'

dataset = SegmentationDataset(root_dir, transform=transform, mask_transform=mask_transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, masks in dataloader:
    print(images.shape)  
    print(masks.shape)  

    batch_size = images.size(0)
    total_pixels = images.numel()
    print(f"Batch size: {batch_size}, Total pixels: {total_pixels}")
