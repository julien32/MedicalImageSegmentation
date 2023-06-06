import numpy as np
import torch
import PIL
import os
import cv2
import torchvision.transforms as transforms
import albumentations as A


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _convert_for_albumentations(img, mask):
  img = img.permute(1,2,0).numpy().astype(np.uint8)
  mask = mask.numpy()[0]
  mask = np.stack([mask for _ in range(3)], axis=-1)

  return img, mask

def _convert_for_sam(img):
  img *= 255
  img = img.permute(1,2,0).numpy().astype(np.uint8)
  return img

def _restore_from_albumentations(img, mask):
  process = transforms.Compose([
              transforms.ToTensor(),
          ])
  process_mask = transforms.Compose(
          [
              transforms.ToTensor(),
          ] )
  img = process(img)
  mask = process_mask(mask[:,:,0])

  return img, mask


class PromptableMetaDataset(torch.utils.data.Dataset):
    """This dataset combines multiple datasets into a single one."""
    def __init__(self, dataset_names, transforms=None):
        self.datasets = [self.dataset_lookup(dataset_name) for dataset_name in dataset_names]
        # build an index hierarchy that allows us to access the individual datasets
        dataset_lengths = [len(dataset) for dataset in self.datasets]
        self.available_indices_per_dataset = [item for sublist in [[j for j in range(l)] for l in dataset_lengths] for item in sublist]
        self.index_to_dataset = [i for i, l in enumerate(dataset_lengths) for j in range(l)]

        self.total_num_samples = sum([len(s) for s in self.datasets])

        assert self.total_num_samples == len(self.available_indices_per_dataset) == len(self.index_to_dataset)

        self.transforms = transforms
    
    @staticmethod
    def dataset_lookup(dataset_name):
        if dataset_name in [
                                '...',
                            ]:
            return MIDOGPromptableDataset("/mnt/shared/lswezel/anomaly_detection", dataset_name, split="test")
        else:
            raise ValueError(f"Dataset {dataset_name} not found.")
    def __len__(self):
        return self.total_num_samples
    

    def __getitem__(self, idx):
        img, mask = self.datasets[self.index_to_dataset[idx]].load_datum(self.available_indices_per_dataset[idx])

        # apply transforms here
        if self.transforms is not None:
            img, mask = _convert_for_albumentations(img, mask)
            transformed = self.transforms(image=img, mask=mask)
            img, mask = _restore_from_albumentations(transformed['image'], transformed['mask'])
        
        # convert to numpy uint8 \in [0, 255] here
        img = _convert_for_sam(img)

        return img, mask



class MIDOGPromptableDataset(torch.utils.data.Dataset):
    # TODO implement this
    """ Placeholder dataset for MIDOG challenge data. """
    def __init__(
        self,
        source,
        classname,
        split="train",
    ):
        self.source = source
        self.classname = classname
        self.split = split
    
        # load file names where we there are masks
        # TODO implement this
        self.img_names = sorted(os.listdir("imgs"))
        self.img_names = sorted(os.listdir("masks"))

    def __len__(self):
        return len(self.img_names)

    def load_datum(self, idx):
        # select image and corresponding mask
        img_name = self.img_names[idx]
        mask_name = self.mask_names[idx]

        # load files into SAM compatible format
        image = cv2.imread(os.path.join(self.source, self.classname, self.split, "bad", img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_torch = transforms.ToTensor()(image) # convert to torch
        image_torch = transforms.Resize((1024, 1024))(image_torch)
        mask = cv2.imread(os.path.join(self.source, self.classname, "ground_truth", self.split, "bad", mask_name), cv2.IMREAD_GRAYSCALE)
        mask_tresholded = cv2.threshold(mask, 0.1, 255, cv2.THRESH_BINARY)[1]
        mask = transforms.ToTensor()(mask_tresholded)
        mask = transforms.Resize((1024, 1024))(mask)

        return image_torch, mask



