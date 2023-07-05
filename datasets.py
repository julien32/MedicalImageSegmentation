import numpy as np
import torch
import PIL
import os
import cv2
import torchvision.transforms as transforms
import albumentations as A
from PIL import Image
import imageio

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
    def __init__(self, src_dir, dataset_names, transforms=None, precision=torch.float32):
        assert type(dataset_names) == list
        self.src_dir = src_dir
        self.datasets = [self.dataset_lookup(self.src_dir, dataset_name) for dataset_name in dataset_names]
        # build an index hierarchy that allows us to access the individual datasets
        dataset_lengths = [len(dataset) for dataset in self.datasets]
        self.available_indices_per_dataset = [item for sublist in [[j for j in range(l)] for l in dataset_lengths] for item in sublist]
        self.index_to_dataset = [i for i, l in enumerate(dataset_lengths) for j in range(l)]

        self.total_num_samples = sum([len(s) for s in self.datasets])

        assert self.total_num_samples == len(self.available_indices_per_dataset) == len(self.index_to_dataset)

        self.transforms = transforms
        self.precision = precision
    
    @staticmethod
    def dataset_lookup(src_dir, dataset_name):
        if dataset_name in [
                                'TCGA_CS_4941_19960909',
                                'TCGA_CS_4942_19970222',
                                'TCGA_CS_4943_20000902',
                                'TCGA_CS_4944_20010208',
                                'TCGA_CS_5393_19990606',
                                'TCGA_CS_5395_19981004',
                                'TCGA_CS_5396_20010302',
                                'TCGA_CS_5397_20010315',
                                'TCGA_CS_6186_20000601',
                                'TCGA_CS_6188_20010812',
                                'TCGA_CS_6290_20000917',
                                'TCGA_CS_6665_20010817',
                                'TCGA_CS_6666_20011109',
                                'TCGA_CS_6667_20011105',
                                'TCGA_CS_6668_20011025',
                                'TCGA_CS_6669_20020102',
                                'TCGA_DU_5849_19950405',
                                'TCGA_DU_5851_19950428',
                                'TCGA_DU_5852_19950709',
                                'TCGA_DU_5853_19950823',
                                'TCGA_DU_5854_19951104',
                                'TCGA_DU_5855_19951217',
                                'TCGA_DU_5871_19941206',
                                'TCGA_DU_5872_19950223',
                                'TCGA_DU_5874_19950510',
                                'TCGA_DU_6399_19830416',
                                'TCGA_DU_6400_19830518',
                                'TCGA_DU_6401_19831001',
                                'TCGA_DU_6404_19850629',
                                'TCGA_DU_6405_19851005',
                                'TCGA_DU_6407_19860514',
                                'TCGA_DU_6408_19860521',
                                'TCGA_DU_7008_19830723',
                                'TCGA_DU_7010_19860307',
                                'TCGA_DU_7013_19860523',
                                'TCGA_DU_7014_19860618',
                                'TCGA_DU_7018_19911220',
                                'TCGA_DU_7019_19940908',
                                'TCGA_DU_7294_19890104',
                                'TCGA_DU_7298_19910324',
                                'TCGA_DU_7299_19910417',
                                'TCGA_DU_7300_19910814',
                                'TCGA_DU_7301_19911112',
                                'TCGA_DU_7302_19911203',
                                'TCGA_DU_7304_19930325',
                                'TCGA_DU_7306_19930512',
                                'TCGA_DU_7309_19960831',
                                'TCGA_DU_8162_19961029',
                                'TCGA_DU_8163_19961119',
                                'TCGA_DU_8164_19970111',
                                'TCGA_DU_8165_19970205',
                                'TCGA_DU_8166_19970322',
                                'TCGA_DU_8167_19970402',
                                'TCGA_DU_8168_19970503',
                                'TCGA_DU_A5TP_19970614',
                                'TCGA_DU_A5TR_19970726',
                                'TCGA_DU_A5TS_19970726',
                                'TCGA_DU_A5TT_19980318',
                                'TCGA_DU_A5TU_19980312',
                                'TCGA_DU_A5TW_19980228',
                                'TCGA_DU_A5TY_19970709',
                                'TCGA_EZ_7264_20010816',
                                'TCGA_FG_5962_20000626',
                                'TCGA_FG_5964_20010511',
                                'TCGA_FG_6688_20020215',
                                'TCGA_FG_6689_20020326',
                                'TCGA_FG_6690_20020226',
                                'TCGA_FG_6691_20020405',
                                'TCGA_FG_6692_20020606',
                                'TCGA_FG_7634_20000128',
                                'TCGA_FG_7637_20000922',
                                'TCGA_FG_7643_20021104',
                                'TCGA_FG_8189_20030516',
                                'TCGA_FG_A4MT_20020212',
                                'TCGA_FG_A4MU_20030903',
                                'TCGA_FG_A60K_20040224',
                                'TCGA_HT_7473_19970826',
                                'TCGA_HT_7475_19970918',
                                'TCGA_HT_7602_19951103',
                                'TCGA_HT_7605_19950916',
                                'TCGA_HT_7608_19940304',
                                'TCGA_HT_7616_19940813',
                                'TCGA_HT_7680_19970202',
                                'TCGA_HT_7684_19950816',
                                'TCGA_HT_7686_19950629',
                                'TCGA_HT_7690_19960312',
                                'TCGA_HT_7692_19960724',
                                'TCGA_HT_7693_19950520',
                                'TCGA_HT_7694_19950404',
                                'TCGA_HT_7855_19951020',
                                'TCGA_HT_7856_19950831',
                                'TCGA_HT_7860_19960513',
                                'TCGA_HT_7874_19950902',
                                'TCGA_HT_7877_19980917',
                                'TCGA_HT_7879_19981009',
                                'TCGA_HT_7881_19981015',
                                'TCGA_HT_7882_19970125',
                                'TCGA_HT_7884_19980913',
                                'TCGA_HT_8018_19970411',
                                'TCGA_HT_8105_19980826',
                                'TCGA_HT_8106_19970727',
                                'TCGA_HT_8107_19980708',
                                'TCGA_HT_8111_19980330',
                                'TCGA_HT_8113_19930809',
                                'TCGA_HT_8114_19981030',
                                'TCGA_HT_8563_19981209',
                                'TCGA_HT_A5RC_19990831',
                                'TCGA_HT_A616_19991226',
                                'TCGA_HT_A61A_20000127',
                                'TCGA_HT_A61B_19991127',
                            ]:
            return Brain_MRI(f"{src_dir}/lgg-mri-segmentation/kaggle_3m", dataset_name) # TODO pass path as argument
        elif dataset_name in [
            'benign',
            'malignant',
            'normal',
        ]:
            return Breast_MRI(f"{src_dir}/Dataset_BUSI_with_GT", dataset_name)
        elif dataset_name in [
            'QaTa-COV19-v1',
            'QaTa-COV19-v2'
        ]:
            if dataset_name == 'QaTa-COV19-v2':
                return Lungs_MRI(f"{src_dir}/QaTa-COV19/", dataset_name + 'Train Set')
            
            return Lungs_MRI(f"{src_dir}/QaTa-COV19/", dataset_name)
            
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


class Brain_MRI(torch.utils.data.Dataset):
    def __init__(self, source, classname=None):
        self.source = source
        self.classname = classname
        img_names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.source) for f in filenames if f.endswith('.tif') and not '_mask' in f]
        # sort out images with empty masks
        mask_sums = [self.check_empty(os.path.splitext(img_name)[0] + '_mask.tif') for img_name in img_names]
        self.img_names = [img_name for img_name, mask_sum in zip(img_names, mask_sums) if mask_sum]

    def check_empty(self, mask_path):
        mask = Image.open(mask_path)
        mask = transforms.ToTensor()(mask)
        return (mask.sum()>0).item()

    def __len__(self):
        return len(self.img_names)

    def load_datum(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.img_names[idx]
        mask_name = os.path.splitext(img_name)[0] + '_mask.tif'
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_torch = transforms.ToTensor()(img)
        img_torch = img_torch*255
        img_torch = transforms.Resize((1024, 1024))(img_torch)
        mask = Image.open(mask_name)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 1e-5).to(mask.dtype)
        mask = transforms.Resize((1024, 1024))(mask)

        return img_torch.to(torch.uint8), mask
    
class Breast_MRI(torch.utils.data.Dataset):
    def __init__(self, source, classname=None):
        self.source = source
        self.classname = classname
        img_names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.source) for f in filenames if f.endswith('.png') and not '_mask' in f]
        # sort out images with empty masks
        mask_sums = [self.check_empty(os.path.splitext(img_name)[0] + '_mask.png') for img_name in img_names]
        self.img_names = [img_name for img_name, mask_sum in zip(img_names, mask_sums) if mask_sum]

    def check_empty(self, mask_path):
        mask = Image.open(mask_path)
        mask = transforms.ToTensor()(mask)
        return (mask.sum()>0).item()

    def __len__(self):
        return len(self.img_names)

    def load_datum(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.img_names[idx]
        mask_name = os.path.splitext(img_name)[0] + '_mask.png'
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_torch = transforms.ToTensor()(img)
        img_torch = img_torch*255
        img_torch = transforms.Resize((1024, 1024))(img_torch)
        mask = Image.open(mask_name)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 1e-5).to(mask.dtype)
        mask = transforms.Resize((1024, 1024))(mask)

        return img_torch.to(torch.uint8), mask
    
        #Path to dataset: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

class Lungs_MRI(torch.utils.data.Dataset):
    def __init__(self, source, classname=None):
        self.source = source
        self.classname = classname
        img_names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.source + '/Images') for f in filenames if f.endswith('.png')]
        # sort out images with empty masks
        mask_sums = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.source + '/Ground-truths') for f in filenames if f.endswith('.png')]
        self.img_names = [img_name for img_name, mask_sum in zip(img_names, mask_sums) if mask_sum]

    def check_empty(self, mask_path):
        mask = Image.open(mask_path)
        mask = transforms.ToTensor()(mask)
        return (mask.sum()>0).item()

    def __len__(self):
        return len(self.img_names)

    def load_datum(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.img_names[idx]
        mask_name = os.path.splitext(img_name)[0] + '.png'
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_torch = transforms.ToTensor()(img)
        img_torch = img_torch*255
        img_torch = transforms.Resize((1024, 1024))(img_torch)
        mask = Image.open(mask_name)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 1e-5).to(mask.dtype)
        mask = transforms.Resize((1024, 1024))(mask)

        return img_torch.to(torch.uint8), mask

    # root_dir = '/Users/julien/Downloads/archive-4/QaTa-COV19/QaTa-COV19-v1'
    # root_dir2 = '/Users/julien/Downloads/archive-4/QaTa-COV19/QaTa-COV19-v2/Train Set'
    #Path to dataset https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset


class Decathlon_Heart(torch.utils.data.Dataset):
    def __init__(self, source='path/to/decathlon', classname=None):
        self.discipline = "Task02_Heart"
        self.source = source
        self.classname = classname
        if not os.path.exists(f"{self.source}/{self.discipline}"):
            self.download_and_extract()
        self.imgs, self.labels = self.get_img_label_names()
    
    def __len__(self):
        return len(self.imgs)

    def download_and_extract(self):
        url = f"https://msd-for-monai.s3-us-west-2.amazonaws.com/{self.discipline}.tar"
        directory = self.source
        command = f"wget -P {directory} {url}"
        os.system(command)
        command = f"tar -xvf {self.discipline}.tar"
        os.system(command)
    
    def get_img_label_names(self):
        with open(f'{self.source}/{self.discipline}/dataset.json', 'r') as f:
            datainfo = json.load(f)
        images = []
        labels = []
        for img_label_pair in datainfo['training']:
            images.append(f"{self.source}/{self.discipline}{img_label_pair['image'].split('.')[1]}.nii.gz")
            labels.append(f"{self.source}/{self.discipline}{img_label_pair['label'].split('.')[1]}.nii.gz")
        return images, labels
    
    def visualize(self, idx, fps=5):
        nii_img = nib.load(self.imgs[idx])
        nii_data = nii_img.get_fdata()
        nii_label = nib.load(self.labels[idx])
        nii_label_data = nii_label.get_fdata()
        dim = nii_data.shape
        gif_data = [np.tile(np.tile(nii_data[:, :, i][..., np.newaxis], (1, 1, 3)), (1, 2, 1)) for i in range(dim[2])]
        for i in range(dim[2]):
            mask = nii_label_data[:, :, i] == 1.0
            gif_data[i][:dim[0], :dim[1], :][mask] = [255, 0, 0]
        imageio.mimsave(f'{self.source}/{self.discipline}/{idx}.gif', gif_data, fps=fps)
        print(f'Visualization saved at destination: {self.source}/{self.discipline}/{idx}.gif')

    def __getitem__(self, idx):
        nii_img = nib.load(self.imgs[idx])
        nii_data = nii_img.get_fdata()
        nii_label = nib.load(self.labels[idx])
        nii_label_data = nii_label.get_fdata()
        label_argmax = np.argmax(np.sum(nii_label_data, axis=(0, 1)))
        image = np.tile(nii_data[:, :, label_argmax][..., np.newaxis], (1, 1, 3))
        image = transforms.ToTensor()(image)
        image = transforms.Resize((1024, 1024))(image)
        image = image.permute(1, 2, 0)
        mask = transforms.ToTensor()(nii_label_data[:, :, label_argmax])
        mask = transforms.Resize((1024, 1024))(mask)
        return image, mask


if __name__ == "__main__":
    # For debugging only
    dataset = PromptableMetaDataset(["TCGA_CS_4941_19960909"])
    print(len(dataset))