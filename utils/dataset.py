import os
import cv2
from torch.utils.data import Dataset

class BuildingDataset(Dataset):
    def __init__(self, df, transform=None, root='train'):
        self.df = df
        self.img_paths = df['file'].apply(lambda x: os.path.join(root, x)).tolist()
        if 'mask' in df:
            self.label_paths = df['mask'].apply(lambda x: os.path.join('mask', x)).tolist()
        else:
            self.label_paths = None
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def read_img(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img
    
    def get_labels(self):
        return (self.df['ratio'] > 0) * 1
    
    def __getitem__(self, idx):
        img = self.read_img(str(self.img_paths[idx])) / 255
        if self.label_paths:
            mask = self.read_img(str(self.label_paths[idx])) // 255
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                t_image = transformed['image']
                t_mask = transformed['mask']
                return t_image.float(), t_mask.float()
            else:
                return img, mask
        else:
            if self.transform:
                transformed = self.transform(image=img)
                t_image = transformed['image']
                return t_image.float()
            else:
                return img
