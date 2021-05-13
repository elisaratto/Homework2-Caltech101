from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import pandas as pd
import numpy as np


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

    

class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        
        label_counter = 0
        label_dict = {}
        labels = []
        img_paths = []

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.split_path = os.path.join(self.root.split('/')[0], self.split) # split file path -> file where to put datas
        
        paths = np.loadtext(self.split_path, dtype=str)
        
        for path in paths:
            fields = path.split('/') #fields[0] = class_name
            if fields[0]!='BACKGROUND_Google': #drop BACKGROUND_Google folder
                if fields[0] in label_dict: #if it's already in the dictionary, since it is ordered, the label will be the current counter
                    labels.append(label_counter); #assign label
                    img_paths.append(path); #to corrispondent img path
                else: #if not in the dictionary, increment counter so you have a new value
                    label_counter += 1
                    labels_dict[fields[0]] = label_counter; #define the label associated to the category "fields[0]"
                    labels.append(label_counter);
                    img_paths.append(path)
        
        self.dataset = pd.DataFrame({'path': img_paths, 'label': labels})
                                     
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        return image, label

        image = pil_loader(os.path.join(self.root, self.dataset.iloc[index, 0]))
        label = self.dataset.iloc[index, 1]
        
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.dataset) # Provide a way to get the length (number of elements) of the dataset
        return length
