import os
import random

import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

                 
def augment(hazy, clean):
                          
    if random.random() < 0.2:             
        augmentation_method = random.choice([0, 1, 2, 3, 4, 5, 6, 7])            
        rotate_degree = random.choice([90, 180, 270])
        
        '''Rotate'''
        if augmentation_method == 0:
            hazy = transforms.functional.rotate(hazy, rotate_degree)
            clean = transforms.functional.rotate(clean, rotate_degree)
            return hazy, clean
        '''Vertical Flip'''
        if augmentation_method == 1:
            vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
            hazy = vertical_flip(hazy)
            clean = vertical_flip(clean)
            return hazy, clean
        '''Horizontal Flip'''
        if augmentation_method == 2:
            horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
            hazy = horizontal_flip(hazy)
            clean = horizontal_flip(clean)
            return hazy, clean
        '''Rotate + Vertical Flip'''
        if augmentation_method == 3:
            hazy = transforms.functional.rotate(hazy, rotate_degree)
            clean = transforms.functional.rotate(clean, rotate_degree)
            vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
            hazy = vertical_flip(hazy)
            clean = vertical_flip(clean)
            return hazy, clean
        '''Rotate + Horizontal Flip'''
        if augmentation_method == 4:
            hazy = transforms.functional.rotate(hazy, rotate_degree)
            clean = transforms.functional.rotate(clean, rotate_degree)
            horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
            hazy = horizontal_flip(hazy)
            clean = horizontal_flip(clean)
            return hazy, clean
        '''Vertical + Horizontal Flip'''
        if augmentation_method == 5:
            vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
            hazy = vertical_flip(hazy)
            clean = vertical_flip(clean)
            horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
            hazy = horizontal_flip(hazy)
            clean = horizontal_flip(clean)
            return hazy, clean
        '''Rotate + Vertical + Horizontal Flip'''
        if augmentation_method == 6:
            hazy = transforms.functional.rotate(hazy, rotate_degree)
            clean = transforms.functional.rotate(clean, rotate_degree)
            vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
            hazy = vertical_flip(hazy)
            clean = vertical_flip(clean)
            horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
            hazy = horizontal_flip(hazy)
            clean = horizontal_flip(clean)
            return hazy, clean
        '''Random Crop + Flip (轻微裁剪增强)'''
        if augmentation_method == 7:
                        
            crop_size = random.randint(5, 10)
            h, w = hazy.size[1], hazy.size[0]                                                 
            if h > crop_size*2 and w > crop_size*2:
                hazy = transforms.functional.crop(hazy, crop_size, crop_size, h-crop_size*2, w-crop_size*2)
                clean = transforms.functional.crop(clean, crop_size, crop_size, h-crop_size*2, w-crop_size*2)
                        
                hazy = transforms.functional.resize(hazy, (h, w))
                clean = transforms.functional.resize(clean, (h, w))
                  
            if random.random() < 0.5:
                horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
                hazy = horizontal_flip(hazy)
                clean = horizontal_flip(clean)
            return hazy, clean
    
                
    return hazy, clean

class DehazeDataset_ADF(Dataset):
    def __init__(self, data_dir, data_name, is_train=True, tag=False):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.is_train = is_train
        self.list_files = []
        
                                          
        filename = 'train.txt' if is_train else 'test.txt'
        for line in open(os.path.join(data_dir, filename)):
            line = line.strip('\n')
            if line != '':
                self.list_files.append(line)
        
        hazy_folder, clean_folder = data_name.split(',')
        self.tag = tag
        self.root_hazy = os.path.join(data_dir, '{}/'.format(hazy_folder))
        self.root_clean = os.path.join(data_dir, '{}/'.format(clean_folder))
        self.file_len = len(self.list_files)

    def __getitem__(self, index):
        if self.is_train:
            name = self.list_files[random.randint(0, self.file_len - 1)]
        else:
            name = self.list_files[index]
        
                           
        if self.tag in ['LHID', 'DHID']:
            clean_name = name.split('_')[0] + '.jpg'
        elif self.tag in ['indoor']:
            clean_name = name.split('_')[0] + '.png'
        elif self.tag in ['dense']:
                                                
            stem, ext = os.path.splitext(name)
            if '_hazy' in stem:
                clean_name = stem.replace('_hazy', '_GT') + ext
            elif 'hazy' in stem:
                clean_name = stem.replace('hazy', 'GT') + ext
            else:
                clean_name = name.replace('.png', '_GT.png')
        else:
                                                             
            stem, ext = os.path.splitext(name)
            candidates = []
            for rep in ['clean', 'gt', 'GT', 'clear', 'clear_gt', 'gt_clean']:
                if 'hazy' in stem:
                    candidates.append(stem.replace('hazy', rep) + ext)
                if '_hazy' in stem:
                    candidates.append(stem.replace('_hazy', f'_{rep}') + ext)
            if '_hazy' in stem:
                candidates.append(stem.replace('_hazy', '') + ext)
            base_variants = list(dict.fromkeys(candidates + [name.replace('hazy', 'clean') if 'hazy' in name else name]))
            ext_alternatives = ['.png', '.jpg', '.jpeg']
            expanded = []
            for cand in base_variants:
                cstem, cext = os.path.splitext(cand)
                expanded.append(cand)
                for e in ext_alternatives:
                    if e != cext:
                        expanded.append(cstem + e)
            seen = set()
            ordered = []
            for c in expanded:
                if c not in seen:
                    seen.add(c)
                    ordered.append(c)
            clean_name = None
            for cand in ordered:
                if os.path.exists(os.path.join(self.root_clean, cand)):
                    clean_name = cand
                    break
            if clean_name is None:
                                                    
                stem, ext = os.path.splitext(name)
                if '_' in stem:
                    prefix = stem.split('_')[0]
                else:
                    prefix = stem                                
                
                                              
                clean_files = os.listdir(self.root_clean)
                for clean_file in clean_files:
                    if clean_file.startswith(prefix) and os.path.isfile(os.path.join(self.root_clean, clean_file)):
                        clean_name = clean_file
                        break
                
                if clean_name is None:
                    clean_name = name                       

        hazy = Image.open(os.path.join(self.root_hazy, name)).convert('RGB')
        clean = Image.open(os.path.join(self.root_clean, clean_name)).convert('RGB')
        
        if self.is_train:
                            
            i, j, h, w = transforms.RandomCrop.get_params(hazy, output_size=(256, 256))
            hazy_ = TF.crop(hazy, i, j, h, w)
            clean_ = TF.crop(clean, i, j, h, w)
                  
            hazy_arg, clean_arg = augment(hazy_, clean_)
            hazy = self.transform(hazy_arg)
            clean = self.transform(clean_arg)
        else:
                         
            hazy = self.transform(hazy)
            clean = self.transform(clean)
            
                                          
                                              
        hazy_high_freq = hazy      

        return hazy, clean, hazy_high_freq, name                                      

    def __len__(self):
        return self.file_len
