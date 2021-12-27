import numpy as np
from PIL import Image
import torch.utils.data as data
import os, random
from torchvision import transforms





class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + '/train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + '/train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + '/train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + '/train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        


class RegDBData(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.train_folder_path = self.data_dir + '/train_py'
        self.transform = transform
        self.ids = os.listdir(self.train_folder_path)

        self.visible_id_pairs = {}
        self.visible_files = []
        for id in self.ids:
            if id[0] == 'T':
                continue
            else:
                images = os.listdir(os.path.join(self.train_folder_path, id))
                for image in images:
                    self.visible_id_pairs[image] = id[2:]
                    self.visible_files.append(image)

    def __getitem__(self, index):
        visible_image = self.visible_files[index]
        label = self.visible_id_pairs[visible_image]

        thermal_image = random.choice(list(os.listdir(self.train_folder_path+ '/T_{}'.format(label))))
        #print('{} {} {}'.format(label, visible_image, thermal_image))
        visible_image = Image.open(os.path.join(self.train_folder_path, 'V_'+label, visible_image))
        thermal_image = Image.open(os.path.join(self.train_folder_path, 'T_'+label, thermal_image))

        visible_image = self.transform(visible_image)
        thermal_image = self.transform(thermal_image)

        return visible_image, thermal_image, int(label)

    def __len__(self):
        return len(self.visible_files)


class TestData(data.Dataset):
    def __init__(self, data_dir, mode='gallery', transform=None):
        self.data_dir = data_dir + '/' + mode
        self.identities = os.listdir(self.data_dir)
        self.images = []
        self.labels = []
        self.transform = transform
        
        for id in self.identities:
            images = os.listdir(os.path.join(self.data_dir, id))
            for img in images:
                self.images.append(os.path.join(self.data_dir, id, img))
                self.labels.append(id)
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        label = self.labels[index]

        img = self.transform(img)

        return img, int(label)

    def __len__(self):
        return len(self.images)