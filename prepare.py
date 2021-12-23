import os
from shutil import copyfile

import sys

data_path = './datasets/RegDB_01/train'
save_path = './datasets/RegDB_01/train_py'
ids = os.listdir(data_path)

for id in ids:
    visible_path = '{}/V_{}'.format(save_path, id)
    thermal_path = '{}/T_{}'.format(save_path, id)
    os.mkdir(visible_path)
    os.mkdir(thermal_path)
    
    images = os.listdir(os.path.join(data_path,id))
  
    for image in images:
        if image[0] == 'V':
            copyfile(os.path.join(data_path, id, image), os.path.join(visible_path, image))
        elif image[0] == 'T':
            copyfile(os.path.join(data_path, id, image), os.path.join(thermal_path, image))

    
    

