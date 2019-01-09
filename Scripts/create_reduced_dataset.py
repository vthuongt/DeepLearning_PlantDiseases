from os.path import join as osjoin
import os
import ipdb
import pathlib
from math import floor
from shutil import copyfile

# copyfile(src, dst)


# src_path = r'C:\Users\vthuo\DeepLearning_PlantDiseases\Scripts\PlantVillage'
src_path = r'PlantVillage'

keep_factor = 0.01

for root, dirs, files in os.walk(src_path):
    # ipdb.set_trace()
    if files:
        dst_path = root.replace('PlantVillage', 'PlantVillage_reduced_extreme')
        pathlib.Path(dst_path).mkdir(parents=True, exist_ok=True)
        print('created: ' + dst_path)
        files = files[:floor(keep_factor*len(files))]

        for f in files:
            copyfile(osjoin(root,f), osjoin(dst_path, f))









