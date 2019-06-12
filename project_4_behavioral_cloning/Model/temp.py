
import cv2
import os
from tqdm import tqdm
import pandas as pd
import PIL
import csv

import re
def parse_filename(name):
    pattern = re.compile('D.*?IMG(.*?).jpg')
    items = re.findall(pattern, name)
    return items

IMG_path = '/home/workspace/CarND-Behavioral-Cloning-P3/Data/data_1/IMG/'
samples=[]
with open('/home/workspace/CarND-Behavioral-Cloning-P3/Data/data_1/driving_log.csv', 'r') as f:
    reader=csv.reader(f)
    for line in reader:
        samples.append(line)

filename = parse_filename(samples[0][0])
filename[0] = filename[0].replace('\\','//')
print(filename[0])