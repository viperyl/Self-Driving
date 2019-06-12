import cv2
import PIL
import csv

IMG_path = '/home/workspace/CarND-Behavioral-Cloning-P3/Data/data/'
samples=[]

with open('/home/workspace/CarND-Behavioral-Cloning-P3/Data/data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        samples.append(line)
    for i in range(0, len(samples)):
        for ii in range(0, 3):
            string = samples[i][ii]
            file_path = IMG_path + string
            file_path = file_path.replace(" ", "")
            samples[i][ii] = file_path
print(file_path)