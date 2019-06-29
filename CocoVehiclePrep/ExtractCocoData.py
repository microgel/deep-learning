######################################################
# @Author: Kaustav Vats (kaustav16048@iiitd.ac.in)
# @Date: Thursday, June 19th 2019, 10:00:37 am
######################################################


from pycocotools.coco import COCO
import numpy as np
import json
import shutil
from tqdm import tqdm
import csv
import cv2

dataDir='COCO_data'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# print(annFile)
coco=COCO(annFile)

# catIds = coco.getCatIds(catNms=['bicycle','car','motorcycle','airplane','bus','train','truck'])

Vehicle = {'2': 'bicycle', '3': 'car', '4':'motorcycle', '5':'airplane', '6':'bus', '8':'truck'}
# cat = [2, 3, 4 ,5 ,6 ,8]
cat = [8]


imgIds = []

# catIds = coco.getCatIds(catNms=['bicycle'])
# imgIds += coco.getImgIds(catIds=catIds)

# catIds = coco.getCatIds(catNms=['car'])
# imgIds += coco.getImgIds(catIds=catIds)

# catIds = coco.getCatIds(catNms=['motorcycle'])
# imgIds += coco.getImgIds(catIds=catIds)

# catIds = coco.getCatIds(catNms=['airplane'])
# imgIds += coco.getImgIds(catIds=catIds)

# catIds = coco.getCatIds(catNms=['bus'])
# imgIds += coco.getImgIds(catIds=catIds)

catIds = coco.getCatIds(catNms=['truck'])
imgIds += coco.getImgIds(catIds=catIds)
# print(imgIds)

print("ImageCount: {}".format(len(imgIds)))


IMAGE_PATH = "./COCO_data/coco_dataset/{}/".format(dataType)
OUTPUT_PATH = "./COCO_data/test/"
# CSV_PATH = "./COCO_data/{}".format(dataDir)
CSV_PATH = "./COCO_data/tru.csv"

dataset = dict()
dataset = json.load(open(annFile, 'r'))

licenses = dict()
# licenses = json.load(open("FILENAME", 'r'))
licenses = dataset['images']
CustomDict = {}

for i in range(len(licenses)):
    CustomDict[licenses[i]['id']] = [licenses[i]['file_name'], licenses[i]['height'], licenses[i]['width']]
# print(CustomDict)

Annotations = dataset['annotations']
print("Total Annotations: {}".format(len(Annotations)))

with open(CSV_PATH, 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])

    for i in tqdm(range(len(Annotations))):
        # print(Annotations[i]['image_id'])
        if int(Annotations[i]['image_id']) in imgIds and int(Annotations[i]['category_id']) in cat:
            # print(Annotations[i]['image_id'])
            Map = CustomDict[Annotations[i]['image_id']]
            image_name = Map[0] 
            line = []
            line.append(image_name)
            line.append(Map[2])
            line.append(Map[1])
            line.append(Annotations[i]['category_id'])
            line += Annotations[i]['bbox']
            # print(line)
            writer.writerow(line)

writeFile.close()
            # break
            # shutil.move(IMAGE_PATH + image_name, OUTPUT_PATH + image_name)



