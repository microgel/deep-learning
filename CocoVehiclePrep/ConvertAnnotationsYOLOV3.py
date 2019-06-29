######################################################
# @Author: Kaustav Vats (kaustav16048@iiitd.ac.in)
# @Date: Friday, June 20st 2019, 10:04:16 am
######################################################

import json 
from tqdm import tqdm

CategoriesID = [2, 3, 4, 5, 6, 8]
count = [0]*6

K = 0
jsonFiles = ["instances_train2017.json", "instances_val2017.json"]
AnnotationsPath = "./annotations/{}".format(jsonFiles[K])

Annotations = json.load(open(AnnotationsPath, "r"))

dataImages = Annotations["annotations"]

ImageDict = {}

for item in dataImages:
    if int(item["category_id"]) in CategoriesID:
        if int(item["category_id"]) == 8:
            item["category_id"] = 6
        else:
            item["category_id"] = int(item["category_id"] - 1)

        ImageDict[int(item["image_id"])] = item

print("Total Images:", len(ImageDict.keys()))

TxToutputFolder = "./TxTcoco/"

dataImages = Annotations["images"]

listImages = ["train.txt", "test.txt"]
file2 = open(listImages[K], "w")

Keys = ImageDict.keys()
for item in tqdm(dataImages):
    if int(item["id"]) in Keys:
        filename = item["file_name"]
        filename_txt = filename.split(".")[0] + ".txt"
        file2.write("data/face-vehicle/{}\n".format(filename))
        # print("[w] {}".format(filename))
        file = open(TxToutputFolder + filename_txt, "a+")

        h = float(item["height"])
        w = float(item["width"])

        data = ImageDict[item["id"]]
        count[int(data["category_id"])-2] += 1
        bbox = data["bbox"]
        x1 = float(bbox[0])
        y1 = float(bbox[1])
        bw = float(bbox[2])
        bh = float(bbox[3])
        x1 = x1 + (bw/2)
        y1 = y1 + (bh/2)

        # Relative values
        x1 = x1/w
        y1 = y1/h
        bw = bw/w
        bh = bh/h

        line = str(data["category_id"]) + " " + str(x1) + " " + str(y1) + " " + str(bw) + " " + str(bh) + "\n"
        file.write(line)
        file.close()
file2.close()

