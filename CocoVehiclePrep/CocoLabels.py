######################################################
# @Author: Kaustav Vats (kaustav16048@iiitd.ac.in)
# @Date: Friday, June 20st 2019, 10:32:46 am
######################################################
# In[]:
import json 
from tqdm import tqdm

CategoriesID = [2, 3, 4, 5, 6, 8]
count = [0]*7

K = 0
jsonFiles = ["instances_train2017.json", "instances_val2017.json"]
AnnotationsPath = "./annotations/{}".format(jsonFiles[K])

Annotations = json.load(open(AnnotationsPath, "r"))

dataImages = Annotations["images"]
ImageDict = {}

# In[]
for item in dataImages:
    ImageDict[int(item["id"])] = item
print(len(ImageDict.keys()))


# In[]
folder = ["Train", "Test"]
TxToutputFolder = "./TxTcoco/{}/".format(folder[K])
dataImages = Annotations["annotations"]
listImages = ["train.txt", "test.txt"]

file2 = open(listImages[K], "w")

for item in tqdm(dataImages):
    if int(item["category_id"]) in CategoriesID:
        if int(item["category_id"]) == 8:
            item["category_id"] = 6
        else:
            item["category_id"] = int(item["category_id"] - 1)

        count[int(item["category_id"])] += 1


        ImgData = ImageDict[int(item["image_id"])]
        file_name = ImgData["file_name"]
        file_name_txt = file_name.split(".")[0] + ".txt"

        file2.write("data/face-vehicle/{}\n".format(file_name))
        
        h = float(ImgData["height"])
        w = float(ImgData["width"])
        bbox = item["bbox"]
        x = float(bbox[0])
        y = float(bbox[1])
        bw = float(bbox[2])
        bh = float(bbox[3])

        x = x + (bw/2)
        y = y + (bh/2)
        x = x/w
        y = y/h
        bw = bw/w
        bh = bh/h
        line = str(item["category_id"]) + " " + str(x) + " " + str(y) + " " + str(bw) + " " + str(bh) + "\n"

        file = open(TxToutputFolder + file_name_txt, "a+")
        file.write(line)
        file.close()
file2.close()
print(count)
#%%
