######################################################
# @Author: Kaustav Vats (kaustav16048@iiitd.ac.in)
# @Date: Tuesday, June 18th 2019, 2:08:06 pm
######################################################

import os
import cv2
import sys

LabelsOutput = ["../face-vehicle/Train/", "../face-vehicle/Test/"]

IMAGES = ["./WIDER_train/images/", "./WIDER_val/images/"]
BBOX = ["wider_face_train_bbx_gt.txt", "wider_face_val_bbx_gt.txt"]
DataTxT = ["face-vehicle-train.txt", "face-vehicle-test.txt"]
K = 1

file = open(BBOX[K], "r+")
data = file.readlines()
file.close()

file2 = open(DataTxT[K], "w")

flag = True
i = 0
total = 0
print("Len: ", len(data))
while i < len(data):
	filepath = data[i].strip("\n")
	# print("[W] {}".format(filepath))
	i += 1
	filename = filepath.split("/")[1]
	txt_name = filename.split(".")[0] + ".txt"

	count = int(data[i].strip("\n"))
	# print(i)
	if count > 0:
		i += 1
		file2.write("data/face-vehicle/" + filename + "\n")
		file = open(LabelsOutput[K] + txt_name, "w")
		image = cv2.imread(IMAGES[K] + filepath, 0)
		# print(image.shape)
		(height, width) = image.shape
		total += 1
		while count > 0:
			line = data[i].strip("\n")
			line = list(map(float, line.split()))
			# print(line)
			outline = "0 "
			x = line[0]
			y = line[1]
			w = line[2]
			h = line[3]
			x = x + (w/2)
			y = y + (h/2)
			x = x/float(width)
			y = y/float(height)
			w = w/float(width)
			h = h/float(height)

			outline += str(float(x)) + " " + str(float(y)) + " " + str(float(w)) + " " + str(float(h)) + "\n"
			file.write(outline)
			count -= 1
			i+=1
		file.close()
	else:
		i += 2
file2.close()
print("Total Images: ", total)
