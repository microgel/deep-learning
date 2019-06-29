
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import pylab

dataDir='..'
dataType='val2017'
annFile="annotations/instances_train2017.json"
coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['bicycle','car','motorcycle','bus' ,'truck'])
imgIds = coco.getImgIds(catIds=catIds)
print(imgIds)
# imgIds = coco.getImgIds(imgIds = [324158])
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

