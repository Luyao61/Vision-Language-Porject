import torch
import numpy as np
import torchvision.models as models
from PIL import Image

# load vgg 19 fropm jcjohnson pretrained file
'''
vgg_19 = torch.load('vgg19-d01eb7cb.pth')
print(vgg_19.summary())
'''

vgg_19 = models.vgg19(pretrained=True).features


# tr_or_val = 'train'
# img_path = "{s}2014/COCO_{s}2014_{d:012d}.jpg".format(s=tr_or_val, d=num,)

def load_img(tr_or_val, img_list):
    dataset = torch.randn(len(img_list), 3, 640, 480)

    for idx, img_idx in enumerate(img_list):
        img_path = "{s}2014/COCO_{s}2014_{d:012d}.jpg".format(s=tr_or_val, d=img_idx,)
        try:
            img = Image.open(img_path)
        except IOError:
            print "Cannot find file: %s"%(img_path)

        img.ToTensor()
        img.Normailze(mean = [ 0.485, 0.456, 0.406 ],
                      std = [ 0.229, 0.224, 0.225 ])

        dataset[idx] = img
    return dataset

val_image_indices = np.genfromtxt('cocoqa/test/img_ids.txt', dtype=int)
val_img = load_img('val', val_image_indices)

print(val_img.size())
