import torch
import numpy as np
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()

# load vgg 19 fropm jcjohnson pretrained file
'''
vgg_19 = torch.load('vgg19-d01eb7cb.pth')
print(vgg_19.summary())
'''

# load the pretrained vgg_19 model; and remove the last layer to get image features
vgg_19 = models.vgg19(pretrained=True)
# print(vgg_19)
# print(list(list(vgg_19.classifier.children())[1].parameters()))
mod = list(vgg_19.classifier.children())
mod.pop()
new_classifier = nn.Sequential(*mod)
# print(list(list(new_classifier.children())[1].parameters()))
vgg_19.classifier = new_classifier
# print(vgg_19)

'''
# remove last fully-connected layer
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
'''

# tr_or_val = 'train'
# img_path = "{s}2014/COCO_{s}2014_{d:012d}.jpg".format(s=tr_or_val, d=num,)


if USE_CUDA:
    vgg_19 = vgg_19.cuda()

def load_img(tr_or_val, img_list):
    dataset = torch.randn(len(img_list), 4096)

    preprocess = transforms.Compose([transforms.Scale(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                   std = [ 0.229, 0.224, 0.225 ])])

    for idx, img_idx in enumerate(img_list):
        img_path = "{s}2014/COCO_{s}2014_{d:012d}.jpg".format(s=tr_or_val, d=img_idx,)
        try:
            img = Image.open(img_path).convert('RGB')
        except IOError:
            print "Cannot find file: %s"%(img_path)

        inputVar = Variable(preprocess(img).unsqueeze(0))
        if USE_CUDA:
            inputVar = inputVar.cuda()
        # print(inputVar)
        outputVar = vgg_19.forward(inputVar)
        # print(outputVar[0])
        dataset[idx] = outputVar.data
        print("%s: %d/%d"%(tr_or_val, idx, len(img_list)))
    torch.save(dataset, '%s_features.pth'%(tr_or_val))

val_image_indices = np.genfromtxt('cocoqa/test/img_ids.txt', dtype=int)
load_img('val', val_image_indices)

val_image_indices = np.genfromtxt('cocoqa/train/img_ids.txt', dtype=int)
load_img('train', val_image_indices)

# print(val_img.size())
