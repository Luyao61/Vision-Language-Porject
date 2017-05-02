import torch
import numpy as np
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()

# max-pooling layer
vgg_19 = models.vgg19(pretrained=True).features

if USE_CUDA:
    vgg_19 = vgg_19.cuda()

def load_img(tr_or_val, img_list_full):
    for i in range(8):
        size = len(img_list_full)/8
        img_list = img_list_full[i*(size):(i+1)*size]

        dataset = torch.randn(len(img_list), 512, 14, 14)

        preprocess = transforms.Compose([transforms.Scale(448),
                                       transforms.CenterCrop(448),
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
            # print(outputVar.size())

            dataset[idx] = outputVar.data
            print("%s: %d/%d"%(tr_or_val, idx, len(img_list)))
        torch.save(dataset, '%s_features_SAN_%d.pth'%(tr_or_val, i))

val_image_indices = np.genfromtxt('cocoqa/test/img_ids.txt', dtype=int)
load_img('val', val_image_indices)

val_image_indices = np.genfromtxt('cocoqa/train/img_ids.txt', dtype=int)
load_img('train', val_image_indices)
