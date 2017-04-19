import torch
import torchvision
from torchvision import models
import torch.nn as nn
import json, string
from torch.autograd import Variable



val_features = torch.load('val_features.pth')
vgg_19 = models.vgg19(pretrained=True)

classifier = nn.Sequential(list(vgg_19.classifier.children())[-1])

imagenetClasses = {int(idx): entry[1] for (idx, entry) in json.load(open('imagenet_class_index.json')).items()}


classifier.eval()
predictions = classifier(Variable(val_features[0].unsqueeze(0)))

probs, indices = (-nn.Softmax()(predictions).data).sort()
probs = (-probs).numpy()[0][:10]; indices = indices.numpy()[0][:10]
preds = [imagenetClasses[idx] + ': ' + str(prob) for (prob, idx) in zip(probs, indices)]

print(string.join(preds, '\n'))
print()


predictions = classifier(Variable(val_features[1].unsqueeze(0)))

probs, indices = (-nn.Softmax()(predictions).data).sort()
probs = (-probs).numpy()[0][:10]; indices = indices.numpy()[0][:10]
preds = [imagenetClasses[idx] + ': ' + str(prob) for (prob, idx) in zip(probs, indices)]

print(string.join(preds, '\n'))
print()

predictions = classifier(Variable(val_features[2].unsqueeze(0)))

probs, indices = (-nn.Softmax()(predictions).data).sort()
probs = (-probs).numpy()[0][:10]; indices = indices.numpy()[0][:10]
preds = [imagenetClasses[idx] + ': ' + str(prob) for (prob, idx) in zip(probs, indices)]

print(string.join(preds, '\n'))
print()




predictions = classifier(Variable(val_features[3].unsqueeze(0)))

probs, indices = (-nn.Softmax()(predictions).data).sort()
probs = (-probs).numpy()[0][:10]; indices = indices.numpy()[0][:10]
preds = [imagenetClasses[idx] + ': ' + str(prob) for (prob, idx) in zip(probs, indices)]

print(string.join(preds, '\n'))
print()


predictions = classifier(Variable(val_features[4].unsqueeze(0)))

probs, indices = (-nn.Softmax()(predictions).data).sort()
probs = (-probs).numpy()[0][:10]; indices = indices.numpy()[0][:10]
preds = [imagenetClasses[idx] + ': ' + str(prob) for (prob, idx) in zip(probs, indices)]

print(string.join(preds, '\n'))
print()
