import io
import time
import argparse
import requests
import torch
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import copy                            
import math
from collections import OrderedDict
import os
from os import listdir
from os.path import isfile, join
from torchvision import datasets
import ipdb


parser = argparse.ArgumentParser(description='PlantVillage Sailency Map Visualization')
parser.add_argument('model_path', help='path to trained model')
parser.add_argument('data_dir', help='path to data dir containing train/ and val/')
parser.add_argument('image_path', help='path to the image')
parser.add_argument('image_class', help='disease name')
parser.add_argument('--output_dir', help='path to output dir', default="output_saliency/")
parser.add_argument('--classes', default=10, type=int, metavar='N', help='number of classes, is deducted from data_dir/train')
parser.add_argument('--arch',  default="vgg13", help='architecture name, default: vgg13')
parser.add_argument('--method',default="guided", help='method for computing gradients in saliency map', choices=['guided', 'naive', 'deconv'])

args = parser.parse_args()

def diff_states(dict_canonical, dict_subset):
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    
    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    
    assert len(not_in_1) == 0
    assert len(not_in_2) == 0

    for name, v1 in dict_canonical.items():
        v2 = dict_subset[name]
        assert hasattr(v2, 'size')
        if v1.size() != v2.size():
            yield (name, v1)                

def load_defined_model(path, num_classes,name):
    model = models.__dict__[name](num_classes=num_classes)
    pretrained_state = torch.load(path)
    new_pretrained_state= OrderedDict()
    
    # for k, v in pretrained_state['state_dict'].items():
    for k, v in pretrained_state.items():
        layer_name = k.replace("module.", "")
        new_pretrained_state[layer_name] = v
        
    #Diff
    diff = [s for s in diff_states(model.state_dict(), new_pretrained_state)]
    if(len(diff)!=0):
        print("Mismatch in these layers :", name, ":", [d[0] for d in diff])
   
    assert len(diff) == 0
    
    #Merge
    model.load_state_dict(new_pretrained_state)
    return model


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],  # where to the numbers come from?
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

# same as preprocess but without normalize
preprocess1 = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   #transforms.Scale(342),
   #transforms.CenterCrop(299),
   transforms.ToTensor(),
])

def load_labels(data_dir,resize=(224,224)):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(max(resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms[x])
             for x in ['train']}  
    return (dsets['train'].classes)


labels=load_labels(args.data_dir)
print("Classes of plant diseases classification :")
print ("-------------------------------")
for label in labels:
    print (label)
print ("-------------------------------")
if args.classes != len(labels):
    print('WARNING: number of classes are not matching. Take the number deduced from input folder %s: %d' %(args.data_dir,len(labels)))
    args.classes = len(labels)



#Load the model
model= load_defined_model(args.model_path,args.classes,args.arch)
use_gpu = torch.cuda.is_available()

from torchvis import util
vis_param_dict, reset_state, remove_handles = util.augment_module(model)


def Saliency_map(image,model,preprocess,ground_truth,use_gpu=False,method=util.GradType.GUIDED):
    vis_param_dict['method'] = method
    img_tensor = preprocess(image)
    img_tensor.unsqueeze_(0)
    if use_gpu:
        img_tensor=img_tensor.cuda()
    input = Variable(img_tensor,requires_grad=True)
    
    if  input.grad is not None:
        input.grad.data.zero_()
    
    model.zero_grad()
    output = model(input)
    ind=torch.LongTensor(1)
    if(isinstance(ground_truth,np.int64)):
        ground_truth=np.asscalar(ground_truth)
    ind[0]=ground_truth
    ind=Variable(ind)
    energy=output[0,ground_truth]
    energy.backward() 
    grad=input.grad
    if use_gpu:
        return np.abs(grad.data.cpu().numpy()[0]).max(axis=0)
    return np.abs(grad.data.numpy()[0]).max(axis=0)


    
def classifyOneImage(model,img_pil,preprocess):
    model.eval()
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    if use_gpu:
        img_tensor = img_tensor.cuda()
        
    img_variable = Variable(img_tensor)
    out = model(img_variable)
    m = nn.Softmax()
    if use_gpu:     
        return m(out).cpu()[0].tolist()
    return m(out)[0].tolist()

    

if args.method == 'guided':
    method=util.GradType.GUIDED
elif args.method == 'naive':
    method=util.GradType.NAIVE
elif args.method == 'deconv':
    method=util.GradType.DECONV
else:
    raise ValueError('Invalid value for args.method = %s' % args.method)


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
        
use_gpu = torch.cuda.is_available()

if use_gpu:
    print("Transfering models to GPU(s)")
    model= torch.nn.DataParallel(model).cuda()
    

tic = time.time()
    
ind=labels.index(args.image_class)
img=Image.open(args.image_path)
model.eval()

#Extract image name without extension
file_name=os.path.basename(args.image_path)
file_name=os.path.splitext(file_name)[0]



if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

pred = classifyOneImage(model, img, preprocess)
pred_class = labels[pred.index(max(pred))]
probabilities_str = ''
for idx, label in enumerate(labels):
    probabilities_str += '%s: %.3f - ' % (label, pred[idx])
# ----------------------------------->
cropped_img = preprocess1(img)
npimg = cropped_img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.savefig(args.output_dir+file_name+"(cropped).png")
plt.title('true label: %s, predicted %s\n %s' %(args.image_class, pred_class, probabilities_str))
plt.show()
img.save(args.output_dir+file_name+"(original).png")
#------------------------------------>


map=Saliency_map(img,model,preprocess,ind,use_gpu,method)

plot_name=args.output_dir+str(file_name)+"("+method.name+").png"
plt.imshow(map,cmap='hot', interpolation='nearest')
plt.title('method: %s' % (method.name,))
plt.savefig(plot_name)
plt.show()       
          

toc=time.time()
print (toc-tic)


print("done")



