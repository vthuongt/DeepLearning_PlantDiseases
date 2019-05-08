import torch
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import copy
import math
from collections import OrderedDict
import os
from torchvision import datasets
from torchvis import util
# import ipdb

use_gpu = torch.cuda.is_available()

# model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))


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
   mean=[0.485, 0.456, 0.406],
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
            normalize
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms[x])
             for x in ['train']}  
    return (dsets['train'].classes)


   
    
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



def Occlusion_exp(image,occluding_size,occluding_stride,model,preprocess,classes,groundTruth):    
    img = np.copy(image)
    height, width,_= img.shape
    output_height = int(math.ceil((height-occluding_size)/occluding_stride+1))
    output_width = int(math.ceil((width-occluding_size)/occluding_stride+1))
    ocludedImages=[]
    for h in range(output_height):
        for w in range(output_width):
            #occluder region
            h_start = h*occluding_stride
            w_start = w*occluding_stride
            h_end = min(height, h_start + occluding_size)
            w_end = min(width, w_start + occluding_size)
            
            input_image = copy.copy(img)
            input_image[h_start:h_end,w_start:w_end,:] =  0
            ocludedImages.append(preprocess(Image.fromarray(input_image)))
            
    L = np.empty(output_height*output_width)
    L.fill(groundTruth)
    L = torch.from_numpy(L) # groundtrutch label vector for all occluded images
    tensor_images = torch.stack([img for img in ocludedImages]) # tensor of occluded images
    dataset = torch.utils.data.TensorDataset(tensor_images,L) 
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=5,shuffle=False, num_workers=8) 

    heatmap=np.empty(0)
    model.eval()
    for data in dataloader:
        images, labels = data
        
        if use_gpu:
            images, labels = (images.cuda()), (labels.cuda(non_blocking=True))
        
        outputs = model(Variable(images))
        m = nn.Softmax(dim=1)
        outputs=m(outputs)
        if use_gpu:   
            outs=outputs.cpu()
        
        # ipdb.set_trace()
        # heatmap = np.concatenate((heatmap,outs[0:outs.size()[0],groundTruth].data.numpy()))
        heatmap = np.concatenate((heatmap,outs[:,groundTruth].data.numpy()))
        
    return heatmap.reshape((output_height, output_width))    



def Saliency_map(image,model,vis_param_dict,preprocess,ground_truth,use_gpu=False,method=util.GradType.GUIDED):
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

