# some useful functions
import os
import shutil
import math
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import skimage

input_sizes = {'MNIST':28*28, 'FashionMNIST':28*28, 'CIFAR10':32*32*3, 'CIFAR100':32*32*3}
output_sizes = {'MNIST':10, 'FashionMNIST':10, 'CIFAR10':10, 'CIFAR100':100}
input_channels = {'MNIST':1, 'FashionMNIST':1, 'CIFAR10':3, 'CIFAR100':3}


def hinge_regression(output, target, epsilon=.1, type='quadratic'):
    power = 1 if type=='linear' else 2
    delta = (output-target).abs()-epsilon
    loss = torch.nn.functional.relu(delta)*delta.pow(power)
    return loss.mean()

def hinge_classification(output,target,epsilon=.5, type='quadratic'):
    power = 1 if type=='linear' else 2
    output_size=output.size(1)
    if output_size==1:
        target = 2*target.double()-1
        print(target,output)
        return 0.5*(epsilon-output*target).mean()
    delta = torch.zeros(output.size(0))
    for i,(out,tar) in enumerate(zip(output,target)):
        tar = int(tar)
        delta[i] = epsilon + torch.cat((out[:tar],out[tar+1:])).max() - out[tar]
    loss = 0.5 * torch.nn.functional.relu(delta).pow(power).mean()
    return loss
    
def normalize(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std[std==0]=1
    return (x-mean)/std

def shuffle_labels(labels, label_noise, num_classes):
    for i in range(len(labels)):
        if np.random.rand()<label_noise:
            labels[i] = np.random.randint(0, num_classes)
    return labels

def get_data(dataset_name, num_classes=None, d=None, n=None, label_noise=0., dataset_path='~/data', device='cuda'):

    dataclass = eval('Fast'+dataset_name)
    tr_data = dataclass(dataset_path, train=True, download=True)
    te_data = dataclass(dataset_path, train=False, download=True)

    if n is not None:
        tr_data.data = tr_data.data[:n]
        tr_data.targets = tr_data.targets[:n]

    if num_classes is None:
        output_size = output_sizes[dataset_name]
    else:
        output_size = num_classes
        tr_data.targets = tr_data.targets%num_classes
        te_data.targets = te_data.targets%num_classes
        
    if label_noise:
        for dataset in [tr_data, te_data]:
            dataset.targets = shuffle_labels(dataset.targets, label_noise, output_size)
        
    return tr_data, te_data, input_sizes[dataset_name], output_size

def resize_data(tr_data, te_data, d):
    
    assert d**.5 == int(d**.5)
    input_size = int(d**.5)
    print('Starting the resizing')
    for dataset in [tr_data, te_data]:
        size, channels, _, _, = dataset.data.size()
        new_data = torch.empty(size, input_size, input_size)
        for i, img in enumerate(dataset.data):
            new_data[i] = torch.from_numpy(skimage.transform.resize(img.mean(0).cpu(), (input_size, input_size)))
        dataset.data = new_data
    print('Finished the resizing')

    return tr_data, te_data

def get_pca(tr_data, te_data, d, normalized = True):
    
    device = tr_data.device
    tr_data.data = tr_data.data.reshape(tr_data.data.size(0),-1)
    te_data.data = te_data.data.reshape(te_data.data.size(0),-1)
    x = tr_data.data.cpu()
    m = x.mean(dim=0)
    std = x.std(dim=0)
    std[std==0]=1
    x = normalize(x)
    u,s,v = torch.svd(torch.t(x))
    if normalized:
        tr_data.data = x @ u[:, :d].to(device) / s[:d].to(device)
        te_data.data = normalize(te_data.data) @ u[:, :d].to(device) / s[:d].to(device)
    else:
        tr_data.data = x @ u[:, :d].to(device).to(device)
        te_data.data = normalize(te_data.data) @ u[:, :d].to(device).to(device)

    tr_data.data = (tr_data.data - tr_data.data.mean())/tr_data.data.std()
    te_data.data = (te_data.data - te_data.data.mean())/te_data.data.std()

    print(tr_data.data.std(dim=0))

    return tr_data, te_data


def get_teacher_labels(data, task, batch_size, noise, num_classes=None, teacher=None, train=True):

    n_batches = int(len(data)//batch_size)
    with torch.no_grad():
        dataset = []
        if task.endswith('classification'):
            for i in range(n_batches):
                x = data[i*batch_size:(i+1)*batch_size]
                y = teacher(x).max(1)[1].squeeze()
                for j in range(len(y)):
                    if np.random.random()<noise:
                        y[j]= np.random.randint(num_classes)
                dataset.append((x,y.long()))
        elif task.endswith('regression'):
            for i in range(n_batches):
                x = data[i*batch_size:(i+1)*batch_size]
                y = teacher(x)+noise*torch.randn((batch_size,1))
                dataset.append((x,y))
        else:
            raise
    return dataset

def rot(x, th):
    with torch.no_grad(): 
        rotation = torch.eye(len(x))
        rotation[:2,:2] = torch.Tensor([[np.cos(th),np.sin(th)],[-np.sin(th), np.cos(th)]]) 
        return rotation @ x

def who_am_i():
    import subprocess
    whoami = subprocess.run(['whoami'], stdout=subprocess.PIPE)
    whoami = whoami.stdout.decode('utf-8')
    whoami = whoami.strip('\n')
    return whoami

def copy_py(dst_folder):
    # and copy all .py's into dst_folder
    if not os.path.exists(dst_folder):
        print("Folder doesn't exist!")
        return
    for f in os.listdir():
        if f.endswith('.py'):
            shutil.copy2(f, dst_folder)

class FastMNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, device="cpu",**kwargs): 
        super().__init__(*args, **kwargs)

        self.device=device
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

class FastFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, *args, device='cpu',**kwargs): 
        super().__init__(*args, **kwargs)

        self.device=device
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target
    
class FastCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, device='cpu', **kwargs):
        super().__init__(*args, **kwargs)

        self.device=device
        self.data = torch.from_numpy(self.data).float().div(255).transpose(3,1)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), torch.LongTensor(self.targets).to(self.device)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target
    
class FastCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.data = torch.from_numpy(self.data).float().div(255).transpose(3,1)
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        self.data, self.targets = self.data.to(self.device), torch.LongTensor(self.targets).to(self.device)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

