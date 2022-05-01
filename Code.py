#!/usr/bin/env python
# coding: utf-8

# In[30]:


import argparse
import os 
import random
import matplotlib.pyplot as plt
from os.path import join
import math 
import glob 
import numpy as np 
import torch.nn as nn
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2

import torchvision.datasets as dset


# In[144]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100 , 32 , 4,1,0 ,bias = False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32 , 16 , 4,2,1 ,bias = False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16 , 8 , 4,2,1 ,bias = False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(8 , 3 , 4,2,1 ,bias = False),
            nn.Tanh()
            )
    def forward(self,noise):
        return self.main(noise)

        
        
        
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )
        
    def forward(self , img):
        return self.main(img)
        

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)        


# In[151]:


def train(batch_size= 64, epochs=200 , lr=0.0002):

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    
    os.makedirs("result/history",exist_ok = True)
    os.makedirs("result/trained_model",exist_ok = True)
    cuda = True if torch.cuda.is_available() else False
    
    # Loss function
    # Initialize generator and discriminator
    Loss = torch.nn.BCELoss()
    G = Generator()
    D = Discriminator()

    G_Loss_history = []
    D_Loss_history = []
    
    if cuda :
        G.cuda()
        D.cuda()
        L.cuda()
    
    # Initialize weights
    G.apply(weights_init)
    D.apply(weights_init)
    
    # Optimizer (betas)  ############
    OPT_G = torch.optim.Adam(G.parameters(), lr=lr*5 ,betas = (0.5,0.999))
    OPT_D = torch.optim.Adam(D.parameters(), lr=lr ,betas = (0.5,0.999)) 
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # read data
    Dataset = dset.ImageFolder(root='./data/',
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.CenterCrop(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    
    
    Dataloader = torch.utils.data.DataLoader(Dataset, batch_size=batch_size, shuffle=True)
    
    # Training 
    for epoch in range(epochs): 
        for i , imgs in enumerate(Dataloader):
            

            # latent_dim = 100
            Real_label = Variable(Tensor(imgs[0].shape[0], 1,1,1).fill_(1.0), requires_grad=False)
            Fake_label = Variable(Tensor(imgs[0].shape[0], 1,1,1).fill_(0.0), requires_grad=False)
            Real_imgs  = Variable(imgs[0].type(Tensor))
            noise      = Variable(Tensor(np.random.normal(0, 1,(imgs[0].shape[0], 100,1,1))))
            
            # Train Generator
            OPT_G.zero_grad()
            Fake_imgs = G.forward(noise)
            G_Loss = Loss ( D.forward(Fake_imgs) , Real_label )
            G_Loss.backward()
            OPT_G.step()
            
            # Train Discriminaator 
            OPT_D.zero_grad()
            Real_img_Loss = Loss( D.forward(Real_imgs         ), Real_label)
            Fake_img_Loss = Loss( D.forward(Fake_imgs.detach()), Fake_label)
            D_Loss = (Real_img_Loss + Fake_img_Loss)/2
            D_Loss.backward()
            OPT_D.step()
            
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                  (epoch, epochs, i, len(Dataloader), D_Loss.item(), G_Loss.item()))
        
            G_Loss_history.append(G_Loss)
            D_Loss_history.append(D_Loss)
            
        # print_interval ############
        save_image(Fake_imgs.data[:25],"result/history/%d.png" % epoch,nrow=5,normalize=True)
        os.makedirs("result/trained_model/%d" % epoch, exist_ok=True)
        torch.save(D.state_dict(), "result/trained_model/%d/D.pt" % epoch)
        torch.save(G.state_dict(), "result/trained_model/%d/G.pt" % epoch)
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_Loss_history,label="G")
    plt.plot(D_Loss_history,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# In[152]:


def generate_image(model_number):
    os.makedirs("result/images_Fake", exist_ok=True)
    cuda = True if torch.cuda.is_available() else False
    
    # Initialize generator and discriminator
    G = Generator()
    D = Discriminator()
    
    # Load trained models
    G.load_state_dict(torch.load("result/trained_model/"+ model_number + '/G.pt'))
    
    noise = Variable(Tensor(np.random.normal(0, 1, (100, 100))))
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Fake_imgs = G.forward(noise)
    
    for i in tqdm(range(0, 100)):
        save_image(Fake_imgs.data[i],"result/images_Fake/%d/%d.png" % model_number,i,normalize=True)


# In[153]:


if __name__ == "__main__" :
    train()


# In[125]:





# In[ ]:





# In[ ]:




