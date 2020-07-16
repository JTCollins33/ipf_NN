from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from CAxis_to_Sine import convert_CAxis_to_Sine
import torch.nn.functional as func


"""Set parameters for network"""
# Root directory for dataset
dataroot = "./datasets/"
# Number of channels in the training images. For color images this is 3
nc = 1
# Number of training epochs
num_epochs = 150
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
#dataset size
dataset_size = 500


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Conv1d(1, 2, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(2)
        self.conv2 = nn.Conv1d(2, 4, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(4)
        self.lin1 = nn.Linear(128, 3)

    def forward(self, input):
        x = func.relu(self.bn1(self.conv1(input)))
        x = func.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 128)
        x = self.lin1(x)
        x = x.view(3)
        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def display_losses(epochs, losses, loss_type):
    plt.clf()
    plt.plot(epochs, losses)
    plt.title("NN Loss After "+str(num_epochs)+" epochs")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss Value")
    plt.savefig("./results/"+str(num_epochs)+"_epochs_loss_plot"+loss_type+".tif")


if __name__ == '__main__':
    """Create dataset """
    sine_list, CAxis_list = convert_CAxis_to_Sine(dataroot, True)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    """Setupt model """
    netG = Generator(ngpu)

    # Initialize L1Loss function
    criterion = nn.L1Loss()


    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    G_losses = []
    dot_products = []
    epochs = []
    iters = 0

    print("Starting Training Loop...")
    """Loop for training"""
    for epoch in range(1, num_epochs+1):
        for i in range(0, dataset_size):
            sine_file_stats = sine_list[i]
            CAxis_file_stats = CAxis_list[i]
            for j in range(0, sine_file_stats.shape[1]):
                current_sine = sine_file_stats[0,j,:]
                current_CAxis = CAxis_file_stats[0,j,:]

                #get tensor into proper shape
                current_sine = torch.reshape(current_sine, (1,1,36))

                netG.zero_grad()
                fake = netG(current_sine)

                dot_product = fake[0]*current_CAxis[0] + fake[1]*current_CAxis[1] + fake[2]*current_CAxis[2]

                #calculate loss and backpropagate
                loss = criterion(fake, current_CAxis)  
                loss.backward()
                optimizerG.step() 
        print("Completed epoch "+str(epoch)+"/"+str(num_epochs))
        G_losses.append(loss.item())
        dot_products.append(dot_product)
        epochs.append(epoch)
    display_losses(epochs, G_losses, "")
    display_losses(epochs, dot_products, "_dot_products")



