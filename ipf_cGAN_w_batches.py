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
from CAxis_to_Sine import convert_CAxis_to_Sine, get_transform
import torch.nn.functional as func
import math


"""Set parameters for network"""
# Root directory for dataset
dataroot = "./datasets/"
# Number of channels in the training images. For color images this is 3
nc = 1
# Number of training epochs
num_epochs = 300
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0
#dataset size
dataset_size = 500
#adjust dot_product loss scaling factor
lambda_dot_product = 100


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm1d(32)
        self.lin1 = nn.Linear(480, 128)
        # self.bn4 = nn.BatchNorm1d(128)
        self.lin2 = nn.Linear(128, 3)


    def forward(self, input):
        x = func.relu(self.bn1(self.conv1(input)))
        x = func.relu(self.bn2(self.conv2(x)))
        x = func.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 480)
        x = func.relu(self.lin1(x))
        x = self.lin2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # self.main = nn.Sequential(
        #     nn.Conv1d(1,2, kernel_size=2, stride=1)
        #     nn.BatchNorm1d(2)
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Sigmoid()
        # )
        self.conv1 = nn.Conv1d(1,2, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm1d(2)
        self.rl1 = nn.LeakyReLU(0.2, inplace=True)
        self.lin1 = nn.Linear(4, 1)
        self.sig1 = nn.Sigmoid()

    def forward(self, input):
        x = self.rl1(self.bn1(self.conv1(input)))
        x = x.view(-1,4)
        x = self.sig1(self.lin1(x))
        return x
        # return torch.mean(self.main(input)).item()

def display_losses(epochs, losses, loss_type):
    plt.clf()
    plt.plot(epochs, losses)
    plt.title("NN Loss After "+str(num_epochs)+" epochs"+loss_type)
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss Value")
    plt.savefig("./results/"+str(num_epochs)+"_epochs_loss_plot"+loss_type+".png")


if __name__ == '__main__':
    """Create dataset """
    sine_list, CAxis_list = convert_CAxis_to_Sine(dataroot, True)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    """Setupt model """
    netG = Generator(ngpu)
    netD = Discriminator(ngpu)

    # Initialize Loss function
    criterionD = nn.BCELoss()
    criterionG = nn.MSELoss()

    real_label = 0
    fake_label = 1

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    dot_products = []
    epochs = []
    iters = 0

    tens_transform = get_transform()

    #shuffle sine list for randomized training
    random.shuffle(sine_list)

    print("Starting Training Loop...")
    """Loop for training"""
    for epoch in range(1, num_epochs+1):
        G_loss_sum = 0.0
        D_loss_sum = 0.0
        dot_product_sum = 0.0
        for i in range(0, dataset_size):
            sine_file_stats = sine_list[i][0]
            CAxis_number = sine_list[i][1]
            CAxis_file_stats = CAxis_list[CAxis_number-1]


            # reshape tensor for input into generator
            current_sine = torch.reshape(sine_file_stats, (sine_file_stats.shape[1], 1, 36))

            current_CAxis = torch.reshape(CAxis_file_stats, (CAxis_file_stats.shape[1], 1, 3))

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            #train with real
            netD.zero_grad()
            real_cpu = current_sine.to(device)
            batch_size = CAxis_file_stats.shape[1]
            label = torch.FloatTensor(batch_size,1).uniform_(0, 0.25).to(device)
            
            output_D = netD(current_CAxis)

            errD_real = criterionD(output_D, label)
            errD_real.backward(retain_graph=True)
            D_x = output_D.mean().item()


            #train with fake
            fake = netG(current_sine)
            for h in range(0, fake.shape[0]):
                magnitude = fake[h,0]*fake[h,0] + fake[h,1]*fake[h,1] + fake[h,2]*fake[h,2]
                one_CAxis = fake[h,:]
                fake[h,:] = torch.div(one_CAxis, math.sqrt(magnitude))

            fake = torch.reshape(fake, (fake.shape[0], 1, 3))

            label = torch.FloatTensor(batch_size,1).uniform_(0.75, 1.0).to(device)
            output_D = netD(fake)
            errD_fake = criterionD(output_D, label)
            errD_fake.backward(retain_graph=True)

            D_G_z1 = output_D.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output_D = netD(fake)
            label = torch.FloatTensor(batch_size,1).uniform_(0, 0.25).to(device)

            #calculate dot product
            dot_sum = 0
            for j in range(0, fake.shape[0]):
                dot_sum += current_CAxis[j,0,0]*fake[j,0,0]+current_CAxis[j,0,1]*fake[j,0,1]+current_CAxis[j,0,2]*fake[j,0,2]
            dot_product = (1.0*dot_sum)/(1.0*fake.shape[0])

            errG = criterionD(output_D, label)
            #adjust generator error
            errG = errG + criterionG(fake, current_CAxis) + dot_product*lambda_dot_product
            errG.backward()
            D_G_z2 = output_D.mean().item()
            optimizerG.step()

            G_loss_sum += errG.item()
            D_loss_sum += errD.item()
            dot_product_sum += dot_product

        print('Epoch [%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            % (epoch, num_epochs,
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        D_losses.append(D_loss_sum/(1.0*dataset_size))
        G_losses.append(G_loss_sum/(1.0*dataset_size))
        dot_products.append(dot_product_sum/(1.0*dataset_size))
        epochs.append(epoch)
    display_losses(epochs, D_losses, "_D_losses")
    display_losses(epochs, G_losses, "_G_losses")
    display_losses(epochs, dot_products, "_dot_products")
