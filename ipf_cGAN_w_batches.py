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
from CAxis_to_Sine import convert_CAxis_to_Sine, convert_CAxis_to_Sine_test, get_transform
import torch.nn.functional as func
import math


"""Set parameters for network"""
# Root directory for dataset
dataroot = "./datasets/"
# Number of channels in the training images. For color images this is 3
nc = 1
# Number of training epochs
num_epochs = 2000
# Learning rate for optimizers
lr = 0.0001
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
#dataset size
dataset_size = 500
#adjust dot_product loss scaling factor
lambda_dot_product = 0.25
#set True if you want to have output files from testing samples
test = False


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
        self.conv = nn.Conv1d(1,2, kernel_size=2, stride=1)
        self.bn = nn.BatchNorm1d(2)
        self.rl = nn.LeakyReLU(0.2, inplace=True)
        self.lin = nn.Linear(4, 1)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        x = self.rl(self.bn(self.conv(input)))
        x = x.view(-1,4)
        x = self.sig(self.lin(x))
        return x

def display_losses(epochs, losses, loss_type):
    plt.clf()
    plt.plot(epochs, losses)
    plt.title("NN Loss After "+str(num_epochs)+" epochs"+loss_type)
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss Value")
    plt.savefig("./results/"+str(num_epochs)+"_epochs_loss_plot"+loss_type+".png")

def print_fake_results(fake, file_num):
    file = open("./result_files/CAxis_fake_results_"+str(file_num)+".txt", "w")

    file.write("CAxisLocation_0,CAxisLocation_1,CAxisLocation_2\n")
    for i in range(0, fake.shape[0]):
        file.write(str(fake[i,0,0].item())+","+str(fake[i,0,1].item())+","+str(fake[i,0,2].item())+"\n")

    file.close()


def check_anti_parallel(real, fake, i, dp):
    anti = False
    mag_real = math.sqrt(real[i,0,0].item()*real[i,0,0].item()+real[i,0,1].item()*real[i,0,1].item()+real[i,0,2].item()*real[i,0,2].item())
    mag_fake = math.sqrt(fake[i,0,0].item()*fake[i,0,0].item()+fake[i,0,1].item()*fake[i,0,1].item()+fake[i,0,2].item()*fake[i,0,2].item())
    cos = dp/(1.0*mag_real*mag_fake)
    if (cos < -1.0):
        cos = -1.0
    elif (cos > 1.0):
        cos = 1.0
    ang = math.acos(cos)
    if (abs(ang)>(math.pi*0.5)):
      anti = True
    return anti


if __name__ == '__main__':
    """Create dataset """
    sine_list, CAxis_list = convert_CAxis_to_Sine(dataroot, True)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    """Setupt model """
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

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
        random.shuffle(sine_list)
        for i in range(0, dataset_size):
            sine_file_stats = sine_list[i][0]
            CAxis_number = sine_list[i][1]
            CAxis_file_stats = CAxis_list[CAxis_number-1]


            # reshape tensor for input into generator
            current_sine = torch.reshape(sine_file_stats, (sine_file_stats.shape[1], 1, 36))

            current_CAxis = torch.reshape(CAxis_file_stats, (CAxis_file_stats.shape[1], 1, 3)).to(device)

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
            errD_real.backward()
            D_x = output_D.mean().item()


            #train with fake
            fake = netG(real_cpu)

            fake = torch.reshape(fake, (fake.shape[0], 1, 3))

            label = torch.FloatTensor(batch_size,1).uniform_(0.75, 1.0).to(device)
            output_D = netD(fake.detach())
            errD_fake = criterionD(output_D, label)
            errD_fake.backward()

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
                dp = current_CAxis[j,0,0]*fake[j,0,0]+current_CAxis[j,0,1]*fake[j,0,1]+current_CAxis[j,0,2]*fake[j,0,2]
                if (check_anti_parallel(current_CAxis, fake, j, dp)):
                    dp = (1.0-dp)
                dot_sum += dp
            dot_product = (1.0*dot_sum)/(1.0*fake.shape[0])

            #adjust generator error
            errG = criterionD(output_D, label) + criterionG(fake, current_CAxis) + (1.0-dot_product)*lambda_dot_product
            errG.backward()
            D_G_z2 = output_D.mean().item()
            optimizerG.step()

            G_loss_sum += errG.item()
            D_loss_sum += errD.item()
            dot_product_sum += dot_product.item()

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

    if (test):
        """Start Testing"""
        print("Starting Testing ...")
        sine_test_list, CAxis_test_list = convert_CAxis_to_Sine_test(dataroot, True)
        for i in range(0, len(sine_test_list)):
            print("Printing output file "+str(i)+"/"+str(len(sine_test_list)))
            current_sine = torch.reshape(sine_test_list[i][0], (sine_test_list[i][0].shape[1], 1, 36))
            current_CAxis = torch.reshape(CAxis_test_list[i], (CAxis_test_list[i].shape[1], 1, 3))
            file_number = sine_test_list[i][1]

            fake = netG(current_sine)
            for h in range(0, fake.shape[0]):
                magnitude = fake[h,0]*fake[h,0] + fake[h,1]*fake[h,1] + fake[h,2]*fake[h,2]
                one_CAxis = fake[h,:]
                fake[h,:] = torch.div(one_CAxis, math.sqrt(magnitude))

            fake = torch.reshape(fake, (fake.shape[0], 1, 3))

            print_fake_results(fake, file_number)
