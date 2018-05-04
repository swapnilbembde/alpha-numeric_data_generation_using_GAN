import os
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np
# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, input_size=32, n_class = 10):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.tanh(self.fc4(x))

        return x

def shows_result(num_epoch, show = False, save = False, path = 'result.png'):
    z_ = torch.randn((4*4, 100))
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_)
    test_images=test_images.cpu()
#    print(test_images.shape())
    G.train()
    test_images = test_images.transpose(test_images.FLIP_TOP_BOTTOM)

    test_images = test_images.rotate(90)


    size_figure_grid = 4
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(4, 4))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(4*4):
        i = k // 4
        j = k % 4
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='Greys_r')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.05, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        flipped = np.fliplr(img)
        return np.rot90(flipped)

def save_images(num_epoch, show = False, save = False, path = 'result.png'):
    z_ = torch.randn((4*4, 100))
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_)
    # print test_images.size()
    G.train()

    size_figure_grid = 4
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(4, 4))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(4*4):
        i = k // 4
        j = k % 4
        ax[i, j].cla()
        ax[i, j].imshow(rotate(test_images[k, :].cpu().data.view(28, 28).numpy()), cmap='Greys_r')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.05, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
# network
G = generator(input_size=100, n_class=28*28)
G.cuda()
epochs = 100

for epoch in range(epochs):

    G.load_state_dict(torch.load('emnist_results/generator_param.pkl'))
    p = 'emnist_results/images/EMNIST_GAN_' + str(epoch + 1) + '.png'
    save_images((epoch+100), show= True, save=False, path=p)