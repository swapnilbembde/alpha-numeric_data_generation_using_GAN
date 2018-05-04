import os
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from Generator import generator
from Discriminator import discriminator
from Visualize import loss_plots

# creating folders for results
if not os.path.isdir('mnist_results'):
    os.mkdir('mnist_results')
if not os.path.isdir('mnist_results/images'):
    os.mkdir('mnist_results/images')


def save_images(num_epoch, show = False, save = False, path = 'result.png'):
    z_ = torch.randn((4*4, 100))
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_)
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
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='Greys_r')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.05, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

################################### Main Code ######################################
print ("training...")

lr = 0.0002
epochs = 100

# data_loader
# transforms.ToTensor() = torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
# Tensor image of size (C, H, W) to be normalized. i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# if data not present then it downloads, takes train part
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=64,shuffle=True)


# networks
G = generator(input_size=100, n_class=28*28)
D = discriminator(input_size=28*28, n_class=1)
G.cuda()
D.cuda()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
for epoch in range(epochs):
    D_losses = []
    G_losses = []
    for load_data, _ in train_loader:

        # training discriminator ############
        # manually setting gradients to zero before mini batches 
        D.zero_grad()

        # format
        load_data = load_data.view(-1, 28 * 28)
        # print load_data.size()[0]
        mini_batch = load_data.size()[0]

        D_real = torch.ones(mini_batch)
        D_fake = torch.zeros(mini_batch)

        # variables in pytorch can directly be accessed
        load_data  = Variable(load_data.cuda())
        D_real = Variable(D_real.cuda())
        D_fake = Variable(D_fake.cuda())

        # first it takes real data 
        D_result = D(load_data)
        # loss calculations due to real data : first term in eqn
        # comparing with ones labels
        D_real_loss = F.binary_cross_entropy(D_result, D_real)
        D_real_scores = D_result

        ## for loss due to generated samples
        noise = torch.randn((mini_batch, 100))
        noise = Variable(noise.cuda())

        G_sample = G(noise)
        D_result = D(G_sample)
        # loss calculations due to generated data : second term in eqn
        # comparing with zero labels
        D_fake_loss = F.binary_cross_entropy(D_result, D_fake)
        D_fake_scores = D_result
        # total D_loss
        D_train_loss = D_real_loss + D_fake_loss

        # training of network
        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data[0])

        # training generator ##############

        # manually setting gradients to zero before mini batches 
        G.zero_grad()

        noise = torch.randn((mini_batch, 100))
        out = torch.ones(mini_batch)

        # variables in pytorch can directly be accessed
        noise = Variable(noise.cuda())
        out = Variable(out.cuda())
        # noise input to generator 
        G_result = G(noise)
        D_result = D(G_result)
        # comparing with ones labels
        # loss calculations due to generated data : generator's loss
        G_train_loss = F.binary_cross_entropy(D_result, out)
        # training of network
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch + 1), epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

    p = 'mnist_results/images/' + str(epoch + 1)+ '.png'
    save_images((epoch+1),  save=True, path=p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))


print("Finished training!")

### showing and saving the results ###############
loss_plots(train_hist, save=True, path='mnist_results/train_hist.png')
torch.save(G.state_dict(), "mnist_results/generator_param.pkl")
torch.save(D.state_dict(), "mnist_results/discriminator_param.pkl")
with open('mnist_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

# creating gif file     
images = []
for i in range(epochs):
    img_name = 'mnist_results/images/' + str(i + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('mnist_results/gif_file.gif', images, fps=5)



