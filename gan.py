import os
import argparse
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
from Visualize import loss_plots,rotate
from tqdm import tqdm

def save_images(num_epoch, show = False, save = False, path = 'result.png',dataset_dir = 'EMNIST'):
    if (dataset_dir == 'MNIST'):
        dim = 4
    else:
        dim = 7

    z_ = torch.randn((dim*dim, 100))
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_)
    G.train()

    size_figure_grid = dim
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(dim, dim))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(dim*dim):
        i = k // dim
        j = k % dim
        ax[i, j].cla()
        ax[i, j].imshow(rotate(test_images[k, :].cpu().data.view(28, 28).numpy()), cmap='Greys_r')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.05, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

################################### Main Code ######################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN")
    parser.add_argument("--dataset_dir", type=str, default="MNIST",  ## directory is name of the dataset
                        help="which dataset")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate for training (default: 0.0002)")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    epochs = args.num_epochs
    lr = args.lr

    # creating folders for results
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.isdir(dataset_dir + '/images'):
        os.mkdir(dataset_dir+'/images')
    # data_loader
    # transforms.ToTensor() = torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    # Tensor image of size (C, H, W) to be normalized. i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # if data not present then it downloads, takes train part
    print "loading dataset ..."
    if dataset_dir == 'EMNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST(dataset_dir +'/data',split = 'bymerge', train=True, download=True, transform=transform),
            batch_size=128)

    if dataset_dir == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(dataset_dir + '/data', train=True, download=True, transform=transform),
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
    for epoch in tqdm(range(epochs)):
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
            # D_real_scores = D_result

            ## for loss due to generated samples
            noise = torch.randn((mini_batch, 100))
            noise = Variable(noise.cuda())

            G_sample = G(noise)
            D_result = D(G_sample)
            # loss calculations due to generated data : second term in eqn
            # comparing with zero labels
            D_fake_loss = F.binary_cross_entropy(D_result, D_fake)
            # D_fake_scores = D_result
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

        p = dataset_dir + '/images/' + str(epoch + 1)+ '.png'
        save_images((epoch+1),  save=True, path=p, dataset_dir = dataset_dir)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))


    print("Finished training!")

    ### showing and saving the results ###############
    loss_plots(train_hist, save=True, path=dataset_dir + '/EMNIST_GAN_train_hist.png')
    torch.save(G.state_dict(), dataset_dir + "/generator_param.pkl")
    torch.save(D.state_dict(), dataset_dir + "/discriminator_param.pkl")
    with open(dataset_dir + '/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    # creating gif file     
    images = []
    for i in range(epochs):
        img_name = dataset_dir + '/images/' + str(i + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(dataset_dir + '/gif_file.gif', images, fps=5)
