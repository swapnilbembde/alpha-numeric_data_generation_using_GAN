import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
## takes values of losses and plots the graph
def loss_plots(hist, show = False, save = False, path = 'losses.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def rotate(img):
        # Used to rotate images
        flipped = np.fliplr(img)
        return np.rot90(flipped)