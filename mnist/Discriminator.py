import torch
import torch.nn as nn
import torch.nn.functional as F

# discriminator from the paper
class discriminator(nn.Module):
    # initializers
    def __init__(self, input_size=32, n_class=10):
        super(discriminator, self).__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, 256)
        self.fc3 = nn.Linear(self.fc2.out_features, n_class)        

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)        
        x = F.sigmoid(self.fc3(x))
        return x
