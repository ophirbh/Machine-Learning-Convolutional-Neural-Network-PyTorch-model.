import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optimizer


class NeuralNet(nn.Module):
    def __init__(self, image_size):
        super(NeuralNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 1000)
        self.fc1 = nn.Linear(1000, 30)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = func.relu(self.fc0(x))
        x = func.relu(self.fc1(x))
        ans = func.log_softmax(x, dim=1)
        return ans
