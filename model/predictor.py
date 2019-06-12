import torchvision.models as Models
from torch import nn

class PredictNet(nn.Module):
    def __init__(self, neure_num):
        super(PredictNet, self).__init__()
        self.mlp = make_layers(neure_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.mlp(x)
        y = self.sigmoid(y)
        return y

def make_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        layers += [nn.Linear(input_dim, output_dim), nn.ReLU(inplace = True)]
        input_dim = output_dim
    return nn.Sequential(*layers)
