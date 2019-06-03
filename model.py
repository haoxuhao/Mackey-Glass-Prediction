#encoding=utf8

import torch 
from torch import nn

class Net(torch.nn.Module):
        def __init__(self, n_feature, nums_hidden, n_output):
            super(Net,self).__init__()
            
            self.backbone = self.make_layers(n_feature, nums_hidden)
            self.output_layer = torch.nn.Linear(nums_hidden[-1], n_output)
            self.output_act = nn.ReLU(True)
            self._init_params()

        def forward(self, x):
            x = self.backbone(x)
            out = self.output_layer(x)
            #out = self.output_act(out)
            return out

        def _init_params(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def make_layers(self, n_features, nums_hidden):
            layers = []
            for i, n_hidden in enumerate(nums_hidden):
                if i==0:
                    layers+=[torch.nn.Linear(n_features, n_hidden), nn.ReLU(inplace=True)]
                else:
                    layers+=[torch.nn.Linear(nums_hidden[i-1], n_hidden), nn.ReLU(inplace=True)]

            return nn.Sequential(*layers)
                    
if __name__=="__main__":
    net = Net(5, [15,15], 1)
    x = torch.rand(3, 5)
    out = net(x)
    print(out.shape)
    print(out)

