import torch.nn as nn
import torch.nn.functional as F

class toy_VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()
        self.features = features
        #self.output_dim = output_dim
        self.avgpool = nn.AdaptiveAvgPool2d(7)

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),            
        )

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0],-1)
        x = self.classifier(h)
        return(x,h)
