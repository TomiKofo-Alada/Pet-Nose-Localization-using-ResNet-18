import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights


class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet18.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.regression_head = nn.Linear(512, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.regression_head(x)
        return x


def get_pet_nose_model():
    print("Loaded pre-trained ResNet-18 with custom head")
    return CustomResNet18()

