import torch
import torchvision
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.models import ResNet18_Weights


# import torchvision.models.resnet


class ResNet18(torch.nn.Module):
    """Resnet."""

    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_features = 512
        self.out_features = 37
        self.backbone = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.fc = torch.nn.Linear(in_features=512, out_features=self.out_features)

    def forward(self, x):
        '''
                x = self.backbone.conv1(x)
                x = self.backbone.relu(self.backbone.bn1(x))
                x = self.backbone.conv1(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)

                return x
        '''
        x = self.backbone.conv1(x)
        x = self.backbone.relu(self.backbone.bn1(x))
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ConvNextTiny(torch.nn.Module):
    """ConvNextTiny based on a Resnet."""

    def __init__(self):
        super(ConvNextTiny, self).__init__()
        self.in_features = 512
        self.out_features = 37
        self.backbone = torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.fc = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features)

        self.classifier_layer_norm = ConvNextTiny.LayerNorm2d(768)
        self.classifier_linear = torch.nn.Linear(in_features=768, out_features=self.out_features)

    def forward(self, x):
        '''
                x = self.backbone.conv1(x)
                x = self.backbone.relu(self.backbone.bn1(x))
                x = self.backbone.conv1(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)

                return x
        '''

        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = self.classifier_layer_norm(x)
        x = torch.flatten(x, 1)
        x = self.classifier_linear(x)

        return x
