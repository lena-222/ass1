import torch
import torchvision
import torchvision.models
import torchvision.models.resnet


class ResNet18(torch.nn.Module):
    def __init__(self):

        #model = torchvision.models.resnet18(pretrained=pretrained)

        #model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1),
        #                                      stride=(1, 1))  # Change final layer to 3 classes
        #self.model = model
        super(ResNet18, self).__init__()
        self.in_features = 512
        self.out_features = 37
        self.backbone = torchvision.models.resnet18(weights=torchvision.ResNet18_Weights.DEFAULT, pretrained=True)
        self.backbone.fc = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features)
        #self.backbone.fc(torchvision.nn.linear(num_classes))

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
        return self.backbone.forward(x)

class ConvNextTiny(torch.nn.Module):
    def __init__(self):
        #model = torchvision.models.convnext_tiny(weights=torchvision.ConvNeXt_Tiny_Weights.DEFAULT,pretrained=pretrained)
        #model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1),
        #                                      stride=(1, 1))  # Change final layer to 3 classes
        #self.model = model
        super(ConvNextTiny, self).__init__()
        self.in_features = 512
        self.out_features = 37
        self.backbone = torchvision.models.convnext_tiny(weights=torchvision.ConvNeXt_Tiny_Weights.DEFAULT, pretrained=True)
        # TODO Google fully connected Schicht Namen googlen.
        self.backbone.fc = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features)

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
        return self.backbone.forward(x)

