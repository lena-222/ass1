import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNNResnet:
    """Faster RCNN based on a Resnet."""

    def __init__(self, num_classes, pretrained=True):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=pretrained)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Stop here if you are fine-tunning Faster-RCNN

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        self.model = model


class DeepLabV3Resnet:
    def __init__(self, num_classes, pretrained=True):
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)

        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes

        self.model = model

class ResNet18:
    def __init__(self, num_classes, pretrained=True):
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)

        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1),
                                              stride=(1, 1))  # Change final layer to 3 classes

        self.model = model


