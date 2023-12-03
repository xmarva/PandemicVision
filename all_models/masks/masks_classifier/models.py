import cv2
import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image


classes = ['no_mask', 'with_mask']

transform = transforms.Compose([
                transforms.Resize(128),
                transforms.RandomCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
                    ])


def get_model(device):
    """
    Load untrained ResNet-34 model from torch  and change last fully connected
    and global average pool layer for binary classification task
    :param device: str 'cuda' or 'cpu' or 'cuda:'number''
    :return: torch model
    """
    model = torchvision.models.resnet34(progress=True).to(device)
    model.avgpool = nn.AdaptiveAvgPool2d(1).cuda()
    model.fc = nn.Linear(512 * torchvision.models.resnet.BasicBlock.expansion, 2).cuda()

    return model


def check_face(model, img):
    """
    forward
    :param model: torch model
    :param img: np.ndarray image of single face
    :return:
        confidence[pred_max]: torch.Tensor with shape 1
        pred_max: np.ndarray with shape 1. Predicted class
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img.astype('uint8'), 'RGB')

    image = transform(image)
    image = image.cuda()

    image = image.unsqueeze(dim=0)
    model.eval()
    pred = model(image)
    confidence = torch.nn.functional.softmax(pred).squeeze()
    pred_max = torch.argmax(confidence).cpu().numpy()

    return confidence[pred_max], pred_max