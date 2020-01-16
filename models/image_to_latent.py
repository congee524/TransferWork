import torch
from torchvision.models import resnet50
from models.LocallyConnected2d import LocallyConnected2d
from PIL import Image
import numpy as np


class ImageToLatent(torch.nn.Module):
    def __init__(self, image_size=256):
        super().__init__()

        self.image_size = image_size
        self.activation = torch.nn.ELU()

        self.resnet = list(resnet50(pretrained=True).children())[:-2]
        self.resnet = torch.nn.Sequential(*self.resnet)
        self.conv2d = torch.nn.Conv2d(2048, 144, kernel_size=1)
        self.localconv = LocallyConnected2d(1, 1, (96, 96), 1).cuda()

        # self.resnet = list(resnet50(pretrained=True).children())[:-2]
        # self.resnet = torch.nn.Sequential(*self.resnet)
        # self.conv2d = torch.nn.Conv2d(2048, 256, kernel_size=1)
        # self.flatten = torch.nn.Flatten()
        # self.dense1 = torch.nn.Linear(16384, 256)
        # self.dense2 = torch.nn.Linear(256, (18 * 512))

    def forward(self, image):
        x = self.resnet(image)
        x = self.conv2d(x)
        x = x.view((-1, 1, 96, 96))
        for _ in range(3):
            x = self.localconv(x)
            # print(x.size())
            # assert False
            x = x.permute(0, 1, 3, 2)
        x = self.localconv(x)
        x = x.view(-1, 18, 512)
        # print(x.size())
        # assert False
        # # torch.Size([16, 3, 256, 256])
        # x = self.resnet(image)  # torch.Size([16, 2048, 8, 8])
        # x = self.conv2d(x)  # torch.Size([16, 256, 8, 8])
        # x = self.flatten(x)  # torch.Size([16, 16384])
        # x = self.dense1(x)  # torch.Size([16, 256])
        # x = self.dense2(x)  # torch.Size([16, 9216])
        # x = x.view((-1, 18, 512))  # torch.Size([16, 18, 512])
        return x


class ImageLatentDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, dlatents, image_size=256, transforms=None):
        self.filenames = filenames
        self.dlatents = dlatents
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        dlatent = self.dlatents[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))

        if self.transforms:
            image = self.transforms(image)

        return image, dlatent

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))

        return image
