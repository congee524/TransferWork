import torch.nn.functional as F
from torchvision.models import vgg16
import torch
from StyleTransfer.net import StyleTransfer


class PostSynthesisProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.min_value = -1
        self.max_value = 1

    def forward(self, synthesized_image):
        synthesized_image = (synthesized_image - self.min_value) * \
            torch.tensor(255).float() / (self.max_value - self.min_value)
        synthesized_image = torch.clamp(
            synthesized_image + 0.5, min=0, max=255)

        return synthesized_image


class VGGProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.image_size = 256
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device="cuda").view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225],
                                device="cuda").view(-1, 1, 1)

    def forward(self, image):
        image = image / torch.tensor(255).float()
        image = F.adaptive_avg_pool2d(image, self.image_size)

        image = (image - self.mean) / self.std

        return image


class ImageProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.image_size = 256

    def forward(self, image):
        image = F.adaptive_avg_pool2d(image, self.image_size)

        return image


class LatentOptimizer(torch.nn.Module):
    def __init__(self, synthesizer, layer=12):
        super().__init__()

        self.synthesizer = synthesizer.cuda().eval()
        self.post_synthesis_processing = PostSynthesisProcessing().cuda().eval()
        self.transfer = StyleTransfer().cuda().eval()
        self.vgg_processing = VGGProcessing().cuda().eval()
        self.img_processing = ImageProcessing()
        self.vgg16 = vgg16(pretrained=True).features[:layer].cuda().eval()

    def forward(self, dlatents, style):
        # should follow the style transfer block
        # torch.Size([16, 3, 1024, 1024])
        generated_image = self.synthesizer(dlatents)
        #print(generated_image)
        generated_image = self.post_synthesis_processing(
            generated_image)  # scale to [0 255] torch.Size([16, 3, 1024, 1024])
        #generated_image = self.transfer(generated_image, style)
        generated_image = self.img_processing(generated_image)
        # generated_image = self.vgg_processing(generated_image)  # 256
        # features = self.vgg16(generated_image)  # generate feature map
        #print(generated_image)
        return generated_image
