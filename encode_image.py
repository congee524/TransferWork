import argparse
from tqdm import tqdm
import numpy as np
import torch
import time
import datetime
import os
import PIL.Image
import cv2
from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import LatentOptimizer, ImageProcessing, PostSynthesisProcessing
from models.image_to_latent import ImageToLatent
from models.losses import LatentLoss
from utilities.hooks import GeneratedImageHook
from utilities.images import load_images, images_to_video, save_image
from utilities.files import validate_path
from utilities.dataset import make_dataset, get_data_loader
from utilities import make_logger

parser = argparse.ArgumentParser(
    description="Find the latent space representation of an input image.")
# parser.add_argument("image_path", help="Filepath of the image to be encoded.")
# parser.add_argument(
#     "dlatent_path", help="Filepath to save the dlatent (WP) at.")
parser.add_argument("--save_optimized_image", default=False,
                    help="Whether or not to save the image created with the optimized latents.", type=bool)
parser.add_argument("--optimized_image_path", default="./output/image/",
                    help="The path to save the image created with the optimized latents.", type=str)
parser.add_argument("--output_dir", default="./output",
                    help="The path to save the output results.", type=str)
parser.add_argument("--video", default=False,
                    help="Whether or not to save a video of the encoding process.", type=bool)
parser.add_argument("--video_path", default="video.avi",
                    help="Where to save the video at.", type=str)
parser.add_argument("--save_frequency", default=10,
                    help="How often to save the images to video. Smaller = Faster.", type=int)
parser.add_argument("--epochs", default=10000,
                    help="Number of epoches.", type=int)
parser.add_argument("--model_type", default="stylegan_ffhq",
                    help="The model to use from InterFaceGAN repo.", type=str)
parser.add_argument("--learning_rate", default=2.5e-6,
                    help="Learning rate for SGD.", type=int)
parser.add_argument("--vgg_layer", default=12,
                    help="The VGG network layer number to extract features from.", type=int)

args, other = parser.parse_known_args()


def optimize_latents():
    # logger
    logger = make_logger("project", args.output_dir, 'log')

    logger.info("Optimizing Latents.")  # optimize to latent
    global_time = time.time()

    # the module generating image
    synthesizer = StyleGANGenerator(args.model_type).model.synthesis
    # the optimizer severs as getting the featuremap of the image
    latent_optimizer = LatentOptimizer(synthesizer, args.vgg_layer)

    # Optimizer only the dlatents.
    for param in latent_optimizer.parameters():  # frozen the parameters
        param.requires_grad_(False)

    if args.video or args.save_optimized_image:  # if need save video or the optimized_image
        # Hook, saves an image during optimization to be used to create video.
        generated_image_hook = GeneratedImageHook(
            latent_optimizer.post_synthesis_processing, args.save_frequency)

    image_to_latent = ImageToLatent().cuda()
    # image_to_latent.load_state_dict(torch.load('output/models'))
    image_to_latent.train()

    dataset = make_dataset(
        folder=False, img_dir='./data/sy_imgs', resolution=1024)
    data = get_data_loader(dataset, batch_size=16, num_workers=4)

    criterion = LatentLoss()
    optimizer = torch.optim.Adam(
        image_to_latent.parameters(), lr=args.learning_rate)

    save_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda')
    imgprocess = ImageProcessing().cuda().eval()
    imgpostprocess = PostSynthesisProcessing().cuda().eval()
    sum_loss = 0

    for epoch in range(1, args.epochs + 1):
        # logger.info("Epoch: [%d]" % epoch)

        for (i, batch) in enumerate(data, 1):
            reference_image = batch.to(device)
            style_image = reference_image.clone()  # [1, 3, 1024, 2024]
            reference_image = imgprocess(reference_image)
            # print(reference_image.size())
            # print(reference_image[0][1]) # 值在0到1之间 tensor([-0.2157, -0.2157, -0.2235,  ...,  0.0431,  0.0431,  0.0431]
            # reference_image = latent_optimizer.vgg_processing(
            #    reference_image)  # normalization
            # print(reference_image.size()) # ([1, 3, 256, 256])
            # print(reference_image[0][1][1]) # 这里出了问题，去了和上面一样的tensor，基本都变成了-2.03
            # assert False
            reference_features = latent_optimizer.vgg16(reference_image)

            # reference_image = reference_image.detach()

            # latents should be get from the gmapping class from stylegan
            latents_to_be_optimized = image_to_latent(reference_image)
            # print(latents_to_be_optimized.size()) # torch.Size([1, 18, 512])
            # print(latents_to_be_optimized)
            # latents_to_be_optimized = latents_to_be_optimized.detach().cuda().requires_grad_(True)
            latents_to_be_optimized = latents_to_be_optimized.cuda().requires_grad_(True)

            generated_image = latent_optimizer(
                latents_to_be_optimized, style_image)
            generated_image_features = latent_optimizer.vgg16(generated_image)
            # print(generated_image_features.size()) # torch.Size([1, 3, 256, 256])
            # print(generated_image_features[0][1])
            # assert False
            loss = criterion(generated_image_features, reference_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        if (epoch) % 1 == 0:
            elapsed = time.time() - global_time
            elapsed = str(datetime.timedelta(
                seconds=elapsed)).split('.')[0]
            logger.info(
                "Elapsed: [%s] Epoch: [%d] Step: %d  Loss: %f"
                % (elapsed, epoch, i, sum_loss / 100))
            print(latents_to_be_optimized[0,:10,:10])
            print()
            sum_loss = 0
            model_save_file = os.path.join(
                save_dir, "ImageToLatent" + "_" + str(epoch) + ".pth")
            torch.save(image_to_latent.state_dict(), model_save_file)
            image_dir = args.optimized_image_path + \
                str(epoch) + "_" + ".jpg"
            save_img = imgpostprocess(
                generated_image[0].detach().cpu()).numpy()
            save_image(save_img, image_dir)

    if args.video:
        images_to_video(generated_image_hook.get_images(), args.video_path)
    if args.save_optimized_image:
        save_image(generated_image_hook.last_image, args.optimized_image_path)


def main():
    # assert(validate_path(args.image_path, "r"))
    # assert(validate_path(args.dlatent_path, "w"))  # is validate path?
    # vgg layer number must be in valid range
    assert(1 <= args.vgg_layer <= 16)
    if args.video:
        assert(validate_path(args.video_path, "w"))  # whether transf to video
    if args.save_optimized_image:
        # whether save the image optimized
        assert(validate_path(args.optimized_image_path, "w"))
    # assert(validate_path(args.save_dir, "r"))  # ? latent_finder

    optimize_latents()


if __name__ == "__main__":
    main()
