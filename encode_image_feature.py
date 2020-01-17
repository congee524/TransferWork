import argparse
import random
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import numpy as np
import torch
import time
import datetime
import os
import PIL.Image
import cv2
from models.feature_mapping import FeatureMapping
from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import LatentOptimizer, ImageProcessing, PostSynthesisProcessing
from models.image_to_latent import ImageToLatent
from models.losses import LatentLoss
from utilities.hooks import GeneratedImageHook
from utilities.images import load_images, images_to_video, save_image
from utilities.files import validate_path
from utilities.dataset import make_dataset, get_data_loader
from utilities import make_logger
from data_create import DataSet
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
parser.add_argument("--epochs", default=2000,
                    help="Number of epoches.", type=int)
parser.add_argument("--model_type", default="stylegan_ffhq",
                    help="The model to use from InterFaceGAN repo.", type=str)
parser.add_argument("--learning_rate", default=2.5e-3,
                    help="Learning rate for SGD.", type=int)
parser.add_argument("--vgg_layer", default=12,
                    help="The VGG network layer number to extract features from.", type=int)

args, other = parser.parse_known_args()

def result_get():
    synthesizer = StyleGANGenerator(args.model_type).model.synthesis
    latent_optimizer = LatentOptimizer(synthesizer, args.vgg_layer)
    image_to_latent = ImageToLatent().cuda()
    FeatureMapper = FeatureMapping().cuda()
    FeatureMapper.load_state_dict(torch.load('./output/models/FeatureMapper_90.pth'))
    image_to_latent.load_state_dict(torch.load('output/models/ImageToLatent_90.pth'))
    X_train=np.zeros((2,256,256,3))
    #img1 = image.load_img('./data/train/tfaces/0020_01.png', target_size=(256, 256))
    #img2 = image.load_img('./data/train/tfaces/0022_01.png', target_size=(256, 256))
    img1 = image.load_img('./data/sy_imgs/example0.png', target_size=(256, 256))
    img2 = image.load_img('./data/sy_imgs/example14.png', target_size=(256, 256))

    img1 = image.img_to_array(img1) / 255.0
    img2 = image.img_to_array(img2) / 255.0
    X_train[0] = img1
    X_train[1] = img2
    device = torch.device('cuda')
    smile = np.array((np.load(
        './InterFaceGAN/boundaries/stylegan_ffhq_smile_w_boundary.npy')))
    smile_w = torch.from_numpy(np.tile(np.concatenate(
        (np.tile(smile, (9, 1)), np.zeros((9, 512))), axis=0), (2, 1, 1))).cuda()
    batchimg = X_train.astype(np.float16)
    reference_image = torch.from_numpy(batchimg).float().to(device)
    reference_image = reference_image.permute(0, 3, 1, 2)
    latents_optimized = image_to_latent(reference_image)
    latents_to_be_optimized = latents_optimized.view(-1, 18 * 512)
    latents_to_be_optimized = FeatureMapper(latents_to_be_optimized)
    latents_to_be_optimized = latents_to_be_optimized.view(-1, 18, 512)
    latents_to_be_optimized = latents_to_be_optimized.cuda().requires_grad_(True)
    l1 = latents_to_be_optimized.clone()
    for j in range(30):
        lx=l1.clone()
        lx+=smile_w/10*j
        gen=latent_optimizer(lx,batchimg)
        for i in range(2):
            image_dir = args.optimized_image_path + \
                        "video_" + str(i) +"_ra_"+str(j)+".jpg"
            save_img = gen[i].detach().cpu().numpy()
            save_image(save_img, image_dir)
    # Using for Presentation

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
    FeatureMapper=FeatureMapping().cuda()
    FeatureMapper.load_state_dict(torch.load('./output/models/FeatureMapper_90.pth'))
    #image_to_latent = torch.load('./output/models/ImageToLatent_105.pth')
    # best 110
    # try below 100
    image_to_latent.load_state_dict(torch.load('output/models/ImageToLatent_90.pth'))
    # for param in image_to_latent.parameters():  # frozen the parameters
    #   param.requires_grad_(False)
    # You can use models from scatch

    #image_to_latent.train() # training for the first step
    FeatureMapper.train()# The feature mapping step
    X_train,Y_train=DataSet()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        image_to_latent.parameters(), lr=args.learning_rate)
    save_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda')
    imgprocess = ImageProcessing().cuda().eval()
    imgpostprocess = PostSynthesisProcessing().cuda().eval()
    sum_loss = 0
    batch_size=8
    for epoch in range(1, args.epochs + 1):
        # logger.info("Epoch: [%d]" % epoch)
        index = [i for i in range(len(X_train))]
        random.shuffle(index)
        X_train = X_train[index]
        Y_train = Y_train[index]
        index_b=[]
        ite=int(len(X_train)/batch_size)
        for i in range(ite):
            batchx=index[i*batch_size:i*batch_size+batch_size]
            batchy=Y_train[i*batch_size:i*batch_size+batch_size]
            index_b.append(batchx)
        for (i, batchind) in enumerate(index_b,1):
            batchimg=X_train[batchind].astype(np.float16)
            batchlats=Y_train[batchind].astype(np.float16)
            reference_image = torch.from_numpy(batchimg).float().to(device)
            reference_image=reference_image.permute(0,3,1,2)
            generated_features=torch.from_numpy(batchlats).float().to(device)
            latents_optimized = image_to_latent(reference_image)
            # On first step , this do not use
            latents_to_be_optimized = latents_optimized.view(-1,18*512)
            latents_to_be_optimized = FeatureMapper(latents_to_be_optimized)
            latents_to_be_optimized = latents_to_be_optimized.view(-1, 18,512)
            latents_to_be_optimized = latents_to_be_optimized.cuda().requires_grad_(True)
            generated_imgs = latent_optimizer(latents_to_be_optimized, batchimg)
            nimgs = latent_optimizer(generated_features, batchimg)
            gen_feat = latent_optimizer.vgg16(imgprocess(generated_imgs))
            n_feat=latent_optimizer.vgg16(imgprocess(nimgs))
            # until here , for the second step (Mapping)
            # loss=criterion(latents_to_be_optimized,generated_features)
            loss = criterion(gen_feat,n_feat)+criterion(generated_imgs,nimgs) # second loss for second step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        if (epoch) % 1 == 0:
            print("epoch ["+str(epoch)+"] finished!")
        if (epoch) % 10== 0:
            elapsed = time.time() - global_time
            elapsed = str(datetime.timedelta(
                seconds=elapsed)).split('.')[0]
            logger.info(
                "Elapsed: [%s] Epoch: [%d] Step: %d  Loss: %f"
                % (elapsed, epoch, i, sum_loss / 300))
            sum_loss = 0
            model_save_file = os.path.join(
                save_dir, "FeatureMapper" + "_" + str(epoch) + ".pth")
            torch.save(FeatureMapper.state_dict(), model_save_file)
            model_save_file = os.path.join(
                save_dir, "ImageToLatent" + "_" + str(epoch) + ".pth")
            torch.save(image_to_latent.state_dict(), model_save_file)
            image_dir = args.optimized_image_path + \
                str(epoch) + "_" + ".jpg"
            save_img = generated_imgs[0].detach().cpu().numpy()
            save_image(save_img, image_dir)
            image_dir = args.optimized_image_path + \
                        str(epoch) + "_r" + ".jpg"
            save_image(nimgs[0].detach().cpu().numpy(), image_dir)
    if args.video:
        images_to_video(generated_image_hook.get_images(), args.video_path)
    if args.save_optimized_image:
        save_image(generated_image_hook.last_image, args.optimized_image_path)
def main():
    assert(1 <= args.vgg_layer <= 16)
    if args.video:
        assert(validate_path(args.video_path, "w"))  # whether transf to video
    if args.save_optimized_image:
        # whether save the image optimized
        assert(validate_path(args.optimized_image_path, "w"))
    #optimize_latents()
    result_get()

if __name__ == "__main__":
    main()
