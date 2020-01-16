# -*- coding: UTF-8 -*-

import os, sys
import numpy as np
from tensorflow.keras.preprocessing import image
import random
def DataSet():
    train_path_face='data/train/tfaces'
    train_path_face_dlatents='data/train/latents'
    # os.listdir(path) 是 python 中的函数，它会列出 path 下的所有文件名
    os.makedirs(train_path_face, exist_ok=True)
    imglist_train_facelatents = os.listdir(train_path_face_dlatents)
    os.makedirs(train_path_face_dlatents, exist_ok=True)
    '''
    # 读取 /test/face 下的所有图片文件名
    os.makedirs(test_path_face, exist_ok=True)
    imglist_test_face = os.listdir(test_path_face)
    os.makedirs(test_path_face_dlatents, exist_ok=True)
    '''
    # 定义两个 numpy 对象，X_train 和 Y_train

    # X_train 对象用来存放训练集的图片。每张图片都需要转换成 numpy 向量形式
    # X_train 的 shape 是 (，256，256，3)
    # resnet50 缺省的输入图片尺寸是 (224,224) ,我们这里设置为(256,256)
    # 3 是图片的通道数（rgb）

    # Y_train 用来存放训练集中每张图片对应的dlatents
    # Y_train 的 shape 是 （，18,512），与StyleGAN的dlatents一致
    X_train = np.empty((len(imglist_train_facelatents), 256, 256, 3))
    Y_train = np.empty((len(imglist_train_facelatents), 18, 512))

    # count 对象用来计数，每添加一张图片便加 1
    count = 0
    # 遍历 /train/face 下所有图片，即训练集下所有的图片
    for latent_name in imglist_train_facelatents:
        # 得到图片的路径
        if latent_name.endswith('.npy') :
            latent_path = os.path.join(train_path_face_dlatents, latent_name)
            # 通过 image.load_img() 函数读取对应的图片，并转换成目标大小
            # image 是 tensorflow.keras.preprocessing 中的一个对象
            latent =np.load(latent_path)
            # 将处理好的图片装进定义好的 X_train 对象中
            Y_train[count] = latent
            # 将对应的标签装进 Y_train 对象中
            # 这里需要载入StyleGAN生成图片s时对应的dlatents
            only_name = os.path.splitext(latent_name)[0]
            img_name = only_name + '.png'
            img_path = os.path.join(train_path_face, img_name)
            img=image.load_img(img_path, target_size=(256, 256))
            # 将图片转换成 numpy 数组，并除以 255 ，归一化
            # 转换之后 img 的 shape 是 （256，256，3）
            img = image.img_to_array(img) / 255.0
            X_train[count] = img
            count += 1
    '''
    # 准备测试集的数据
    X_test = np.empty((len(imglist_test_face), 256, 256, 3))
    Y_test = np.empty((len(imglist_test_face), 18, 512))
    count = 0
    for img_name in imglist_test_face:
        if img_name.endswith('png') or img_name.endswith('jpg'):
            img_path = os.path.join(test_path_face, img_name)
            img = image.load_img(img_path, target_size=(256, 256))
            img = image.img_to_array(img) / 255.0
            X_test[count] = img

            only_name = os.path.splitext(img_name)[0]
            img_name = only_name + '.npy'
            img_path = os.path.join(test_path_face_dlatents, img_name)
            Y_test[count] = np.load(img_path)
            count += 1
    '''
    # 打乱训练集中的数据
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    '''
    # 打乱测试集中的数据
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]
    '''
    return X_train, Y_train

def main():
    X_train, Y_train= DataSet()
if __name__ == "__main__":
    main()