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
    X_train = np.empty((len(imglist_train_facelatents), 256, 256, 3))
    Y_train = np.empty((len(imglist_train_facelatents), 18, 512))
    count = 0
    for latent_name in imglist_train_facelatents:
        if latent_name.endswith('.npy') :
            latent_path = os.path.join(train_path_face_dlatents, latent_name)
            latent =np.load(latent_path)
            Y_train[count] = latent
            only_name = os.path.splitext(latent_name)[0]
            img_name = only_name + '.png'
            img_path = os.path.join(train_path_face, img_name)
            img=image.load_img(img_path, target_size=(256, 256))
            img = image.img_to_array(img) / 255.0
            X_train[count] = img
            count += 1
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    return X_train, Y_train

def main():
    X_train, Y_train= DataSet()
if __name__ == "__main__":
    main()