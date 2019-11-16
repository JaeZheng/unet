from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
import random

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        # 每个图像减去训练集图像的均值再除以标准差
        img = (img - 68.58) / 46.29
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def my_train_data_loader(batch_size, num_image, train_path, image_folder, mask_folder, target_size=(128,128), as_gray=True):
    for i in range(num_image):
        crop_rows, crop_cols = target_size
        batch = np.array((batch_size, crop_cols, crop_rows))
        img = io.imread(os.path.join(train_path, image_folder, "%d.bmp"%i),as_gray = as_gray)
        mask = io.imread(os.path.join(train_path, mask_folder, "%d.bmp"%i),as_gray = as_gray)
        img = (img - 68.58) / 46.29
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        batch_img, batch_mask = [], []
        input_rows, input_cols = img.shape
        for _ in range(batch_size):
            start_x = random.randint(0, input_rows-crop_rows-1)
            start_y = random.randint(0, input_cols-crop_cols-1)
            crop_patch = img[start_x:start_x+crop_rows, start_y:start_y+crop_cols]
            crop_patch = np.reshape(crop_patch,crop_patch.shape+(1,))
            crop_mask = mask[start_x:start_x+crop_rows, start_y:start_y+crop_cols]
            crop_mask = np.reshape(crop_mask,crop_mask.shape+(1,))
            batch_img.append(crop_patch)
            batch_mask.append(crop_mask)
        batch_img, batch_mask = np.array(batch_img), np.array(batch_mask)
        yield (batch_img, batch_mask)


def my_test_data_loader(num_image, test_path, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.bmp"%i),as_gray = as_gray)
        img = (img - 68.58) / 46.29
        yield img


def preddict_single_image(model, image, target_size=(128,128)):
    crop_rows, crop_cols = target_size
    input_rows, input_cols = image.shape
    x1, y1, dx, dy, stride = 0, 0, crop_rows, crop_cols, 16
    pred_cnt = np.zeros((input_rows, input_cols, 1), dtype=np.int8)
    pred_result = np.zeros((input_rows, input_cols, 1))
    while x1 <= input_rows-dx:
        while y1 <= input_cols-dy:
            crop_patch = image[x1:x1+dx, y1:y1+dy]
            crop_patch = np.reshape(crop_patch,crop_patch.shape+(1,))
            crop_patch = np.reshape(crop_patch,(1,)+crop_patch.shape)
            crop_result = model.predict(crop_patch)
            crop_result = np.reshape(crop_result, (crop_rows, crop_cols, 1))
            crop_result[crop_result > 0.5] = 1
            crop_result[crop_result <= 0.5] = 0
            pred_cnt[x1:x1+dx, y1:y1+dy] += 1
            pred_result[x1:x1+dx, y1:y1+dy] += crop_result
            y1 += stride
        y1 = 0
        x1 += stride
    pred_result = pred_result/pred_cnt
    pred_result[pred_result > 0.5] = 1
    pred_result[pred_result <= 0.5] = 0
    return pred_result


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        # cv2.imwrite("./tmp/img.png", img[0])
        # cv2.imwrite("./tmp/mask.png", mask[0])
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        # cv2.imwrite("./tmp/img_adjust.png", img[0])
        # cv2.imwrite("./tmp/mask_adjust.png", mask[0])
        yield (img,mask)


def testGenerator(test_path,num_image = 30,target_size = (512,512),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.bmp"%i),as_gray = as_gray)
        img = (img - 68.58) / 46.29
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.bmp"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        img[img<=0.5] = 0
        img[img>0.5] = 1
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


def meanIOU(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, 0.5), K.floatx())
    y_true = K.cast(K.greater(y_true, 0.5), K.floatx())
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return K.switch(K.equal(union, 0), 1.0, intersection / union)