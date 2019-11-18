#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : JaeZheng
# @Time    : 2019/11/11 17:19
# @File    : finetune.py

from model import *
from data import *
import os
import keras.backend.tensorflow_backend as KTF
from data import meanIOU
# 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 自适应分配显存
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)


# prepare the 2D model
input_channels, input_rows, input_cols, input_deps = 1, 64, 64, 32
num_class, activate = 2, 'sigmoid'
weight_dir = 'Vnet-genesis_thyroid_us.h5'
models_genesis = unet(input_size=(128,128,1))
print("Load pre-trained Models Genesis weights from {}".format(weight_dir))
models_genesis.load_weights(weight_dir)
x = models_genesis.get_layer('conv2d_23').output
print(models_genesis.input.shape)
print(x.shape)
final_convolution = Conv2D(1, 1)(x)
output = Activation(activate)(final_convolution)
model = Model(inputs=models_genesis.input, outputs=output)
adam_optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam_optimizer, loss = 'binary_crossentropy', metrics = ['accuracy', meanIOU])

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
# data_gen_args = dict()
myGene = trainGenerator(1,'data/thyroid/train/','image','label',data_gen_args,save_to_dir = None, target_size=(400,496))
# myGene = my_train_data_loader(16,50,'data/thyroid/train','image','label',target_size=(128,128))

# model = unet(pretrained_weights='Vnet-genesis_thyroid_us.h5')
model_checkpoint = ModelCheckpoint('finetune_unet_thyroid.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=500,epochs=500,callbacks=[model_checkpoint])

if os.path.exists("finetune_unet_thyroid.hdf5.txt"):
    os.remove("finetune_unet_thyroid.hdf5.txt")
with open("finetune_unet_thyroid.hdf5.txt",'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

# model = load_model('unet_thyroid.hdf5')
# testGene = testGenerator("data/thyroid/test",num_image=59, target_size=(400,496))
# results = model.predict_generator(testGene,59,verbose=1)
# saveResult("data/thyroid/test",results)
testGene = my_test_data_loader(59, "data/thyroid/test")
cnt = 0
for img in testGene:
    result = predict_single_image(model, img, target_size=(128,128))
    saveSingleResult("data/thyroid/test", result, cnt)
    cnt += 1

import test