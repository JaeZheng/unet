from model import *
from data import *
import os
from keras.models import load_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from data import meanIOU

# 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 自适应分配显存
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
# data_gen_args = dict()
myGene = trainGenerator(6,'data/thyroid/train','image','label',data_gen_args,save_to_dir=None,target_size=(400,496))

model = unet(input_size=(400,496,1))

if os.path.exists("unet_thyroid.hdf5.txt"):
    os.remove("unet_thyroid.hdf5.txt")
with open("unet_thyroid.hdf5.txt",'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

model_checkpoint = ModelCheckpoint('unet_thyroid.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=10,callbacks=[model_checkpoint])

# model = load_model('unet_thyroid.hdf5', custom_objects={'meanIOU':meanIOU})
testGene = testGenerator("data/thyroid/test", num_image=59,target_size=(400,496))
results = model.predict_generator(testGene,59,verbose=1)
saveResult("data/thyroid/test",results)

import test