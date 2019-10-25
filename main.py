from model import *
from data import *
import os
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/thyroid/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_thyroid.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=5,callbacks=[model_checkpoint])

# model = load_model('unet_thyroid.hdf5')
testGene = testGenerator("data/thyroid/test",num_image=159)
results = model.predict_generator(testGene,159,verbose=1)
saveResult("data/thyroid/test",results)