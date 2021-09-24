import sys

from functions import *
from hyperparameters import  * #show_images,plot_history,factor,noise,testName,numEpochsPretrain ,patiencePretrain ,lrPretrain ,batch_size_valuePretrain ,no_schedule,schedulePretrain,model_namePretrain,optimizer_namePretrain ,loss_acronymPretrain ,max_poolingPretrain,numEpochs ,patience,lr,batch_size_value ,schedule ,model_name ,optimizer_name ,loss_acronym ,max_pooling,repetitions,train_encoder,bottleneck_freezing,GPU
import os
import tensorflow as tf
#load the required example functions from pix2pix tf.pack

from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
import os
import random
print('Se ha cargado las dependencias correctamente')
if GPU_availability:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU;
print('Se ha seleccionado la GPU:',GPU)

try:
  os.mkdir(testName)
except:
 print('Already created folder')

try:
  os.mkdir('Modelo')
except:
 print('Already created folder')
try:
  os.mkdir('Modelo/Iteraciones1')
except:
 print('Already created folder')
try:
  os.mkdir('Modelo/Iteraciones2')
except:
 print('Already created folder')

nameSavingFile=str((testName+'.xlsx'))


def set_seed(seedValue=42):
  """Sets the seed on multiple python modules to obtain results as
  reproducible as possible.
  Args:
  seedValue (int, optional): seed value.
  """
  random.seed(a=seedValue)
  np.random.seed(seed=seedValue)
  tf.random.set_seed(seedValue)
  os.environ["PYTHONHASHSEED"]=str(seedValue)
  
set_seed()

print( tf.__version__ )

#@title
import os

print('Cargando los data sets...')
# Read the list of file names
train_input_filenames1 = [x for x in os.listdir( train_input_path1 ) if x.endswith(".png")]
train_input_filenames1.sort()

train_label_filenames1 = [x for x in os.listdir( train_label_path1 ) if x.endswith(".png")]
train_label_filenames1.sort()

print( 'Dataset1 input images loaded: ' + str( len(train_input_filenames1)) )
print( 'Dataset1 label images loaded: ' + str( len(train_label_filenames1)) )


#@title
from skimage.util import img_as_ubyte
from skimage import io
from matplotlib import pyplot as plt

# read training images and labels
train_img1 = [ img_as_ubyte( io.imread( train_input_path1 + '/' + x ) )/255.0 for x in train_input_filenames1 ]
train_lbl1 = [ img_as_ubyte( io.imread( train_label_path1 + '/' + x ) )/255.0 for x in train_label_filenames1 ]

# display first image
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow( train_img1[0], 'gray' )
plt.title( 'Full-size training image' );
# and its "ground truth"
plt.subplot(1, 2, 2)
plt.imshow( train_lbl1[0], 'gray' )
plt.title( 'Ground truth' );
plt.savefig("Dataset1_training_imgs.png", bbox_inches='tight', pad_inches=0)

#@title

# Read the list of file names
test_input_filenames1 = [x for x in os.listdir( test_input_path1 ) if x.endswith(".png")]
test_input_filenames1.sort()

test_label_filenames1 = [x for x in os.listdir( test_label_path1 ) if x.endswith(".png")]
test_label_filenames1.sort()

print( 'Test input images loaded: ' + str( len(test_input_filenames1)) )
print( 'Test label images loaded: ' + str( len(test_label_filenames1)) )
# Read test images
test_img1 = [ img_as_ubyte( io.imread( test_input_path1 + '/' + x ) )/255. for x in test_input_filenames1 ]
test_lbl1 = [ img_as_ubyte( io.imread( test_label_path1 + '/' + x,as_gray=True) )/255. for x in test_label_filenames1 ]


# Display corresponding first patch at low resolution
plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow( test_img1[0], 'gray' )
plt.title( 'Test image' )
# Side by side with its "ground truth"
plt.subplot(1, 2, 2)
plt.imshow( test_lbl1[0], 'gray' )
plt.title( 'Ground truth' )
plt.savefig("Dataset1_test_imgs.png", bbox_inches='tight', pad_inches=0)
#@title
#Loading train and validation splits
from sklearn.model_selection import train_test_split
train_img1, val_img1, train_lbl1, val_lbl1 = train_test_split(train_img1,
                                                        train_lbl1,
                                                        train_size=1-0.1,
                                                        test_size=0.1,
                                                        random_state=42, shuffle=False)


"""Now we will check the number of images and masks:"""

#@title
from skimage.util import img_as_ubyte
from skimage import io
from matplotlib import pyplot as plt
from skimage.transform import resize

#@title


# Read the list of file names
train_input_filenames2 = [x for x in os.listdir( train_input_path2 ) if x.endswith(".png")]
train_input_filenames2.sort()

train_label_filenames2 = [x for x in os.listdir( train_label_path2 ) if x.endswith(".png")]
train_label_filenames2.sort()

print( 'Input images loaded: ' + str( len(train_input_filenames2)) )
print( 'Label images loaded: ' + str( len(train_label_filenames2)) )

from skimage.util import img_as_ubyte
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt

# read training images and labels
train_img2 = [ resize((img_as_ubyte( io.imread( train_input_path2 + '/' + x ,as_gray=True) )),(768, 1024),
                       anti_aliasing=True) for x in train_input_filenames2 ]
train_lbl2 = [ resize((img_as_ubyte( io.imread( train_label_path2 + '/' + x ,as_gray=True) )),(768, 1024),
                       anti_aliasing=True)for x in train_label_filenames2 ]

# display first image
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow( train_img2[0], 'gray' )

plt.title( 'Full-size training image' );
# and its "ground truth"
plt.subplot(1, 2, 2)
plt.imshow( train_lbl2[0], 'gray' )
plt.title( 'Ground truth' );
plt.savefig("Dataset2_train_imgs.png", bbox_inches='tight', pad_inches=0)

# Read the list of file names
test_input_filenames2 = [x for x in os.listdir( test_input_path2 ) if x.endswith(".png")]
test_input_filenames2.sort()

test_label_filenames2 = [x for x in os.listdir( test_label_path2 ) if x.endswith(".png")]
test_label_filenames2.sort()

print( 'Test input images loaded: ' + str( len(test_input_filenames2)) )
print( 'Test label images loaded: ' + str( len(test_label_filenames2)) )
# Read test images
test_img2 = [  resize((img_as_ubyte( io.imread( test_input_path2 + '/' + x ,as_gray=True) )),(768, 1024),
                       anti_aliasing=True) for x in test_input_filenames2 ]
test_lbl2 = [  resize((img_as_ubyte( io.imread( test_label_path2 + '/' + x ,as_gray=True) )),(768, 1024),
                       anti_aliasing=True) for x in test_label_filenames2 ]


# Display corresponding first patch at low resolution
plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow( test_img2[0], 'gray' )
plt.title( 'Test image' )
# Side by side with its "ground truth"
plt.subplot(1, 2, 2)
plt.imshow( test_lbl2[0], 'gray' )
plt.title( 'Ground truth' )
plt.savefig("Dataset2_test_imgs.png", bbox_inches='tight', pad_inches=0)

del test_input_filenames1,test_input_filenames2,test_input_path1,test_input_path2

#@title
#Loading train and validation splits
from sklearn.model_selection import train_test_split
train_img2, val_img2, train_lbl2, val_lbl2 = train_test_split(train_img2,
                                                        train_lbl2,
                                                        train_size=1-0.1,
                                                        test_size=0.1,
                                                        random_state=42, shuffle=False)

"""## Preparing the pre-training data

"""

#@title
import random
percentage_data=1 #parameter to be changed in between 0-1 to reduce  randomly the number of annotated patches to be used during training

train_img_patches1,train_lbl_patches1=reduce_number_imgs(create_patches(train_img1,(256,256)),create_patches(train_lbl1,(256,256)),percentage_data=percentage_data,normalize=False)

for i in range(0,3):
  plt.figure(figsize=(10,5))
  plt.subplot(3, 2, 1)
  plt.imshow( train_img_patches1[i] ,'gray')
  plt.colorbar()
  plt.title( 'Input training patch' );
  # and its "ground truth"
  plt.subplot(3, 2, 2)
  plt.imshow( train_lbl_patches1[i] ,'gray')
  plt.title( 'Ground truth patch' );
plt.savefig("Dataset1_train_patches_imgs.png", bbox_inches='tight', pad_inches=0)

val_img_patches1,val_lbl_patches1=reduce_number_imgs(create_patches(val_img1,(256,256)),create_patches(val_lbl1,(256,256)),percentage_data=percentage_data,normalize=False)

#@title
import random
percentage_data=1 #parameter to be changed in between 0-1 to reduce  randomly the number of annotated patches to be used during training
#from SelfSupervisedLearning.general_functions import reduce_number_imgs,add_Gaussian_Noise


train_img_patches2,train_lbl_patches2=reduce_number_imgs(create_patches(train_img2,(256,256)),create_patches(train_lbl2,(256,256)),percentage_data=percentage_data,normalize=False)

for i in range(0,3):
  plt.figure(figsize=(10,5))
  plt.subplot(3, 2, 1)
  plt.imshow( train_img_patches2[i] )
  plt.colorbar()
  plt.title( 'Input training patch' );
  # and its "ground truth"
  plt.subplot(3, 2, 2)
  plt.imshow( train_lbl_patches2[i] ,'gray')
  plt.title( 'Ground truth patch' );
plt.savefig("Dataset2_train_patches_imgs.png", bbox_inches='tight', pad_inches=0)

val_img_patches2,val_lbl_patches2=reduce_number_imgs(create_patches(val_img2,(256,256)),create_patches(val_lbl2,(256,256)),percentage_data=percentage_data,normalize=False)

"""#### Creating noisy patches for the pretraining step mixing both datasets"""

#Create test patches to include in the pretraining
test_img_patches1,test_lbl_patches1=reduce_number_imgs(create_patches(test_img1,(256,256)),create_patches(test_lbl1,(256,256)),percentage_data=percentage_data,normalize=False)
test_img_patches2,test_lbl_patches2=reduce_number_imgs(create_patches(test_img2,(256,256)),create_patches(test_lbl2,(256,256)),percentage_data=percentage_data,normalize=False)

#@title
import numpy as np
import matplotlib.pyplot as plt
#low resolution patches 


#from SelfSupervisedLearning.general_functions import add_Gaussian_Noise,crappify
 #Here we mix both datasets so that we can perform the pretraining step
train_img_patches=train_img_patches1+train_img_patches2+test_img_patches1+test_img_patches2
val_img_patches=val_img_patches1+val_img_patches2

"""**There are 3 options to perform the pretraining step:**
* `hide_fragments`: is meant to perform inpainting it hides fragments of the image by setting them to zero, we can choose the size of the fragments as well as the percentage of the image hidden
* `add_Gaussian_Noise`: is meant to perform denoising by adding Normal Gaussian Noise N(0,$\sigma$) to the images. 
* `crappify`: is meant to simulate superresolution so that we start by adding noise to the image and then downsizing and upsizing it.
"""

#@title
# We will use these patches as "ground truth" for the pretraining step
noisy_train_img=[crappify(x,resizing_factor=factor,add_noise=True,noise_level=noise) for x in train_img_patches]
noisy_val_img=[crappify(x,resizing_factor=factor,add_noise=True,noise_level=noise) for x in val_img_patches]

"""Now we are going to reduce the number of annotated images used for training. This is meant to simulate the case where few annotated images are available as they often require from expert labelling but there's available a bunch of same-context non-annotated images. 

The numbers used in here for the percentages are arbitrary and meant to simulate the percentages used by the DenoiSeg paper in their study.

## Custom loss functions
To better train our segmentation networks, we can define new loss functions:
* [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) which is a proper segmentation metric.

* Weighted BCE and Dice.

* SEG metric which is based in a weighted categorical crossentropy loss where some of the labels(i.e border which are underepresented in the image) are highly weighted in the loss computation.
"""

#@title
from tensorflow.keras import losses
import tensorflow as tf

"""## One-cycle policy for super-convergence
We implement the learning rate on a one-cycle policy as per Leslie Smith's paper (https://arxiv.org/pdf/1803.09820.pdf) for fast convergence.
"""

#@title
#from SelfSupervisedLearning.oneCycle import CosineAnnealer,OneCycleScheduler
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tensorflow.keras.callbacks import Callback

"""## Network definitions
Next, we define our U-Net-like networks. In particular, we define three type of architectures:
* `MobileNetEncoder`is an encoder based in a MobileNet V2 architecture where the bottleneck would be the output of the MobileNet and 4 resolution levels in the expanding path(based in the tutorial from TF https://www.tensorflow.org/tutorials/generative/pix2pix )
* `UNet` is a regular U-Net with 4 resolution levels in the contracting path, a bottleneck, and 4 resolution levels in the expanding path.
* `ResUNet` is a residual U-Net with a user-defined depth (number of levels + bottleneck).
"""

#@title
# Network definitions

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, Conv1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Add, concatenate, Lambda

#@title

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import transform

#@title
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from SelfSupervisedLearning.Network_training import train

#@title
import numpy as np
from skimage.segmentation import find_boundaries
#from SelfSupervisedLearning.DenoiSeg_functions import pixel_sharing_bipartite,intersection_over_union,matching_iou,measure_precision,measure_seg,compute_labels,seg,precision,matching_overlap,convert_to_oneHot,add_boundary_label,onehot_encoding,normalize,denormalize,zero_out_train_data

import numpy as np
from numba import jit
from scipy import ndimage
from tqdm import tqdm, tqdm_notebook

"""###Pretraining - Super resolution

The idea is to **pretrain the network by using the noisy patches** previously created as the input and the **pretraining ground_truth would be the original images**. 

This procedure is meant to **enhance the initial weights of our model to afterwards improve its segmentation performance and its transferability to the target domain**.

As loss function, we use the mean squared error (MSE) between the expected and the predicted pixel values, and we also include the mean absolute error (MAE) as a control metric.

For this step **we will use all the training data**, as even simulating scarcity of labelled data, the unlabelled data might still be available for being used in SSL super resolution.

Furthermore we will evaluate the PSNR SSNR
"""

#@title
# Prepare the training data and create data generators
import numpy as np

# training input
X_train = np.asarray(noisy_train_img)
X_train = np.expand_dims( X_train, axis=-1 ) # add extra dimension
print(X_train[0].shape)

# training ground truth
Y_train = np.asarray(train_img_patches)#here we define our ground_truth
Y_train = np.expand_dims( Y_train, axis=-1 ) # add extra dimension
print(Y_train[0].shape)
# validation input
X_val = np.asarray(noisy_val_img)
X_val = np.expand_dims( X_val, axis=-1 ) # add extra dimension
print(X_train[0].shape)

# validation ground truth
Y_val = np.asarray(val_img_patches)#here we define our ground_truth
Y_val = np.expand_dims( Y_val, axis=-1 ) # add extra dimension
print(Y_train[0].shape)

#@title
values=[]
X_train=X_train
for i in range(len(X_train[:,0,0,0])):
  values.append(np.max(X_train[i,:,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
values=[]
X_val=X_val
for i in range(len(X_val[:,0,0,0])):
  values.append(np.max(X_val[i,:,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
values=[]
Y_train=Y_train
for i in range(len(Y_train[:,0,0,0])):
  values.append(np.max(Y_train[i,:,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
values=[]
Y_val=Y_val
for i in range(len(Y_val[:,0,0,0])):
  values.append(np.max(Y_val[i,:,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title


# Display corresponding first 3 patches
plt.figure(figsize=(15,15))
plt.subplot(1, 2, 1)
plt.imshow( Y_train[0,:,:,0], 'gray' )
plt.title( ' Original image' )
# Side by side with its "ground truth"
plt.subplot(1, 2, 2)
plt.imshow( X_train[0,:,:,0], 'gray' )
plt.title( ' Noisy training image' )
plt.figure(figsize=(15,15))
plt.subplot(1, 2, 1)
plt.imshow( Y_train[2102,:,:,0], 'gray' )
plt.title( ' Original image' )
# Side by side with its "ground truth"
plt.subplot(1, 2, 2)
plt.imshow( X_train[2102,:,:,0], 'gray' )
plt.title( 'Noisy training image' )
plt.savefig("Pretraining_patches_imgs.png", bbox_inches='tight', pad_inches=0)

#@title


history,model=train(X_train,Y_train,X_val,Y_val,numEpochsPretrain,1,patiencePretrain,lrPretrain,lrPretrain,batch_size_valuePretrain,schedulePretrain,model_name,optimizer_namePretrain,loss_acronymPretrain,max_poolingPretrain,train_encoder=True,Denoising=True,preTrain=True)
print('Pre-trained weights are ready to be used!')

model.save('Pretrained_model.h5')

#Evaluar métricas denoising sobre entrenamiento, validación y test

psnr_array_noise = []


print('\n# Generate predictions for all validation samples')

for i in range(0, len(X_val)):
  pred = X_val[i][:,:,0];
  psnr_array_noise.append(metrics.peak_signal_noise_ratio(pred, Y_val[i][:,:,0]));
psnr_mean_noise = np.mean(psnr_array_noise)

ssim_array_noise = []
for i in range(0, len(X_val)):
  pred = X_val[i][:,:,0];
  ssim_array_noise.append(metrics.structural_similarity(pred, Y_val[i][:,:,0]));
ssim_mean_noise = np.mean(ssim_array_noise)

print("PSNR original:", psnr_mean_noise)
print("SSIM original:", ssim_mean_noise)
file1 = open(testName+'.txt',"a")

# \n is placed to indicate EOL (End of Line)
file1.write("PSNR original: "+ str(psnr_mean_noise)+'\n')
file1.write("SSIM original:"+ str(ssim_mean_noise)+'\n')
file1.close() #to change file access modes
  
#Evaluar métricas denoising sobre entrenamiento, validación y test

psnr_array = []


print('\n# Generate predictions for all validation samples')
predictions = model.predict(X_val)
for i in range(0, len(predictions)):
  pred = np.clip( predictions[i][:,:,0], a_min=0, a_max=1 );
  psnr_array.append(metrics.peak_signal_noise_ratio(pred, Y_val[i][:,:,0]));
psnr_mean = np.mean(psnr_array)

ssim_array = []
for i in range(0, len(predictions)):
  pred = np.clip( predictions[i][:,:,0], a_min=0, a_max=1 );
  ssim_array.append(metrics.structural_similarity(pred, Y_val[i][:,:,0]));
ssim_mean = np.mean(ssim_array)

print("PSNR reconstructed:", psnr_mean)
print("SSIM reconstructed:", ssim_mean)
# \n is placed to indicate EOL (End of Line)
file1 = open(testName+'.txt',"a")

file1.write("PSNR reconstructed: "+ str(psnr_mean)+'\n')
file1.write("SSIM reconstructed:"+ str(ssim_mean)+'\n')
file1.close() #to change file access modes
  
"""Now it would be interesting to visualize our output data to check what does the output of our net look like. """

#@title

print('predictions shape:', predictions.shape)
# Display corresponding first 3 patches
plt.figure(figsize=(15,15))
plt.subplot(3, 3, 1)
plt.imshow( val_img_patches[0], 'gray' )
plt.title( 'Validation original image' )
# Side by side with its "ground truth"
plt.subplot(3, 3, 2)
plt.imshow( noisy_val_img[0], 'gray' )
plt.title( 'Image with added noise' )
# and its prediction
plt.subplot(3, 3, 3)
plt.imshow( predictions[0,:,:,0], cmap='gray' )
plt.title( 'Denoised image' )

plt.subplot(3, 3, 4)
plt.imshow( val_img_patches[1], 'gray' )
plt.title( 'Validation original image' )
# Side by side with its "ground truth"
plt.subplot(3, 3, 5)
plt.imshow( noisy_val_img[1], 'gray' )
plt.title( 'Image with added noise' )
# and its prediction
plt.subplot(3, 3, 6)
plt.imshow( predictions[1,:,:,0], cmap= 'gray' )
plt.title( 'Denoised image' )

plt.subplot(3, 3, 7)
plt.imshow( val_img_patches[310], 'gray' )
plt.title( 'Validation original image' )
# Side by side with its "ground truth"
plt.subplot(3, 3, 8)
plt.imshow( noisy_val_img[310], 'gray' )
plt.title( 'Image with added noise' )
# and its prediction
plt.subplot(3, 3, 9)
plt.imshow( predictions[310,:,:,0], cmap= 'gray' )
plt.title( 'Denoised image' )
plt.savefig('Pretraining_predictions.png')

del model,noisy_val_img,noisy_train_img,X_train,Y_train,X_val,Y_val,predictions

"""## Prepare data for training
Now in order to prepare the training for the real task we'll take again the images but this time:

* **Our inputs are the original patches** with no noise
* And the **groundtruth would be the already labelled patches**

As it is possible to observe the Background GT is empty as in this case all the cells fill the whole image, therefore makes no sense to use it and will only cause the model to train slower and maybe confuse its predictions.

### Training with 100% labels
"""

#@title
X_train = np.asarray(train_img_patches1)
X_train = np.expand_dims( X_train, axis=-1 )
Y_train = np.asarray(train_lbl_patches1)
Y_train = np.expand_dims( Y_train, axis=-1 )
X_val = np.asarray(val_img_patches1)
X_val = np.expand_dims( X_val, axis=-1 )
Y_val = np.asarray(val_lbl_patches1)
Y_val = np.expand_dims( Y_val, axis=-1 )

#@title
values=[]
X_train=X_train
for i in range(len(X_train[:,0,0,0])):
  values.append(np.max(X_train[i,:,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
values=[]
X_val=X_val
for i in range(len(X_val[:,0,0,0])):
  values.append(np.max(X_val[i,:,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
values=[]
Y_train=Y_train
for i in range(len(Y_train[:,0,0])):
  values.append(np.max(Y_train[i,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
values=[]
Y_val=Y_val
for i in range(len(Y_val[:,0,0])):
  values.append(np.max(Y_val[i,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile
#from SelfSupervisedLearning.Data_aug import get_train_val_generators,random_90rotation
#from SelfSupervisedLearning.lr_finder import lr_finder
#from SelfSupervisedLearning.metrics_loss_functions import jaccard_index_final
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras
# Recrea exactamente el mismo modelo solo desde el archivo
model = keras.models.load_model('Pretrained_model.h5')
total_seg100=[]
total_prec100=[]
#from SelfSupervisedLearning.DenoiSeg_functions import threshold_optimization

for i in range(0,repetitions):
    history,model1=train(X_train,Y_train,X_val,Y_val,numEpochs,1,patience,lr,lr*1e-1,batch_size_value,schedule,model_name,optimizer_name,loss_acronym,max_pooling,train_encoder=train_encoder,preTrain=False,Denoising=False,pre_load_weights=True,pretrained_model=model,plot_history=plot_history,bottleneck_freezing=bottleneck_freezing)
    # Evaluate the model on the test data using `evaluate`
    model1.save((testName+'/Iteraciones1/lucchi_FineTunedModel'+str(i)+'.h5'))
    print('\n# Evaluate on test data with all training data in loop:',i)
    


import cv2
#from SelfSupervisedLearning.general_functions import append_blackborder,append_pot2

X_test = [  np.expand_dims( append_pot2(x), axis=-1 )  for x in test_img1 ];
Y_test = [  append_pot2(x)  for x in test_lbl1 ];
test_lbl=[  append_pot2(x)  for x in test_lbl1 ];
X_datagen = ImageDataGenerator()
Y_datagen = ImageDataGenerator()
X_test=np.asarray(X_test)
Y_test = np.asarray(Y_test)
Y_test = np.expand_dims( Y_test, axis=-1 )

model1.evaluate(x=X_test,y=Y_test)

IoU_Lucchi2Lucchi_temp=[]
IoU_Lucchi2Lucchi=[]
model_input_filenames1 = [x for x in os.listdir(testName+'/Iteraciones1' ) if x.endswith(".h5")]
for w in  model_input_filenames1 :
  model1 = keras.models.load_model(testName+'/Iteraciones1/'+w)
  for i in range(0,len(X_test)):
    print('Evaluating test image',i)
    normalizedImg = X_test[i][:,:,:];
    prediction = model1.predict(normalizedImg[np.newaxis,:,:]);
    image=prediction[0,:,:,:];
    
    IoU_Lucchi2Lucchi_temp.append(jaccard_index_final(test_lbl[i],image[:,:,0]));

  IoU_Lucchi2Lucchi.append(np.mean(np.nan_to_num(IoU_Lucchi2Lucchi_temp)))

print('The average IoU in test set is: ',IoU_Lucchi2Lucchi)
try:
    file1 = open(testName+'.txt',"a")
    file1.write('Lucchi-Lucchi IoU:'+ str(np.mean(IoU_Lucchi2Lucchi))+'\n')
    file1.write('Lucchi-Lucchi std:'+ str(np.std(IoU_Lucchi2Lucchi))+'\n')

except:
    print('No se ha podido copiar en el txt el IoU')
#@title
file1.close() #to change file access modes
predictions=[]
for i in range(0,len(X_test[0:5])):
      #print('Evaluating test image',i)
      normalizedImg = X_test[i][:,:,:]
      prediction = model1.predict(normalizedImg[np.newaxis,:,:]);
      image=prediction[0,:,:,:]>0.5;
      predictions.append(image);
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.imshow(predictions[0][:,:,0])
plt.title('Predicted labels')
plt.subplot(2,2,2)
plt.imshow(test_lbl[0].astype(int)[:,:])
plt.title('GT labels')
plt.subplot(2,2,3)
plt.imshow(predictions[1][:,:,0])
plt.title('Predicted labels')
plt.subplot(2,2,4)
plt.imshow(test_lbl[1].astype(int)[:,:])
plt.title('GT labels')
plt.savefig('Model_Lucchi_Predictions_Lucchi.png')

X_test = [  np.expand_dims( append_pot2(x), axis=-1 )  for x in test_img2 ];
Y_test = [  append_pot2(x)  for x in test_lbl2 ];
test_lbl=[  append_pot2(x)  for x in test_lbl2 ];

IoU_Lucchi2Kashturi_temp=[]
IoU_Lucchi2Kashturi=[]
for w in  model_input_filenames1 :
  model1 = keras.models.load_model(testName+'/Iteraciones1/'+w)
  for i in range(0,len(X_test)):

    #print('Evaluating test image',i)
    normalizedImg = X_test[i][:,:,:]
    prediction = model1.predict(normalizedImg[np.newaxis,:,:]);
    image=prediction[0,:,:,:];
    
    IoU_Lucchi2Kashturi_temp.append(jaccard_index_final(test_lbl[i],image[:,:,0]));

  IoU_Lucchi2Kashturi.append(np.nanmean(IoU_Lucchi2Kashturi_temp))

print('The average IoU in test set is: ',np.mean(IoU_Lucchi2Kashturi))
try:
    file1 = open(testName + '.txt',"a")
    file1.write('Lucchi-Kashturi IoU:')
    file1.write(str(np.mean(IoU_Lucchi2Kashturi))+'\n')
    file1.write(str(np.std(IoU_Lucchi2Kashturi))+'\n')

   
except:
    print('No se ha podido copiar en el txt el IoU en el dataset 2')
#@title
file1.close() #to change file access modes
#@title
predictions=[]
for i in range(0,len(X_test)):
    #print('Evaluating test image',i)
    normalizedImg = X_test[i][:,:,:]
    prediction = model1.predict(normalizedImg[np.newaxis,:,:]);
    image=prediction[0,:,:,:]>0.5;
    predictions.append(image);
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.imshow(predictions[0][:,:,0])
plt.title('Predicted labels')
plt.subplot(2,2,2)
plt.imshow(test_lbl2[0])
plt.title('GT labels')
plt.subplot(2,2,3)
plt.imshow(predictions[1][:,:,0])
plt.title('Predicted labels')
plt.subplot(2,2,4)
plt.imshow(test_lbl2[1])
plt.title('GT labels')
plt.savefig('Model_Lucchi_Predictions_Kashturi.png')

#@title
X_train = np.asarray(train_img_patches2)
X_train = np.expand_dims( X_train, axis=-1 )
Y_train = np.asarray(train_lbl_patches2)
Y_train = np.expand_dims( Y_train, axis=-1 )
X_val = np.asarray(val_img_patches2)
X_val = np.expand_dims( X_val, axis=-1 )
Y_val = np.asarray(val_lbl_patches2)
Y_val = np.expand_dims( Y_val, axis=-1 )

#@title
values=[]
X_train=X_train
for i in range(len(X_train[:,0,0,0])):
  values.append(np.max(X_train[i,:,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
values=[]
X_val=X_val
for i in range(len(X_val[:,0,0,0])):
  values.append(np.max(X_val[i,:,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
values=[]
Y_train=Y_train
for i in range(len(Y_train[:,0,0])):
  values.append(np.max(Y_train[i,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
values=[]
Y_val=Y_val
for i in range(len(Y_val[:,0,0])):
  values.append(np.max(Y_val[i,:,:]))
print('The range of max values is between:',np.min(values),'and',np.max(values))

#@title
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile
#from SelfSupervisedLearning.Data_aug import get_train_val_generators,random_90rotation
#from SelfSupervisedLearning.lr_finder import lr_finder
#from SelfSupervisedLearning.metrics_loss_functions import jaccard_index_final
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#@title

total_seg100=[]
total_prec100=[]
#from SelfSupervisedLearning.DenoiSeg_functions import threshold_optimization

for i in range(0,repetitions):
    history,model2=train(X_train,Y_train,X_val,Y_val,numEpochs,1,patience,lr,lr*1e-1,batch_size_value,schedule,model_name,optimizer_name,loss_acronym,max_pooling,train_encoder=train_encoder,preTrain=False,Denoising=False,pre_load_weights=True,pretrained_model=model,plot_history=plot_history,bottleneck_freezing=bottleneck_freezing)
    # Evaluate the model on the test data using `evaluate`
    model2.save((testName+'/Iteraciones2/kashturi_FineTunedModel'+str(i)+'.h5'))
    print('\n# Evaluate on test data with all training data in loop:',i)

X_test = [  np.expand_dims( append_pot2(x), axis=-1 )  for x in test_img2 ];
Y_test = [  append_pot2(x)  for x in test_lbl2 ];
test_lbl=[  append_pot2(x)  for x in test_lbl2 ];

IoU_Kashturi2Kashturi_temp=[]
IoU_Kashturi2Kashturi=[]
model_input_filenames2 = [x for x in os.listdir(testName+'/Iteraciones2' ) if x.endswith(".h5")]
for w in  model_input_filenames2 :
  model2 = keras.models.load_model(testName+'/Iteraciones2/'+w)
  for i in range(0,len(X_test)):
    #print('Evaluating test image',i)
    normalizedImg = X_test[i][:,:,:]
    prediction = model2.predict(normalizedImg[np.newaxis,:,:]);
    image=prediction[0,:,:,:];
    
    IoU_Kashturi2Kashturi_temp.append(jaccard_index_final(test_lbl[i],image[:,:,0]));

  IoU_Kashturi2Kashturi.append(np.nanmean(np.nan_to_num(IoU_Kashturi2Kashturi_temp)))

print('The average SEG in test set is: ',IoU_Kashturi2Kashturi)
try:
    file1 = open(testName + '.txt',"a")
    file1.write('Kashturi-Kashturi IoU:')
    file1.write(str(np.mean(IoU_Kashturi2Kashturi))+'\n')
    file1.write('Kashturi-Kashturi std:')
    file1.write(str(np.std(IoU_Kashturi2Kashturi))+'\n')
    

   
except:
    print('No se ha podido copiar en el txt el IoU en el dataset 2')
#@title
file1.close() #to change file access modes
predictions=[]
for i in range(0,len(X_test[0:5])):
      #print('Evaluating test image',i)
      normalizedImg = X_test[i][:,:,:]
      prediction = model2.predict(normalizedImg[np.newaxis,:,:]);
      image=prediction[0,:,:,:]>0.5;
      predictions.append(image);
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.imshow(predictions[0][:,:,0])
plt.title('Predicted labels')
plt.subplot(2,2,2)
plt.imshow(test_lbl[0][:,:])
plt.title('GT labels')
plt.subplot(2,2,3)
plt.imshow(predictions[3][:,:,0])
plt.title('Predicted labels')
plt.subplot(2,2,4)
plt.imshow(test_lbl[3][:,:])
plt.title('GT labels')
plt.savefig('Model_Kashturi_Predictions_Kashturi.png')

X_test = [  np.expand_dims( append_pot2(x), axis=-1 )  for x in test_img1 ];
Y_test = [  append_pot2(x)  for x in test_lbl1 ];
test_lbl=[  append_pot2(x)  for x in test_lbl1 ];

IoU_Kashturi2Lucchi_temp=[]
IoU_Kashturi2Lucchi=[]
for w in  model_input_filenames2 :
  model2 = keras.models.load_model(testName+'/Iteraciones2/'+w)
  for i in range(0,len(X_test)):

    #print('Evaluating test image',i)
    normalizedImg = X_test[i][:,:,:]
    prediction = model2.predict(normalizedImg[np.newaxis,:,:]);
    image=prediction[0,:,:,:];
    
    IoU_Kashturi2Lucchi_temp.append(jaccard_index_final(test_lbl[i],image[:,:,0]));

  IoU_Kashturi2Lucchi.append(np.nanmean(IoU_Kashturi2Lucchi_temp))
print('The average SEG in test set is: ',np.mean(IoU_Kashturi2Lucchi))
try:
    file1 = open(testName + '.txt',"a")
    file1.write('Kashturi-Lucchi++ IoU:')
    file1.write(str(np.mean(IoU_Kashturi2Lucchi))+'\n')
    file1.write('Kashturi-Lucchi++ std:')
    file1.write(str(np.std(IoU_Kashturi2Lucchi))+'\n')

    
except:
    print('No se ha podido copiar en el txt el IoU en el dataset 2')
#@title
file1.close() #to change file access modes
predictions=[]
for i in range(0,len(X_test)):
    #print('Evaluating test image',i)
    normalizedImg = X_test[i][:,:,:]
    prediction = model2.predict(normalizedImg[np.newaxis,:,:]);
    image=prediction[0,:,:,:]>0.5;
    predictions.append(image);
plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.imshow(predictions[0][:,:,0])
plt.title('Predicted labels')
plt.subplot(2,2,2)
plt.imshow(test_lbl[0])
plt.title('GT labels')
plt.subplot(2,2,3)
plt.imshow(predictions[1][:,:,0])
plt.title('Predicted labels')
plt.subplot(2,2,4)
plt.imshow(test_lbl[1]),
plt.title('GT labels')
plt.savefig('Model_Kashturi_Predictions_Lucchi++.png')



