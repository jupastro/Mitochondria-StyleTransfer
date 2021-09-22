#@title **SSL task**
#@title ## Markdown
#@markdown * `factor`: represents the downsampling performed in the image if set to 1 there's only denoising
#@markdown *`noise`:represents the amount of Gaussian noise to be added to the image 

#@markdown ---
#GPU selection 
GPU_availability=False
show_images=False
GPU="0"
#Parameters to be modified:
plot_history=False
factor=4  #@param {type:"integer"}
noise=0.2 #@param {type:"number"}
testName='EM_Seg' #@param {type:'string'}

 #@title **Pretraining Hyperparameters**

# === PreTraining parameters ===
# number of epochs
numEpochsPretrain =  200#@param {type:"integer"}
# patience
patiencePretrain =  200#@param {type:"integer"}
# learning rate
lrPretrain = 5e-4 #@param {type:"number"}
# batch size
batch_size_valuePretrain =  5#@param {type:"integer"}
# use one-cycle policy for super-convergence? Reduce on plateau?
no_schedule = None #@param {type:"raw"}
schedulePretrain = 'no_schedule' #@param [ "no_schedule","'oneCycle'","'reduce'"] {type:"raw"}

# Network architecture: UNet, ResUNet,MobileNetEncoder
model_namePretrain = 'AttentionUNET'#@param ['UNet','MobileNetEncoder','AttentionUNET']
# Optimizer name: 'Adam', 'SGD'
optimizer_namePretrain = 'Adam'#@param ['Adam','SGD']{type:"string"}
# Loss function name: 'BCE', 'Dice', 'W_BCE_Dice'
loss_acronymPretrain = 'mse' #@param ['mae','mse']{type:"string"}
max_poolingPretrain=True #@param {type:"boolean"}

#@title **Training Hyperparameters**

# === Training parameters ===
# number of epochs
numEpochs = 10 #@param {type:"integer"}
# patience
patience = 10 #@param {type:"integer"}
# learning rate
lr =5e-4 #@param {type:"number"}
# batch size
batch_size_value = 1#@param {type:"integer"}
# use one-cycle policy for super-convergence? Reduce on plateau?
schedule = no_schedule #@param [ "no_schedule","'oneCycle'","'reduce'"] {type:"raw"}
# Network architecture: UNet, ResUNet,MobileNetEncoder
model_name = 'AttentionUNET' #@param ['UNet','MobileNetEncoder','AttentionUNET']
# Optimizer name: 'Adam', 'SGD'
optimizer_name = 'Adam' #@param ['Adam','SGD']{type:"string"}
# Loss function name: 'BCE', 'Dice', 'W_BCE_Dice'
loss_acronym = 'BCE' #@param ['BCE','Dice','SEG']{type:"string"}
# create the network and compile it with its optimizer
max_pooling=True #@param {type:"boolean"}

repetitions=1 #@param {type:"slider", min:1, max:30, step:1}
train_encoder=False #@param {type:"boolean"}
bottleneck_freezing=True #@param {type:"boolean"}


# Paths to the initial training images and their corresponding labels
train_input_path1 = 'Lucchi++/Train_In'
train_label_path1 = 'Lucchi++/Train_Out'
test_input_path1 = 'Lucchi++/Test_In'
test_label_path1 = 'Lucchi++/Test_Out'


train_input_path2 = 'Kasthuri++/Train_In'
train_label_path2 = 'Kasthuri++/Train_Out'
test_input_path2 = 'Kasthuri++/Test_In'
test_label_path2 = 'Kasthuri++/Test_Out'