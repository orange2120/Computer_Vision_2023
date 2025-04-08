################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
# exp_name   = 'sgd_da' # name of experiment
exp_name   = 'adam_da'

# Model Options
# model_type = 'resnet18' # 'mynet' or 'resnet18'
model_type = 'mynet'

# Learning Options

# For resnet18
# epochs     = 50           # train how many epochs
# batch_size = 32           # batch size for dataloader 
# use_adam   = False        # Adam or SGD optimizer
# lr         = 1e-2         # learning rate
# milestones = [16, 32, 45] # reduce learning rate at 'milestones' epochs

# For myNet
epochs     = 50
batch_size = 32
use_adam   = True
lr         = 1e-3
milestones = [16, 32, 45]