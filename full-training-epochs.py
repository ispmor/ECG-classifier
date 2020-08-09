#!/usr/bin/env python
# coding: utf-8

# In[3]:


import neptune

from nbeats_pytorch.model import NBeatsNet
from torch.nn import functional as F
import nbeats_additional_functions as naf
import os
import torch
from torch import optim
from config import default_net_params as dnp

print("Starting training script")

checkpoint_name_BASE = "nbeats_checkpoint"
checkpoint_training_base = "_training"

data_dir = os.path.dirname(os.getcwd()) + "/data/"
models = os.path.dirname(os.getcwd()) + "/models/"
training_models = os.path.dirname(os.getcwd()) + "/models/training/"

if not os.path.exists(models):
    os.mkdir(models)
if not os.path.exists(training_models):
    os.mkdir(training_models)



d = [x[0] for x in os.walk(data_dir)]
dirs = []
for directory_name in d:
    if ',' not in directory_name:
        directory_name = directory_name.split("/")[-1]
        if all(x not in directory_name for x in [",", "training", "test"]):
            dirs.append(directory_name)
dirs = dirs[1:]

threshold = 0.0001
limit = 600
plot_eval = False

# Bart

cuda1 = torch.cuda.set_device(1)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(device)
print("Selected device: %s" % (torch.cuda.get_device_name(1)))
torch.pin_memory=False

print("Considered classes: %s" % (dirs))

neptune.init('puszkarb/ecg-dyplom')


# In[4]:


def train_full_grad_steps(data, device, net, optimiser, test_losses, training_checkpoint, size):
    global_step = naf.load(training_checkpoint, net, optimiser)
    local_step = 0
    for x_train_batch, y_train_batch in data:
        global_step += 1
        local_step += 1
        optimiser.zero_grad()
        net.train()
        _, forecast = net(x_train_batch.clone().detach())#.to(device))
        loss = F.mse_loss(forecast, y_train_batch.clone().detach())#.to(device))
        loss.backward()
        optimiser.step()
        if global_step > 0 and global_step % 100 == 0:
            with torch.no_grad():
                print("Training batches passed: %d" % (local_step), end="\r")
                naf.save(training_checkpoint, net, optimiser, global_step)
        if local_step > 0 and local_step % size == 0:
            print(local_step)
            return global_step


# In[ ]:

forecast_length = dnp["forecast_length"]
batch_size = dnp["batch_size"]
backcast_length = dnp["backcast_length"]
hidden = dnp["hidden_layer_units"]

for folder_name in dirs:
    print("Folder loop")
    experiment = neptune.create_experiment(name=folder_name + f'-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}')
   

    name = folder_name.split("/")[-1]
    checkpoint_name = name + "_" + checkpoint_name_BASE+ f'-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}'
    training_checkpoint = name + checkpoint_training_base + f'-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}' + ".th"


    if os.path.isfile(training_checkpoint):
        continue

    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length= forecast_length,
                    thetas_dims=[7, 8],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=hidden,
                    share_weights_in_stack=False,
                    device=device)
    net.cuda()
    optimiser = optim.Adam(net.parameters())

    test_losses = []
    old_eval = 100
    the_lowest_error = [100]
    old_checkpoint = ""
    actual_class_dir = data_dir + name + "/"
    print("N-Beats training for class: %s" % (actual_class_dir))
    epoch = 0

    for (_, dirs, files) in os.walk(actual_class_dir):
        print("\t\t\t _, dirs, files loop, epoch: %d\t\n" % (epoch))
        difference = 1000
        for fil in files:
            print("\t\t FIle loop, \t\t epoch: %d\n" % (epoch))
            i = 0
            if 'mat' in fil:
                continue

            print(actual_class_dir, fil, end="\r")
            
            if epoch > 100 or difference < threshold:
                break

            data, x_train, y_train, x_test, y_test, norm_constant = naf.one_file_training_data(actual_class_dir,
                                                                                               fil,
                                                                                               forecast_length,
                                                                                               backcast_length,
                                                                                               batch_size,
                                                                                               device)
            

            while i < 5:  #difference > threshold and
                i += 1
                epoch += 1
                print(epoch)
                global_step = train_full_grad_steps(data,
                                                        device,
                                                        net,
                                                        optimiser,
                                                        test_losses, 
                                                        training_models+training_checkpoint,
                                                        x_train.shape[0])
                
                train_eval = naf.evaluate_training(backcast_length,
                                                   forecast_length,
                                                   net,
                                                   norm_constant,
                                                   test_losses,
                                                   x_train,
                                                   y_train,
                                                   the_lowest_error,
                                                   device)
                experiment.log_metric('train_loss', train_eval)
                
               
                new_eval = naf.evaluate_training(backcast_length,
                                                 forecast_length,
                                                 net,
                                                 norm_constant,
                                                 test_losses,
                                                 x_test,
                                                 y_test, 
                                                 the_lowest_error,
                                                 device,
                                                 plot_eval=False,
                                                 class_dir=name,
                                                 step=i)
                experiment.log_metric('eval_loss', new_eval)
                
                print("\nFile: %s\t Training Loop: %d/%d, New evaluation sccore: %f" % (fil, i, limit, new_eval), end="\r")
                if new_eval < old_eval:
                    difference = old_eval - new_eval
                    old_eval = new_eval
                    with torch.no_grad():
                        if old_checkpoint:
                            os.remove(models+old_checkpoint)
                        new_checkpoint_name = str(checkpoint_name[:-3] + str(len(test_losses)) + ".th")
                        naf.save(models + new_checkpoint_name, net, optimiser, global_step)
                        old_checkpoint = new_checkpoint_name
                        
    experiment.stop()
