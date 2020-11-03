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
from config import leads_dict_available
from config import default_net_params as dnp
from config import exp_net_params as exp
from config import epoch_limit
from config import leads_dict 
import sys

print("Starting training script")

checkpoint_name_BASE = "nbeats_checkpoint"
checkpoint_training_base = "_training"

data_dir = os.path.dirname(os.getcwd()) + "/data/"
models = os.path.dirname(os.getcwd()) + "/models/"
training_models = os.path.dirname(os.getcwd()) + "/models/training/"
neptune.init('puszkarb/ecg-dyplom')


classes_variation = True

if not os.path.exists(models):
    os.mkdir(models)
if not os.path.exists(training_models):
    os.mkdir(training_models)
selected_classes = ['STE']
lead = 3
if len(sys.argv) > 1:
    if not sys.argv[1].isnumeric() and len(sys.argv) < 3:
        selected_classes = sys.argv[1].strip('[]').split(',')
        classes_variation = True
    elif not sys.argv[1].isnumeric() and len(sys.argv) > 2:
        selected_classes = sys.argv[1].strip('[]').split(',')
        classes_variation = True
        lead = int(sys.argv[2])
    else:
        lead = int(sys.argv[1])

if leads_dict_available:
    best_leads = leads_dict


d = [x[0] for x in os.walk(data_dir)]
dirs = []
for directory_name in d:
    if ',' not in directory_name:
        directory_name = directory_name.split("/")[-1]
        if all(x not in directory_name for x in [",", "training", "test"]):
            dirs.append(directory_name)
dirs = dirs[1:]

if classes_variation:
    dirs = [x for x in dirs if x in selected_classes]

threshold = 0.0001
limit = 50
plot_eval = False

# Bart

cuda1 = torch.cuda.set_device(1)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(device)
print("Selected device: %s" % (torch.cuda.get_device_name(1)))
torch.pin_memory=False

print("Considered classes: %s" % (dirs))






# In[4]:


def train_full_grad_steps(data, device, net, optimiser, test_losses, training_checkpoint, size):
    global_step = naf.load(training_checkpoint, net, optimiser)
    local_step = 0
    each_epoch_plot = True
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
                print("Training batches passed: %d" % (local_step))
                naf.save(training_checkpoint, net, optimiser, global_step)
        if local_step > 0 and local_step % size == 0:
            print(local_step)
            return global_step


# In[ ]:

forecast_length = dnp["forecast_length"]
batch_size = dnp["batch_size"]
backcast_length = dnp["backcast_length"]
hidden = dnp["hidden_layer_units"]
nb_blocks_per_stack = dnp["nb_blocks_per_stack"]
thetas_dim_array = [[2,4]]
for thetas_dim in thetas_dim_array:
    print(thetas_dim, thetas_dim_array)
    for folder_name in dirs:
        print(folder_name, dirs)
        if leads_dict_available:
            lead = leads_dict[folder_name]
        experiment = neptune.create_experiment(name=folder_name + f'bl{nb_blocks_per_stack}-thetas[{thetas_dim[0]},{thetas_dim[1]}]-btch{batch_size}-h{hidden}-l{lead+ 1}')
        print(experiment)
   

        name = folder_name.split("/")[-1]
        checkpoint_name = name + "_" + checkpoint_name_BASE+ f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}'
        training_checkpoint = name + checkpoint_training_base + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}' + ".th"


        if os.path.isfile("/home/puszkar/ecg/models/training/" + training_checkpoint):
            os.remove("/home/puszkar/ecg/models/training/" + training_checkpoint)

        net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length= forecast_length,
                    thetas_dims=thetas_dim,
                    nb_blocks_per_stack=nb_blocks_per_stack,
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

        for (_, dirs2, files) in os.walk(actual_class_dir):
            print("\t _, dirs2, files loop, epoch: %d\t\n" % (epoch))
            difference = 1000
            for fil in files:
                print("\t\t FIle loop, epoch: %d\n" % (epoch))
                plot_file = True
                i = 0
                if 'mat' in fil:
                    continue

                print("Reading files from: %s, file loaded: %s" % (actual_class_dir, fil))
            
                if epoch >= epoch_limit : #or difference < threshold:
                    break

                data, x_train, y_train, x_test, y_test, norm_constant, diagnosis = naf.one_file_training_data(actual_class_dir,
                                                                                               fil,
                                                                                               forecast_length,
                                                                                               backcast_length,
                                                                                               batch_size,
                                                                                               device,
                                                                                               lead=lead)
            

                while i < 2: #old was 5  #difference > threshold and
                    i += 1
                    epoch += 1
                    print("Actual epoch: ", epoch, "\nActual inside file loop: ", i)
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
                                                   device,
                                                   experiment=experiment)
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
                                                 experiment=experiment,
                                                 plot_eval=True,
                                                 class_dir=name,
                                                 step=i,
                                                 file_name=fil)
                    experiment.log_metric('eval_loss', new_eval)
                    experiment.log_text('file_name', fil)
                    experiment.log_text('direcotry', actual_class_dir)
                    experiment.log_text('diagnosis_from_file', diagnosis)
                    if plot_file:
                        naf.plot_singlas(x_test, y_test, fil, diagnosis, experiment)
                        plot_file = False
                
                    print("\nFile: %s\t Training Loop: %d/%d, New evaluation sccore: %f" % (fil, i, limit, new_eval))
                    if new_eval < old_eval:
                        difference = old_eval - new_eval
                        old_eval = new_eval
                        with torch.no_grad():
                            if old_checkpoint:
                                os.remove(models+old_checkpoint)
                            new_checkpoint_name = str(checkpoint_name + ".th")
                            naf.save(models + new_checkpoint_name, net, optimiser, global_step)
                            old_checkpoint = new_checkpoint_name
                            
        experiment.stop()
    
