import neptune

from nbeats_pytorch.model import NBeatsNet
import nbeats_additional_functions as naf
import os
import torch
from torch import optim

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

# input = input.cuda()
# target = target.cuda()


neptune.init('puszkarb/ecg-dyplom')



forecast_length = 200
backcast_length = 6 * forecast_length
batch_size = 64
hidden = 16

for folder_name in dirs:
    experiment = neptune.create_experiment(name=folder_name + f'-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}')
   

    name = folder_name.split("/")[-1]
    checkpoint_name = name + "_" + checkpoint_name_BASE+ f'-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}'
    training_checkpoint = name + checkpoint_training_base + f'-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}' + ".th"


    if os.path.isfile(training_checkpoint):
        continue

    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length=forecast_length,
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
    iteration = 0

    for (_, dirs, files) in os.walk(actual_class_dir):
        difference = 1000
        for file in files:
            i = 0
            if 'mat' in file:
                continue

            iteration += 1
            if iteration > 100 or difference < threshold:
                break

            data, x_train, y_train, x_test, y_test, norm_constant = naf.one_file_training_data(actual_class_dir,
                                                                                               file,
                                                                                               forecast_length,
                                                                                               backcast_length,
                                                                                               batch_size,
                                                                                               device)
            

            while i < limit:  #difference > threshold and
                i += 1
                
                global_step = naf.train_full_grad_steps(data,
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
                
                print("File: %s\t Training Loop: %d/%d, New evaluation sccore: %f" % (file, i, limit, new_eval), end="\r")
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

