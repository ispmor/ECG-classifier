from nbeats_pytorch.model import NBeatsNet
import nbeats_additional_functions as naf
import os
import torch
from torch import optim

checkpoint_name_BASE = "nbeats_checkpoint.th"

data_dir = os.getcwd() + "/data/"

d = [x[0] for x in os.walk(data_dir)]
dirs = []
for directory_name in d:
    if ',' not in directory_name:
        dirs.append(directory_name)
dirs = dirs[1:]
print(dirs)

device = torch.device('cpu')
forecast_length = 500
backcast_length = 3 * forecast_length
batch_size = 256

for folder_name in dirs:
    name = folder_name.split("/")[-1]
    checkpoint_name = name + "_" + checkpoint_name_BASE

    if os.path.isfile(checkpoint_name):
        continue

    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[7, 8],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=128,
                    share_weights_in_stack=False,
                    device=device)
    optimiser = optim.Adam(net.parameters())

    test_losses = []
    actual_class_dir = data_dir + "/" + name + "/"
    for (_, dirs, files) in os.walk(actual_class_dir):
        iteration = 0
        for file in files:

            if 'mat' in file:
                continue

            iteration += 1
            print(iteration)
            if iteration > 30:
                break

            data, x_test, y_test, norm_constant = naf.one_file_training_data(actual_class_dir, file, forecast_length,
                                                                         backcast_length, batch_size)

            for i in range(10):
                naf.eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test)
                naf.train_100_grad_steps(checkpoint_name, data, device, net, optimiser, test_losses)
