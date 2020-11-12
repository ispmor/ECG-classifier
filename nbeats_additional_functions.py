import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas
import torch
from datetime import date
import config
from scipy import stats
from torch.nn import functional as F
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

today = date.today().strftime("%d%m%Y")
def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


def data_generator(x, y, batch_size):
    while True:
        for xy_pair in split((x, y), batch_size):
            yield xy_pair


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def batcher(dataset, batch_size, infinite=False):
    while True:
        x, y = dataset
        for x_, y_ in zip(split(x, batch_size), split(y, batch_size)):
            yield x_, y_
        if not infinite:
            break


def load(checkpoint_name, model, optimiser):
    if os.path.exists(checkpoint_name):
        checkpoint = torch.load(checkpoint_name, map_location=torch.device('cuda:1'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        grad_step = checkpoint['grad_step']
        #print(f'Restored checkpoint from {checkpoint_name}.')
        return grad_step
    return 0


def save(checkpoint_name, model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict()
    }, checkpoint_name)


def train_100_grad_steps(checkpoint_name, data, device, net, optimiser, test_losses):
    global_step = load(checkpoint_name, net, optimiser)
    for x_train_batch, y_train_batch in data:
        global_step += 1
        optimiser.zero_grad()
        net.train()
        _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        if global_step > 0 and global_step % 100 == 0:
            with torch.no_grad():
                save(checkpoint_name, net, optimiser, global_step)
            break


def fit(checkpoint_name, net, optimiser, data_generator, on_save_callback, device, max_grad_steps=10000):
    initial_grad_step = load(checkpoint_name, net, optimiser)
    for grad_step, (x, target) in enumerate(data_generator):
        grad_step += initial_grad_step
        optimiser.zero_grad()
        net.train()
        backcast, forecast = net(torch.tensor(x, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(target, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        if grad_step % 1000 == 0 or (grad_step < 1000 and grad_step % 100 == 0):
            with torch.no_grad():
                save(checkpoint_name, net, optimiser, grad_step)
                if on_save_callback is not None:
                    on_save_callback(x, target, grad_step)
        if grad_step > max_grad_steps:
            print('Finished.')
            break


def eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test, visualise=False):
    net.eval()
    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    singular_loss = F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item()
    test_losses.append(singular_loss)
    
    
    if visualise:
        subplots = [221, 222, 223, 224]
        p = forecast.detach().cpu().numpy()
        plt.figure(1)
        for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
            ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
            plt.subplot(subplots[plot_id])
            plt.grid()
            plot_scatter(range(0, backcast_length), xx, color='b')
            plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
            plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        plt.show()
        
        

def evaluate_training(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test, the_lowest_error, device, experiment, plot_eval=False, class_dir="", step=0, file_name=""):
    net.eval()
    _, forecast = net(x_test.clone().detach())
    
    if device.type == 'cuda:1':
        singular_loss = F.mse_loss(forecast, y_test.clone().detach()).item()
    else :
        singular_loss = F.mse_loss(forecast, y_test.clone().detach()).item()
        
    test_losses.append(singular_loss)
    if singular_loss < the_lowest_error[-1]:
            the_lowest_error.append(singular_loss)           
           
    
    if plot_eval:
        p = forecast.detach().cpu().numpy()
        subplots = [221, 222, 223, 224]
        fig = plt.figure(1, figsize=(12, 10))
        for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
            ff, xx, yy = p[i], x_test.detach().cpu().numpy()[i], y_test.detach().cpu().numpy()[i]
            plt.subplot(subplots[plot_id])
            plt.grid()
            plt.ylabel("Values normalised")
            plt.xlabel("Time (1/500 s)")
            plt.title(file_name)
            plot_scatter(range(0, backcast_length), xx, color='b')
            plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
            plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        experiment.log_image('epoch_test_eval_visualisation', fig)
        if not os.path.exists(f"/home/puszkar/ecg/results/images/training_eval/{today}v1"):
            os.mkdir(f"/home/puszkar/ecg/results/images/training_eval/{today}v1")
        plt.savefig(f"/home/puszkar/ecg/results/images/training_eval/{today}v1/{class_dir}_latest_eval.png")
        plt.close()
    
    return singular_loss
    
def train_full_grad_steps(data, device, net, optimiser, test_losses, training_checkpoint, size):
    global_step = load(training_checkpoint, net, optimiser)
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
                save(training_checkpoint, net, optimiser, global_step)
                return global_step
        if local_step > 0 and local_step % size == 0:
            return global_step
    
def plot_singlas(x, y, file_name, diagnosis, experiment):
    backcast_length = config.default_net_params["backcast_length"]
    forecast_length = config.default_net_params["forecast_length"]
    subplots = [221, 222, 223, 224]
    fig = plt.figure(1, figsize=(12, 10))
    plt.title(diagnosis)
    for plot_id, i in enumerate(np.random.choice(range(len(x)), size=4, replace=False)):
        xx, yy = x.detach().cpu().numpy()[i], y.detach().cpu().numpy()[i]
        plt.subplot(subplots[plot_id])
        plt.grid()
        plt.ylabel("Values")
        x_label_with_true = f"File name: {file_name}, diagnosis: {diagnosis}"
        plt.xlabel(x_label_with_true)
        plot_scatter(range(0, backcast_length), xx, color='b')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
    experiment.log_image('file-visualisation', fig)
    plt.close()



def get_avg_score(net, x_test, y_test, name, experiment, plot_counter=0, plot_title=""):
    backcast_length = config.default_net_params["backcast_length"]
    forecast_length = config.default_net_params["forecast_length"]
    net.eval()
    _, forecast = net(x_test.clone().detach())
    singular_loss = F.mse_loss(forecast, y_test.clone().detach()).item()
    
    y_test_numpy = np.nan_to_num(y_test.clone().detach().cpu().numpy())
    forecast_numpy = np.nan_to_num(forecast.clone().detach().cpu().numpy())
    '''
    y_test_1d = np.append(y_test_numpy[0], [x[-1] for x in y_test_numpy])
    forecast_1d = np.append(forecast_numpy[0], [x[-1] for x in forecast_numpy])
    singular_loss, path = fastdtw(y_test_numpy, forecast_numpy, dist=euclidean)
    experiment.log_metric(f'dtw-distance-{name}', singular_loss)
    '''
    experiment.log_metric(f'mse-loss-distance-{name}', singular_loss)
    
    if plot_counter > 0:
        if not os.path.exists(f"/home/puszkar/ecg/results/images/{today}v1"):
            os.mkdir(f"/home/puszkar/ecg/results/images/{today}v1")
        p = forecast_numpy
        subplots = [221, 222, 223, 224]
        fig = plt.figure(1, figsize=(12, 10))
        plot_title = plot_title + f"----Used Network: {name}" 
        plt.title(plot_title)
        for plot_id, i in enumerate([1,100, 500, 1000]):
            ff, xx, yy = p[i], x_test.detach().cpu().numpy()[i], y_test_numpy[i]
            plt.subplot(subplots[plot_id])
            plt.grid()
            plt.ylabel("Values normalised")
            x_label_with_true = f"Time (1/500 s), true label: {plot_title}"
            plt.xlabel(x_label_with_true)
            plot_scatter(range(0, backcast_length), xx, color='b')
            plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
            plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
        experiment.log_image('signal-visualisation', fig)
        plt.close()

    return singular_loss


def one_file_training_data(data_dir, file, forecast_length, backcast_length, batch_size, cuda, lead=3):
    normal_signal_data = []

    x = wfdb.io.rdsamp(data_dir + file[:-4])
    normal_signal_data.append(x[0][:, lead])
    normal_signal_data = [y for sublist in normal_signal_data for y in sublist]
    normal_signal_data = np.array(normal_signal_data)
    normal_signal_data.flatten()
    print(x[1])
    diagnosis = x[1]['comments'][2]

    norm_constant = np.max(normal_signal_data)
    norm_min = np.min(normal_signal_data)
    #print(norm_constant)
    normal_signal_data = (normal_signal_data - norm_min) / (norm_constant - norm_min)  # leak to the test set here.

    x_train_batch, y = [], []
    dist_to_one = 1 - normal_signal_data[0]
    normal_signal_data += dist_to_one
    for i in range(backcast_length, len(normal_signal_data) - forecast_length):
        x_train_batch.append(normal_signal_data[i - backcast_length:i])
        y.append(normal_signal_data[i:i + forecast_length])

    x_train_batch = torch.tensor(x_train_batch, device=cuda, dtype=torch.float)  # [..., 0]
    y = torch.tensor(y, device=cuda,  dtype=torch.float)  # [..., 0]

    if len(x_train_batch) > 7500:
        x_train_batch = x_train_batch[0:7500]
        y = y[0:7500]
        
        
    c = int(len(x_train_batch) * 0.8)
    x_train, x_test, y_train, y_test = train_test_split(x_train_batch, y, test_size=0.005, random_state=17)
    data = data_generator(x_train, y_train, batch_size)

    return data,x_train, y_train, x_test, y_test, norm_constant, diagnosis


def organise_data(data, data_header, forecast_length, backcast_length, batch_size , cuda, lead=3):
    normal_signal_data = []

    normal_signal_data.append(data[lead])
    normal_signal_data = [y for sublist in normal_signal_data for y in sublist]
    normal_signal_data = np.array(normal_signal_data)
    normal_signal_data.flatten()


    norm_constant = np.max(normal_signal_data)
    norm_min = np.min(normal_signal_data)
    #print(norm_constant)
    normal_signal_data = (normal_signal_data - norm_min) / (norm_constant - norm_min)  # leak to the test set here.

    dist_to_one = 1 - normal_signal_data[0]
    normal_signal_data += dist_to_one
    x, y = [], []
    for i in range(backcast_length, len(normal_signal_data) - forecast_length):
        x.append(normal_signal_data[i - backcast_length:i])
        y.append(normal_signal_data[i:i + forecast_length])

    x = torch.tensor(x, device=cuda, dtype=torch.float)  # [..., 0]
    y = torch.tensor(y, device=cuda, dtype=torch.float)  # [..., 0]

    if len(x) > 7500:
        x = x[0:7500]
        y = y[0:7500]
    
    true_label = data_header[-4]

    return x, y, true_label
