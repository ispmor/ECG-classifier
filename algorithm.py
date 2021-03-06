import nbeats_additional_functions as naf
from nbeats_pytorch.model import NBeatsNet
import os
import torch
from torch import optim
from config import default_net_params, exp_net_params 
from config import leads_dict


class Model:
    def __init__(self, device=torch.device('cpu'), counter=90):
        self.forecast_length = default_net_params["forecast_length"]
        self.backcast_length = default_net_params["backcast_length"]
        self.batch_size =  default_net_params["batch_size"]
        self.classes =  exp_net_params["classes"]
        self.nb_blocks_per_stack = exp_net_params["nb_blocks_per_stack"]
        self.thetas = exp_net_params["thetas_dim"]
        self.hidden_layer_units = exp_net_params["hidden_layer_units"]
        self.device = device
        self.nets = {}
        self.scores = {}
        self.scores_norm = {}
        self.scores_final = {}
        self.scores_final_sorted={}
        self.plots_counter = counter
        self.checkpoint_name_BASE = "nbeats_checkpoint"

    def load(self, path=""):
        for d in self.classes:
            checkpoint = d + "_" + self.checkpoint_name_BASE+ f'bl{self.nb_blocks_per_stack}-f{self.forecast_length}-b{self.backcast_length}-btch{self.batch_size}-h{self.hidden_layer_units}.th'
            if path:
                checkpoint = path + checkpoint
            
            print(f'From where to load models: {path}, and the current checkpoint is: {checkpoint}')
            if not os.path.exists(checkpoint):
                print("-" * 50)
                print("* /n * /n * /n * /n * /n * /t THERE IS NO CHECKPOINT LIKE THIS! * /n * /n * /n * /n * ")
                print("-" * 50)
                raise ValueError("wrong checkpoint name!")
            else:
                print("-" * 50)
                print("*/n*/n*/n*/n*/n*/t CHECKPOINT IS VALID WOOOHOO ! */n*/n*/n*/n*")
                print("-" * 50)
                
            net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                            forecast_length=self.forecast_length,
                            thetas_dims=self.thetas,
                            nb_blocks_per_stack=self.nb_blocks_per_stack,
                            backcast_length=self.backcast_length,
                            hidden_layer_units=self.hidden_layer_units,
                            share_weights_in_stack=False,
                            device=self.device)
            optimiser = optim.Adam(net.parameters())

            naf.load(checkpoint, net, optimiser)
            net.to(self.device)
            self.nets[d] = net

    def predict(self, data, data_header, experiment, leads_dict_available = False, lead=3, file_name=""):
        signals_dict={}
        if leads_dict_available:
            for c in self.classes:
                signals_dict[c] = naf.organise_data(data, data_header, self.forecast_length, self.backcast_length, self.batch_size, self.device, lead=leads_dict[c])
        else:
            x, y, true_label = naf.organise_data(data, data_header, self.forecast_length, self.backcast_length, self.batch_size, self.device, lead=lead)
        for c in self.classes:
            if leads_dict_available:
                x, y, true_label = signals_dict[c]
            self.scores[c] = naf.get_avg_score(self.nets[c], x , y, c, experiment, self.plots_counter, plot_title=f"from:{file_name}, {true_label}")
            self.plots_counter -= 1

        scores = list(self.scores.values())

        print(self.scores)
        max_score = max(scores)
        min_score = min(scores)
        result = {}
        for c in self.classes:
            self.scores_norm[c] = (self.scores[c] - min_score) / (max_score - min_score)
            self.scores_final[c] = 1 - self.scores_norm[c]            
            result[c] = 0
        
        counter = 0
        self.scores_final_sorted = {k: v for k, v in sorted(self.scores_final.items(), key=lambda item: item[1], reverse=True)}
        
        for cl, value in self.scores_final_sorted.items():
            if counter < 3:
                result[cl] = 1 
                counter += 1
            else:
                counter = 0
                break
            
            
            #if self.scores_final[c] == 1:
                #result[c] = 1
        print(self.scores_norm)
        print(self.scores_final)

        return result









