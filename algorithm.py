import nbeats_additional_functions as naf
from nbeats_pytorch.model import NBeatsNet
import os
import torch
from torch import optim


class Model:
    def __init__(self, device=torch.device('cpu')):
        self.forecast_length = 200
        self.backcast_length = 6 * self.forecast_length
        self.batch_size = 64# 256
        self.classes = ['LBBB', 'STD', 'Normal', 'RBBB', 'AF', 'I-AVB', 'STE', 'PAC', 'PVC']
        self.device = device
        self.nets = {}
        self.scores = {}
        self.scores_norm = {}
        self.scores_final = {}

    def load(self, path=""):
        for d in self.classes:
            checkpoint = d + "_nbeats_checkpoint.th"
            if path:
                checkpoint = path + checkpoint
                
            net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                            forecast_length=self.forecast_length,
                            thetas_dims=[7, 8],
                            nb_blocks_per_stack=3,
                            backcast_length=self.backcast_length,
                            hidden_layer_units=16,#128,
                            share_weights_in_stack=False,
                            device=self.device)
            optimiser = optim.Adam(net.parameters())

            naf.load(checkpoint, net, optimiser)
            net.to(self.device)
            self.nets[d] = net

    def predict(self, data, data_header):
        x, y = naf.organise_data(data, data_header, self.forecast_length, self.backcast_length, self.batch_size, self.device)
        for c in self.classes:
            self.scores[c] = naf.get_avg_score(self.nets[c], x , y, c)

        scores = list(self.scores.values())

        print(self.scores)
        max_score = max(scores)
        min_score = min(scores)
        result = {}
        for c in self.classes:
            self.scores_norm[c] = (self.scores[c] - min_score) / (max_score - min_score)
            self.scores_final[c] = 1 - self.scores_norm[c]
            result[c] = 0
            if self.scores_final[c] > 0.99:
                result[c] = 1
        print(self.scores_norm)
        print(self.scores_final)

        return result









