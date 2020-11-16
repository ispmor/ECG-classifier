#!/usr/bin/env python
from algorithm import Model
import torch


def run_12ECG_classifier(data,header_data,classes,model, experiment, leads_dict_available = False, lead=3, file_name = ""):

    num_classes = len(classes)
    current_label = []
    current_score = []

    label = model.predict(data, header_data, experiment, leads_dict_available, lead, file_name=file_name)
    score = model.scores_final
    print(header_data[-4])

    keys = sorted(label.keys())
    print(keys)

    for key in keys:
        current_label.append(label[key])
        current_score.append(score[key])

    print(current_label)
    print(current_score)

    return list(current_label), list(current_score)


def load_12ECG_model(path=""):
    # load the model from disk 
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    loaded_model = Model(device=device)
    loaded_model.load(path=path)

    return loaded_model
