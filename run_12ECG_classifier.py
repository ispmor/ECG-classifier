#!/usr/bin/env python
from algorithm import Model


def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = []
    current_score = []

    label = model.predict(data, header_data)
    score = model.scores_final
    print(header_data[-4])

    keys = sorted(label.keys())
    print(keys)

    for key in keys:
        current_label.append(label[key])
        current_score.append(score[key])

    print("CURRRRRENT")
    print(current_label)
    print(current_score)

    return list(current_label), list(current_score)


def load_12ECG_model(path=""):
    # load the model from disk 
    loaded_model = Model()
    loaded_model.load(path=path)

    return loaded_model
