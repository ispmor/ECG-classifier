default_net_params = dict(
    backcast_length = 1200,
    forecast_length = 200,
    batch_size = 64,
    classes = ['LBBB', 'STD', 'Normal', 'RBBB', 'AF', 'I-AVB', 'STE', 'PAC', 'PVC'],
    backcast_multiplier = 6,
    hidden_layer_units=16,
    nb_blocks_per_stack=3,
    thetas_dim=[7,8]
)


exp_net_params = dict(
    backcast_length = 1200,
    forecast_length = 200,
    batch_size = 64,
    classes = ['LBBB', 'STD', 'Normal', 'RBBB', 'AF', 'I-AVB', 'STE', 'PAC', 'PVC'],
    backcast_multiplier = 6,
    hidden_layer_units=16,
    nb_blocks_per_stack= 8,
    thetas_dim=[16,32]
)

epoch_limit = 50


leads_dict_available= True
# leads are provided REAL - 1
leads_dict = {
    'AF': 2,
    'I-AVB':1,
    'LBBB' : 11,
    'Normal':2,
    'PAC':1,
    'PVC':1,
    'RBBB':0,
    'STD':2,
    'STE':4  
}