default_net_params = dict(
    backcast_length = 1200,
    forecast_length = 200,
    batch_size = 64,
    classes = ['LBBB', 'STD', 'Normal', 'RBBB', 'AF', 'I-AVB', 'STE', 'PAC', 'PVC'],
    backcast_multiplier = 6,
    hidden_layer_units=16
)