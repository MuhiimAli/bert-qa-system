sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'val_f1',
        'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'uniform',
            # 'min': 3.0e-5,
            # 'max': 3.33e-5

            # 'min': 3.34e-5,
            # 'max': 3.66e-5


            'min': 3.67e-5,
            'max': 4.0e-5
        },
        'seed': {
            'value': 42
        }
    }
}