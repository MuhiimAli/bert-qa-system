sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'val_precision',
        'goal': 'maximize'   
    },
    'parameters': {
        'batch_size': {
            'values': [8,64]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 3e-5,
            'max': 4e-5
        },
        'num_epochs': {
            'values': [2,3]
        }
    }
}