sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'val_f1',
        'goal': 'maximize'   
    },
    'parameters': {
        'batch_size': {
            'values': [4, 8, 16, 32, 64]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 3e-5,
            'max': 4e-5
        },
        'num_epochs': {
            'values': [1, 2,3]
        }
    }
}