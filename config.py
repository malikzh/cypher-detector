import torch

def get_configuration():
    return {
        'BATCH_SIZE': 128,
        'LEARNING_RATE': 0.0001,
        'WEIGHT_DECAY': 0.001,
        'EPOCHS': 50,
        'SEQUENCE_LENGTH': 8,
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
        'BYTES_QUANTITY': 256
    }