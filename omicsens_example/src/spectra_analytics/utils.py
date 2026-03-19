import torch


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])

def R2score(predictions, target):
    return 1 - ((predictions - target) ** 2).mean() / target.var()