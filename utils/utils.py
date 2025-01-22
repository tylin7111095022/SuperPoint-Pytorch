import torch

def dict2Array(data:dict):
    for k, v in data.items():
        if isinstance(v, dict):
            dict2Array(v)
        elif isinstance(v, torch.Tensor):
            data[k] = v.cpu().numpy().squeeze()
    
    return data

def list2array(data:list):
    data = [d.cpu().numpy() for d in data if isinstance(d, torch.Tensor)]

    return data
    