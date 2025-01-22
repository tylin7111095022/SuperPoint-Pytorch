from .magic_point import MagicPoint
from .superpoint import SuperPointNet
from .superpoint_bn import SuperPointBNNet

def getModel(config, device):
    if config['model']['name'] == 'superpoint':
        model = SuperPointBNNet(config['model'], device=device, using_bn=config['model']['using_bn'])
    elif config['model']['name'] == 'magicpoint':
        model = MagicPoint(config['model'], device=device)
    else:
        raise Exception('The model is not implemented.')
    
    return model

__all__ = [getModel] 