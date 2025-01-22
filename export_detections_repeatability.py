#-*-coding:utf-8-*-
import os
import yaml
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from dataset.customDataset import CustomEvalDataset
from dataset.synthetic_shapes import SyntheticShapes
from model import getModel
from utils.utils import dict2Array


if __name__=="__main__":
    ##
    with open('./config/detection_repeatability.yaml', 'r', encoding='utf8') as fin:
        config = yaml.safe_load(fin)

    output_dir = config['data']['export_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if config['data']['name']=='synthetic':
        dataset_ = SyntheticShapes(config['data'], task='training', device=device)
    elif config['data']['name'] != 'synthetic':
        dataset_ = CustomEvalDataset(config['data'], device=device)

    p_dataloader = DataLoader(dataset_, batch_size=1, shuffle=False, collate_fn=dataset_.batch_collator)

    net = getModel(config, device=device)

    net.load_state_dict(torch.load(config['model']['pretrained_model'], map_location=device))
    net.to(device).eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(p_dataloader)):
            prob1 = net(data['img'])
            prob2 = net(data['warp_img'])
            
            pred = {'prob':prob1['det_info']['prob_nms'], 'warp_prob':prob2['det_info']['prob_nms'],
                    'homography': data['homography']}
            
            pred = dict2Array(pred)
            data = dict2Array(data)
            pred['img'] = data['img']
            pred['warp_img'] = data['warp_img']

            filename = data['name'][0] if 'name' in data else str(i)
            filepath = os.path.join(output_dir, f'{filename}.npz')
            np.savez_compressed(filepath, **pred)

    print('Done')