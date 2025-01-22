import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.customDataset import CustomEvalDataset
from model import getModel
from utils.archive import *
from utils.utils import dict2Array
import time
import shutil

if __name__ == '__main__':

    with open('./config/export_descriptors.yaml', 'r', encoding='utf8') as fin:
        config = yaml.safe_load(fin)

    output_dir = config['data']['export_dir']
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print('rm dir:{}'.format(output_dir))
    os.makedirs(output_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    datasetEval = CustomEvalDataset(config['data'], device=device)
    dataloaderEval = DataLoader(datasetEval, batch_size=1, shuffle=False, collate_fn=datasetEval.batch_collator)

    net = getModel(config, device=device)
    net.load_state_dict(torch.load(config['model']['pretrained_model'],map_location=device))
    net.to(device).eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloaderEval)):
            pred1 = net(data['img'])
            pred2 = net(data['warp_img'])

            pred = {'prob': pred1['det_info']['prob_nms'],
                    'warped_prob': pred2['det_info']['prob_nms'],
                    'desc': pred1['desc_info']['desc'],
                    'warped_desc': pred2['desc_info']['desc'],
                    'homography': data['homography']}
            
            pred = dict2Array(pred)
            data = dict2Array(data)
            pred['img'] = data['img']
            pred['warp_img'] = data['warp_img']
            pred = {k: np.transpose(v,(1,2,0)) if k=='warped_desc' or k=='desc' else v for k, v in pred.items()}

            filename = data['name'][0] if 'name' in data else str(i)

            print('number of keypoints {}'.format(len(np.where(pred['prob']>0)[0])))
            filepath = os.path.join(output_dir, '{}.bin'.format(filename))
            pickle_save(filepath, pred)
            time.sleep(1)
            #np.savez_compressed(filepath, **pred)
    print('Done')



