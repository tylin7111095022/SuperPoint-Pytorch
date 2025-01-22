#-*-coding:utf8-*-
import os
import glob
import torch
import yaml
import matplotlib.pyplot as plt
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import DataLoader
import kornia
# from utils.params import dict_update
from dataset.utils.homographic_augmentation import homographic_aug_pipline, ratio_preserving_resize, sample_homography
from dataset.utils.photometric_augmentation import *
from utils.keypoint_op import compute_keypoint_map
from utils.utils import list2array

class CustomTrainDataset(torch.utils.data.Dataset):

    def __init__(self, config, is_train, device='cpu'):

        super(CustomTrainDataset, self).__init__()
        self.device = device
        self.is_train = is_train
        self.resize = tuple(config['resize']) if config.get('resize') else None
        self.photo_augmentor = PhotoAugmentor(config['augmentation']['photometric']) if config.get('augmentation', {}).get('photometric', {}) else None
        # load config
        self.config = config #dict_update(getattr(self, 'default_config', {}), config)
        # get images
        if self.is_train:
            self.samples = self._init_data(config['image_train_path'], config['label_train_path'])
        else:
            self.samples = self._init_data(config['image_test_path'], config['label_test_path'])


    def _init_data(self, image_path, label_path=None):
        ##
        if not isinstance(image_path,list):
            image_paths, label_paths = [image_path,], [label_path,]
        else:
            image_paths, label_paths = image_path, label_path

        image_types = ['jpg','jpeg','bmp','png']
        samples = []
        for im_path, lb_path in zip(image_paths, label_paths):
            for it in image_types:
                temp_im = glob.glob(os.path.join(im_path, '*.{}'.format(it)))
                if lb_path is not None:
                    temp_lb = [os.path.join(lb_path, os.path.basename(imp).split('.')[0]+'.npy') for imp in temp_im]
                else:
                    temp_lb = [None,]*len(temp_im)
                temp = [{'image':imp, 'label':lb} for imp, lb in zip(temp_im, temp_lb)]
                samples += temp
        ##
        return samples

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        '''load raw data'''
        data_path = self.samples[idx]#raw image path of processed image and point path
        name = os.path.basename(data_path['image']).split('.')[0]
        img = cv2.imread(data_path['image'], 0) #Gray image
        if self.resize:
            img = cv2.resize(img, self.resize[::-1])
        pts = None if data_path['label'] is None else np.load(data_path['label']).astype(np.float32) #N*2,yx

        # init data dict
        img_tensor = torch.as_tensor(img.copy(), dtype=torch.float, device=self.device)
        kpts_tensor = None if pts is None else torch.as_tensor(pts, device=self.device)
        kpts_map = None if pts is None else compute_keypoint_map(kpts_tensor, img.shape, device=self.device)
        valid_mask = torch.ones(img.shape, device=self.device)

        data = {'raw':{'name': name,
                       'img': img_tensor,
                       'kpts': kpts_tensor,
                       'kpts_map':kpts_map,
                       'mask':valid_mask},
                'warp':None,
                'homography':torch.eye(3,device=self.device)}
        
        data['warp'] = deepcopy(data['raw'])

        # photometric
        photo_enable = None
        if self.photo_augmentor is not None:
            photo_enable = self.config['augmentation']['photometric']['train_enable'] if self.is_train else self.config['augmentation']['photometric']['test_enable']

        # homographic
        homo_enable = self.config['augmentation']['homographic']['train_enable'] if self.is_train else self.config['augmentation']['homographic']['test_enable']

        if homo_enable: # homographic augmentation
            # return dict{warp:{img:[H,W], point:[N,2], valid_mask:[H,W], homography: [3,3]; tensors}}
            data_homo = homographic_aug_pipline(data['warp']['img'],
                                                data['warp']['kpts'],
                                                self.config['augmentation']['homographic'],
                                                device=self.device)
            data.update(data_homo)
            data['warp']['name'] = 'warp_' + data['raw']['name']

        if photo_enable:
            photo_img = data['warp']['img'].cpu().numpy().round().astype(np.uint8)
            if self.photo_augmentor:
                photo_img = self.photo_augmentor(photo_img)
            data['warp']['img'] = torch.as_tensor(photo_img, dtype=torch.float,device=self.device)

        ##normalize
        data['raw']['img'] = data['raw']['img']/255.
        data['warp']['img'] = data['warp']['img']/255.

        return data # name: str, img:HW, kpts:N2, kpts_map:HW, valid_mask:HW, homography:HW

    def batch_collator(self, samples):
        """
        :param samples:a list, each element is a dict with keys
        like `img`, `img_name`, `kpts`, `kpts_map`,
        `valid_mask`, `homography`...
        img:H*W, kpts:N*2, kpts_map:HW, valid_mask:HW, homography:HW
        :return:
        """
        sub_data = {'img': [], 'kpts_map': [],'mask': []} # remove kpts
        batch = {'raw':sub_data, 'warp':deepcopy(sub_data), 'homography': [], 'name': []}
        for s in samples:
            batch['homography'].append(s['homography'])
            batch['name'].append(s['raw']['name'])
            for k in sub_data:
                if k=='img':
                    batch['raw'][k].append(s['raw'][k].unsqueeze(dim=0))
                    if 'warp' in s:
                        batch['warp'][k].append(s['warp'][k].unsqueeze(dim=0))
                else:
                    batch['raw'][k].append(s['raw'][k])
                    if 'warp' in s:
                        batch['warp'][k].append(s['warp'][k])
        ##
        batch['homography'] = torch.stack(batch['homography'])
        for k0 in ('raw','warp'):
            for k1 in sub_data:
                if None in batch[k0][k1]:
                    continue 
                batch[k0][k1] = torch.stack(batch[k0][k1])

        return batch
    
class CustomEvalDataset(torch.utils.data.Dataset):
    default_config = {
        'dataset': 'custom',  # or 'coco'
        'alteration': 'all',  # 'all', 'i' for illumination or 'v' for viewpoint
        'cache_in_memory': False,
        'truncate': None,
        'preprocessing': {
            'resize': False
        }
    }

    def __init__(self, config, device='cpu'):

        super(CustomEvalDataset, self).__init__()
        self.device = device
        self.config = config
        self.files = self._init_dataset()

    def _init_dataset(self, imageType:str='png'):
        ##
        dataset_folder = self.config['data_dir']
        if not isinstance(dataset_folder,list):
            image_paths = [dataset_folder]
        files = []
        for im_path in image_paths:
            temp_im = glob.glob(os.path.join(im_path, '*.{}'.format(imageType)))
            imgs = [cv2.imread(p, 0) for p in temp_im] # img shape: H, W
            imgs_t = [torch.as_tensor(img.copy(), dtype=torch.float, device=self.device) for img in imgs]

            homographies = [sample_homography(np.array(img.shape), self.config['homography_adaptation']['homographies'], device=self.device) for img in imgs]
            warpedImgs = [kornia.warp_perspective(img[None, None, :, :], h, img.shape, align_corners=True).squeeze() for img, h in zip(imgs_t, homographies) ]
            homographies = list2array(homographies)
            warpedImgs = list2array(warpedImgs)
            
            temp = [{'image':img, "warp":warp, "homography": homo, 'name': os.path.basename(path).split('.')[0]} for img, warp, homo, path in zip(imgs ,warpedImgs, homographies,temp_im) ]

            files += temp
        
        return files
    
    def _preprocess(self, image):
        if len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = ratio_preserving_resize(image, self.config['preprocessing']['resize'])
        return image

    def _adapt_homography_to_preprocessing(self, zip_data):
        '''缩放后对应的图像的homography矩阵
        :param zip_data:{'shape':原图像HW,
                         'warped_shape':warped图像HW,
                         'homography':原始变换矩阵}
        :return:对应当前图像尺寸的homography矩阵
        '''
        if isinstance(zip_data['homography'], torch.Tensor):
            zip_data['homography'] = zip_data['homography'].cpu().numpy()

        H = zip_data['homography'].astype(np.float32)
        source_size = zip_data['shape'].astype(np.float32)#h,w
        source_warped_size = zip_data['warped_shape'].astype(np.float32)#h,w
        target_size = np.array(self.config['preprocessing']['resize'],dtype=np.float32)#h,w

        # Compute the scaling ratio due to the resizing for both images
        s = np.max(target_size/source_size)
        up_scale = np.diag([1./s, 1./s, 1])
        warped_s = np.max(target_size/source_warped_size)
        down_scale = np.diag([warped_s, warped_s, 1])

        # Compute the translation due to the crop for both images
        pad_y, pad_x = (source_size*s - target_size)//2.0

        translation = np.array([[1, 0, pad_x],
                                [0, 1, pad_y],
                                [0, 0, 1]],dtype=np.float32)
        pad_y, pad_x = (source_warped_size*warped_s - target_size) //2.0

        warped_translation = np.array([[1,0, -pad_x],
                                       [0,1, -pad_y],
                                       [0,0,1]], dtype=np.float32)
        H = warped_translation @ down_scale @ H @ up_scale @ translation
        return H

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        img = self.files[idx]['image']
        warped_img = self.files[idx]['warp']
        homography = self.files[idx]['homography']
        name = self.files[idx]['name']

        if self.config['preprocessing']['resize']:
            img_shape = img.shape
            warped_shape = warped_img.shape
            homographyOption = {'homography': homography, 'shape': np.array(img_shape), 'warped_shape': np.array(warped_shape)}
            homography = self._adapt_homography_to_preprocessing(homographyOption)

        img = self._preprocess(img)
        warped_img = self._preprocess(warped_img)

        ##to tenosr
        img = torch.as_tensor(img,dtype=torch.float32, device=self.device)#HW
        warped_img = torch.as_tensor(warped_img, dtype=torch.float32, device=self.device)#HW
        homography = torch.as_tensor(homography, device=self.device)#HW
        ##normalize
        img = img/255.
        warped_img = warped_img/255.

        data = {'img': img, 'warp_img': warped_img, 'homography': homography, 'name': name}

        return data

    def batch_collator(self, samples):
        """
        :param samples:a list, each element is a dict with keys
        like `img`, `img_name`, `kpts`, `kpts_map`,
        `valid_mask`, `homography`...
        img:H*W, kpts:N*2, kpts_map:HW, valid_mask:HW, homography:HW
        :return:batch data
        """
        assert (len(samples) > 0 and isinstance(samples[0], dict))
        batch = {'img':[], 'warp_img':[], 'homography': [], 'name': []}
        for s in samples:
            for k,v in s.items():
                if 'img' in k:
                    batch[k].append(v.unsqueeze(dim=0)) # add channel axis
                else:
                    batch[k].append(v)
        ##
        for k in batch:
            if k == 'name':
                continue
            batch[k] = torch.stack(batch[k],dim=0)
        return batch
    
def main():
    with open('config/superpoint_angiogram_train.yaml','r') as fin:
        config = yaml.safe_load(fin)

    dataset = CustomTrainDataset(config['data'],True)
    dataloader = DataLoader(dataset,collate_fn=dataset.batch_collator,batch_size=1,shuffle=True)

    for i,d in enumerate(dataloader):
        if i>=5:
            break
        img = (d['raw']['img']*255).cpu().numpy().squeeze().astype(np.int64).astype(np.uint8)
        img_warp = (d['warp']['img']*255).cpu().numpy().squeeze().astype(np.int64).astype(np.uint8)
        img = cv2.merge([img, img, img])
        img_warp = cv2.merge([img_warp, img_warp, img_warp])
        ##
        kpts = np.where(d['raw']['kpts_map'].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(np.int64)
        for kp in kpts:
            cv2.circle(img, (kp[1], kp[0]), radius=3, color=(0,255,0))
        kpts = np.where(d['warp']['kpts_map'].squeeze().cpu().numpy())
        kpts = np.vstack(kpts).T
        kpts = np.round(kpts).astype(np.int64)
        for kp in kpts:
            cv2.circle(img_warp, (kp[1], kp[0]), radius=3, color=(0,255,0))

        mask = d['raw']['mask'].cpu().numpy().squeeze().astype(np.int64).astype(np.uint8)*255
        warp_mask = d['warp']['mask'].cpu().numpy().squeeze().astype(np.int64).astype(np.uint8)*255

        img = cv2.resize(img, (640,480))
        img_warp = cv2.resize(img_warp,(640,480))

        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.subplot(2,2,2)
        plt.imshow(mask)
        plt.subplot(2,2,3)
        plt.imshow(img_warp)
        plt.subplot(2,2,4)
        plt.imshow(warp_mask)
        plt.show()

    print('Done')



if __name__=='__main__':
    main()