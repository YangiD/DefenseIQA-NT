# A test demo of HyperIQA and HyperIQA+NT

import torch
import torchvision
import models
from PIL import Image
import numpy as np
import os
import argparse
from torch.autograd import grad
from torch.autograd import Variable
from FGSM_demo import norm


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def fix_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(config):
    fix_seed(919)
    crop_dir = './images_fixedcrop'
    print('crop_dir',crop_dir)

    # HyperIQA
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.train(False)
    modelpath = './checkpoints/livec_bs16_grad[0]_weight[0.0].pth'
    model_dict = torch.load(modelpath)
    if 'model' in model_dict:
        model_hyper.load_state_dict(model_dict['model'])
    else:
        model_hyper.load_state_dict(model_dict)
    # HyperIQA+NT
    model_hyper_NT = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper_NT.train(False)
    modelpath = './checkpoints/livec_bs16_grad[1]_weight[0.001]_h[0.01].pth'
    model_dict_NT = torch.load(modelpath)
    if 'model' in model_dict:
        model_hyper_NT.load_state_dict(model_dict_NT['model'])
    else:
        model_hyper_NT.load_state_dict(model_dict_NT)
    
    transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
    
    imgname = config.img_name
    norm_hyperIQA = []
    norm_hyperIQA_NT = []
    ori_score = []
    NT_score = []
    # 25 patches for each original image
    for i in range(config.patch_size):
        ori_normi = None
        NT_normi = None

        imgpath = os.path.join(crop_dir,imgname[:-4]+'_'+str(i)+imgname[-4:])
        img = pil_loader(imgpath)
        img = transforms(img)
        img = torch.tensor(img.cuda()).unsqueeze(0)
        img.requires_grad_(True)
        
        # HyperIQA -- predicted socre / gradient norm
        paras = model_hyper(img)
        model_target = models.TargetNet(paras).cuda()
        pred_score = model_target(paras['target_in_vec'])
        ori_grad = grad(pred_score, img, grad_outputs=torch.ones_like(pred_score))[0]
        ori_grad = ori_grad.view(ori_grad.shape[0], -1) 
        ori_norm = ori_grad.norm(1, dim=-1, keepdim=True)
        
        # HyperIQA+NT -- predicted score / gradient norm
        NT_paras = model_hyper_NT(img)
        model_target_NT = models.TargetNet(NT_paras).cuda()
        NT_pred = model_target_NT(NT_paras['target_in_vec'])
        NT_grad = grad(NT_pred, img, grad_outputs=torch.ones_like(NT_pred))[0]
        NT_grad = NT_grad.view(NT_grad.shape[0], -1) 
        NT_norm = NT_grad.norm(1, dim=-1, keepdim=True)
        
        if ori_normi is None:
            ori_normi = ori_norm
            NT_normi = NT_norm
        else:
            ori_normi = torch.cat((ori_normi, ori_norm),dim=0)
            NT_normi = torch.cat((NT_normi, NT_norm), dim=0)

        ori_score.append(pred_score.detach().cpu().numpy().astype(np.float))
        NT_score.append(NT_pred.detach().cpu().numpy().astype(np.float))

        ori_normi = ori_normi.detach().cpu().numpy()
        ori_normi = ori_normi.astype(np.float)
        ori_normi = ori_normi.mean()
        norm_hyperIQA.append(ori_normi)
        
        NT_normi = NT_normi.detach().cpu().numpy()
        NT_normi = NT_normi.astype(np.float)
        NT_normi = NT_normi.mean()
        norm_hyperIQA_NT.append(NT_normi)
    
    # the average gradient norm over 25 patches
    norm_hyperIQA = np.array(norm_hyperIQA).mean()
    norm_hyperIQA_NT = np.array(norm_hyperIQA_NT).mean()
    ori_score = np.array(ori_score).mean()
    NT_score = np.array(NT_score).mean()

    print('For {}, the L_1 norm of output\'s gradint in term of the input image:'.format(config.img_name))
    print('HyperIQA:{:.4f}'.format(norm_hyperIQA))
    print('HyperIQA+NT:{:.4f}'.format(norm_hyperIQA_NT))
    print('For {}, the predicted score of the image:'.format(config.img_name))
    print('HyperIQA:{:.4f}'.format(ori_score))
    print('HyperIQA+NT:{:.4f}'.format(NT_score))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=25, help='optional patch size: 10, 25')
    parser.add_argument('--img_name', type=str, default='123.bmp', help='optional image name in this demo: 123.bmp, 447.bmp')
    config = parser.parse_args()
    main(config)
