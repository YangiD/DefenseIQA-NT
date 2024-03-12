# A test demo of FGSM attack on HyperIQA and HyperIQA+NT

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
import numpy as np
from hyperIQAclass import HyperIQA
import argparse


def fix_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
'''
FGSM--untargeted attack
model: the NR-IQA model
x: input image
pred_score: predicted socre of the unattaked image
eps: the l_inf norm bound of the perturbation
alpha: the step size in the I-FGSM attack
iteration: the iteration number of the I-FGSM attack
'''
def IFGSM_IQA_untarget(model, x, pred_score, eps=0.05, alpha=0.01, iteration=10, x_val_min=0, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            tmp_adv = norm(x_adv)
            score_adv = model(tmp_adv)
            if pred_score > 50: # L_{mid}
                loss = score_adv
            else:
                loss = - score_adv
            # loss = torch.abs(score_adv - pred_score) # L_{mae}
            # loss = - torch.pow(score_adv - pred_score, 2) # L_{mse}
            model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward(retain_graph=True)

            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = torch.where(x_adv > x+eps, x+eps, x_adv)
            x_adv = torch.where(x_adv < x-eps, x-eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        score_org = model(norm(x))
        score_adv = model(norm(x_adv))

        return x_adv, score_adv, score_org

def norm(x):
    mean = torch.ones((3,224,224)).cuda()
    std = torch.ones((3,224,224)).cuda()
    mean[0,:,:]=0.485
    mean[1,:,:]=0.456
    mean[2,:,:]=0.406
    std[0,:,:]=0.229
    std[1,:,:]=0.224
    std[2,:,:]=0.225 
    
    x = (x - mean) / std
    
    return x

# save images
def save(pert_image, path):
    pert_image = torch.round(pert_image * 255) / 255
    quantizer = transforms.ToPILImage()
    pert_image = quantizer(pert_image.squeeze())
    pert_image.save(path)
    
    return pert_image


def main(config):
    fix_seed(919)

    transform_w_norm = transforms.Compose([
        transforms.RandomCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])

    transform_wo_norm = transforms.Compose([
        transforms.RandomCrop(size=224),
        transforms.ToTensor()
    ])

    # model to be attacked
    use_cuda = True
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if config.attacked_model=='NT':
        load_path = './checkpoints/livec_bs16_grad[1]_weight[0.001]_h[0.01].pth'
    else:
        load_path = './checkpoints/livec_bs16_grad[0]_weight[0.0].pth'
    model = HyperIQA(load_path).to(device)

    images_mini = [config.img_name]
    mos_dir = {'123.bmp':67.29775,'447.bmp':25.33146}
    mos_mini = [mos_dir[config.img_name]]
    mini_list = [i for i in range(len(images_mini))]


    img_folder = './images_fixedcrop'
    moses = []
    pred_scores = []
    pred_scores_ori = []
    eps = config.epsilon
    iter = config.step
    alpha = config.alpha
    if config.attacked_model=='NT':
        save_dir = './adv_images_patches/IFGSM_spar1e-3_{}_{}_{}'.format(eps, iter, config.patch_num)
        print('Attacking baseline+NT model.')
    else:
        save_dir = './adv_images_patches/IFGSM_baseline_{}_{}_{}'.format(eps, iter, config.patch_num)
        print('Attacking baseline model.')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(len(mini_list)):
        score_i = []
        score_ori_i = []
        moses.append(mos_mini[i])
        for j in range(config.patch_num):
            img_path = os.path.join(img_folder, images_mini[i][:-4]+'_'+str(j)+images_mini[i][-4:])
            image = pil_loader(img_path)
            image = transform_wo_norm(image)
            image = image.unsqueeze(0)
            image = image.cuda()
        
            pred_score = model(image)
            pert_image, pert_score, org_score = IFGSM_IQA_untarget(model, image, pred_score, eps=eps, alpha=alpha, iteration=iter)
             
            save_path = os.path.join(save_dir, images_mini[i][:-4]+'_'+str(j)+images_mini[i][-4:])
            save(pert_image, save_path)
            score_i.append(pert_score.detach().cpu().numpy())
            score_ori_i.append(org_score.detach().cpu().numpy())
            
        score_i = np.array(score_i)
        score_i = np.mean(score_i)
        pred_scores.append(score_i)
        
        score_ori_i = np.array(score_ori_i)
        score_ori_i = np.mean(score_ori_i)
        pred_scores_ori.append(score_ori_i)
        
    pred_scores = np.array(pred_scores).squeeze()
    moses = np.array(moses).squeeze()
    pred_scores_ori = np.array(pred_scores_ori).squeeze()
    
    print('For {}, MOS (without normalization):{}'.format(config.img_name,moses))
    print('Predicted Score (Unattacked):{}'.format(pred_scores_ori))
    print('Predicted Score (Attacked):{}'.format(pred_scores))
    print('Adversarial samples are saved to:',save_dir)
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', dest='epsilon', type=float, default=0.03, help='the scale of FGSM attacks')
    parser.add_argument('--alp', dest='alpha', type=float, default=0.01, help='the step size of FGSM attacks')
    parser.add_argument('--step', dest='step', type=int, default=1, help='the number of steps')
    parser.add_argument('--patch_num', type=int, default=25, help='optional patch size: 10, 25')
    parser.add_argument('--img_name', type=str, default='123.bmp', help='optional image name in this demo: 123.bmp, 447.bmp')
    parser.add_argument('--attacked_model', type=str, default='NT', help='optional attacked model in this demo: NT, baseline')
    
    config = parser.parse_args()
    main(config)