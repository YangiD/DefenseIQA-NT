import torch
from scipy import stats
import numpy as np
import models
import data_loader
from torch.autograd import grad

class HyperIQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()

        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        self.batch_size = config.batch_size
        self.dataset = config.dataset
        self.if_grad = config.if_grad
        if self.if_grad:
            self.h = config.h
            self.weight = config.weight

    # The regularization term proposed in paper: Defense Against Adversarial Attacks on No-Reference Image Quality Models with Gradient Norm Regularization
    def loss_grad(self, images):
       
        images = images.cuda()
        images.requires_grad_(True)

        paras_cur = self.model_hyper(images)
        model_cur = models.TargetNet(paras_cur).cuda()
        pred_cur = model_cur(paras_cur['target_in_vec'])
        
        dx = grad(pred_cur, images, grad_outputs=torch.ones_like(pred_cur), retain_graph=True)[0]
        images.requires_grad_(False)
        
        v = dx.view(dx.shape[0], -1)
        v = torch.sign(v)
        
        v = v.view(dx.shape).detach()
        x2 = images + self.h*v

        paras_pert = self.model_hyper(x2)
        model_pert = models.TargetNet(paras_pert).cuda()
        pred_pert = model_pert(paras_pert['target_in_vec'])

        dl = (pred_pert - pred_cur)/self.h # This is the finite difference approximation of the directional derivative of the loss
        
        loss = dl.pow(2).mean()/2

        return loss
    
    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, label in self.train_data:
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())

                self.solver.zero_grad()

                # Generate weights for target network
                paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                # Use weighted loss if training with NT
                l1_loss = self.l1_loss(pred.squeeze(), label.float().detach())
                if self.if_grad:
                    grad_loss = self.loss_grad(img)
                    loss = l1_loss + self.weight * grad_loss
                else:
                    loss = l1_loss
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                if self.if_grad:
                    save_name = './checkpoints/{}_bs{}_grad[1]_weight[{}].pth'.format(self.dataset,self.batch_size,self.weight)
                else:
                    save_name = './checkpoints/{}_bs{}_grad[0]_weight[0.0].pth'.format(self.dataset,self.batch_size)
                checkpoint = {
                    'model': self.model_hyper.state_dict(),
                    'optimizer': self.solver.state_dict()
                }
                torch.save(checkpoint, save_name)
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in data:
            # Data.
            img = torch.tensor(img.cuda())
            label = torch.tensor(label.cuda())

            paras = self.model_hyper(img)
            model_target = models.TargetNet(paras).cuda()
            model_target.train(False)
            pred = model_target(paras['target_in_vec'])

            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_hyper.train(True)
        return test_srcc, test_plcc