import torch
from torch import nn
import models

class HyperIQA(nn.Module):
    def __init__(self, modelpath=None):
        super(HyperIQA, self).__init__()
        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(False)
        model_path = modelpath
        print('Load from:',model_path)
        model_dict = torch.load(model_path)
        if 'model' in model_dict:
            self.model_hyper.load_state_dict((model_dict['model']))
        else:
            self.model_hyper.load_state_dict((model_dict))
     

    def forward(self,img): 
        paras = self.model_hyper(img)
        # Building target network
        model_target = models.TargetNet(paras).cuda()
        model_target.train(False)
        for param in model_target.parameters():
            param.requires_grad = False
        # Quality prediction
        pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
        return pred