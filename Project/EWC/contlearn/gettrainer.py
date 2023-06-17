from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import torch.utils.data
from tqdm import tqdm


def var2device(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

def one_epoch_baseline(model: nn.Module, data_loader: torch.utils.data.DataLoader, lr = 1e-3):
    model.train()
    epoch_loss = 0
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    
    for input, target in data_loader:
        # no need for the channel dim
        # bs,1,h,w -> bs,h,w
        # input = input.squeeze(1) # 수정: 내 데이터는 컬러이미지. 채널 3개라 필요x
        input, target = var2device(input), var2device(target)
        
        optimizer.zero_grad()
        output = model(input)
                
        loss = F.cross_entropy(output, target.long()) # 수정 # int도 안 되고 long으로 해야 해결됨..
        
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    correct = 0
    
    for input, target in data_loader:
        input, target = var2device(input), var2device(target)
        #input = input.squeeze(1)  # 수정: 채널 3개라서 이 소스 필요없음. 주석처리
        output = model(input)
        
        #target = target.squeeze(dim=1)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum() # max(~)[1]: value와 index 중 index를 받아오는 것
    return correct / len(data_loader.dataset)
'''
def elastic_weight_consolidation_training(
    model, 
    epochs, 
    train_loader,
    test_loader,
    test2_loader = None,
    use_cuda=True, 
):
    
    """
    This function saves the training curve data consisting
    training set loss and validation set accuracy over the
    course of the epochs of training using the 
    elastic_weight_consolidation method
    
    I set this up such that if you provide 2 test sets,you
    can watch the test accuracy change together during training
    on train_loder
    """
    
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
        
    train_loss, val_acc, val2_acc = [], [], []
    
    for epoch in tqdm(range(epochs)):

        epoch_loss = one_epoch_baseline(model,train_loader)
        train_loss.append(epoch_loss)
        
        acc = test(model,test_loader)
        val_acc.append(acc.detach().cpu().numpy())
        
        if test2_loader is not None:
            acc2 = test(model,test2_loader)
            val2_acc.append(acc2.detach().cpu().numpy())
            
    return train_loss, val_acc, val2_acc, model 
'''


#====================================================================

class EWC(object):
    """
    Class to calculate the Fisher Information Matrix
    used in the Elastic Weight Consolidation portion of the loss function
    """
    
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model #pretrained model
        self.dataset = dataset #samples from the old task or tasks
        
        # n is the string name of the parameter matrix p, aka theta, aka weights
        # in self.params we reference all of those weights that are open to being updated by the gradient.
        # n은 파라미터 행렬 p를 의미하며, 이 모든 파라미터는 기울기에 의해 업데이트 됨            
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        
        # make a copy of the old weights, ie theta_A,star, ie 𝜃∗A, in the loss equation
        # we need this to calculate (𝜃 - 𝜃∗A)^2 because self.params will be changing 
        # upon every backward pass and parameter update by the optimizer
        # 이전 가중치 복사본을 만들고 (𝜃∗A), 오차 방정식에서 (𝜃 - 𝜃∗A)^2가 계산되도록 함.
        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = var2device(p.data)
        
        # calculate the fisher information matrix 
        self._precision_matrices = self._diag_fisher()    # 에러지점
        # 에러 내용: RuntimeError: Expected 4-dimensional input for 4-dimensional weight [32, 3, 3, 3], 
        #                         but got 3-dimensional input of size [3, 224, 224] instead

    def _diag_fisher(self):
        
        # save a copy of the zero'd out version of
        # each layer's parameters of the same shape
        # to precision_matrices[n]
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = var2device(p.data)
            
        # new task의 loss 규제와 EWC lsos 규제를 합침
        # we need the model to calculate the gradient but
        # we have no intention in this step to actually update the model
        # that will have to wait for the combining of this EWC loss term
        # with the new task's loss term
        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            # remove channel dim, these are greyscale, not color rgb images
            # bs,1,h,w -> bs,h,w
            #input = input.squeeze(1)   # 수정:채널 3개라 주석처리
            input = input.unsqueeze(0)
            input = var2device(input)
            output = self.model(input).view(1, -1)  # 배치별로 flatten
            label = output.max(1)[1].view(-1)       # 가장 큰 index (즉 argmax)들을 flatten
            
            # calculate loss and backprop
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
    


def one_epoch_ewc(
    ewc: EWC, 
    importance: float,
    model: nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    lr = 1e-3,
):
    model.train()
    epoch_loss = 0
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    for input, target in data_loader:
        # no need for the channel dim
        # bs,1,h,w -> bs,h,w
        #input = input.squeeze(1) # 수정: 채널 3개라서 필요 없음
        input, target = var2device(input), var2device(target)
        #target = target.squeeze(dim=1)
        
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target.long()) + importance * ewc.penalty(model)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)
