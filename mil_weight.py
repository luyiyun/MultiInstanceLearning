import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np
import progressbar as pb


class Weighter:
    '''
    储存每次计算得到的proba，然后利用这些proba来计算weight。
    实现方式是利用了数字化的bag id和instance id，然后利用这些id来作为index构建
        矩阵来储存proba，这样即便于计算也便于更新proba
    '''

    def __init__(self, dataset, device=torch.device('cuda:0')):
        self.df = dataset.df
        # 因为bagid和instanceid都是唯一的，所有可以使用稀疏tensor来储存已经计算
        #   好的proba, 这里第一个维度是bag_id，第二个维度时instance_id
        i = torch.tensor(self.df[['bag_id', 'instance_id']].values).long().t()
        # 初始所有样本的概率是0.5
        v = torch.full((i.size(1),), 0.5)
        v_mask = torch.full((i.size(1),), 1).float()
        self.p_tensor = torch.sparse.FloatTensor(i, v).to_dense().to(device)
        # 设置一个mask tensor，用来防止在每个bag的instance数量不同时，会自动补充
        #   导致对instance少的bag的过多计数
        self.mask_tensor = torch.sparse.FloatTensor(
            i, v_mask).to_dense().to(device)

    def __call__(self, batch_proba, batch_target, bags_id, instances_id):
        batch_proba = batch_proba.squeeze()
        self.batch_add(batch_proba, batch_target, bags_id, instances_id)
        w = self.batch_weight(batch_proba, batch_target, bags_id, instances_id)
        return w

    def batch_add(self, batch_proba, batch_target, bags_id, instances_id):
        ''' 记录本次batch得到的概率，用于计算之后batch的weight使用 '''
        with torch.no_grad():
            self.p_tensor[bags_id, instances_id] = batch_proba

    def batch_weight(self, batch_proba, batch_target, bags_id, instances_id):
        ''' 输出当前batch的weight '''
        with torch.no_grad():
            log_inv_proba_sum = (1 - self.p_tensor).log().mul(
                self.mask_tensor).sum(dim=1)
            delta = 1 / (log_inv_proba_sum[bags_id] - batch_proba.log()).exp() - 1
            weight = batch_proba / (batch_proba + delta)
            weight = torch.max(weight, (1 - batch_target).float())
        return weight


def test():
    import torchvision.transforms as transforms

    from datasets import MilData
    from networks import NormalCnn

    neg_dir = './DATA/TCT/negative'
    pos_dir = './DATA/TCT/positive'
    transfer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    exam_dat = MilData.from2dir(neg_dir, pos_dir, transfer=transfer)
    dataloader = data.DataLoader(exam_dat, batch_size=8, shuffle=True)

    weighter = Weighter(dataloader.dataset)
    net = NormalCnn().cuda()
    for batch in dataloader:
        batch = [b.cuda() for b in batch]
        imgs, targets, bag_id, inst_id = batch
        proba = net(imgs)
        w = weighter(proba, targets, bag_id, inst_id)
        print(w)
        break


if __name__ == "__main__":
    test()
