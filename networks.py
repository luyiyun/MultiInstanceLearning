from itertools import chain

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.models as models
import progressbar as pb


def cat_objs(objs):
    one_obj = objs[0]
    if isinstance(one_obj, torch.Tensor):
        if objs[-1].dim() == 0:
            objs[-1] = objs[-1].reshape(1)
        return torch.cat(objs, dim=0)
    elif isinstance(one_obj, (list, tuple)):
        return list(chain.from_iterable(objs))
    else:
        return objs


def collect_results(batch_list):
    return [cat_objs(objs) for objs in zip(*batch_list)]


class NormalCnn(nn.Module):

    def __init__(self):
        super(NormalCnn, self).__init__()
        backbones = list(models.resnet50(True).children())
        self.backbone = nn.Sequential(*backbones[:-1])
        self.final = nn.Sequential(
            nn.Linear(backbones[-1].in_features, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        x = x.squeeze()
        return self.final(x)

    def predict(self, dataloader, bar=True):
        self.cuda()
        results = []
        with torch.no_grad():
            if bar:
                dataloader = pb.progressbar(dataloader, prefix='Predict: ')
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    imgs = batch[0]
                    others = list(batch[1:])
                else:
                    imgs = batch
                    others = []
                imgs = imgs.cuda()
                proba = self.__call__(imgs).squeeze()
                result_batch = [proba] + others
                results.append(result_batch)
        return collect_results(results)


def calculate_weight(proba, patient):
    ''' 计算在p_i / (p_i + delta)中的那个delta '''
    log_inv_proba = np.log(1 - proba)
    df = pd.DataFrame(dict(log_inv_proba=log_inv_proba, patient=patient))
    agg = df.groupby('patient')['log_inv_proba'].transform('sum').values
    delta = np.exp(log_inv_proba - agg) - 1
    score = proba / (proba + delta)
    return score


def calcaulate_weight_for_batch(
    net, dataloader, imgs, patients, patient_id, verbose=True
):
    '''
    为每个batch的元素计算其对应的梯度权重
    '''
    temp_dataloader_positive = dataloader.choice_by_patients(patients)
    proba, _, patients2, patient_id2 = net.predict(
        temp_dataloader_positive, bar=verbose)
    proba_arr = proba.cpu().numpy()
    scores = calculate_weight(proba_arr, patients2)

    df1 = pd.DataFrame(dict(patient=patients, patient_id=patient_id))
    df2 = pd.DataFrame(
        dict(score=scores, patient=patients2, patient_id=patient_id2))
    df = df1.merge(df2, how='left', on=['patient', 'patient_id'])
    df.fillna(1., inplace=True)
    return df['score'].values


def test():
    from datasets import TctMilData, MilDataloader
    import torchvision.transforms as transforms

    neg_dir = 'G:/dataset/TCT/negative'
    pos_dir = 'G:/dataset/TCT/positive'
    exam_dat = TctMilData.from2dir(neg_dir, pos_dir)
    train, test = exam_dat.split_by_patients(
        shuffle=True, train_transfer=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor()
        ]),
        test_transfer=transforms.Compose({
            transforms.Resize(224),
            transforms.ToTensor()
        })
    )
    train_dataloader = MilDataloader(train, 8, shuffle=True)
    # test_dataloader = MilDataloader(test, 8)

    net = NormalCnn()
    for batch in train_dataloader:
        imgs, targets, patients, patient_id = batch
        weight = calcaulate_weight_for_batch(
            net, train_dataloader, imgs, patients, patient_id)
        print(weight)
        break


if __name__ == "__main__":
    test()
