import os
import copy
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import progressbar as pb
import pandas as pd
import argparse

from datasets import TctMilData, MilDataloader
from networks import calcaulate_weight_for_batch, NormalCnn
import metrics as mm


class NoneScheduler:
    def __init__(self, optimizer):
        pass

    def step(self):
        pass


def predict(model, dataloader, device=torch.device('cuda:0'), bar=True):
    # 记录model之前所处的状态，以便于在使用其预测完成之后恢复其状态
    ori_phase = model.training
    ori_device = next(model.parameters()).device

    model.eval()
    model.to(device)
    with torch.no_grad():
        if bar:
            dataloader = pb.progressbar(dataloader, prefix='Predict: ')
        preds = []
        patient_ids = []
        file_names = []
        for batch in dataloader:
            # 因为这里是多实例学习的任务，所以每次迭代返回的batch中除了图像外，
            #   一定还有这张图像所属的病人id和这张图像的名称，所以batch一定是
            #   一个sequence
            imgs = batch[0].to(device)
            imgs = batch.to(device)
            patient_id, file_name = batch[-2], batch[-1]
            pred = model(imgs)
            preds.append(pred)
            patient_ids += list(patient_id)
            file_names += list(file_name)
        preds = torch.cat(preds, dim=0).cpu().numpy()
    res_df = pd.DataFrame({
        'score': preds, 'patient_id': patient_ids, 'file_name': file_names})
    # 完成预测任务后将模型的状态返回之前
    model.train(ori_phase)
    model.to(ori_device)
    return res_df


def evaluate(
    model, dataloader, criterion, metrics,
    device=torch.device('cuda:0'), bar=True
):
    # 记录之前的状态
    ori_phase = model.training
    ori_device = next(model.parameters()).device
    # 将模型调整至预测模式
    model.eval()
    model.to(device)
    # 准备一个字典来储存评价结果
    history = {}
    for m in metrics:
        m.reset()
    if bar:
        dataloader = pb.progressbar(dataloader, prefix='Test: ')
    for batch_x, batch_y, batch_ids, batch_files in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.no_grad():
            scores = model(batch_x)
            loss = criterion(scores, batch_y)
            for m in metrics:
                if isinstance(m, mm.Loss):
                    m.add(loss.cpu().item(), batch_x.size(0))
                else:
                    m.add(scores.squeeze(), batch_y, batch_ids)
    for m in metrics:
        history[m.__class__.__name__] = m.value()
    print(
        "Test results: " +
        ", ".join([
            '%s: %.4f' % (m.__class__.__name__, history[m.__class__.__name__])
            for m in metrics
        ])
    )
    model.train(ori_phase)
    model.to(ori_device)
    return history


def train(
    model, criterion, optimizer, dataloaders, scheduler=NoneScheduler(None),
    epoch=100, device=torch.device('cuda:0'), l2=0.0,
    metrics=(mm.Loss(),), standard_metric_index=1,
    clip_grad=False
):
    # 构建几个变量来储存最好的模型
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0
    best_metric_name = metrics[standard_metric_index].__class__.__name__ + \
        '_valid'
    # 构建dict来储存训练过程中的结果
    history = {
        m.__class__.__name__+p: []
        for p in ['_train', '_valid']
        for m in metrics
    }
    model.to(device)

    for e in range(epoch):
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
                prefix = "Train: "
            else:
                model.eval()
                prefix = "Valid: "
            # progressbar
            format_custom_text = pb.FormatCustomText(
                'Loss: %(loss).4f', dict(loss=0.))
            widgets = [
                prefix, " ",
                pb.Counter(),
                ' ', pb.SimpleProgress(
                    format='(%s)' % pb.SimpleProgress.DEFAULT_FORMAT
                ),
                ' ', pb.Bar(),
                ' ', pb.Timer(),
                ' ', pb.AdaptiveETA(),
                ' ', format_custom_text
            ]
            iterator = pb.progressbar(dataloaders[phase], widgets=widgets)

            for m in metrics:
                m.reset()
            import ipdb; ipdb.set_trace()
            for batch_x, batch_y, batch_ids, batch_files in iterator:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    proba = model(batch_x)  # 注意模型输出的需要时proba
                    # 计算每个样本的权重
                    weights = calcaulate_weight_for_batch(
                        model, dataloaders[phase], batch_x, batch_ids,
                        batch_files, verbose=False
                    )
                    # 这个criterion不能reduction
                    loss_es = criterion(proba, batch_y.to(torch.float))
                    # 使用计算的权重
                    loss = (loss_es * weights).mean()
                    # 只给weight加l2正则化
                    if l2 > 0.0:
                        for p_n, p_v in model.named_parameters():
                            if p_n == 'weight':
                                loss += l2 * p_v.norm()
                    if phase == 'train':
                        loss.backward()
                        if clip_grad:
                            nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1)
                        optimizer.step()
                with torch.no_grad():
                    for m in metrics:
                        if isinstance(m, mm.Loss):
                            m.add(loss.cpu().item(), batch_x.size(0))
                            format_custom_text.update_mapping(loss=m.value())
                        else:
                            m.add(proba.squeeze(), batch_y, batch_ids)

            for m in metrics:
                history[m.__class__.__name__+'_'+phase].append(m.value())
            print(
                "Epoch: %d, Phase:%s, " % (e, phase) +
                ", ".join([
                    '%s: %.4f' % (
                        m.__class__.__name__,
                        history[m.__class__.__name__+'_'+phase][-1]
                    ) for m in metrics
                ])
            )

            if phase == 'valid':
                epoch_metric = history[best_metric_name][-1]
                if epoch_metric > best_metric:
                    best_metric = epoch_metric
                    best_model_wts = copy.deepcopy(model.state_dict())

    print("Best metric: %.4f" % best_metric)
    model.load_state_dict(best_model_wts)
    return model, history


def check_update_dirname(dirname):
    if os.path.exists(dirname):
        dirname += '-'
        check_update_dirname(dirname)
    else:
        os.makedirs(dirname)
        return dirname


def main():

    # config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--save', default='./save',
        help='保存的文件夹路径，如果有重名，会在其后加-来区别'
    )
    parser.add_argument(
        '-is', '--image_size', default=224, type=int,
        help='patch会被resize到多大，默认时224 x 224'
    )
    parser.add_argument(
        '-ts', '--test_size', default=0.2, type=float,
        help='测试集的大小，默认时0.2'
    )
    parser.add_argument(
        '-bs', '--batch_size', default=64, type=int,
        help='batch size，默认时64'
    )
    parser.add_argument(
        '-nw', '--num_workers', default=12, type=int,
        help='多进程数目，默认时12'
    )
    parser.add_argument(
        '-lr', '--learning_rate', default=0.0001, type=float,
        help='学习率大小，默认时0.0001'
    )
    parser.add_argument(
        '-e', '--epoch', default=10, type=int,
        help='epoch 数量，默认是10'
    )
    parser.add_argument(
        '--reduction', default='mean',
        help='聚合同一bag的instances时的聚合方式，默认时mean'
    )
    args = parser.parse_args()
    save = args.save
    image_size = (args.image_size, args.image_size)
    test_size = args.test_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.learning_rate
    epoch = args.epoch
    reduction = args.reduction

    # ----- 读取数据 -----
    neg_dir = 'G:/dataset/TCT/negative'
    pos_dir = 'G:/dataset/TCT/positive'

    dat = TctMilData.from2dir(neg_dir, pos_dir)
    train_transfer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transfer = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dat, valid_dat = dat.split_by_patients(
        test_size, train_transfer=train_transfer, test_transfer=test_transfer)
    dataloaders = {
        'train': MilDataloader(
            train_dat, batch_size=batch_size, num_workers=num_workers,
            shuffle=True
        ),
        'valid': MilDataloader(
            valid_dat, batch_size=batch_size, num_workers=num_workers,
        )
    }

    # ----- 构建网络和优化器 -----
    net = NormalCnn()
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scorings = [
        mm.Loss(), mm.ROCAUC(reduction=reduction),
        mm.BalancedAccuracy(reduction=reduction),
        mm.F1Score(reduction=reduction),
        mm.Precision(reduction=reduction),
        mm.Recall(reduction=reduction),
        mm.Accuracy(reduction=reduction)
    ]

    # ----- 训练网络 -----
    net, hist = train(
        net, criterion, optimizer, dataloaders, epoch=epoch, metrics=scorings
    )

    # 保存结果
    dirname = check_update_dirname(save)
    torch.save(net.state_dict(), os.path.join(dirname, 'model.pth'))
    pd.DataFrame(hist).to_csv(os.path.join(dirname, 'train.csv'))
    with open(os.path.join(dirname, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f)


if __name__ == "__main__":
    main()
