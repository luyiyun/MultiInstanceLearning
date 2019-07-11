import torch
import numpy as np
import pandas as pd
from scipy.special import softmax

from sklearn import metrics


class Loss:
    def __init__(self):
        super(Loss, self).__init__()
        self.reset()

    def reset(self):
        self.running_loss = 0.
        self.num_samples = 0

    def add(self, batch_loss, batch_size):
        self.running_loss += batch_loss * batch_size
        self.num_samples += batch_size

    def value(self):
        return self.running_loss / self.num_samples


class SklearnMeter:
    def __init__(self, func, tensor=None, proba2int=True, reduction='mean'):
        super(SklearnMeter, self).__init__()
        self.proba2int = proba2int
        self.func = func
        self.tensor = tensor
        self.reduction = reduction
        self.reset()

    def __call__(self, output, target, patientid):
        self.reset()
        self.add(output, target, patientid)
        res = self.value()
        self.reset()
        return res

    def reset(self):
        self.outputs = []
        self.targets = []
        self.ids = []

    def add(self, output, target, patientid):
        if self.tensor is None:
            self.tensor = isinstance(output, torch.Tensor)
        if self.tensor:
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
        self.outputs.append(output)
        self.targets.append(target)
        self.ids += patientid

    def value(self):
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        self.results = pd.DataFrame({
            'output': self.outputs,
            'target': self.targets,
            'patientid': self.ids
        })
        self.reduction_res = self.results.groupby('patientid').agg({
            'output': self.reduction,
            'target': lambda x: x.iloc[0]
        })
        if self.proba2int:
            self.reduction_res['output'] = self.reduction_res['output'] > 0.5
            self.reduction_res['output'] = self.reduction_res['output'].astype(
                'int')
        return self.func(
            self.reduction_res['target'].values,
            self.reduction_res['output'].values
        )


class Accuracy(SklearnMeter):
    def __init__(
        self, proba2int=True, tensor=None, reduction='mean', **kwargs
    ):
        def func(y_true, y_pred):
            return metrics.accuracy_score(y_true, y_pred, **kwargs)
        super(Accuracy, self).__init__(func, tensor, proba2int, reduction)


class BalancedAccuracy(SklearnMeter):
    def __init__(
        self, proba2int=True, tensor=None, reduction='mean', **kwargs
    ):
        def func(y_true, y_pred):
            return metrics.balanced_accuracy_score(y_true, y_pred, **kwargs)
        super(BalancedAccuracy, self).__init__(
            func, tensor, proba2int, reduction)


class F1Score(SklearnMeter):
    def __init__(
        self, proba2int=True, tensor=None, reduction='mean', **kwargs
    ):
        def func(y_true, y_pred):
            return metrics.f1_score(y_true, y_pred, **kwargs)
        super(F1Score, self).__init__(func, tensor, proba2int, reduction)


class Precision(SklearnMeter):
    def __init__(
        self, proba2int=True, tensor=None, reduction='mean', **kwargs
    ):
        def func(y_true, y_pred):
            return metrics.precision_score(y_true, y_pred, **kwargs)
        super(Precision, self).__init__(func, tensor, proba2int, reduction)


class Recall(SklearnMeter):
    def __init__(
        self, proba2int=True, tensor=None, reduction='mean', **kwargs
    ):
        def func(y_true, y_pred):
            return metrics.recall_score(y_true, y_pred, **kwargs)
        super(Recall, self).__init__(func, tensor, proba2int, reduction)


class ROCAUC(SklearnMeter):
    def __init__(
        self, tensor=None, score2proba=False, reduction='mean', **kwargs
    ):
        def func(y_true, y_pred):
            if 'average' in kwargs:
                # 将y_true变成one-hot向量
                max_num = y_true.max()
                eye_matrix = np.eye(max_num+1)
                y_true = eye_matrix[y_true]
            if score2proba:
                y_pred = softmax(y_pred, axis=1)
            return metrics.roc_auc_score(y_true, y_pred, **kwargs)
        super(ROCAUC, self).__init__(
            func, tensor, proba2int=False, reduction=reduction)


def test():
    patientids = np.arange(107)
    targets = np.random.randint(2, size=107)
    patientids_es = []
    targets_es = []
    for p, t in zip(patientids, targets):
        patientids_es += [p] * 332
        targets_es += [t] * 332
    proba = np.random.rand(len(targets_es))

    for cc in [Accuracy, BalancedAccuracy, F1Score, Precision, ROCAUC, Recall]:
        m = cc()
        print('%s: %.4f' % (cc.__name__, m(proba, targets_es, patientids_es)))


if __name__ == "__main__":
    test()
