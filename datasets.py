import os

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


class MilData(data.Dataset):
    '''
    用于多实例学习的数据集，其中bag id和instance id都会被转换成数字，其对应关系
    储存在相应的LabelEncoder对象中。
    '''
    def __init__(
        self, df, transfer=None, encoder=True, bags_le=None, instances_le=None
    ):
        '''
        df：需要有4列，分别是imgf、target、bag_id、instance_id，分别是图像文件
            的完整路径、图像的标签(实际上是bag的标签，一个bag中的所有图像共用)、
            bag的id和bag中每个instance的id
        transfer：对图像进行与处理的transforms。
        '''
        super(MilData, self).__init__()
        assert df.columns.isin(
            ['imgf', 'target', 'bag_id', 'instance_id']).all()
        self.df = df
        self.transfer = transfer
        # 对bag_id和instance_id进行编码
        if encoder:
            self.bags_le, self.instances_le = LabelEncoder(), LabelEncoder()
            self.df['bag_id'] = self.bags_le.fit_transform(
                self.df['bag_id'].values)
            self.df['instance_id'] = self.instances_le.fit_transform(
                self.df['instance_id'].values)
        else:
            assert bags_le is not None
            assert instances_le is not None
            self.bags_le, self.instances_le = bags_le, instances_le

    def __getitem__(self, indx):
        img = Image.open(self.df['imgf'].iloc[indx])
        if self.transfer is not None:
            img = self.transfer(img)
        target, bag_id, instance_id = self.df.iloc[indx, 1:].values
        return img, target, bag_id, instance_id

    def __len__(self):
        return len(self.df)

    @staticmethod
    def from2dir(neg_dir, pos_dir, transfer=None, imgtype='jpg'):
        ''' 静态方法，从两个路径中读取文件，返回一个MilData数据 '''
        img_paths = []
        target = []
        bag_ids = []
        instance_ids = []
        for t, label_dir in zip([0, 1], [neg_dir, pos_dir]):
            for bag_folder in os.listdir(label_dir):
                bag_dir = os.path.join(label_dir, bag_folder)
                for img_f in os.listdir(bag_dir):
                    if img_f.endswith(imgtype):
                        img_path = os.path.join(bag_dir, img_f)
                        img_paths.append(img_path)
                        target.append(t)
                        bag_ids.append(bag_folder + '_' + str(t))
                        instance_ids.append(img_f)
        df = pd.DataFrame({
            'imgf': img_paths, 'target': target, 'bag_id': bag_ids,
            'instance_id': instance_ids
        })
        return MilData(df, transfer=transfer)

    def split_by_bag(
        self, test_size=0.1, valid_size=None, train_transfer=None,
        test_transfer=None, valid_transfer=None, **kwargs
    ):
        ''' 根据bag id来划分train和test, 可以有valid '''
        uniques = self.df[['bag_id', 'target']].drop_duplicates()

        # 根据size来分割数据集
        train, test = train_test_split(
            uniques, test_size=test_size,
            stratify=uniques['target'].values, **kwargs
        )
        if valid_size is not None:
            train, valid = train_test_split(
                train, test_size=valid_size/(1-test_size),
                stratify=train['target'].values, **kwargs
            )
        # 选择出每个部分对应的df
        train_mask = self.df['bag_id'].isin(train['bag_id'].values).values
        test_mask = self.df['bag_id'].isin(test['bag_id'].values).values
        train_df = self.df.loc[train_mask]
        test_df = self.df.loc[test_mask]
        if valid_size is not None:
            valid_mask = self.df['bag_id'].isin(valid['bag_id'].values).values
            valid_df = self.df.loc[valid_mask]
        # 将结果放在一起输出
        datasets = [
            MilData(
                train_df, train_transfer, encoder=False, bags_le=self.bags_le,
                instances_le=self.instances_le
            ),
            MilData(
                test_df, test_transfer, encoder=False, bags_le=self.bags_le,
                instances_le=self.instances_le
            )
        ]
        if valid_size is not None:
            datasets.insert(
                1, MilData(
                    valid_df, valid_transfer, encoder=False,
                    bags_le=self.bags_le, instances_le=self.instances_le
                )
            )
        return tuple(datasets)

    def cvsplit_by_bag(
        self, cv, test_size=0.1, valid_size=None, train_transfer=None,
        test_transfer=None, valid_transfer=None, random_seed=None
    ):
        ''' 根据bag id来划分train和test, 可以有valid, 进行交叉验证 '''
        uniques = self.df[['bag_id', 'target']].drop_duplicates()
        sss = StratifiedShuffleSplit(
            cv, test_size=test_size, random_state=random_seed)

        for train_indx, test_indx in sss.split(
            uniques.values, uniques['target'].values
        ):
            # 根据size来分割数据集
            train = uniques.iloc[train_indx, :]
            test = uniques.iloc[test_indx, :]

            if valid_size is not None:
                train, valid = train_test_split(
                    train, test_size=valid_size/(1-test_size),
                    stratify=train['target'].values,
                    random_state=random_seed, shuffle=True
                )
            # 选择出每个部分对应的df
            train_mask = self.df['bag_id'].isin(train['bag_id'].values).values
            test_mask = self.df['bag_id'].isin(test['bag_id'].values).values
            train_df = self.df.loc[train_mask]
            test_df = self.df.loc[test_mask]
            if valid_size is not None:
                valid_mask = self.df['bag_id'].isin(
                    valid['bag_id'].values).values
                valid_df = self.df.loc[valid_mask]
            # 将结果放在一起输出
            datasets = [
                MilData(
                    train_df, train_transfer, encoder=False,
                    bags_le=self.bags_le, instances_le=self.instances_le
                ),
                MilData(
                    test_df, test_transfer, encoder=False,
                    bags_le=self.bags_le, instances_le=self.instances_le
                )
            ]
            if valid_size is not None:
                datasets.insert(
                    1, MilData(
                        valid_df, valid_transfer, encoder=False,
                        bags_le=self.bags_le, instances_le=self.instances_le
                    )
                )
            yield tuple(datasets)


def test():
    import torchvision.transforms as transforms

    neg_dir = './DATA/TCT/negative'
    pos_dir = './DATA/TCT/positive'
    transfer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    exam_dat = MilData.from2dir(neg_dir, pos_dir, transfer=transfer)
    # dataloader = data.DataLoader(exam_dat, batch_size=8, shuffle=True)

    # for batch in dataloader:
    #     imgs, targets, bag_id, inst_id = batch
    #     print(imgs)
    #     print(targets)
    #     print(bag_id)
    #     print(inst_id)
    #     break
    i = 1
    for train, valid, test in exam_dat.cvsplit_by_bag(10, 0.1, 0.1):
        print(i)
        print("%d, %d, %d" % (len(train), len(valid), len(test)))
        print('train bag id:')
        print(train.df.bag_id.value_counts())
        print('valid bag id:')
        print(valid.df.bag_id.value_counts())
        print('test bag id:')
        print(test.df.bag_id.value_counts())
        print('')
        i += 1


if __name__ == "__main__":
    test()
