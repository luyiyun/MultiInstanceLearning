import os

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch.utils.data as data


class TctMilData(data.Dataset):
    ''' 用于多实例学习的数据集 '''
    def __init__(self, df, transfer=None):
        '''
        df：需要有3列，分别是imgf、target、patient、imgid，分别是图像文件的完整路径、
            图像所属病人的标签、病人的id和病人每张图片的id，病人的id在所有病人里
            必须是唯一的。
        transfer：对图像进行与处理的transforms。
        '''
        super(TctMilData, self).__init__()
        assert df.columns.isin(['imgf', 'target', 'patient', 'imgid']).all()
        self.df = df
        self.transfer = transfer

    def __getitem__(self, indx):
        img = Image.open(self.df['imgf'].iloc[indx])
        if self.transfer is not None:
            img = self.transfer(img)
        target = self.df['target'].iloc[indx]
        patient = self.df['patient'].iloc[indx]
        imgid = self.df['imgid'].iloc[indx]
        return img, target, patient, imgid

    def __len__(self):
        return len(self.df)

    @staticmethod
    def from2dir(neg_dir, pos_dir, transfer=None, imgtype='jpg'):
        ''' 静态方法，从两个路径中读取文件，返回一个TctMilData数据 '''
        img_paths = []
        target = []
        patient = []
        imgid = []
        for t, label_dir in zip([0, 1], [neg_dir, pos_dir]):
            for patient_folder in os.listdir(label_dir):
                patient_dir = os.path.join(label_dir, patient_folder)
                for img_f in os.listdir(patient_dir):
                    if img_f.endswith(imgtype):
                        img_path = os.path.join(patient_dir, img_f)
                        img_paths.append(img_path)
                        target.append(t)
                        patient.append(patient_folder + '_' + str(t))
                        imgid.append(img_f)
        df = pd.DataFrame({
            'imgf': img_paths, 'target': target, 'patient': patient,
            'imgid': imgid
        })
        return TctMilData(df, transfer=transfer)

    def split_by_patients(
        self, test_size=0.2, train_transfer=None, test_transfer=None, **kwargs
    ):
        ''' 根据patient id来划分train和test '''
        uniques = self.df[['patient', 'target']].drop_duplicates()
        train, test = train_test_split(
            uniques, test_size=test_size,
            stratify=uniques['target'].values, **kwargs
        )
        train_mask = self.df['patient'].isin(train['patient'].values).values
        test_mask = self.df['patient'].isin(test['patient'].values).values
        train_df = self.df.loc[train_mask]
        test_df = self.df.loc[test_mask]
        return (
            TctMilData(train_df, train_transfer),
            TctMilData(test_df, test_transfer)
        )


class MilDataloader(data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        super(MilDataloader, self).__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers
        )

    def choice_by_patients(self, patients, only_pos=True, samples=False):
        use_df = self.dataset.df.loc[self.dataset.df.patient.isin(patients)]
        if only_pos:
            use_df = use_df.loc[use_df['target'] == 1]
        if samples:
            use_df = use_df.groupby('patient').apply(
                lambda x: x.sample(samples)).reset_index(drop=True)
        use_data = TctMilData(use_df, transfer=self.dataset.transfer)
        return MilDataloader(
            use_data, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers
        )


def test():
    import torchvision.transforms as transforms

    neg_dir = 'G:/dataset/TCT/negative'
    pos_dir = 'G:/dataset/TCT/positive'
    exam_dat = TctMilData.from2dir(neg_dir, pos_dir)
    train, test = exam_dat.split_by_patients(
        shuffle=True, train_transfer=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(512),
            transforms.ToTensor()
        ]), test_transfer=transforms.ToTensor()
    )
    print(len(exam_dat))
    print(len(train))
    print(len(train))
    print(train[0])
    print(train[0][0].shape)
    print(test[0][0].shape)

    dataloader = MilDataloader(train, 2, shuffle=True)
    for batch in dataloader:
        imgs, targets, patients = batch
        print(imgs.shape)
        print(targets.shape)
        print(patients)
        break

    use_patients = ['p1_0', 'p1_1', '3221801885_0']
    subdataloader = dataloader.choice_by_patients(use_patients)
    for batch in subdataloader:
        imgs, targets, patients = batch
        print(imgs.shape)
        print(targets.shape)
        print(patients)
        break


if __name__ == "__main__":
    test()
