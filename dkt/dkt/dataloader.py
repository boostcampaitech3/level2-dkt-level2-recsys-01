import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        #cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag'] 
        cate_cols = ['assessmentItemID', 'testId', 'tag'] 

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            #모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test


        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        # df['Timestamp'] = df['Timestamp'].apply(convert_time)
        cont_cols = ['test_mean', 'assessment_mean', 'tag_mean']
        #cont_cols = ['user_correct_answer', 'user_total_answer', 'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']#,'elapsed_assessmentItemID','test_elapsed_assessmentItemID','elapsed_testId']
        #cont_cols = [ 'test_mean','assessment_mean', 'tag_mean', 'test_sum','tag_sum','assessment_sum']
        df[cont_cols] = df[cont_cols].astype(np.float32)
        return df

    def __feature_engineering(self, df, is_train, train_df=None):
        if is_train:
            train_df = df

        df.rename(columns={'userID': 'user', 'answerCode': 'answer', 'KnowledgeTag': 'tag'}, inplace=True)
        df.sort_values(by=['user','Timestamp'], inplace=True)
        correct_t = train_df.groupby(['testId'])['answer'].agg(['mean', 'sum'])
        correct_t.columns = ["test_mean", 'test_sum']
        correct_k = train_df.groupby(['tag'])['answer'].agg(['mean', 'sum'])
        correct_k.columns = ["tag_mean", 'tag_sum']
        correct_a = train_df.groupby(['assessmentItemID'])['answer'].agg(['mean', 'sum'])
        correct_a.columns = ["assessment_mean", 'assessment_sum']

        df.sort_values(by=['user','Timestamp'], inplace=True)

        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['tag'], how="left")
        df = pd.merge(df, correct_a, on=['assessmentItemID'], how="left")

    
    # #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    #     df['user_correct_answer'] = df.groupby('user')['answer'].transform(lambda x: x.cumsum().shift(1))
    #     df['user_total_answer'] = df.groupby('user')['answer'].cumcount()
    #     df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

    #     #유저별 카테고리별
    #     df['user_category_correct_answer'] = df.groupby(['user', 'category'])['answer'].transform(lambda x: x.cumsum().shift(1))
    #     df['user_category_total_answer'] = df.groupby(['user', 'category'])['answer'].cumcount()
    #     df['user_category_acc'] = df['user_category_correct_answer']/df['user_category_total_answer']

    #     #유저별 태그별
    #     df['user_tag_correct_answer'] = df.groupby(['user', 'tag'])['answer'].transform(lambda x: x.cumsum().shift(1))
    #     df['user_tag_total_answer'] = df.groupby(['user', 'tag'])['answer'].cumcount()
    #     df['user_tag_acc'] = df['user_tag_correct_answer']/df['user_tag_total_answer']

    #     # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    #     # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    #     correct_t = df.groupby(['testId'])['answer'].agg(['mean', 'sum'])
    #     correct_t.columns = ["test_mean", 'test_sum']
    #     correct_k = df.groupby(['tag'])['answer'].agg(['mean', 'sum'])
    #     correct_k.columns = ["tag_mean", 'tag_sum']

    #     elapsed_user = df.groupby(['user'])['elapsed'].mean()
    #     elapsed_user.rename('elapsed_user', inplace=True)
    #     test_elapsed_user = df.groupby(['user'])['test_elapsed'].mean()
    #     test_elapsed_user.rename('test_elapsed_user', inplace=True)
    #     elapsed_testId = df.groupby(['testId'])['elapsed'].mean()
    #     elapsed_testId.rename('elapsed_testId', inplace=True)
    #     test_elapsed_testId = df.groupby(['testId'])['test_elapsed'].mean()
    #     test_elapsed_testId.rename('test_elapsed_testId', inplace=True)
    #     elapsed_category = df.groupby(['category'])['elapsed'].mean()
    #     elapsed_category.rename('elapsed_category', inplace=True)
    #     test_elapsed_category = df.groupby(['category'])['test_elapsed'].mean()
    #     test_elapsed_category.rename('test_elapsed_category', inplace=True)
    #     elapsed_user_testId = df.groupby(['user', 'testId'])['elapsed'].mean()
    #     elapsed_user_testId.rename('elapsed_user_testId', inplace=True)
    #     test_elapsed_user_testId = df.groupby(['user', 'testId'])['test_elapsed'].mean()
    #     test_elapsed_user_testId.rename('test_elapsed_user_testId', inplace=True)
    #     elapsed_user_category = df.groupby(['user', 'category'])['elapsed'].mean()
    #     elapsed_user_category.rename('elapsed_user_category', inplace=True)
    #     test_elapsed_user_category = df.groupby(['user', 'category'])['test_elapsed'].mean()
    #     test_elapsed_user_category.rename('test_elapsed_user_category', inplace=True)
    #     elapsed_assessmentItemID = df.groupby(['assessmentItemID'])['elapsed'].mean()
    #     elapsed_assessmentItemID.rename('elapsed_assessmentItemID', inplace=True)
    #     test_elapsed_assessmentItemID = df.groupby(['assessmentItemID'])['test_elapsed'].mean()
    #     test_elapsed_assessmentItemID.rename('test_elapsed_assessmentItemID', inplace=True)
    #     elapsed_tag = df.groupby(['tag'])['elapsed'].mean()
    #     elapsed_tag.rename('elapsed_tag', inplace=True)
    #     test_elapsed_tag = df.groupby(['tag'])['test_elapsed'].mean()
    #     test_elapsed_tag.rename('test_elapsed_tag', inplace=True)
    #     elapsed_user_tag = df.groupby(['user', 'tag'])['elapsed'].mean()
    #     elapsed_user_tag.rename('elapsed_user_tag', inplace=True)
    #     test_elapsed_user_tag = df.groupby(['user', 'tag'])['test_elapsed'].mean()
    #     test_elapsed_user_tag.rename('test_elapsed_user_tag', inplace=True)

    #     df = pd.merge(df, correct_t, on=['testId'], how="left")
    #     df = pd.merge(df, correct_k, on=['tag'], how="left")
    #     df = pd.merge(df, elapsed_user, on=['user'], how="left")
    #     df = pd.merge(df, test_elapsed_user, on=['user'], how="left")
    #     df = pd.merge(df, elapsed_testId, on=['testId'], how="left")
    #     df = pd.merge(df, test_elapsed_testId, on=['testId'], how="left")
    #     df = pd.merge(df, elapsed_category, on=['category'], how="left")
    #     df = pd.merge(df, test_elapsed_category, on=['category'], how="left")
    #     df = pd.merge(df, elapsed_user_testId, on=['user', 'testId'], how="left")
    #     df = pd.merge(df, test_elapsed_user_testId, on=['user', 'testId'], how="left")
    #     df = pd.merge(df, elapsed_user_category, on=['user', 'category'], how="left")
    #     df = pd.merge(df, test_elapsed_user_category, on=['user', 'category'], how="left")
    #     df = pd.merge(df, elapsed_assessmentItemID, on=['assessmentItemID'], how="left")
    #     df = pd.merge(df, test_elapsed_assessmentItemID, on=['assessmentItemID'], how="left")
    #     df = pd.merge(df, elapsed_tag, on=['tag'], how="left")
    #     df = pd.merge(df, test_elapsed_tag, on=['tag'], how="left")
    #     df = pd.merge(df, elapsed_user_tag, on=['user', 'tag'], how="left")
    #     df = pd.merge(df, test_elapsed_user_tag, on=['user', 'tag'], how="left")

        return df


    def load_data_from_file(self, file_name, is_train=True, train_df=None):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)     #, nrows=100000)
        if not is_train:
            train_df = pd.read_csv(os.path.join(self.args.data_dir, train_df))
        df = self.__feature_engineering(df, is_train, train_df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir, 'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir, 'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir, 'KnowledgeTag_classes.npy')))

        df = df.sort_values(by=['user', 'Timestamp'], axis=0)
        columns = ['user', 'assessmentItemID', 'testId', 'answer', 'tag']

        #cont_cols = ['user_correct_answer', 'user_total_answer', 'user_acc']#, 'test_mean', 'test_sum', 'tag_mean','tag_sum','elapsed_assessmentItemID','test_elapsed_assessmentItemID','elapsed_testId']
        #cont_cols = [ 'test_mean','assessment_mean', 'tag_mean', 'test_sum','assessment_sum','tag_sum']
        cont_cols = ['test_mean', 'assessment_mean', 'tag_mean']
        columns += cont_cols
        self.args.n_cont = len(cont_cols)
        group = (
            df[columns].groupby('user').apply(
                lambda r: (
                    r['testId'].values,
                    r['assessmentItemID'].values,
                    r['tag'].values,
                    r['answer'].values,
                    r[cont_cols].values
                )
            )
        )
        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name, train_df=None):
        self.test_data = self.load_data_from_file(file_name, is_train= False, train_df=train_df)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args
    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct, conts = row[0], row[1], row[2], row[3], row[4]

        cate_cols = [test, question, tag, correct, conts]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            if len(col.size()) == 1:
                pre_padded = torch.zeros(max_seq_len)
            else:
                pre_padded = torch.zeros(max_seq_len, len(col[0]), dtype=torch.float)
            pre_padded[-len(col):] = col

            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader