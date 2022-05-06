import json

import pandas as pd

from features import Features

class GBMDataset:
    def __init__(self, data_path, descript=None):
        features = Features()
        self.features = features.FEAT
        self.cat_features = features.CAT_FEAT
        self.num_features = list(set(self.features) - set(self.cat_features))
        self.df = pd.read_csv(data_path)
        
        self.descript = f"{len(self.features)}_features"
        if descript is not None:
            self.descript += f"_{descript}"
    
    def _get_features_from(self, feat_path):
        with open(feat_path, 'r') as f:
            feature_dict = json.load(f)
        return feature_dict
    
    def split_data(self, pseudo_labeling=False):
        self._convert_cat_features_dtype()
        self._convert_num_features_dtype()
        
        train_valid_df = self.get_train_data(pseudo_labeling)
        
        train = train_valid_df[train_valid_df.user == train_valid_df.user.shift(-1)]
        valid = train_valid_df[train_valid_df.user != train_valid_df.user.shift(-1)]
        
        X_train, y_train = train[self.features], train["answer"]
        X_valid, y_valid = valid[self.features], valid["answer"]
        
        return X_train, X_valid, y_train, y_valid
    
    def get_train_data(self, pseudo_labeling=False):
        if pseudo_labeling:
            # self.df[self.df.answer == -1] = correct answer
            pass
        return self.df[self.df.answer != -1]
    
    def get_test_data(self):
        return self.df[self.df.answer == -1][self.features]
    
    def _convert_cat_features_dtype(self):
        for cat_feat in self.cat_features:
            self.df[cat_feat] = self.df[cat_feat].astype("category")
    
    def _convert_num_features_dtype(self):
        for num_feat in self.num_features:
            self.df[num_feat] = self.df[num_feat].astype(float)


if __name__ == "__main__":
    dataset = GBMDataset(data_path="/opt/ml/input/data/feature_engineering/processed_data.csv",
                         descript="test")
    print("GBMDataset made")
    # print(dataset.df.columns)
    X_train, X_valid, y_train, y_valid = dataset.split_data()
    print(f"shape of data: {X_train.shape, X_valid.shape, y_train.shape, y_valid.shape}")
    print(f"test data(X_test) shape: {dataset.get_test_data().shape}")
    assert len(dataset.features) == X_valid.shape[1]
