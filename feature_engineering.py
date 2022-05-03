import os
import json
import time
from typing import Final

import pandas as pd


USER: Final = "user"
TIMESTAMP: Final = "Timestamp"

ANSWER: Final = "answer"
CATEGORY: Final = "category"
TAG: Final = "tag"
TEST_ID: Final = "testId"
TEST_NUMBER: Final = "test"
T_ELAPSED: Final = "test_elapsed"
ITEM_ID = "assessmentItemID"
ITEM_NUMBER: Final = "item"


class FeatureEngineer:
    def __init__(self, df, descript=None, sep="_"):
        '''
        df : pandas.DataFrame
        descript : short description about feature engineering
        '''
        self.org_df = df.sort_values(by=[USER, TIMESTAMP])
        self.user_features = list()
        self.item_features = list()
        self.features = list()
        self.descript = descript
        self.sep = sep
        
        self.final_df = None
    
    def get_user_features(self):
        return self.user_features
    
    def get_item_features(self):
        return self.item_features
    
    def _get_cumulative_value_by_group(self, grouping_targets, feature, df=None):
        if df is None:
            df = self.org_df
        group = df.groupby(grouping_targets)[feature]
        cumsum = group.transform(lambda x : x.cumsum().shift(1))
        cumcount = group.cumcount()
        cummean = cumsum / cumcount
        return cumsum, cumcount, cummean
    
    def _get_agg_value_by_group(self, grouping_targets, features, aggs, df=None):
        if len(features) != len(aggs):
            raise ValueError("length of 'features' and 'aggs' should be same!")
            
        if df is None:
            df = self.org_df
            
        group_df = (
            df.groupby(grouping_targets)
            .agg({feat: func 
                for feat, func in zip(features, aggs)})
        )
        
        return group_df
    
    def _get_column_names(self, grouping_targets, features, aggs):
        column_names = list()
        prefix = self.sep.join(grouping_targets)
        
        for feat, func in zip(features, aggs):
            if isinstance(func, list):
                column_names.extend(
                    [self.sep.join([prefix, feat, f]) for f in func]
                )
            else:
                column_names.append(self.sep.join([prefix, feat, func]))
                
        return column_names
    
    def _get_last_prob_feature(self):
        feat_df = self.org_df.groupby([TEST_ID]).agg({ITEM_NUMBER: "max"})
        feat_df.rename(columns={ITEM_NUMBER: "last_prob_no"}, inplace=True)
        
        feat_df = pd.merge(self.org_df, feat_df, on=TEST_ID, how="left")
        
        return feat_df[ITEM_NUMBER] / feat_df["last_prob_no"]
    
    def _explore_user_features(self):
        '''
        사용자에 대한 feature engineering을 하는 메소드.
        사용자의 feature는 시간이 지날 수록 업데이트 되는 경향이 있다. (ex. 공부 양과 시간에 따른 사용자의 역량 변화)
        따라서 '정답률'과 '학습시간'의 변화를 feature로 주었다.
        '''
        user_df = self.org_df.copy()
        
        grouping_targets = [[USER], [USER, CATEGORY], [USER, TAG], [USER, TEST_ID]]
        column_prefixes = [self.sep.join(targets) for targets in grouping_targets]
    
        # 시간에 따른 정답률 feature
        for prefix, target in zip(column_prefixes, grouping_targets):
            cum_feats = self._get_cumulative_value_by_group(target, ANSWER)
            user_df[prefix + "_correct_answer"] = cum_feats[0]
            user_df[prefix + "_total_answer"] = cum_feats[1]
            user_df[prefix + "_acc"] = cum_feats[2]
        
        ## 시간에 따른 학습시간 feature (test_elapsed cumsum, mean)
        for prefix, target in zip(column_prefixes, grouping_targets[1:]):
            cum_feats = self._get_cumulative_value_by_group(target, T_ELAPSED)
            user_df[prefix + "_cum_telapsed"] = cum_feats[0]
            user_df[prefix + "_mean_telapsed"] = cum_feats[2]
        
        self.user_features = list(set(user_df.columns.values) - set(self.org_df.columns.values))
        return user_df[self.user_features]
    

    def _explore_item_features(self):
        '''
        item 즉, 문제에 대한 feature engineering을 하는 메소드.
        '''
        item_df = self.org_df.copy()
        
        # item feature 별 평균 정답률과 평균 풀이에 걸린 시간
        grouping_targets = [[CATEGORY], [TAG], [TEST_ID], [ITEM_ID]]
        features = [ANSWER, T_ELAPSED]
        aggs = [["mean", "sum"], "mean"]
        
        for targets in grouping_targets:
            group_df = self._get_agg_value_by_group(grouping_targets=targets,
                                                    features=features,
                                                    aggs=aggs)
            column_names = self._get_column_names(targets, features, aggs)
            group_df.columns = column_names
            item_df = pd.merge(item_df, group_df, on=targets, how="left")
        
        # last_prob feature (해당 문제가 마지막 문항 번호인지 )
        # == 해당 문제 번호 / 해당 문제가 속한 시험의 가장 마지막 번호
        item_df["last_prob"] = self._get_last_prob_feature()
        
        self.item_features = list(set(item_df.columns.values) - set(self.org_df.columns.values))
        return item_df[self.item_features]
    
    def run_feature_engineering(self, save=False):
        print("user fe started...")
        start = time.time()
        user_feature_df = self._explore_user_features()
        print(f"user fe done!\n\ttaken: {time.time() - start}")
        
        print("item fe started...")
        start = time.time()
        item_feature_df = self._explore_item_features()
        print(f"item fe done!\n\ttaken: {time.time() - start}")
        
        self.final_df = pd.concat([self.org_df, user_feature_df, item_feature_df], axis=1)
        if save:
            print("saving preprocessed_data...")
            self.save()
        return self.final_df
    
    def save(self, save_dir="/opt/ml/input/data/feature_engineering/"):
        os.makedirs(save_dir, exist_ok=True)
        
        if self.final_df is None:
            raise Exception("You should call '.run_feature_enginnering()' function first!")
            
        feature_config = {
            "description": self.descript,
            "user_features": self.user_features,
            "item_features": self.item_features
        }
        
        with open(os.path.join(save_dir, "features.json"), "w", encoding="utf-8") as f:
            json.dump(feature_config, f)
            
        self.final_df.to_csv(os.path.join(save_dir, "processed_data.csv"), index=False)
    

if __name__ == '__main__':
    start = time.time()
    org_df = pd.read_csv("/opt/ml/input/data/preprocessed_data.csv")
    fe = FeatureEngineer(org_df, descript="test")
    final_df = fe.run_feature_engineering()
    fe.save()
    print(f"taken time: {time.time() - start}")