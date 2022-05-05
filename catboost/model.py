import os
import json

import pandas as pd
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier

class CatBoostModel:
    def __init__(self, args, output_dir):
        self.output_dir = output_dir
        self.model = CatBoostClassifier(iterations=args.iteration,
                                        random_seed=args.seed,
                                        custom_metric=["AUC", "Accuracy"],
                                        eval_metric="AUC",
                                        early_stopping_rounds=args.early_stopping,
                                        train_dir=self.output_dir,
                                        
                                        task_type="GPU",
                                        devices="0")
    
    def fit(self, X, y, cat_features, eval_set, verbose=100):
        self.model.fit(X, y, 
                       cat_features=cat_features, 
                       eval_set=eval_set,
                       verbose=verbose)
    
    def save_features(self, features, cat_features):
        # feature names
        with open(os.path.join(self.output_dir, "features.json"), "w") as f:
            feature_dict = {
                "num_feats": len(features),
                "FEAT": features,
                "CAT_FEAT": cat_features
            }
            json.dump(feature_dict, f)
        
        # save feature importance (both text & image)
        importance_df = (
            pd.DataFrame({
                "feature_name": self.model.feature_names_,
                "importances": self.model.feature_importances_})
            .sort_values("importances", ascending=False)
            .reset_index(drop=True))
        
        importance_df.to_csv(os.path.join(self.output_dir, "featrue_importances.csv"))
        importance_df.plot.barh(x="feature_name", y="importances", figsize=(15, 20)).invert_yaxis()
        plt.savefig(os.path.join(self.output_dir, "feature_importances.png"))
    
    def inference(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]