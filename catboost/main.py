import argparse
import os
import random

import pandas as pd
import numpy as np

from dataset import GBMDataset
from model import CatBoostModel


def seed_everythings(seed):
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    # dataset setting
    print(">>> load dataset...")
    dataset = GBMDataset(args.data_dir, feat_path=args.feat_path)
    X_train, X_valid, y_train, y_valid = dataset.split_data()
    print("<<< done!\n")

    output_dir = os.path.join(args.output_dir_prefix, dataset.descript)
    os.makedirs(output_dir, exist_ok=True)

    # model setting & run
    print(">>>load model with configurations...")
    model = CatBoostModel(args, output_dir=output_dir)
    print("<<< done! now start training\n")
    model.fit(
        X_train, y_train,
        cat_features=dataset.cat_features,
        eval_set=(X_valid, y_valid),
        verbose=args.verbose,
    )
    # save model feature information
    model.save_features(dataset.features, dataset.cat_features)
    if args.save_model:
        model.save_model(output_dir)

    # inference & make submission file
    submission = pd.read_csv(args.inference_dir)
    submission.prediction = model.inference(dataset.get_test_data())
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/feature_engineering/processed_data.csv")
    parser.add_argument("--feat_path", type=str, default="./feature_config.json")
    parser.add_argument("--inference_dir", type=str, default="/opt/ml/input/data/sample_submission.csv")

    parser.add_argument("--output_dir_prefix", type=str, default="/opt/ml/output/catboost")
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--early_stopping", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--save_model", type=bool, default=False)
    
    args = parser.parse_args()
    
    seed_everythings(args.seed)
    main(args)
