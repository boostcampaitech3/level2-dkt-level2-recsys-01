import argparse
import os
import random
from sys import float_info

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
    dataset = GBMDataset(args.data_dir, descript=args.feature_descript)
    X_train, X_valid, y_train, y_valid = dataset.split_data()
    
    print(f"# of features: {len(dataset.features)}")
    args.output_dir = os.path.join(args.output_dir, dataset.descript + "_" + args.exp_descript)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"The result will be saved in {args.output_dir}")
    print("<<< done!\n")

    # model setting & run
    print(">>>load model with configurations...")
    model = CatBoostModel(args, output_dir=args.output_dir)
    print("<<< done! now start training\n")
    model.fit(
        X_train, y_train,
        cat_features=dataset.cat_features,
        eval_set=(X_valid, y_valid),
        verbose=args.verbose,
    )
    # save model feature information
    model.save_features(dataset.features, dataset.cat_features, args.feature_descript)
    if args.save_model:
        model.save_model(args.output_dir)

    # inference & make submission file
    submission = pd.read_csv(args.inference_dir)
    submission.prediction = model.inference(dataset.get_test_data())
    submission.to_csv(os.path.join(args.output_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/feature_engineering/processed_data.csv")
    parser.add_argument("--inference_dir", type=str, default="/opt/ml/input/data/sample_submission.csv")
    parser.add_argument("--feature_descript", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="/opt/ml/output/catboost")
    parser.add_argument("--lr",type=float, default=None)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--early_stopping", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--save_model", type=bool, default=False)
    
    parser.add_argument("--exp_descript", type=str, default=None)
    
    args = parser.parse_args()
    
    seed_everythings(args.seed)
    main(args)
