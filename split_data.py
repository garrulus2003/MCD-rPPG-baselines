import argparse
import os
import pandas as pd
import shutil

from sklearn.model_selection import train_test_split

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataframe',
        type=str,
        required=True,
        help="Path to the directory with gt file (.csv)"
    )
    
    parser.add_argument(
        '--output_dataframe_train',
        type=str,
        default=None,
        help="Path to store train df"
    )

    parser.add_argument(
        '--output_dataframe_test',
        type=str,
        default=None,
        help="Path to store test df"
    )

    parser.add_argument(
        '--test_fraction',
        type=float,
        default=0.2,
        help="Fraction of test subset"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=43,
        help="Random seed"
    )
    
    return parser.parse_args(args)

def run(args=None):
    args = parse_args(args)

    # split and store dataframes
    data = pd.read_csv(args.dataframe)
    ps = data['patient_id'].unique()
    train_ps, test_ps = train_test_split(
        ps, 
        test_size=args.test_fraction, 
        random_state=args.seed
    )
    train = data[data['patient_id'].isin(train_ps)]
    test = data[data['patient_id'].isin(test_ps)]

    # get filenames
    if args.output_dataframe_train is None:
        args.output_dataframe_train = args.dataframe[:-4] + '_train.csv'

    if args.output_dataframe_test is None:
        args.output_dataframe_test = args.dataframe[:-4] + '_test.csv'

    train.to_csv(args.output_dataframe_train, index=False)
    test.to_csv(args.output_dataframe_test, index=False)
    
    
if __name__ == '__main__':
    run()