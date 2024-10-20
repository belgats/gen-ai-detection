import pandas as pd
import random
import logging
import argparse
from os.path import join, dirname, basename
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import csv

import sys
sys.path.append('.')

from scorer.task import evaluate
from format_checker.task import check_format

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

def run_majority_baseline(data_fpath, test_fpath, results_fpath):
    train_df = pd.read_json(data_fpath, lines=True, dtype=object, encoding="utf-8")
    test_df = pd.read_json(test_fpath, lines=True, dtype=object, encoding="utf-8")

    pipeline = DummyClassifier(strategy="most_frequent")

    text_head = "essay"
    id_head = "id"

    pipeline.fit(train_df[text_head], train_df['label'])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict(test_df[text_head])

        results_file.write("id\tlabel\n")

        for i, line in test_df.iterrows():
            label = predicted_distance[i]

            results_file.write("{}\t{}\n".format(line[id_head], label))


def run_random_baseline(data_fpath, results_fpath):
    gold_df = pd.read_json(data_fpath, lines=True, dtype=object, encoding="utf-8")
    label_list=list(set(gold_df['label'].to_list()))

    id_head = "id"

    with open(results_fpath, "w") as results_file:
        results_file.write("id\tlabel\n")
        for i, line in gold_df.iterrows():
            results_file.write('{}\t{}\n'.format(line[id_head],random.choice(label_list)))


def run_ngram_baseline(train_fpath, test_fpath, results_fpath):
    train_df = pd.read_json(train_fpath, lines=True, dtype=object, encoding="utf-8")
    test_df = pd.read_json(test_fpath, lines=True, dtype=object, encoding="utf-8")

    text_head = "essay"
    id_head = "id"

    pipeline = Pipeline([
        ('ngrams', TfidfVectorizer(ngram_range=(1, 1),lowercase=True,use_idf=True,max_df=0.95, min_df=3,max_features=5000)),
        ('clf', SVC(C=1, gamma='scale', kernel='linear', random_state=0))
    ])
    pipeline.fit(train_df[text_head], train_df['label'])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict(test_df[text_head])
        results_file.write("id\tlabel\n")
        for i, line in test_df.iterrows():
            label = predicted_distance[i]
            results_file.write("{}\t{}\n".format(str(line[id_head]), label))

def run_baselines(train_fpath, test_fpath):
    majority_baseline_fpath = join(ROOT_DIR,
                                 f'data/majority_baseline_{basename(test_fpath)}')
    run_majority_baseline(train_fpath, test_fpath, majority_baseline_fpath)

    if check_format(majority_baseline_fpath):
        acc, precision, recall, f1 = evaluate(majority_baseline_fpath, test_fpath)
        logging.info(f"Majority Baseline F1-macro: {f1}")


    random_baseline_fpath = join(ROOT_DIR, f'data/random_baseline_{basename(test_fpath)}')
    run_random_baseline(test_fpath, random_baseline_fpath)

    if check_format(random_baseline_fpath):
        acc, precision, recall, f1 = evaluate(random_baseline_fpath, test_fpath)
        logging.info(f"Random Baseline F1-macro: {f1}")

    ngram_baseline_fpath = join(ROOT_DIR, f'data/ngram_baseline_{basename(test_fpath)}')
    run_ngram_baseline(train_fpath, test_fpath, ngram_baseline_fpath)
    if check_format(ngram_baseline_fpath):
        acc, precision, recall, f1 = evaluate(ngram_baseline_fpath, test_fpath)
        logging.info(f"Ngram Baseline F1-macro: {f1}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", "-t", required=True, type=str,
                        help="The absolute path to the training data")
    parser.add_argument("--dev-file-path", "-d", required=True, type=str,
                        help="The absolute path to the dev data")

    args = parser.parse_args()
    run_baselines(args.train_file_path, args.dev_file_path)
