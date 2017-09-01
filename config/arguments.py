import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("-cf", "--config_file", default="config/default.yml", type=str, help="Configuration Files")
parser.add_argument("-m", "--mode", default="train", type=str, help="train / test", choices=["train", "test"])
parser.add_argument("-d", "--device", default="gpu", type=str, help="CPU / GPU", choices=["cpu", "gpu"])
parser.add_argument("-dd", "--data_dir", default="data/sst2-padding/", type=str, help="Training / Test data dir")
parser.add_argument("-td", "--train_dir", default="save", type=str, help="training base dir")
parser.add_argument("-bd", "--best_dir", default="save_best", type=str, help="best model base dir")
parser.add_argument("-vf", "--vocab_file", default="vocab", type=str, help="file having reverse vocabulary")
parser.add_argument("-w2vf", "--w2v_file", default="w2v.pickle", type=str, help="file having word2vec embeddings")
parser.add_argument("-seed", "--seed", default=1, type=int, help="value of the random seed")
parser.add_argument("-id", "--job_id", default="save_0", type=str, help="Run ID")


def modify_arguments(args):
    args.train_dir = os.path.join(args.train_dir, args.job_id)
    args.best_dir = os.path.join(args.best_dir, args.job_id)
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.exists(args.best_dir):
        os.makedirs(args.best_dir)
