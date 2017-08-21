import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-cf", "--config_file", default="config/default.yml", type=str, help="Configuration Files")
parser.add_argument("-m", "--mode", default="train", type=str, help="Train / Eval", choices=["train", "eval"])
parser.add_argument("-d", "--device", default="gpu", type=str, help="CPU / GPU", choices=["cpu", "gpu"])
