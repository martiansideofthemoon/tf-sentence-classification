import os
import yaml

import tensorflow as tf
import numpy as np

from config.arguments import parser


def main():
    args = parser.parse_args()
    with open(args.config_file, 'r') as stream:
        args.config = yaml.load(stream)
    train(args)


def train(args):



if __name__ == '__main__':
    main()
