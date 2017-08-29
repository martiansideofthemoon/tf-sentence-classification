import cPickle
import glob
import logging
import os
import sys
import time
import yaml

import tensorflow as tf
import numpy as np

from bunch import bunchify

from config.arguments import modify_arguments, parser
from model import SentimentModel


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    args = parser.parse_args()
    modify_arguments(args)
    tf.set_random_seed(args.seed)
    with open(args.config_file, 'r') as stream:
        args.config = bunchify(yaml.load(stream))
    if args.mode == 'train':
        train(args)
    else:
        test(args)


def load_train_data(args):
    train_files = glob.glob(os.path.join(args.data_dir, "train*.tfrecords"))
    logger.info("%d training file(s) used", len(train_files))
    number_of_instances = 0
    for i, train_file in enumerate(train_files):
        number_of_instances += sum([1 for _ in tf.python_io.tf_record_iterator(train_file)])
        # Using ceil below since we allow for smaller final batch
    num_batches = int(np.ceil(number_of_instances / float(args.config.batch_size)))
    return train_files, num_batches


def load_eval_data(args, split='dev'):
    with open(os.path.join(args.data_dir, split + ".pickle"), 'rb') as f:
        eval_data = cPickle.load(f)
    logger.info("Total # of eval samples - %d", len(eval_data))
    return eval_data


def load_vocab(args):
    vocab_file = os.path.join(args.data_dir, args.vocab_file)
    with open(vocab_file, 'r') as f:
        rev_vocab = f.read().split('\n')
    vocab = {v: i for i, v in enumerate(rev_vocab)}
    return vocab, rev_vocab


def initialize_weights(sess, model, args, mode='train'):
    ckpt = tf.train.get_checkpoint_state(args.train_dir)
    ckpt_best = tf.train.get_checkpoint_state(args.best_dir)
    if mode == 'test' and ckpt_best:
        logger.info("Reading best model parameters from %s", ckpt_best.model_checkpoint_path)
        model.saver.restore(sess, ckpt_best.model_checkpoint_path)
        steps_done = int(ckpt_best.model_checkpoint_path.split('-')[-1])
        # Since local variables are not saved
        sess.run([
            tf.local_variables_initializer()
        ])
    elif mode == 'train' and ckpt:
        logger.info("Reading model parameters from %s", ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
        # Since local variables are not saved
        sess.run([
            tf.local_variables_initializer()
        ])
    else:
        steps_done = 0
        sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ])
    return steps_done


def evaluate(sess, model_dev, data, args):
    batch_size = args.config.batch_size
    num_batches = int(np.ceil(len(data) / batch_size))
    correct = 0
    for i in range(num_batches):
        split = data[i * batch_size:(i + 1) * batch_size]
        seq_len = np.array([x['sentence_len'] for x in split])
        max_seq_len = np.max(seq_len)
        labels = np.array([x['label'] for x in split])
        sents = [np.array(x['sentence']) for x in split]
        sentences = np.array([np.lib.pad(x, (0, max_seq_len - len(x)), 'constant') for x in sents])
        feed_dict = {
            model_dev.inputs.name: sentences,
            model_dev.seq_len.name: seq_len
        }
        outputs = sess.run(model_dev.softmax, feed_dict=feed_dict)
        outputs = np.argmax(outputs, axis=1)
        correct += np.sum(outputs == labels)
    return correct


def test(args):
    if args.device == "gpu":
        cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=2)
    else:
        cfg_proto = None
    with tf.Session(config=cfg_proto) as sess:
        # Loading the vocabulary files
        vocab, rev_vocab = load_vocab(args)
        args.vocab_size = len(rev_vocab)
        # Creating test model
        with tf.variable_scope("model", reuse=None):
            model_test = SentimentModel(args, None, mode='eval')
        # Reload model from checkpoints, if any
        steps_done = initialize_weights(sess, model_test, args, mode='test')
        logger.info("loaded %d completed steps", steps_done)
        test_set = load_eval_data(args, split='test')
        correct = evaluate(sess, model_test, test_set, args)
        percent_correct = float(correct) * 100.0 / len(test_set)
        logger.info("Correct Predictions - %.4f", percent_correct)


def train(args):
    if args.device == "gpu":
        cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=2)
    else:
        cfg_proto = None
    with tf.Session(config=cfg_proto) as sess:
        # Loading the vocabulary files
        vocab, rev_vocab = load_vocab(args)
        args.vocab_size = len(rev_vocab)

        # Loading all the training data
        logger.info("Loading training data from %s", args.data_dir)
        train_files, num_batches = load_train_data(args)
        logger.info("Total # of minibatches - %d", num_batches)
        queue = tf.train.string_input_producer(train_files, shuffle=True)

        # Creating training model
        with tf.variable_scope("model", reuse=None):
            model = SentimentModel(args, queue, mode='train')
        # Reload model from checkpoints, if any
        steps_done = initialize_weights(sess, model, args, mode='train')
        logger.info("loaded %d completed steps", steps_done)

        # Load the w2v embeddings
        # and mode == static / non-static
        if steps_done == 0:
            with open(os.path.join(args.data_dir, args.w2v_file), 'rb') as f:
                w2v = cPickle.load(f)
            w2v_array = np.array([np.array(w2v[x]) for x in rev_vocab])
            feed_dict = {
                model.w2v_embeddings.name: w2v_array
            }
            sess.run(model.load_embeddings, feed_dict=feed_dict)
            logger.info("loaded word2vec values")

        # Reusing weights for evaluation model
        with tf.variable_scope("model", reuse=True):
            model_eval = SentimentModel(args, None, mode='eval')
        dev_set = load_eval_data(args, split='dev')

        # This need not be zero due to incomplete runs
        epoch = model.epoch.eval()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        percent_best = 0.0
        while epoch < args.config.max_epochs:
            epoch_start = time.time()
            if coord.should_stop():
                break
            logger.info("Epochs done - %d", epoch)
            for i in range(1, num_batches + 1):
                output_feed = [
                    model.updates,
                    model.clip,
                    model.losses
                ]
                _, _, losses = sess.run(output_feed)
                if i % 100 == 0:
                    logger.info(
                        "minibatches done %d. Training Loss %.4f. Time elapsed in epoch %.4f.",
                        i, losses, (time.time() - epoch_start) / 3600.0
                    )
                if i % args.config.eval_frequency == 0 or i == num_batches:
                    logger.info("Evaluating model after %d minibatches", i)
                    correct = evaluate(sess, model_eval, dev_set, args)
                    percent_correct = float(correct) * 100.0 / len(dev_set)
                    logger.info("Correct Predictions - %.4f", percent_correct)
                    if percent_correct > percent_best:
                        percent_best = percent_correct
                        logger.info("Saving Best Model")
                        checkpoint_path = os.path.join(args.best_dir, "sentence.ckpt")
                        model.best_saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
                    # Also save the model for continuing in future
                    checkpoint_path = os.path.join(args.train_dir, "sentence.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
            checkpoint_path = os.path.join(args.train_dir, "sentence.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
            # Update epoch counter
            sess.run(model.epoch_incr)
            epoch += 1
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
