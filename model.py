import logging
import math
import tensorflow as tf


def random_uniform(limit):
    return tf.random_uniform_initializer(-limit, limit)


class SentimentModel(object):
    def __init__(self, args, queue=None, mode='train'):
        self.logger = logger = logging.getLogger(__name__)
        self.config = config = args.config
        # Epoch variable and its update op
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)
        self.queue = queue
        self.embedding_size = e_size = config.embedding_size
        self.num_classes = num_classes = config.num_classes
        self.batch_size = batch_size = config.batch_size
        self.keep_prob = keep_prob = config.keep_prob
        self.clipped_norm = clipped_norm = config.clipped_norm

        # Learning rate variable and it's update op
        self.learning_rate = tf.get_variable(
            "lr", shape=[], dtype=tf.float32, trainable=False,
            initializer=tf.constant_initializer(config.lr)
        )
        self.global_step = tf.Variable(0, trainable=False)

        if mode == 'train':
            # Using queues for training
            self.inputs, self.labels, self.seq_len, self.segment_id = self.get_queue_batch()
        else:
            # Feeding inputs for evaluation
            self.inputs = tf.placeholder(tf.int64, [self.batch_size, None])
            self.labels = tf.placeholder(tf.int64, [self.batch_size])
            self.seq_len = tf.placeholder(tf.int64, [self.batch_size])
            self.segment_id = tf.placeholder(tf.int64, [self.batch_size])

        # Logic for embeddings
        self.w2v_embeddings = tf.placeholder(tf.float32, [args.vocab_size, e_size])
        if args.config.cnn_mode == 'static':
            embeddings = tf.get_variable(
                "embedding", [args.vocab_size, e_size],
                initializer=random_uniform(0.25),
                trainable=False
            )
        else:
            embeddings = tf.get_variable(
                "embedding", [args.vocab_size, e_size],
                initializer=random_uniform(0.25),
                trainable=True
            )
        # Used in the static / non-static configurations
        self.load_embeddings = embeddings.assign(self.w2v_embeddings)
        # Looking up input embeddings
        input_vectors = tf.nn.embedding_lookup(embeddings, self.inputs)

        # Apply a convolutional layer
        self.input_vectors = input_vectors = tf.expand_dims(input_vectors, axis=3)
        conv_outputs = []
        self.debug = []
        for i, filter_specs in enumerate(config.conv_filters):
            size = filter_specs['size']
            channels = filter_specs['channels']
            debug = {}
            with tf.variable_scope("conv%d" % i):
                # Convolution Layer begins
                debug['filter'] = conv_filter = tf.get_variable(
                    "conv_filter%d" % i, [size, e_size, 1, channels],
                    initializer=random_uniform(0.01)
                )
                debug['bias'] = bias = tf.get_variable(
                    "conv_bias%d" % i, [channels],
                    initializer=tf.zeros_initializer()
                )
                debug['conv_out'] = output = tf.nn.conv2d(input_vectors, conv_filter, [1, 1, 1, 1], "VALID") + bias
                time_size = tf.shape(output)[1]
                # Apply sequence length mask
                modified_seq_lens = tf.nn.relu(self.seq_len - size + 1)
                mask = tf.sequence_mask(modified_seq_lens, maxlen=time_size, dtype=tf.float32)
                debug['mask'] = mask = tf.expand_dims(tf.expand_dims(mask, axis=2), axis=3)
                debug['mask_out'] = output = tf.multiply(output, mask)
                # Applying non-linearity
                output = tf.nn.relu(output)
                # Pooling layer, max over time for each channel
                debug['output'] = output = tf.reduce_max(output, axis=[1, 2])
                conv_outputs.append(output)
                self.debug.append(debug)

        # Concatenate all different filter outputs before fully connected layers
        conv_outputs = tf.concat(conv_outputs, axis=1)
        total_channels = conv_outputs.get_shape()[-1]

        # Adding a dropout layer during training
        if mode == 'train':
            conv_outputs = tf.nn.dropout(conv_outputs, keep_prob=keep_prob)

        # Apply a fully connected layer
        with tf.variable_scope("full_connected"):
            self.W = W = tf.get_variable(
                "fc_weight", [total_channels, num_classes],
                initializer=random_uniform(math.sqrt(6.0 / (total_channels.value + num_classes)))
            )
            self.clipped_W = clipped_W = tf.clip_by_norm(W, clipped_norm)
            self.b = b = tf.get_variable(
                "fc_bias", [num_classes],
                initializer=tf.zeros_initializer()
            )
            if mode == 'train':
                self.logits = tf.matmul(conv_outputs, W) + b
            else:
                self.logits = keep_prob * tf.matmul(conv_outputs, W) + b

        # Declare the loss function
        self.softmax = tf.nn.softmax(self.logits)
        one_hot_labels = tf.one_hot(self.labels, num_classes)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=one_hot_labels
        )
        self.losses = tf.reduce_sum(self.loss) / batch_size

        if config.optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(
                learning_rate=1.0,
                rho=0.95,
                epsilon=1e-6
            )
        else:
            opt = tf.train.AdamOptimizer(self.learning_rate)

        if mode == 'train':
            for variable in tf.trainable_variables():
                logger.info("%s - %s", variable.name, str(variable.get_shape()))
        # Apply optimizer to minimize loss
        self.updates = opt.minimize(self.losses, global_step=self.global_step)

        # Clip fully connected layer's norm
        with tf.control_dependencies([self.updates]):
            self.clip = W.assign(clipped_W)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    def get_queue_batch(self):
        reader = tf.TFRecordReader()
        _, serialized = reader.read(self.queue)
        context_features = {
            "sentence_len": tf.FixedLenFeature([], tf.int64),
            "label": tf.FixedLenFeature([], tf.int64),
            "order_id": tf.FixedLenFeature([], tf.int64)
        }
        sequence_features = {
            "sentence": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
        }
        context, sequence = tf.parse_single_sequence_example(
            serialized=serialized,
            context_features=context_features,
            sequence_features=sequence_features
        )

        inputs = [sequence['sentence'], context['label'], context['sentence_len'], context['order_id']]

        # The code below is used to shuffle the input sequence
        # reference - https://github.com/tensorflow/tensorflow/issues/5147#issuecomment-271086206
        dtypes = list(map(lambda x: x.dtype, inputs))
        shapes = list(map(lambda x: x.get_shape(), inputs))
        self.random_queue = queue = tf.RandomShuffleQueue(2000, 1999, dtypes)
        enqueue_op = queue.enqueue(inputs)
        qr = tf.train.QueueRunner(queue, [enqueue_op])
        tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
        inputs = queue.dequeue()
        for tensor, shape in zip(inputs, shapes):
            tensor.set_shape(shape)

        return tf.train.batch(
            tensors=inputs,
            batch_size=self.batch_size,
            capacity=2000,
            num_threads=1,
            # does the padding to get all lengths the same, to the maximum length of sequence in minibatch
            dynamic_pad=True,
            allow_smaller_final_batch=True
        )
