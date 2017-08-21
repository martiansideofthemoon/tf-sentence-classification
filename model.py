import tensorflow as tf


class SentimentModel(object):
    def __init__(self, config, queue):
        # Epoch variable and its update op
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)
        self.queue = queue
        self.embedding_size = e_size = config.embedding_size

        # Learning rate variable and it's update op
        self.learning_rate = tf.get_variable(
            "lr", shape=[], dtype=tf.float32, trainable=False,
            initializer=tf.constant_initializer(config.lr)
        )
        self.global_step = tf.Variable(0, trainable=False)

        self.inputs, self.labels, self.seq_len = self.get_queue_batch()

        # Logic for embeddings
        w2v_embeddings = tf.placeholder(tf.float32, [config.vocab_size, e_size])
        embeddings = tf.get_variable("embedding", [config.vocab_size, e_size])
        # Used in the static / non-static configurations
        self.load_embeddings = embeddings.assign(w2v_embeddings)
        # Looking up input embeddings
        input_vectors = tf.nn.embedding_lookup(embeddings, self.input_data)

        # Apply a convolutional layer
        input_vectors = tf.expand_dims(input_vectors, axis=3)
        for i, (size, channels) in enumerate(config.conv_filters):
            conv_filter = tf.get_variable("conv_filter%d" % i, [size, e_size, 1, channels])
            bias = tf.get_variable("conv_bias%d" % i, [channels])
            tf.nn.conv2d(input_vectors, )

        # Apply a fully connected layer

        if config.optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(
                learning_rate=1.0,
                rho=0.95,
                epsilon=1e-6
            )
        else:
            opt = tf.train.AdamOptimizer(self.learning_rate)

        # Apply optimizer to minimize loss

        # Clip fully connected layer's norm

    def get_queue_batch(self):
        reader = tf.TFRecordReader()
        _, serialized = reader.read(self.queue)
        context_features = {
            "sentence_len": tf.FixedLenFeature([], tf.int64),
            "label": tf.FixedLenFeature([], tf.int64)
        }
        sequence_features = {
            "sentence": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
        }
        context, sequence = tf.parse_single_sequence_example(
            serialized=serialized,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return tf.train.batch(
            tensors=[sequence['sentence'], context['label'], context['sentence_len']],
            batch_size=self.batch_size,
            capacity=2000,
            num_threads=1,
            # does the padding to get all lengths the same, to the maximum length of sequence in minibatch
            dynamic_pad=True,
            allow_smaller_final_batch=True
        )
