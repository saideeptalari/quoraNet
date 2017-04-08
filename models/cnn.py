import tensorflow as tf


def textCNN(question, dropout, filter_sizes=(2,3,5), num_filters=75,**kwargs):

    question = tf.expand_dims(question, -1)   #None, 25, 300, 1
    sequence_length = question.shape[1]
    embedding_size = question.shape[2]
    outputs = []
    for c,filter_size in enumerate(filter_sizes):
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W1 = tf.get_variable("weight1_{}".format(c),
                             shape=filter_shape,
                             initializer=tf.random_normal_initializer(stddev=0.1))

        b1 = tf.get_variable("bias1_{}".format(c), shape=[num_filters],
                             initializer=tf.constant_initializer(0.1))

        conv_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(question, W1, strides=[1,1,1,1], padding="VALID"), b1))
        pool_1 = tf.nn.max_pool(conv_1, [1, sequence_length-filter_size+1, 1,1], strides=[1,2,2,1],padding="VALID")
        outputs.append(pool_1)

    total_num_filters = len(filter_sizes)*num_filters
    output_concatenated = tf.concat(outputs,3)
    return tf.reshape(output_concatenated, [-1, total_num_filters])