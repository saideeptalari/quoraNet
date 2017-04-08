import tensorflow as tf

def multiLayerPerceptron(question, dropout,**kwargs):
    # build variables
    w1 = tf.get_variable("weight_1", shape=[300, 128], dtype=tf.float32,
                         initializer=tf.random_normal_initializer(stddev=1.0))
    b1 = tf.get_variable("bias_1", shape=[128], dtype=tf.float32,
                         initializer=tf.constant_initializer(value=1))

    w2 = tf.get_variable("weight_2", shape=[128, 128], dtype=tf.float32,
                         initializer=tf.random_normal_initializer(stddev=1.0))
    b2 = tf.get_variable("bias_2", shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(value=1))

    w3 = tf.get_variable("weight_3", shape=[128, 128], dtype=tf.float32,
                         initializer=tf.random_normal_initializer(stddev=1.0))
    b3 = tf.get_variable("bias_3", shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(value=1))

    z1 = tf.matmul(question, w1) + b1
    h1 = tf.nn.relu(z1)
    h1 = tf.nn.dropout(h1, dropout)

    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    z2 = tf.reduce_sum([h1, h2], axis=0)
    h2 = tf.nn.dropout(z2, dropout)

    h3 = tf.nn.relu(tf.matmul(z2, w3) + b3)

    return tf.concat([h1, h2, h3], axis=1)