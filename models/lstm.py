import tensorflow as tf
from tensorflow.contrib import rnn

def textLSTM(question, dropout, num_units=5, num_layers=3,**kwargs):
    batch_size = kwargs["batch_size"]
    sequence_length = [question.get_shape()[1].value]*batch_size

    cell = rnn.LSTMCell(num_units= num_units, state_is_tuple=True)
    cell = rnn.MultiRNNCell([cell]*num_layers, state_is_tuple=True)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell, cell, question,sequence_length=sequence_length, dtype=tf.float32)

    output_fw, output_bw = outputs

    encoded = tf.stack([output_fw, output_bw], axis=3)
    encoded = tf.reshape(encoded, [-1, sequence_length[0]*num_units*2])
    encoded = tf.nn.dropout(encoded, dropout)

    return encoded

