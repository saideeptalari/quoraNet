import tensorflow as tf
import numpy as np
import pandas as pd
import os
import spacy

class Doc2VecLoader(object):
    def __init__(self, input="quora_full.tsv",path="data", outputs=("quora.npz",),
                 encode_labels=False):
        path = os.path.abspath(path)
        if not os.path.exists(os.path.join(path, outputs[0])):
            df = pd.read_csv(os.path.join(path, input), delimiter="\t")
            df.dropna(inplace=True)
            df["question1"] = df["question1"].apply(lambda x: unicode(x))
            df["question2"] = df["question2"].apply(lambda x: unicode(x))

            question_1 = df["question1"].values
            question_2 = df["question2"].values
            targets = df["is_duplicate"].values

            nlp = spacy.load("en")

            question_1 = np.array([question.vector for question in nlp.pipe(question_1, n_threads=50)])
            question_2 = np.array([question.vector for question in nlp.pipe(question_2, n_threads=50)])

            np.savez("quora.npz", question_1=question_1, question_2=question_2, targets=targets)

        quora_npz = np.load(os.path.join(path, outputs[0]))
        self.question_1 = quora_npz["question_1"]
        self.question_2 = quora_npz["question_2"]
        self.targets = quora_npz["targets"]

        if encode_labels:
            self.targets = self._encode(self.targets)
        else:
            self.targets = self.targets.reshape(-1,1)


        #Placeholders
        self.Q1 = tf.placeholder(tf.float32, shape=[None, 300])
        self.Q2 = tf.placeholder(tf.float32, shape=[None, 300])
        self.dropout_keep = tf.placeholder(tf.float32)

        if encode_labels:
            self.Y = tf.placeholder(tf.float32, shape=[None,2])
        else:
            self.Y = tf.placeholder(tf.float32, shape=[None,1])

    def _encode(self,y):
        Y = np.zeros((y.shape[0], len(np.unique(y))))
        for i in xrange(y.shape[0]):
            Y[i, y[i]] = 1
        return Y


    def get_original_data(self):
        return (self.question_1, self.question_2, self.targets)

    def get_placeholders(self):
        return (self.Q1, self.Q2, self.Y, self.dropout_keep)



