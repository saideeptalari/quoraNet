import tensorflow as tf
import numpy as np
import pandas as pd
import os
import spacy
import cPickle

class Word2VecLoader(object):
    def __init__(self, path="data", input="quora_full.tsv",
                 outputs=("quora_indexes.npz","vocab.npz","embed.npz","vocab_processor.pkl"),
                 encode_labels=False,max_sequence_length=60):

        self.vocabulary_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length)
        path = os.path.abspath(path)
        if not os.path.exists(os.path.join(path, outputs[0])):
            print "[+] Saving data to disk..."
            df = pd.read_csv(os.path.join(path, input), delimiter="\t")
            df.dropna(inplace=True)
            df["question1"] = df["question1"].apply(lambda x: unicode(x))
            df["question2"] = df["question2"].apply(lambda x: unicode(x))

            question_1 = df["question1"].values
            question_2 = df["question2"].values
            targets = df["is_duplicate"].values

            all_questions = np.hstack([question_1, question_2])
            self.vocabulary_processor.fit(all_questions)
            self.vocabulary_ = self.vocabulary_processor.vocabulary_._mapping
            self.vocabulary_ = sorted(self.vocabulary_.items(), key=lambda x: x[1])
            self.vocabulary_ = np.array([word for word,_ in self.vocabulary_])
            question_1 = np.array(list(self.vocabulary_processor.transform(question_1)))
            question_2 = np.array(list(self.vocabulary_processor.transform(question_2)))
            np.savez(os.path.join(path,outputs[0]), question_1=question_1, question_2=question_2, targets=targets)
            np.savez(os.path.join(path,outputs[1]), vocabulary=self.vocabulary_)
            self.vocabulary_processor.save(os.path.join(path, outputs[3]))
            nlp = spacy.load("en")
            initial_embeddings = np.array([word.vector for word in nlp.pipe(self.vocabulary_.tolist(), n_threads=50)])
            print "Embeddings shape", initial_embeddings.shape
            np.savez(os.path.join(path, outputs[2]), embeddings=initial_embeddings)

        quora_npz = np.load(os.path.join(path, outputs[0]))
        vocab_npz = np.load(os.path.join(path, outputs[1]))
        embed_npz = np.load(os.path.join(path, outputs[2]))
        self.vocabulary_processor = cPickle.load(open(os.path.join(path, outputs[3])))
        self.question_1 = quora_npz["question_1"]
        self.question_2 = quora_npz["question_2"]
        self.targets = quora_npz["targets"]

        if encode_labels:
            self.targets = self._encode(self.targets)
        else:
            self.targets = self.targets.reshape(-1,1)

        self.vocabulary_ = vocab_npz["vocabulary"]

        self.initial_embeddings = embed_npz["embeddings"]

        #Placeholders
        self.Q1 = tf.placeholder(tf.int32, shape=[None, max_sequence_length])
        self.Q2 = tf.placeholder(tf.int32, shape=[None, max_sequence_length])
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

