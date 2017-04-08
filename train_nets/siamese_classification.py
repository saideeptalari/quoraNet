import tensorflow as tf
from sklearn.utils import shuffle

class SiameseClassificationNet(object):

    def __init__(self, dataLoader, model):
        encode_labels = False
        if "classification" in self.__class__.__name__.lower():
            encode_labels = True
        self.dataLoader = dataLoader(encode_labels=encode_labels)
        self.model = model

    def get_batches(self,num_samples, batch_size):
        for i in range(0, num_samples, batch_size):
            yield i, i + batch_size

    def train(self, epochs=100, n_samples=None, batch_size=128,embeddings=False,
              split_percent=0.8, save_every=10, save_path="output/model.ckpt",retrain=False):

        question_1, question_2, targets = self.dataLoader.get_original_data()
        Q1, Q2, Y, dropout_keep = self.dataLoader.get_placeholders()

        ques_1 = Q1
        ques_2 = Q2

        if embeddings:
            W = self.dataLoader.initial_embeddings
            ques_1 = tf.nn.embedding_lookup(W, ques_1)
            ques_2 = tf.nn.embedding_lookup(W, ques_2)

        with tf.variable_scope("siamese") as scope:
            question1 = self.model(ques_1, dropout_keep, batch_size=batch_size)
            scope.reuse_variables()
            question2 = self.model(ques_2, dropout_keep, batch_size=batch_size)
        saver = tf.train.Saver()

        features = tf.concat(
            [question1, question2, tf.subtract(question1, question2), tf.multiply(question1, question2)], 1)
        feature_length = 4 * question1.get_shape().as_list()[1]

        num_hidden1 = 256
        num_hidden2 = 256

        W3 = tf.get_variable(
            "W3",
            shape=[feature_length, num_hidden1],
            initializer=tf.contrib.layers.xavier_initializer())

        b3 = tf.Variable(tf.constant(0.1, shape=[num_hidden1]), name="b3")
        H3 = tf.nn.relu(tf.nn.xw_plus_b(features, W3, b3, name="hidden"))

        W4 = tf.get_variable(
            "W4",
            shape=[num_hidden1, num_hidden2],
            initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.constant(0.1, shape=[num_hidden2]), name="b4")
        H4 = tf.nn.relu(tf.nn.xw_plus_b(H3, W4, b4, name="hidden"))

        W5 = tf.get_variable(
            "W5",
            shape=[num_hidden2, 2],
            initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.constant(0.1, shape=[2]), name="b5")
        l2_loss = tf.constant(0.0)

        l2_loss += tf.nn.l2_loss(W5)
        l2_loss += tf.nn.l2_loss(b5)

        scores = tf.nn.xw_plus_b(H4, W5, b5, name="scores")
        predictions = tf.argmax(scores, 1, name="predictions")

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=Y)) + 0.1 * l2_loss
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        y_truth = tf.argmax(Y, 1, name="y_truth")
        correct_predictions = tf.equal(predictions, y_truth, name="correct_predictions")
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            if retrain:
                saver.restore(sess, save_path=save_path)

            if n_samples is None:
                n_samples = len(targets)

            split = int(split_percent * n_samples)
            n_batches = split / batch_size

            for i in range(epochs):
                question_1, question_2, targets = shuffle(question_1, question_2, targets)

                question_1_train, question_1_test = question_1[:split], question_1[split:n_samples]
                question_2_train, question_2_test = question_2[:split], question_2[split:n_samples]
                targets_train, targets_test = targets[:split], targets[split:n_samples]


                avg_loss = 0
                avg_acc = 0

                for start, end in self.get_batches(split, batch_size):
                    _, cost, acc = sess.run([optimizer, loss, accuracy],
                                            feed_dict={Q1: question_1_train[start:end], Q2: question_2_train[start:end],
                                                       Y: targets_train[start:end], dropout_keep: 0.80})

                    avg_acc += acc * 100
                    avg_loss += cost

                print "Epoch: {}, loss: {}, accuracy: {}".format(i + 1, avg_loss / n_batches, avg_acc / n_batches)

                if i%save_every==0:
                    print "[+] Saving model checkpoint"
                    saver.save(sess, save_path=save_path)

            cost, accu = sess.run([loss, accuracy],
                                  feed_dict={Q1: question_1, Q2: question_2, Y: targets,
                                             dropout_keep: 1.0},)
            print "Total cost: {}, Total Accuracy: {}".format(cost, accu * 100)