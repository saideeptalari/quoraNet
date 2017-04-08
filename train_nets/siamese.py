import tensorflow as tf
from sklearn.utils import shuffle

class SiameseNet(object):

    def __init__(self, dataLoader, model):
        self.dataLoader = dataLoader()
        self.model = model

    def contrastive_loss(self, distances, labels, margin=0.25):
        one = tf.constant(1.0)
        between_class = tf.multiply(one-labels, tf.square(distances))
        max_part = tf.square(tf.maximum(margin - distances, 0))
        within_class = tf.multiply(labels, max_part)

        loss = 0.5*tf.reduce_mean(within_class + between_class)

        return loss

    def euclidean_distance(self,question1, question2):
        return tf.sqrt(tf.reduce_sum(tf.square(question1 - question2), axis=1, keep_dims=True))

    def accuracy(self, distances, labels):
        predictions = tf.sign(distances)
        labels_int = tf.cast(labels, tf.int32)
        predictions_int = tf.cast(predictions, tf.int32)

        correct_predictions = tf.cast(tf.equal(labels_int, predictions_int), tf.float32)
        accuracy = tf.reduce_mean(correct_predictions)
        return accuracy

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
        distances = self.euclidean_distance(question1, question2)
        loss = self.contrastive_loss(distances, Y)
        accuracy = self.accuracy(distances, Y)

        optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

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


