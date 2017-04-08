from data_loaders import Doc2VecLoader, Word2VecLoader
from train_nets import SiameseNet, SiameseClassificationNet
from models import multiLayerPerceptron, textCNN, textLSTM

train_net = SiameseClassificationNet(Word2VecLoader, textLSTM)

train_net.train(epochs=20, batch_size=128, save_path="output/model_lstm_clf.ckpt", embeddings=True)

