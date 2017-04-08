# quoraNet

### Run

`python train.py`

This will train the neural network with given parameters explained below.

The training consists of various blocks called

* Data Loaders
* Models
* Train Nets

### Data Loaders

These take care of data loading for various nets and also creates valid placeholders, encodes and embeds the data etc. 

#### 1. Doc2Vec 

Creates/uses **GloVe** vectors and transform each document into shape of $(1, 300)$. So it will transform $N$ documents into $(N,300)$

#### 2.Word2Vec

Creates/uses **GloVe** vectors and transform each word into shape of $(1,300)$. So if a document have 10 words then it will transform it into $(1,10,300)$. So concretely if we have $N$ documents of sequence length $S$ then it will transform it into the shape $(N,S,300)$.

### Models

#### 1. Multi layer Perceptron

This implements a simple 3 layer perceptron, and an added dropout. This model takes Doc2Vec as data loader. This finally built to output a vector which will be used further for building the Train Nets.

#### 2.Text CNN

This implements a convolution neural network applied to text, inspired from a paper by Yoon Kim https://arxiv.org/pdf/1408.5882.pdf

This takes Word2Vec as data loader and performs a single convolution operation followed by maxpooling with varying filter sizes and fixed number of filters. Finally it produces a vector which can be flexible to use with any Train Net.

#### 3.Text LSTM

This implements a **Bi-directional Stacked LSTM** which takes in Word2Vec as a data loader and passes it through the LSTM network and finally the outputs are stacked and reshaped to produce a vector which is useful for building an Train Net later.

### Train Nets

These are the actual training nets that takes in the output from Models and implements various training techniques to finally train the end model.

#### 1. Siamese

This takes in the input from the model for both the questions. Now that it has two vectors and it applies a distance function like euclidean and the objective function is set to minimise **contrastive loss** between the distance (d) and the actual label (y) as follows.

$$Loss = \frac{1}{2N} \sum_{n=1}^{N}(y)d^2 + (1-y) max(margin -d,0)^2$$

#### 2. Siamese Classification

This takes an vectors from model for both the questions, let say $y_1$ and $y_2$ are the output vectors from the models. Then it concatenate in a way shown below

$$y = [y_1,y_2,y_1-y_2,y_1*y_2]$$

And it's objective function is set to minimize **cross entropy** loss between the predicted logits and target labels.

### Accuracy

The most performed model is the combination of **Doc2Vec+textLSTM+Siamese Classification**. It gave around **81%** accuracy when trained for $2000$ epochs with **Adam Optimizer** with *learning_rate=0.001*.

