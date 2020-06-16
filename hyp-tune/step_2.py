import numpy as np
from step_1 import load_pickle
# from step_1 import save_pickle
import warnings
from step_1 import create_batches
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import tensorflow as tf


class Parameters():
    def __init__(self, V, n_train):
        self.window_size = 5
        self.n_nodes_hl = 50
        self.n_dimensions = 60
        self.alpha = 0.1
        self.batch_size = 30
        self.n_batches = n_train / self.batch_size
        self.in_size = [self.n_batches, self.window_size * self.n_dimensions]  #no of batches = 668921
        self.V = V
        self.out_size = [3, V]
        self.n_epochs = 3


class BengioModel:
    def __init__(self, params):
        self.x = tf.compat.v1.placeholder(tf.float32, shape=params.in_size, name="x")
        self.y = tf.compat.v1.placeholder(tf.float32, shape=params.out_size, name="y")
        self.alpha = tf.compat.v1.placeholder(tf.float32, shape=[1], name="alpha")
        with tf.compat.v1.variable_scope("foo") as scope:
            self.C = tf.compat.v1.get_variable("C", [params.window_size * params.n_dimensions, params.V])
            self.W = tf.compat.v1.get_variable("W", [params.window_size * params.n_dimensions, params.V])
            self.H = tf.compat.v1.get_variable("H", [params.window_size * params.n_dimensions, params.n_nodes_hl])
            self.d = tf.compat.v1.get_variable("d", [1, params.n_nodes_hl])
            self.U = tf.compat.v1.get_variable("U", [params.n_nodes_hl, params.V])
            self.b = tf.compat.v1.get_variable("b", [1, params.V])

    def forward(self, C):
        a1 = tf.multiply(self.U, tf.tanh(tf.add(self.d, (tf.matmul(self.C, self.H)))))
        direct = tf.add(tf.add(tf.matmul(self.W, self.x), self.b), a1)
        yhat = tf.nn.softmax(direct)
        return yhat


def run_model(data, model, running_mode):
    train_data = create_batches(data, params.batch_size)
    # train_data = tf.data.Dataset.from_tensor_slices(train_data)
    yhat = model.forward(train_data)
    cost = tf.math.divide(tf.math.reduce_sum(tf.math.log(yhat), axis=1), params.n_batches)
    optimizer = tf.train.GradientDescentOptimizer(params.alpha)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(train_data[1].shape)
        for epoch in range(params.n_epochs):
            epoch_loss = 0
            for i in range(params.batch_size - params.window_size):
                # x, y = train_data.train.next_batch(params.window_size)
                x, y = train_data[:, i: i+params.window_size], train_data[:, i+params.window_size]
                _, c = sess.run([optimizer, cost], feed_dict={x: x})
                epoch_loss += c
            print("Epoch", epoch, "completed out of", params.n_epochs, "loss : ", epoch_loss)


if __name__ == "__main__":
    vocab_dict = load_pickle("vocab_dict_step4.pkl")
    train_data = load_pickle("train_step4.pkl")
    params = Parameters(len(vocab_dict), len(train_data))
    bengio_model = BengioModel(params)
    run_model(train_data, bengio_model, 'train')