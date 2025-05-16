from time import time
import numpy as np
import keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from sklearn.cluster import KMeans

def autoencoder(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    x = Input(shape=(dims[0],), name='input')
    h = x
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name=f'encoder_{i}')(h)
    h = Dense(dims[-1], kernel_initializer=init, name=f'encoder_{n_stacks - 1}')(h)
    y = h
    for i in range(n_stacks - 1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name=f'decoder_{i}')(y)
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)
    return Model(inputs=x, outputs=y, name='AE'), Model(inputs=x, outputs=h, name='encoder')

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_clusters

class DEC:
    def __init__(self, dims, n_clusters=10, alpha=1.0, init='glorot_uniform'):
        self.dims = dims
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp'):
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        csv_logger = callbacks.CSVLogger(f'{save_dir}/pretrain_log.csv')
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        self.autoencoder.save_weights(f'{save_dir}/ae_weights.h5')

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, maxiter=20000, batch_size=256, tol=1e-3, update_interval=140):
        print("Fitting KMeans on encoder output...")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        features = self.encoder.predict(x)
        y_pred = kmeans.fit_predict(features)
        return y_pred

def group(data, n_clusters, labels):
    cluster_group = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        cluster_group[label].append(data[i])
    return cluster_group

def main(data, target):
    if len(data.shape) != 2:
        raise ValueError(f"Expected data with shape (samples, features), but got {data.shape}")

    nc = 3  # Number of clusters (can be tuned)
    print(f"Data shape: {data.shape}, running DEC clustering with {nc} clusters...")
    dec = DEC(dims=[data.shape[-1], 500, 500, 2000, 10], n_clusters=nc)
    cluster_labels = dec.fit(data)

    data_list = data.tolist()
    cluster_group = group(data_list, nc, cluster_labels)
    cluster_group_target = group(target.flatten().tolist(), nc, cluster_labels)

    return cluster_group, cluster_group_target
