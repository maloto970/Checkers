import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model 
from tensorflow.keras.losses import Loss

class DeepCheckersHandler:

    def __init__(self):
        self.model = DeepCheckers()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.batch = []
        self.train_loss = tf.keras.metrics.Mean(
            name='train_loss'
        )
        self.model.compile(self.optimizer,self.loss,self.train_loss)

    def move(self, board, player):
        move = tf.reshape(self.model(tf.convert_to_tensor(board)), (1,))
        return move

    def choose_move(self, move, player):
        self.batch.append([move, player])

    def train_on_batch(self,winner):
        self.batch = tf.transpose([[m,1] if p==winner else [m,0] for [m,p] in self.batch])
        self.model.fit(self.batch[0], self.batch[1], epochs=10)
        
    """ @tf.function
    def train_step(preds, results):
        with tf.GradientTape as tape:
            loss = self.loss(results, preds)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
    """

class ReinforceLoss(Loss):
    def call(self, moves, won):
        moves = tf.convert_to_tensor(moves)
        return tf.reduce_mean([1 if w else -1 for w in won], axis=-1)


class DeepCheckers(Model):
    
    def __init__(self):
        super(DeepCheckers, self).__init__()
        filters = 8
        kernel_size = 2
        channel_size = (8-kernel_size+1)**2
        
        self.conv = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            input_shape=(8,8,1),
            kernel_regularizer='l2'
        )
        self.flatten = Flatten()
        self.d1 = Dense(channel_size*filters, activation='relu',
            kernel_regularizer='l2')
        self.d2 = Dense(2*channel_size*filters, activation='relu',
            kernel_regularizer='l2')
        self.d3 = Dense(32, activation='relu',
            kernel_regularizer='l2')
        self.move = Dense(1, activation='sigmoid')
    
    """
    def scale_move(self,x):
        scaling = tf.tensordot(x, 8.0, axes=0)
        integers = tf.cast(tf.math.round(scaling), dtype=tf.int32)
        return integers
    """

    def call(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.move(x)
    
    



