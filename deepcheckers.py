from numpy.core.arrayprint import _none_or_positive_arg
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model 
from tensorflow.keras.losses import Loss
from tensorflow.python import training
from tensorflow.python.ops.math_ops import truediv
import tensorflow_addons as tfa
import os
import numpy as np
import random
import datetime

class DeepCheckersHandler:

    def __init__(self):
        self.model = DeepCheckersMove()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.batch_nr = 0
        self.batch_X = []
        self.batch_m = []
        self.batch_t = []
        self.batch_last_five = []
        self.train_loss = tf.keras.metrics.Mean(
            name='train_loss'
        )
        
        #self.model.compile(self.optimizer,self.loss,self.train_loss)

    def move(self, board, move=[], last_five=[], training=False):
        board = tf.convert_to_tensor(board)
        move = tf.cast(tf.convert_to_tensor(move), tf.float32)
        last_five = tf.cast(tf.convert_to_tensor(last_five), tf.float32)
        m = tf.reshape(self.model(board, move, last_five, training), (board.shape[0],))
        return m

    def train_step(self, boards, moves, last_five, targets):
        with tf.GradientTape() as tape:
            ratings = self.move(boards, moves, last_five, True)
            loss = self.loss(ratings, targets)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, ratings

    def append_batch(self, batch_X, batch_m, batch_last_five, batch_t):
        self.batch_X.extend(batch_X)
        self.batch_m.extend(batch_m)
        self.batch_t.extend(batch_t)
        self.batch_last_five.extend(batch_last_five)
    
    def train_on_batch(self):
        training_batch_X = []
        training_batch_m = []
        training_batch_t = []
        training_batch_last_five = []
        for i in range(100):
            index = random.randint(0,len(self.batch_X)-1)
            training_batch_X.append(self.batch_X.pop(index)[0])
            training_batch_m.append(self.batch_m.pop(index))
            training_batch_t.append(self.batch_t.pop(index))
            training_batch_last_five.append(self.batch_last_five.pop(index))
        

        #train_writer = tf.summary.create_file_writer(os.getcwd() + '/logs/train/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        train_writer = tf.summary.create_file_writer(os.getcwd() + '/logs/train')
        stats_writer = tf.summary.create_file_writer(os.getcwd() + '/logs/stats')
        ratings = []
        with train_writer.as_default():
            loss = 0
            for epoch in range(20):
                loss, ratings = self.train_step(training_batch_X, training_batch_m, training_batch_last_five, training_batch_t)
            tf.summary.scalar('Q_mse', loss, self.batch_nr)

        with stats_writer.as_default():
            tf.summary.scalar('prediction_deviation', np.std(ratings), self.batch_nr)
            tf.summary.scalar('prediction_mean', np.mean(ratings), self.batch_nr)
            
            test_boards = training_batch_X[:4]
            ratings_rep = []
            ratings_non_rep = []
            for sample, b in enumerate(test_boards):
                
                piece = None
                for i,r in enumerate(b):
                    b = tf.squeeze(b)
                    for j,c in enumerate(r):
                        if c == -1:
                            piece = (i,j)
                if piece != None:
                    y,x = piece[0], piece[1]
                    move = [y, x, 0, 0]
                    enter_move = [0, 0, 0, 0]
                    if y <= 1:
                        enter_move[0] = y+2
                        enter_move[2] = y+1
                    else:
                        enter_move[0] = y-2
                        enter_move[2] = y-1
                    if x <= 1:
                        enter_move[1] = x
                        enter_move[3] = x+1
                    else:
                        enter_move[1] = x
                        enter_move[3] = x-1
                    move = [enter_move[2], enter_move[3], y, x]
                    move2 = [y,x,enter_move[2],enter_move[3]]
                    last5 = [enter_move,move,move2,move,move2]
                    ratings_rep.append(self.move(tf.reshape([b],(1,8,8,1)), [move], tf.reshape([last5],(1,20))))
                    ratings_non_rep.append(self.move(tf.reshape([b],(1,8,8,1)), [training_batch_m[sample]], [training_batch_last_five[sample]]))

            if ratings_rep == []:
                ratings_rep = [0]
                ratings_non_rep = [0] 
                
            tf.summary.scalar('prediction_repetition', np.mean(ratings_rep), self.batch_nr)
            tf.summary.scalar('prediction_non_repetition_diff', np.mean(ratings_non_rep)-np.mean(ratings_rep), self.batch_nr)



        if self.batch_nr % 20 == 0:
            self.model.save_weights(os.getcwd() + "/modelweights_move_mc.ckpt")

        self.batch_nr += 1

    """ @tf.function
    def train_step(preds, results):
        with tf.GradientTape as tape:
            loss = self.loss(results, preds)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
    """


class DeepCheckersMove(Model):
    def __init__(self):
        super(DeepCheckersMove, self).__init__()
        move_rep_units = 20
        move_rep_past_units = 10
        filters = 4
        kernel_size = 2
        channel_size = (8-kernel_size+1)**2
        
        #Block 1
        self.bn1_a = tf.keras.layers.BatchNormalization()
        self.conv1_a = Conv2D(
            filters=filters*2,
            kernel_size=(1,1),
            input_shape=(8,8),
            kernel_regularizer='l2',
            kernel_initializer=tf.keras.initializers.HeNormal()
        )
        self.bn2_a = tf.keras.layers.BatchNormalization()
        self.conv2_a = Conv2D(
            filters=filters,
            kernel_size=(2,2),
            strides=(2,2),
            kernel_regularizer='l2',
            padding='same',
            kernel_initializer=tf.keras.initializers.HeNormal()
        )
        self.bn3_a = tf.keras.layers.BatchNormalization()
        self.shortcut = Conv2D(
            filters = filters,
            kernel_size=(2,2),
            strides = (2,2),
            kernel_regularizer='l2',
            kernel_initializer=tf.keras.initializers.HeNormal()
        )
        #------------------------------------------------------------
        
        self.flatten = Flatten()
        
        #Block2
        self.bn1_b = tf.keras.layers.BatchNormalization()
        self.move_rep = Dense(move_rep_units,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer='l2')
        #---------------------------------------------------------------

        #Block3
        self.bn1_c = tf.keras.layers.BatchNormalization()
        self.move_rep_past = Dense(move_rep_past_units, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer='l2') 
        #--------------------------------------------------------------

        #Block4
        self.bn1_d = tf.keras.layers.BatchNormalization()
        self.d1 = Dense(channel_size*2*filters+move_rep_units*2, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer='l2')
        
        self.bn2_d = tf.keras.layers.BatchNormalization()
        self.d2 = Dense(channel_size*filters/3, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer='l2')
        
        self.bn3_d = tf.keras.layers.BatchNormalization()
        self.d3 = Dense(32, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer='l2')
        
        self.bn4_d = tf.keras.layers.BatchNormalization()
        self.shortcut2 = Dense(32,
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer='l2')
        #----------------------------------------------------------------

        self.move = Dense(1, activation='linear')
        
        """
        self.model = DeepCheckers(move_rep_units, filters, kernel_size, channel_size)
        if os.path.isfile(os.getcwd() + "/modelweights.ckpt.data-00000-of-00001"):
            self.model.built = True
            self.model.load_weights(os.getcwd() + "/modelweights.ckpt")
            print("Loaded weights for old model")
        """

    def call(self, x, xm, xf, training=False):
        #TODO add param BN training
        x_1 = self.conv1_a(x)
        x_1 = self.bn1_a(x_1, training=training)
        x_1 = tf.nn.relu(x_1)
        x_1 = self.conv2_a(x_1)
        x_1 = self.bn2_a(x_1, training=training)
        x_1 = tf.nn.relu(x_1)
        x = self.shortcut(x)
        x = self.bn3_a(x, training=training)
        x_1 += x
        
        xm_1 = self.move_rep(xm)
        xm_1 = self.bn1_b(xm_1, training=training)
        xm_1 = tf.nn.relu(xm_1)
        #xm_1 += xm

        xf_1 = self.move_rep_past(xf)
        xf_1 = self.bn1_c(xf_1, training=training)
        xf_1 = tf.nn.relu(xf_1)
        #xf_1 += xf

        x_1 = self.flatten(x_1)
        x_1 = tf.concat([x_1, xm_1],1)
        x_1 = tf.concat([x_1,xf_1],1)

        x_2 = self.d1(x_1)
        x_2 = self.bn1_d(x_2, training=training)
        x_2 = tf.nn.relu(x_2)
        x_2 = self.d2(x_2)
        x_2 = self.bn2_d(x_2, training=training)
        x_2 = tf.nn.relu(x_2)
        x_2 = self.d3(x_2)
        x_2 = self.bn3_d(x_2, training=training)
        x_2 = tf.nn.relu(x_2)
        x_1 = self.shortcut2(x_1)
        x_1 = self.bn4_d(x_1)
        x_2 += x_1

        x_3 = self.move(x_2)

        return x_3


class DeepCheckers(Model):
    
    def __init__(self, move_rep_units, filters, kernel_size, channel_size):
        super(DeepCheckers, self).__init__()
        
        
        self.conv = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            input_shape=(8,8,1),
            kernel_regularizer='l2',
            kernel_initializer=tf.keras.initializers.HeNormal()
        )
        self.flatten = Flatten()
        
        self.d1 = Dense(channel_size*filters, activation='relu',
            input_shape=(1,move_rep_units + channel_size*filters),
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer='l2')
        self.d2 = Dense(2*channel_size*filters, activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer='l2')
        self.d3 = Dense(32, activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer='l2')
        self.move = Dense(1, activation='linear')
    
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
    
    



