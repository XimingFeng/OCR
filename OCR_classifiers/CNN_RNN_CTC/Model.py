import tensorflow as tf

class Model:
    '''
    A model that has three sequence of data processing
        (1)CNN
            Extract feature map, each CNN layer is followed by a Non-Linear Relu
        (2)RNN
            A Long Short-Term Memory which will produce a sequence
        (3)CTC
            Computer loss according to the output of the
    '''
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
        pass

    def cnn_layers(self, X, kernel_sizes, filter_nums, pool_sizes):
        '''
        set up CNN layers according to the kernel size, filter numbers and pooling sizes
        :param X: Input data
        :param kernel_sizes: The list of sizes of kernel in each layer.
                            Here the row and column sizes are the same
        :param filter_nums: The list of number of filters in each layer
                            This is also the number of features
        :param pool_sizes: The size of each pooling layer.
                            The same as kernel of conv layer, the row and column number are the same
        :return: a number of cnn layers
        '''
        layer_input = tf.placeholder(tf.float32)
        layer_num = len(kernel_sizes)
        for i in range(layer_num):
            conv_layer = tf.layers.conv2d(
                inputs=layer_input,
                filters=filter_nums[i],
                kernel_size=[kernel_sizes[i], kernel_sizes[i]],
                padding="same",
                activation=tf.nn.relu
            )
            pool_layer = tf.layers.max_pooling2d(inputs=conv_layer,
                                           pool_size=[pool_sizes[i], pool_sizes[i]],
                                                 strides=2)
        cnn_model = pool_layer
        return cnn_model

    def rnn_layers(self, X):
        tf.contrib.rnn.LSTMCell(num_units=)
        pass



    def train(self):
        pass

    def loss(self):
        pass