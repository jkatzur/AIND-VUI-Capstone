from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    """ TODO: Add batch normalization - see https://arxiv.org/abs/1502.03167 
        Most notably, 'Batch Normalization achieves the same accuracy with 14 times fewer training steps,
        and beats the original model by a significant margin.' This is achieved by normalizing distribution
        of layers between inputs. 
    """
    bn_rnn = BatchNormalization(name='bn_rnn')(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    # this adds a convolutional layer up front to transform the input data prior to the RNN
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add recurrent layer
    rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization(name='bn_rnn')(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    recur_layer_data = input_data
    # TODO: Add recurrent layers, each with batch normalization
    for recur_layer in range(0,recur_layers):
        recur_layer_name = 'recur_layer_' + str(recur_layer)
        deep_rnn = GRU(units, activation='relu', return_sequences=True, 
                 implementation=2, name=recur_layer_name)(recur_layer_data)
        bnn_layer_name = 'bnn_layer_' + str(recur_layer) 
        recur_layer_data = BatchNormalization(name=bnn_layer_name)(deep_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(recur_layer_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, 
                 implementation=2, name=recur_layer_name), merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    recur_layer_data = input_data
    # First, I chose to enable deep rnns because the deep rnn approach performed better than the standard rnn
    for recur_layer in range(0,recur_layers):
        recur_layer_name = 'recur_layer_' + str(recur_layer)
        """
            I chose to make my deep rnns using BiDirectional rnns becase the basic BiDirectional RNN performed
            exceedingly well compared to the simple_rnn model, even without any batch normalization - which was
            crucial in getting the rnn_model to perform well.
        
            I also tweaked my deep rnn from above to add dropout_W and dropout_U to the rnns. Dropout_W drops
            some input data at the start of the model. My intuition is to avoid overfitting between layers
            so I included this at .1 = 10% of input data dropped each run. 
            
            I also include dropout_U. Dropout_U drops values out between recurrent connections. This helps us
            avoid overfitting within a given epoch for each RNN. 
            
            Given RNNs penchant for overfitting I wanted to experiment with both options. The idea to include
            dropouts came from the documentation and the referred paper here: https://arxiv.org/pdf/1512.05287.pdf
        
        """
        deep_bi_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, 
                 implementation=2, name=recur_layer_name, dropout_W=0.1, dropout_U=.2), merge_mode='concat')(recur_layer_data)
        bnn_layer_name = 'bnn_layer_' + str(recur_layer)
        # Batch normalization is included because we saw how effective this would be
        recur_layer_data = BatchNormalization(name=bnn_layer_name)(deep_bi_rnn)
    
    time_dense = TimeDistributed(Dense(output_dim))(recur_layer_data)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: x
    print(model.summary())
    return model