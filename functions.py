'''

Cameron Fabbri
Commonly used functions for Tensorflow

'''

import tensorflow as tf

def _variable_with_weight_decay(name, shape, stdev, wd):
   var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
   if wd:
      weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
      weight_decay.set_shape([])
      tf.add_to_collection('losses', weight_decay)
   return var

'''
   Convolutional layer

'''
def _conv_layer(inputs, kernel_size, stride, num_channels, name):
   with tf.variable_scope(name) as scope:
      input_channels = inputs.get_shape()[3]
      weights = _variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, input_channels, num_features, stdev=0.1, wd=0.0005)
      conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)
      return conv_biased

'''
   Transpose Convolutional layer (deconvolution)

'''
def _transpose_conv_layer(inputs, kernel_size, stride, num_features, name):
   with tf.variable_scope(name) as scope:
      input_channels = inputs.get_shape()[3]
    
      weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_features,input_channels], stddev=0.1, wd=FLAGS.weight_decay)
      biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.1))
      batch_size = tf.shape(inputs)[0]
      output_shape = tf.pack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
      conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)
      return conv_biased

'''
   Fully Connected Layer
   Inputs:
      inputs:
      hidden_units:
      flatten:
'''
def _fc_layer(inputs, hidden_units, flatten, name):
   with tf.variable_scope(name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flatten:
         dim = input_shape[1]*input_shape[2]*input_shape[3]
         inputs_processed = tf.reshape(inputs, [-1,dim])
      else:
         dim = input_shape[1]
         inputs_processed = inputs

      weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=0.01, wd=FLAGS.weight_decay)
      biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(0.01))
      
      return tf.add(tf.matmul(inputs_processed,weights), biases, name=name)


'''
   Leaky RELU

'''
def lrelu(x, leak=0.1, name='lrelu'):
   return tf.maximum(x, leak*x)

