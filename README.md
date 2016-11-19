# tensorflow_ops

Available Functions:

- Convolution
- Transpose Convolution (Deconvolution)
- Fully Connected Layer
- Leaky RELU

`pip install --user -e .`

or

`sudo pip install -e .`


## Usage

```python
>>> from tensorflow_ops import _conv_layer, _fc_layer
_conv_layer(...)
_fc_layer(...)
```

or

```python
>>> import tensorflow_ops as tfo
tfo._conv_layer(...)
tfo._fc_layer(...)
```


## Functions

#### Convolution
`_conv_layer(inputs, kernel_size, stride, num_channels, name)`

`inputs`: an input tensor (i.e an image)

`kernel_size`: size of the kernel (window size)

`stride`: stride

`num_channels`: number of channels or feature maps

`name`: name for variable scope

Returns: `tf.nn.bias_add(conv, biases)` 



#### Fully Connected
`_fc_layer(inputs, hidden_units, flatten, name)`

`inputs`: an input tensor (i.e the output of a conv layer)

`hiddent units`: the number of hidden units to use (i.e the output shape)

`flatten`: True/False flatten the tensor before reading in (required for images)

`name`: name for variable scope



#### Leaky RELU
`lrelu(x, leak=0.1, name='lrelu')`

`x`: an input tensor to perform LRELU on





