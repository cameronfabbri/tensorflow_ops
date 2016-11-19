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

`_conv_layer(inputs, kernel_size, stride, num_channels, name)`

`inputs`: an input tensor (i.e an image)

`kernel_size`: size of the kernel (window size)

`stride`: stride

`num_channels`: number of channels or feature maps

`name`: name for variable scope


