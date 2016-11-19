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

```>>> from tensorflow_ops import _conv_layer```
_conv_layer(...)
```

or

```>>> import tensorflow_ops as tfo
tfo.conv_layer(...)
```


## Functions

`_conv_layer(inputs, kernel_size, stride, num_channels, name)`

