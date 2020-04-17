import numpy as np
from math import *

'''
    Linear

    Implementation of the linear layer (also called fully connected layer),
    which performs linear transformation on input data: y = xW + b.

    This layer has two learnable parameters:
        weight of shape (input_channel, output_channel)
        bias   of shape (output_channel)
    which are specified and initalized in the init_param() function.

    In this assignment, you need to implement both forward and backward
    computation.

    Arguments:
        input_channel  -- integer, number of input channels
        output_channel -- integer, number of output channels
'''
class Linear(object):

    def __init__(self, input_channel, output_channel):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.init_param()
        self.cache = None

    def init_param(self):
        self.weight = (np.random.randn(self.input_channel,self.output_channel) * sqrt(2.0/(self.input_channel+self.output_channel))).astype(np.float32)
        self.bias = np.zeros((self.output_channel))

    '''
        Forward computation of linear layer. (3 points)

        Note:  You may want to save some intermediate variables to class
        membership (self.) for reuse in backward computation.

        Arguments:
            input  -- numpy array of shape (N, input_channel)

        Output:
            output -- numpy array of shape (N, output_channel)
    '''
    def forward(self, input):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        output = np.dot(input, self.weight) + self.bias
        self.cache = (input, self.weight, self.bias)

        return output

    '''
        Backward computation of linear layer. (3 points)

        You need to compute the gradient w.r.t input, weight, and bias.
        You need to reuse variables from forward computation to compute the
        backward gradient.

        Arguments:
            grad_output -- numpy array of shape (N, output_channel)

        Output:
            grad_input  -- numpy array of shape (N, input_channel), gradient w.r.t input
            grad_weight -- numpy array of shape (input_channel, output_channel), gradient w.r.t weight
            grad_bias   -- numpy array of shape (output_channel), gradient w.r.t bias
    '''
    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        x, w, b = self.cache

        grad_input = grad_output.dot(w.T) # gradient output * switched input (w). grad_input.shape = x.shape
        grad_weight = x.T.dot(grad_output) # switched input (x) * gradient output. grad_weight.shape = w.shape
        grad_bias = grad_output.sum(axis=0) # local gradient for add operation is +1. grad_bias.shape = b.shape

        return grad_input, grad_weight, grad_bias

'''
    BatchNorm1d

    Implementation of batch normalization (or BN) layer, which performs
    normalization and rescaling on input data.  Specifically, for input data X
    of shape (N,input_channel), BN layers first normalized the data along batch
    dimension by the mean E(x), variance Var(X) that are computed within batch
    data and both have shape of (input_channel).  Then BN re-scales the
    normalized data with learnable parameters beta and gamma, both having shape
    of (input_channel).
    So the forward formula is written as:

        Y = ((X - mean(X)) /  sqrt(Var(x) + eps)) * gamma + beta

    At the same time, BN layer maintains a running_mean and running_variance
    that are updated (with momentum) during forward iteration and would replace
    batch-wise E(x) and Var(x) for testing. The equations are:

        running_mean = (1 - momentum) * E(x)   +  momentum * running_mean
        running_var =  (1 - momentum) * Var(x) +  momentum * running_var

    During test time, since the batch size could be arbitrary, the statistics
    for a batch may not be a good approximation of the data distribution.
    Thus, we instead use running_mean and running_var to perform normalization.
    The forward formular is modified to:

        Y = ((X - running_mean) /  sqrt(running_var + eps)) * gamma + beta

    Overall, BN maintains 4 learnable parameters with shape of (input_channel),
    running_mean, running_var, beta, and gamma.  In this assignment, you need
    to complete the forward and backward computation and handle the cases for
    both training and testing.

    Arguments:
        input_channel -- integer, number of input channel
        momentum      -- float,   the momentum value used for the running_mean and running_var computation
'''
class BatchNorm1d(object):

    def __init__(self, input_channel, momentum = 0.9):
        self.input_channel = input_channel
        self.momentum = momentum
        self.eps = 1e-3
        self.init_param()
        self.cache = None

    def init_param(self):
        self.r_mean = np.zeros((self.input_channel)).astype(np.float32)
        self.r_var = np.ones((self.input_channel)).astype(np.float32)
        self.beta = np.zeros((self.input_channel)).astype(np.float32)
        self.gamma = (np.random.rand(self.input_channel) * sqrt(2.0/(self.input_channel))).astype(np.float32)

    '''
        Forward computation of batch normalization layer and update of running
        mean and running variance. (3 points)

        You may want to save some intermediate variables to class membership
        (self.) and you should take care of different behaviors during training
        and testing.

        Arguments:
            input -- numpy array (N, input_channel)
            train -- bool, boolean indicator to specify the running mode, True for training and False for testing
    '''
    def forward(self, input, train):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        if train:
        	EX = np.mean(input, axis=0)
        	VarX = np.var(input, axis=0)
        	std = np.sqrt(VarX + self.eps)
        	X_EX = input - EX
        	Z = X_EX / std

        	output = Z * self.gamma + self.beta
        	self.r_mean = (1.0 - self.momentum) * EX + self.momentum * self.r_mean
        	self.r_var = (1.0 - self.momentum) * VarX + self.momentum * self.r_var

        	self.cache = (X_EX, std, Z, self.gamma)
        else: 
        	output = ((input - self.r_mean) / np.sqrt(self.r_var + self.eps)) * self.gamma + self.beta
        return output

    '''
        Backward computationg of batch normalization layer. (3 points)
        You need to compute gradient w.r.t input data, gamma, and beta.

        It is recommend to follow the chain rule to first compute the gradient
        w.r.t to intermediate variables, in order to simplify the computation.

        Arguments:
            grad_output -- numpy array of shape (N, input_channel)

        Output:
            grad_input -- numpy array of shape (N, input_channel), gradient w.r.t input
            grad_gamma -- numpy array of shape (input_channel), gradient w.r.t gamma
            grad_beta  -- numpy array of shape (input_channel), gradient w.r.t beta
    '''
    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        X_EX, std, Z, gamma = self.cache

        N, C = grad_output.shape
        grad_beta = grad_output.sum(axis=0)
        grad_gamma = np.sum(grad_output * Z, axis=0)
        dz = grad_output * gamma
        dinvVar = np.sum(dz * X_EX, axis=0)
        dXE1 = dz / std
        dstd = -1.0 / (std ** 2) * dinvVar
        dVar = 0.5 * 1. / std * dstd
        dsqXE = 1.0 / N * np.ones((N, C)) * dVar
        dXE2 = 2.0 * X_EX * dsqXE
        dx1 = dXE1 + dXE2
        dE = -1.0 * np.sum(dXE1 + dXE2, axis=0)
        dx2 = 1.0 / N * np.ones((N, C)) * dE
        grad_input = dx1 + dx2

        return grad_input, grad_gamma, grad_beta

'''
    ReLU

    Implementation of ReLU (rectified linear unit) layer.  ReLU is the
    non-linear activation function that sets all negative values to zero.
    The formua is: y = max(x,0).

    This layer has no learnable parameters and you need to implement both
    forward and backward computation.

    Arguments:
        None
'''
class ReLU(object):
    def __init__(self):
        self.cache = None

    '''
        Forward computation of ReLU. (3 points)

        You may want to save some intermediate variables to class membership
        (self.)

        Arguments:
            input  -- numpy array of arbitrary shape

        Output:
            output -- numpy array having the same shape as input.
    '''
    def forward(self, input):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        output = np.maximum(input, 0)
        self.cache = input
        return output

    '''
        Backward computation of ReLU. (3 points)

        You can either modify grad_output in-place or create a copy.

        Arguments:
            grad_output -- numpy array having the same shape as input

        Output:
            grad_input  -- numpy array has the same shape as grad_output. gradient w.r.t input
    '''
    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        x = self.cache
        grad_input = grad_output * (x > 0)

        return grad_input

'''
    CrossEntropyLossWithSoftmax

    Implementation of the combination of softmax function and cross entropy
    loss.  In classification tasks, we usually first apply the softmax function
    to map class-wise prediciton scores into a probability distribution over
    classes.  Then we use cross entropy loss to maximise the likelihood of
    the ground truth class's prediction.  Since softmax includes an exponential
    term and cross entropy includes a log term, we can simplify the formula by
    combining these two functions together, so that log and exp operations
    cancel out.  This way, we also avoid some precision loss due to floating
    point numerical computation.

    If we ignore the index on batch size and assume there is only one grouth
    truth per sample, the formula for softmax and cross entropy loss are:

        Softmax: prob[i] = exp(x[i]) / \sum_{j}exp(x[j])
        Cross_entropy_loss:  - 1 * log(prob[gt_class])

    Combining these two functions togther, we have:

        cross_entropy_with_softmax: -x[gt_class] + log(\sum_{j}exp(x[j]))

    In this assignment, you will implement both forward and backward
    computation.

    Arguments:
        None
'''
class CrossEntropyLossWithSoftmax(object):
    def __init__(self):
        self.cache = None

    '''
        Forward computation of cross entropy with softmax. (3 points)

        Tou may want to save some intermediate variables to class membership
        (self.)

        Arguments:
            input    -- numpy array of shape (N, C), the prediction for each class, where C is number of classes
            gt_label -- numpy array of shape (N), it is an integer array and the value range from 0 to C-1 which
                        specify the ground truth class for each input
        Output:
            output   -- numpy array of shape (N), containing the cross entropy loss on each input
    '''
    def forward(self, input, gt_label):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        N = input.shape[0]

        # Modify input to stablize the computation
        score = input - np.max(input, axis=1, keepdims=True)

        # -x[gt_class] + log(\sum_{j}exp(x[j]))
        output = -score[np.arange(N), gt_label] + np.log(np.exp(score).sum(axis=1))

        self.cache = (N, gt_label, score)

        return output

    '''
        Backward computation of cross entropy with softmax. (3 points)

        It is recommended to resue the variable(s) in forward computation
        in order to simplify the formula.

        Arguments:
            grad_output -- numpy array of shape (N)

        Output:
            output   -- numpy array of shape (N, C), the gradient w.r.t input of forward function
    '''
    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        N, gt_label, score = self.cache

        grad_input = np.exp(score) / np.exp(score).sum(axis=1, keepdims=True) # softmax probability
        grad_input[np.arange(N), gt_label] -= 1

        return grad_input

'''
    im2col (3 points)

    Consider 4 dimensional input tensor with shape (N, C, H, W), where:
        N is the batch dimension,
        C is the channel dimension, and
        H, W are the spatial dimensions.

    The im2col functions flattens each slidding kernel-sized block
    (C * kernel_h * kernel_w) on each sptial location, so that the output has
    the shape of (N, (C * kernel_h * kernel_w), out_H, out_W) and we can thus
    formuate the convolutional operation as matrix multiplication.

    The formula to compute out_H and out_W is the same as to compute the output
    spatial size of a convolutional layer.

    Arguments:
        input_data  -- numpy array of shape (N, C, H, W)
        kernel_h    -- integer, height of the sliding blocks
        kernel_w    -- integer, width of the sliding blocks
        stride      -- integer, stride of the sliding block in the spatial dimension
        padding     -- integer, zero padding on both size of inputs

    Returns:
        output_data -- numpy array of shape (N, (C * kernel_h * kernel_w), out_H, out_W)
'''
def pad_border_im2col(image, padding):
    N, C, H, W = image.shape
    img = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
    img[:, :, padding:(H+padding), padding:(W+padding)] = image
    return img


def im2col(input_data, kernel_h, kernel_w, stride, padding):
    ########################
    # TODO: YOUR CODE HERE #
    ########################
    N, C, H, W = input_data.shape

    kh, kw, s, p = kernel_h, kernel_w, stride, padding

    input_padded = pad_border_im2col(input_data, p)
    out_C = C * kh * kw
    out_H = (H + 2 * p - kh) // s + 1
    out_W = (W + 2 * p - kw) // s + 1
    output_data = np.zeros((N, out_C, out_H, out_W)).astype(np.float32)

    for oh in range(out_H):
        for ow in range(out_W):
            output_data[:, :, oh, ow] = input_padded[:, :, oh*s:oh*s+kh, ow*s:ow*s+kw].reshape(N, out_C)

    return output_data


'''
    col2im (3 points)

    Consider a 4 dimensional input tensor with shape:
        (N, (C * kernel_h * kernel_w), out_H, out_W)
    where:
        N is the batch dimension,
        C is the channel dimension,
        out_H, out_W are the spatial dimensions, and
        kernel_h and kernel_w are the specified kernel spatial dimension.

    The col2im function calculates each combined value in the resulting array
    by summing all values from the corresponding sliding kernel-sized block.
    With the same parameters, the output should have the same shape as
    input_data of im2col.  This function serves as an inverse subroutine of
    im2col, so that we can formuate the backward computation in convolutional
    layers as matrix multiplication.

    Arguments:
        input_data  -- numpy array of shape (N, (C * kernel_H * kernel_W), out_H, out_W)
        kernel_h    -- integer, height of the sliding blocks
        kernel_w    -- integer, width of the sliding blocks
        stride      -- integer, stride of the sliding block in the spatial dimension
        padding     -- integer, zero padding on both size of inputs

    Returns:
        output_data -- output_array with shape (N, C, H, W)
'''
def trim_border_col2im(image, padding):
    _, _, H, W = image.shape
    img = np.copy(image[:, :, padding:(H-padding), padding:(W-padding)])
    return img

def col2im(input_data, kernel_h, kernel_w, stride=1, padding=0, mod_vals=(0,0)):
    ########################
    # TODO: YOUR CODE HERE #
    ########################

    # 'mod_vals' is added as an additional input parameter
    # the parameter 'mod_vals' is a tuple that holds (mod_H, mod_W)
    # where  mod_H = (H + 2 * padding - kernal_h) % s
    # and    mod_W = (W + 2 * padding - kernal_w) % s

    N, out_C, out_H, out_W = input_data.shape
    kh, kw, s, p = kernel_h, kernel_w, stride, padding
    C = out_C // (kh * kw)
    H = (out_H-1)*s + kh - 2*padding + mod_vals[0]
    W = (out_W-1)*s + kw - 2*padding + mod_vals[1]

    output_data = np.zeros((N, C, H+2*p, W+2*p)).astype(np.float32)

    for oh in range(out_H):
        for ow in range(out_W):
            output_data[:, :, oh*s:oh*s+kh, ow*s:ow*s+kw] += input_data[:, :, oh, ow].reshape(N, C, kh, kw)

    if p != 0:
        output_data = trim_border_col2im(output_data, p)

    return output_data



'''
    Conv2d

    Implementation of convolutional layer.  This layer performs convolution
    between each sliding kernel-sized block and convolutional kernel.  Unlike
    the convolution you implemented in HW1, where you needed flip the kernel,
    here the convolution operation can be simplified as cross-correlation (no
    need to flip the kernel).

    This layer has 2 learnable parameters, weight (convolutional kernel) and
    bias, which are specified and initalized in the init_param() function.
    You need to complete both forward and backward functions of the class.
    For backward, you need to compute the gradient w.r.t input, weight, and
    bias.  The input arguments: kernel_size, padding, and stride jointly
    determine the output shape by the following formula:

        out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1

    You need to use im2col, col2im inside forward and backward respectively,
    which formulates the sliding window computation in a convolutional layer as
    matrix multiplication.

    Arguments:
        input_channel  -- integer, number of input channel which should be the same as channel numbers of filter or input array
        output_channel -- integer, number of output channel produced by convolution or the number of filters
        kernel_size    -- integer or tuple, spatial size of convolution kernel. If it's tuple, it specifies the height and
                          width of kernel size.
        padding        -- zero padding added on both sides of input array
        stride         -- integer, stride of convolution.
'''
class Conv2d(object):
    def __init__(self, input_channel, output_channel, kernel_size, padding = 0, stride = 1):
        self.output_channel = output_channel
        self.input_channel = input_channel
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride
        self.init_param()
        self.cache = None

    def init_param(self):
        self.weight = (np.random.randn(self.output_channel, self.input_channel, self.kernel_h, self.kernel_w) * sqrt(2.0/(self.input_channel + self.output_channel))).astype(np.float32)
        self.bias = np.zeros(self.output_channel).astype(np.float32)

    '''
        Forward computation of convolutional layer. (3 points)

        You should use im2col in this function.  You may want to save some
        intermediate variables to class membership (self.)

        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, output_chanel, out_H, out_W)
    '''
    def forward(self, input):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        N, C, H, W = input.shape
        kh, kw, s, p, w, b = self.kernel_h, self.kernel_w, self.stride, self.padding, self.weight, self.bias

        out_C = self.output_channel
        out_H = (H + 2 * p - kh) // s + 1
        out_W = (W + 2 * p - kw) // s + 1
        mod_H = (H + 2 * p - kh) % s
        mod_W = (W + 2 * p - kw) % s

        cols = im2col(input, kh, kw, s, p)
        c_flat = cols.reshape(N, C*kh*kw, out_H*out_W) # flattened for vectorized operation
        w_tiled = np.tile(w.reshape(out_C, C*kh*kw), (N, 1, 1)) # flattened and tiled
        b_tiled = np.tile(b, (N, 1))

        output = (w_tiled.dot(c_flat).reshape(N, out_C, out_H*out_W) + b.reshape((N, out_C, 1))).reshape(N, out_C, out_H, out_W)
        self.cache = (c_flat, (mod_H, mod_W))

        return output

    '''
        Backward computation of convolutional layer. (3 points)

        You need col2im and saved variables from forward() in this function.

        Arguments:
            grad_output -- numpy array of shape (N, output_channel, out_H, out_W)

        Ouput:
            grad_input  -- numpy array of shape(N, input_channel, H, W), gradient w.r.t input
            grad_weight -- numpy array of shape(output_channel, input_channel, kernel_h, kernel_w), gradient w.r.t weight
            grad_bias   -- numpy array of shape(output_channel), gradient w.r.t bias
    '''

    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        cols_flat, mod_vals = self.cache
        N, out_C, out_H, out_W = grad_output.shape
        C, kh, kw = self.input_channel, self.kernel_h, self.kernel_w
        s, p, w, b = self.stride, self.padding, self.weight, self.bias

        # grad_weight
        grad_output_flat = grad_output.reshape(N, out_C, out_H*out_W)
        cols_flat_T = cols_flat.transpose(0, 2, 1)
        grad_weight = grad_output_flat.dot(cols_flat_T).reshape(w.shape)

        # grad_bias
        grad_bias = np.sum(grad_output, axis=(0, 2, 3))

        # grad_input
        w_tiled = np.tile(w.reshape(out_C, C*kh*kw), (N, 1, 1)).transpose(0, 2, 1) # flattened, tiled, transposed
        grad_input_cols = w_tiled.dot(grad_output_flat).reshape(N, C*kh*kw, out_H, out_W)
        grad_input = col2im(grad_input_cols, kh, kw, s, p, mod_vals=mod_vals)

        return grad_input, grad_weight, grad_bias

'''
    MaxPool2d

    Implementation of max pooling layer.  For each sliding kernel-sized block,
    maxpool2d computes the spatial maximum along each channels.  This layer has
    no learnable parameters.

    You need to complete both forward and backward functions of the layer.
    For backward, you need to compute the gradient w.r.t input.  Similar as
    conv2d, the input argument, kernel_size, padding and stride jointly
    determine the output shape by the following formula:

        out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1

    You may use im2col, col2im inside forward and backward, respectively.

    Arguments:
        kernel_size    -- integer or tuple, spatial size of convolution kernel. If it's tuple, it specifies the height and
                          width of kernel size.
        padding        -- zero padding added on both sides of input array
        stride         -- integer, stride of convolution.
'''
class MaxPool2d(object):
    def __init__(self, kernel_size, padding = 0, stride = 1):
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride
        self.cache = None

    '''
        Forward computation of max pooling layer. (3 points)

        You should use im2col in this function.  You may want to save some
        intermediate variables to class membership (self.)

        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, input_channel, out_H, out_W)
    '''
    def forward(self, input):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        N, C, H, W = input.shape
        kh, kw, p, s = self.kernel_h, self.kernel_w, self.padding, self.stride
        out_H = (H + 2 * p - kh) // s + 1
        out_W = (W + 2 * p - kw) // s + 1
        mod_H = (H + 2 * p - kh) % s
        mod_W = (W + 2 * p - kw) % s

        input_reshape = input.reshape(N * C, 1, H, W)
        input_cols = im2col(input_reshape, kh, kw, s, p)
        output = np.max(input_cols, axis=1).reshape(N, C, out_H, out_W)

        self.cache = ((N, C, H, W, out_H, out_W), (mod_H, mod_W), input_cols)

        return output

    '''
        Backward computation of max pooling layer. (3 points)

        You should use col2im and saved variable(s) from forward().

        Arguments:
            grad_output -- numpy array of shape (N, input_channel, out_H, out_W)

        Ouput:
            grad_input  -- numpy array of shape(N, input_channel, H, W), gradient w.r.t input
    '''
    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        shape_vals, mod_vals, input_cols = self.cache
        N, C, H, W, out_H, out_W = shape_vals
        kh, kw, p, s = self.kernel_h, self.kernel_w, self.padding, self.stride

        cols_argmax = input_cols.reshape(N*C, kh*kw, out_H*out_W)
        cols_argmax = cols_argmax.transpose(0, 2, 1).reshape(N*C*out_H*out_W, kh*kw)
        cols_argmax = np.argmax(cols_argmax, axis=1)

        grad_cols = np.zeros((N*C*out_H*out_W, kh*kw)).astype(np.float32)
        grad_output_flat = grad_output.reshape((N*C*out_H*out_W))
        grad_cols[np.arange(N*C*out_H*out_W), cols_argmax] = grad_output_flat
        grad_cols = grad_cols.reshape(N*C, out_H*out_W, kh*kw).transpose(0, 2, 1)
        grad_cols = grad_cols.reshape(N, C*kh*kw, out_H, out_W)

        grad_input = col2im(grad_cols, kh, kw, s, p, mod_vals=mod_vals).reshape((N, C, H, W))
        return grad_input
