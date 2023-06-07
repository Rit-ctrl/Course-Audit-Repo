from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C,H,W = input_dim
        pad = (filter_size-1)//2
        stride = 1 

        
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.params['w1'] = np.random.normal(0,scale = weight_scale,size = (num_filters,C,filter_size,filter_size))
        self.params['b1'] = np.zeros(num_filters,dtype=self.dtype)
        self.params['w2'] = np.random.normal(0,scale=weight_scale,size=((H//2)*(W//2)*num_filters,hidden_dim))
        self.params['b2'] = np.zeros(shape=(hidden_dim),dtype= self.dtype)
        self.params['w3'] = np.random.normal(0,scale=weight_scale,size = (hidden_dim,num_classes))
        self.params['b3'] = np.zeros(shape=(num_classes),dtype=self.dtype)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['w1'], self.params['b1']
        W2, b2 = self.params['w2'], self.params['b2']
        W3, b3 = self.params['w3'], self.params['b3']

        #(w - 2* )

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        N = X.shape[0]
        #conv - relu - 2x2 max pool - affine - relu - affine - softmax
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out,conv_cache = conv_forward_fast(X,self.params['w1'],self.params['b1'],conv_param)
        out,relu_cache_1 = relu_forward(out)
        out,max_cache = max_pool_forward_fast(out,pool_param) 
        # fc_in = out.reshape(N,-1)
        # print(out.shape)
        out,fc_relu_cache = affine_relu_forward(out,self.params['w2'],self.params['b2'])
        out,affine_out_cache = affine_forward(out,self.params['w3'],self.params['b3'])

        scores = out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        #conv - relu - 2x2 max pool - affine - relu - affine - softmax

        loss,dsoft = softmax_loss(scores,y)
        loss += 0.5*self.reg*(np.sum(np.square(self.params['w2'])) + np.sum(np.square(self.params['w3'])))

        daffine_out,grads['w3'],grads['b3']= affine_backward(dsoft,affine_out_cache)
        df_fc_relu,grads['w2'],grads['b2'] = affine_relu_backward(daffine_out,fc_relu_cache)
        dpool = max_pool_backward_fast(df_fc_relu,max_cache)
        drelu = relu_backward(dpool,relu_cache_1)
        dx,grads['w1'],grads['b1'] = conv_backward_fast(drelu,conv_cache)

        grads['w2'] += self.reg*self.params['w2'] #0.5 cancels out 2 ie 2*reg*w1 * 0.5
        grads['w3'] += self.reg*self.params['w3']

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
