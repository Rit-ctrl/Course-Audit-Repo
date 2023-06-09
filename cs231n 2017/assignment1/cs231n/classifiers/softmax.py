import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = np.dot(X,W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    batch_scores = scores[i,:]
    shift_scores = batch_scores - np.max(batch_scores) #for stability https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    exp_scores = np.exp(shift_scores)
    softmax_score = exp_scores[y[i]]/np.sum(exp_scores)

    loss+= -np.log(softmax_score)

    for j in range(num_classes):
      softmax_score_class = exp_scores[j]/np.sum(exp_scores)

      if j == y[i]:
        dW[:,j] += (softmax_score_class - 1)*X[i]
      else:
        dW[:,j] += softmax_score_class * X[i]

    
  dW = (dW/num_train) + 2*reg*W
  loss = (loss/num_train) + reg*np.sum(W*W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # 

  scores = np.dot(X,W)
  shift_scores = scores - np.max(scores,axis = 1).reshape(-1,1)
  exp_scores = np.exp(shift_scores)
  softmax_scores = exp_scores /np.sum(exp_scores,axis=1).reshape(-1,1)

  mask = softmax_scores
  mask[range(X.shape[0]),y] += -1 
  dW = np.dot(X.T,mask)

  # loss = np.sum(-np.log(softmax_scores[range(X.shape[0]),y]))
  correct_class_scores = np.choose(y, shift_scores.T)  # Size N vector
  loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores), axis=1))
  loss = np.sum(loss)

  loss = loss/num_train + reg* np.sum(W*W)
  dW = dW/num_train + 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

