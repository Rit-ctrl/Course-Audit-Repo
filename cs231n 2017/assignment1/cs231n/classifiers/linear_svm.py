import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    incorrect_optimizable_classes = 0
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        incorrect_optimizable_classes +=1
        dW[:,j] += X[i,:]
        loss += margin
    
    dW[:,y[i]] += -1*X[i,:]*incorrect_optimizable_classes

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW = dW/num_train + 2*reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X,W)
  correct_class_scores = np.choose(y,scores.T) #choosing correct class scores
  
  mask = np.ones(scores.shape,dtype= bool)
  mask[range(scores.shape[0]),y] = False
  margin_sc = scores[mask].reshape(scores.shape[0],scores.shape[1]-1)

  margin = margin_sc - correct_class_scores.reshape(scores.shape[0],-1) + 1

  margin[margin<0] = 0 # don't need less than zero for loss calc

  loss = np.sum(margin)/num_train
  loss = loss + np.sum(W**2)*reg

  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  real_margin = scores - correct_class_scores.reshape(scores.shape[0],-1) + 1
  pos_margin = (real_margin>0).astype(float) # take only >0 margins for gradient calc
  pos_margin[range(scores.shape[0]),y] = -1*(pos_margin.sum(1) - 1) # sum-1 for ignoring true class , this will take care of gradient of y[i] term in sj - y[i] + 1 

  dW = np.dot(X.T,pos_margin)/num_train + 2*reg*W


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
