from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, regtype='L2'):
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
    - regtype: Regularization type: L1 or L2

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # here some guide
    # find scores for each data points by W @ X
    # find softwax loss by e ^ true_score / sum ( e ^ all_scores)
    # find gradient by simple d ( 1 - d)

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):

      # find score for each train data with X * W
      # (1, D) * (D, C) -> (1, C)
      scores = np.dot(X[i], W)
      
      # because we have e^scores term in loss 
      # we do not want make it huge 
      # subtract max from all score then make e^scores small
      scores -= np.max(scores)

      # softmax loss
      loss += - np.log(np.exp(scores[y[i]]) / np.sum(np.exp(scores)))

      # find gradients
      for c in range(num_class):
        dW[:, c] += (np.exp(scores[c]) * X[i]) / np.sum(np.exp(scores)) 

      # for the correct class
      dW[:, y[i]] -= X[i]

    # because of we take mean of all train data
    loss /= num_train
    dW   /= num_train

    # add regularization loss
    if regtype == "L1":
      loss += reg * np.sum(np.abs(W))

      # derivative of |w|
      #  d|w|     {  1      w > 0
      # ------ =  { 
      #  |w|      { -1      w < 0
      # 

      dw = np.ones(W.shape)
      dw[ W < 0 ] = -1

      dW += dw

    else:
      loss += reg * np.sum(W * W)
      dW   += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train, num_feat = X.shape
    num_class = W.shape[1]
    
    # (N, D), (D, C) => (N, C)
    scores = np.dot(X, W)

    # numeric stability for exponential
    # (N, C) = (N, C) - (N, 1)
    scores -= scores.max(axis = 1, keepdims = True)

    # find exponents of scores
    scores_exp = np.exp(scores)  
    scores_probs = scores_exp / np.sum( scores_exp, axis = 1, keepdims = True )

    # losses
    loss = -np.log(scores_probs[np.arange(num_train), y])

    # loss is a single number
    loss = np.sum(loss)

    dW = scores_probs.reshape(num_train, -1)
    dW[np.arange(num_train), y] -= 1

    dW = np.dot(np.transpose(X).reshape(num_feat, num_train), dW)

    # Mean
    loss /= num_train
    dW   /= num_train

    if regtype == "L1":
      
      # l1 regularization
      loss += reg * np.sum(np.abs(W))

      dw = np.ones(W.shape)
      dw[ W < 0 ] = -1

      dW += dw

    else:

      # l2 regularization
      loss += reg * np.sum(np.multiply(W, W))
      dW   += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
