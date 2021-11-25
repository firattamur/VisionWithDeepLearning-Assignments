from builtins import range
from builtins import object
import numpy as np

from comp411.layers import *
from comp411.layer_utils import *


class ThreeLayerNet(object):
    """
    A three-layer fully-connected neural network with Leaky ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of tuple of (H1, H2) yielding the dimension for the
    first and second hidden layer respectively, and perform classification over C classes.

    The architecture should be affine - leakyrelu - affine - leakyrelu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=(64, 32), num_classes=10,
                weight_scale=1e-3, reg=0.0, alpha=1e-3):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A tuple giving the size of the first and second hidden layer respectively
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - alpha: negative slope of Leaky ReLU layers
        """
        self.params = {}
        self.reg = reg
        self.alpha = alpha

        ############################################################################
        # TODO: Initialize the weights and biases of the three-layer net. Weights  #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1', second layer                    #
        # weights and biases using the keys 'W2' and 'b2',                         #
        # and third layer weights and biases using the keys 'W3' and 'b3.          #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        """
        Dimension calculations:
      
        input_dim = 3 * 32 * 32
        hidden1_dim = 64
        hidden2_dim = 32
        output_dim = 10

        W1: (input_dim, hidden1_dim)
        b1: (hidden1_dim)
        
        W2: (hidden1_dim, hidden2_dim)
        b2: (hidden2_dim, )

        W3: (hidden2_dim, output_dim)
        b3: (output_dim, )
        """

        hidden1_dim = hidden_dim[0]
        hidden2_dim = hidden_dim[1]
        output_dim  = num_classes

        # initialize weights and biases
        self.params["W1"] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden1_dim))
        self.params["b1"] = np.zeros(hidden1_dim)

        self.params["W2"] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden1_dim, hidden2_dim))
        self.params["b2"] = np.zeros(hidden2_dim)

        self.params["W3"] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden2_dim, output_dim))
        self.params["b3"] = np.zeros(output_dim)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer net, computing the  #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if y is not None:
          Nx = X.shape[0]
          Ny = y.shape[0]

          assert Nx == Ny, "Nx, Ny must be equal!"

        Z1,     Z1_cache    = affine_lrelu_forward(X,  self.params["W1"], self.params["b1"], lrelu_param={"alpha": self.alpha})
        Z2,     Z2_cache    = affine_lrelu_forward(Z1, self.params["W2"], self.params["b2"], lrelu_param={"alpha": self.alpha})
        scores, score_cache = affine_lrelu_forward(Z2, self.params["W3"], self.params["b3"], lrelu_param={"alpha": self.alpha})

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer net. Store the loss#
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # find softmax loss
        loss, softloss_grad = softmax_loss(scores, y)

        # add regularization loss
        loss += 0.5 * self.reg * np.sum(np.multiply(self.params["W1"], self.params["W1"]))
        loss += 0.5 * self.reg * np.sum(np.multiply(self.params["W2"], self.params["W2"]))
        loss += 0.5 * self.reg * np.sum(np.multiply(self.params["W3"], self.params["W3"]))

        # find gradients from backward pass
        dscores, dw3, db3 = affine_lrelu_backward(softloss_grad, score_cache)
        dz2,     dw2, db2 = affine_lrelu_backward(dscores, Z2_cache)
        dz1,     dw1, db1 = affine_lrelu_backward(dz2, Z1_cache)

        grads["W3"] = dw3 + self.reg * self.params["W3"]
        grads["b3"] = db3

        grads["W2"] = dw2 + self.reg * self.params["W2"]
        grads["b2"] = db2

        grads["W1"] = dw1 + self.reg * self.params["W1"]
        grads["b1"] = db1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    LeakyReLU nonlinearities, and a softmax loss function. This will also implement
    dropout optionally. For a network with L layers, the architecture will be

    {affine - leakyrelu - [dropout]} x (L - 1) - affine - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the ThreeLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                dropout=1, reg=0.0, alpha=1e-2,
                weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - reg: Scalar giving L2 regularization strength.
        - alpha: negative slope of Leaky ReLU layers
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deterministic so we can gradient check the
          model.
        """
        self.use_dropout = dropout != 1
        self.reg = reg
        self.alpha = alpha
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # input layer
        self.params[f"W1"] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dims[0]))
        self.params[f"b1"] = np.zeros(hidden_dims[0])

        # hidden layers
        for layer in range(1, len(hidden_dims)):
          self.params[f"W{layer+1}"] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[layer-1], hidden_dims[layer]))
          self.params[f"b{layer+1}"] = np.zeros(hidden_dims[layer])
          
        # output layer
        self.params[f"W{self.num_layers}"] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dims[len(hidden_dims)-1], num_classes))
        self.params[f"b{self.num_layers}"] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as ThreeLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for dropout param since it
        # behaves differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        z = np.copy(X)
        cache_list = []

        for layer in range(1, self.num_layers):
          w = self.params[f"W{layer}"]
          b = self.params[f"b{layer}"]

          z, cache = affine_lrelu_forward(z, w, b, lrelu_param={"alpha": self.alpha})
          cache_list.append(cache)

        # last layer
        w = self.params[f"W{self.num_layers}"]
        b = self.params[f"b{self.num_layers}"]

        scores, scores_cache = affine_lrelu_forward(z, w, b, lrelu_param={"alpha": self.alpha})
        cache_list.append(scores_cache)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # calculate softmax loss
        loss, loss_grad = softmax_loss(scores, y)

        # add regularization loss for each weights
        for layer in range(1, self.num_layers + 1):
          w = self.params[f"W{layer}"]
          loss += 0.5 * self.reg * np.sum(np.multiply(w, w))
        
        # calculate gradients

        # gradient of output layer
        dscores = loss_grad
        dscores, dw, db = affine_lrelu_backward(dscores, cache_list[self.num_layers-1])

        grads[f"W{self.num_layers}"] = dw + self.reg * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db

        dout = dscores

        for i in range(self.num_layers - 2, -1, -1):
          dx, dw, db = affine_lrelu_backward(dout, cache_list[i])

          grads[f"W{i+1}"] = dw + self.reg * self.params[f"W{i+1}"]
          grads[f"b{i+1}"] = db

          dout = dx

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
