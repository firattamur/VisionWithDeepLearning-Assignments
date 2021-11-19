from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_reshaped = x.reshape(x.shape[0], -1)
        
    Nx, Dx = x_reshaped.shape
    
    Dw, Mw = w.shape
    
    Mb     = b.shape[0]
    
    # simple dimension check
    assert Dx == Dw, "D must be equal for x and w!"
    assert Mw == Mb, "M must be equal for w and b!"
    
    out = np.dot(x_reshaped, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_shape    = x.shape
    x_reshaped = x.reshape(x.shape[0], -1)
        
    Nx, Dx = x_reshaped.shape
    
    Dw, Mw = w.shape
    
    Mb     = b.shape[0]
    
    # simple dimension check
    assert Dx == Dw, "D must be equal for x and w!"
    assert Mw == Mb, "M must be equal for w and b!"
    
    # dL/dx = dL/dout * dout/dx -> dout * w
    dx = np.dot(dout, np.transpose(w))
    dx = dx.reshape(x_shape)

    # dL/dw = dL/dout * dout/dw -> dout * x
    dw = np.dot(np.transpose(x_reshaped), dout)

    # dL/db = dL/dout * dout/db -> dout * 1
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(x, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # dL/dx = dL/dout * dout/dx -> dout * (1 if x > 0 else 0)
    dx = np.copy(dout)
    dx[x <= 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def leaky_relu_forward(x, lrelu_param):
    """
    Computes the forward pass for a layer of leaky rectified linear units (Leaky ReLUs).

    Input:
    - x: Inputs, of any shape
    - lrelu_param: Dictionary with the following key:
        - alpha: scalar value for negative slope

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: (x, lrelu_param).
            Input x, of same shape as dout,
            lrelu_param, needed for backward pass.
    """
    out = None
    alpha = lrelu_param.get('alpha', 2e-3)
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # not copying is big big mistake do not forget this
    out = np.copy(x)
    out[x <= 0] = out[x <= 0] * alpha

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, lrelu_param)
    return out, cache


def leaky_relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of leaky rectified linear units (Leaky ReLUs).
    Note that, the negative slope parameter (i.e. alpha) is fixed in this implementation.
    Therefore, you should not calculate any gradient for alpha.
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: (x, lr_param)

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    (x, lr_param) = cache
    alpha = lr_param["alpha"]
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.copy(dout)
    dx[x <= 0] = dx[x <= 0] * alpha

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out  = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) > p) * 1
        out  = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    Note that, the filter is not flipped as in the regular convolution operation
    in signal processing domain. Therefore, technically this implementation
    is a cross-correlation.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # get convolution parameters
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    
    # check dimensions
    Nx, Cx, Hx, Wx = x.shape
    Fw, Cw, Hw, Ww = w.shape
    Fb             = b.shape[0]

    # tests
    assert Cx == Cw,  "Channel dimension must be equal for x and w!"
    assert Fw == Fb,  "Filter count must be equal for w and b!"

    # calculate output dimension
    Hout = 1 + (Hx + 2 * pad - Hw) / stride
    Wout = 1 + (Wx + 2 * pad - Ww) / stride

    # padding x with 0
    # because we have 4D tensor (ax0, ax1, ax2, ax3) 
    # but we want to pad only axes ax2 and ax3
    # we need to set padding width 0 for ax0 and ax1
    # and padding with pad for ax2, ax3
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, ), (pad, pad)), mode='constant')

    Nx_pad, Cx_pad, Hx_pad, Wx_pad = x_padded.shape

    assert Nx_pad == Nx, "Nx is changed after padding!"
    assert Cx_pad == Cx, "Cx is changed after padding"

    """
    Here, I am following notes from class.

    Process for convolution will be:

    X: 

      [x1*  x2*  x3  x4 ]
      [x5*  x6*  x7  x8 ]
      [x9  x10 x11 x12]
      [x13 x14 x15 x16]

    convert X:

      [x1 x2 ...]
      [x2 x3 ...]
      [x5 x6 ...]
      [x6 x7 ...]

    w: 

      [w1 w2]
      [w3 w4] 

    convert w to : 

      [w1]
      [w2]
      [w3]
      [w4]

    """

    # convert filter to (count, filter)
    w_reshaped = w.reshape(Fw, -1)

    for i in range(Nx):

      x_reshape = np.zeros((Cw * Hw * Ww, Hout * Wout))
      x_col = 0

      for h in range(0, Hx_pad - Hw + 1, stride):
        for w in range(0, Wx_pad - Ww + 1, stride):
          x_reshape[:, x_col] = x_padded[i, :, h:h+Hw, w:w+Ww].reshape(Cw * Hw * Ww)
          x_col += 1

      out[i] = (np.transpose(x_reshape).dot(w_reshaped) + b.reshape(Fb, 1)).reshape(Fw, Hout, Wout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    ###########################################################################
    x, w, b, conv_param = cache
    padding, stride = conv_param['pad'], conv_param['stride']
    f, c, filter_height, filter_width = w.shape
    n, _, output_height, output_width = dout.shape

    # pad x
    N, C, H, W = x.shape
    pad_horiz = np.zeros((N, C, H, padding))
    x_horiz_padded = np.concatenate((pad_horiz, x, pad_horiz), axis=3)
    pad_vert = np.zeros((N, C, padding, x_horiz_padded.shape[3]))
    x_padded = np.concatenate((pad_vert, x_horiz_padded, pad_vert), axis=2)

    dx_padded = np.zeros(x_padded.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    w_flat = w.reshape((f, -1))

    for i in range(output_height):
        for j in range(output_width):
            dout_slice = dout[:, :, i, j]

            dx_slice_flattened = dout_slice.dot(w_flat)
            dx_slice = dx_slice_flattened.reshape((n, c, filter_height, filter_width))
            dx_padded[:, :, i * stride: i * stride + filter_height, j * stride: j * stride + filter_width] += dx_slice

            x_padded_slice = x_padded[:, :, i * stride: i * stride + filter_height, j * stride: j * stride + filter_width]
            x_slice_flattened = x_padded_slice.reshape((n, -1))

            dw += dout_slice.T.dot(x_slice_flattened).reshape(dw.shape)
            db += dout_slice.sum(axis=0)

    # crop padding from dx
    dx = dx_padded[:, :, padding:-padding, padding:-padding]
    ###########################################################################
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    Nx, Cx, Hx, Wx = x.shape

    # get parameters for pooling
    stride = pool_param["stride"]
    Hp     = pool_param["pool_height"]
    Wp     = pool_param["pool_width"]

    # calculate result output dimension
    Hout = 1 + (Hx - Hp) / stride
    Wout = 1 + (Wx - Wp) / stride

    # create empty output array
    out = np.zeros((Nx, Cx, Hout, Wout))

    # for each data point in batch
    for i in range(Nx):

      x_reshape = np.zeros((Cx, Hout * Wout))
      x_col = 0

      for h in range(0, Hx - Hp + 1, stride):
        for w in range(0, Wx - Wp + 1, stride):
          pool = x[i, :, h:h+Hp, w:w+Wp].reshape(Cx, Hp * Wp)
          x_reshape[:, x_col] = pool.max(axis=1)

          x_col += 1

      out[i] = x_reshape.reshape(Cx, Hout, Wout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pool_param = cache

    Nout, Cout, Hout, Wout = dout.shape
    Nx, Cx, Hx, Wx = x.shape

    # check dimensions
    assert Nout == Nx, "Nout, Nx must be equal!"
    assert Cout == Cx, "Cout, Cx must be equal!"

    stride = pool_param["stride"]

    Hp = pool_param["pool_height"]
    Wp = pool_param["pool_width"]

    dx = np.zeros(x.shape)

    for i in range(Nout):

      dout_i = dout[i].reshape[Cx, Hout * Wout]
      dout_count = 0

      for h in range(0, Hx - Hp + 1, stride):
        for w in range(0, Wx - Wp + 1, stride):
          
          # we need to find max indices in pooling
          # because we will only allow these indices to have gradients
          pool = x[i, :, h:h+Hp, w:w+Wp].reshape(Cx, Hp * Wp)
          max_indices = pool.argmax(axis=1)

          dout_w = dout_i[:, dout_count]
          dout_count += 1

          # create gradients matrix for filter
          dpool = np.zeros(pool.shape)

          # pass gradients to only elements with max values
          dpool[np.arange(Cx), max_indices] = dout_w

          # update dx gradient with pool gradients
          dx[i, :, h:h+Hp, w:w+Wp] += dpool.reshape(Cx, Hp, Wp)
          
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
