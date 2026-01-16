import numpy as np
from typing import Dict, Tuple, Iterable, Optional, Any

"""
Linear Models skeleton for Deep Learning HW1.

This module defines the classes and functions that students need to complete.
The notebook will import these definitions instead of defining them inline.

Each function or method marked with `raise NotImplementedError` needs to be
implemented by the student.
"""


class LinearClassifier:
    """
    Base class for linear classifiers.  Stores weights and provides the
    interface for prediction, training, and computing accuracy.  Subclasses
    should override ``predict`` and ``loss`` to implement specific
    classification algorithms.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the classifier with a small random weight matrix.

        Parameters
        ----------
        X : np.ndarray
            Training data matrix of shape (N, D), where N is the number
            of samples and D is the number of features (possibly including
            a bias term).  The weight matrix will have shape (D, C).

        y : np.ndarray
            Array of shape (N,) containing integer class labels in the
            range [0, C-1], where C is the number of distinct classes.
        """
        N, D = X.shape
        C = int(np.max(y)) + 1
        # Initialize weights with small random values
        self.W = 0.001 * np.random.randn(D, C)
        self.num_classes = C
        self.num_features = D

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given data using the classifier's weights.

        This default implementation should be overridden by subclasses.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (M, D) of input data.

        Returns
        -------
        np.ndarray
            Array of shape (M,) of predicted class labels.
        """
        scores = X @ self.W
        y_pred = np.argmax(scores, axis=1)
        #print(y_pred)
        return y_pred
        raise NotImplementedError("predict method must be implemented in subclass")

    def calc_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy on a dataset.

        Accuracy is the fraction of instances that are classified correctly.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (M, D).

        y : np.ndarray
            True labels of shape (M,).

        Returns
        -------
        float
            Accuracy as a float in the range [0, 1].
        """
        accuracy = 0.0
        y_pred = self.predict(X)
        tp_value = np.sum(y_pred == y)

        accuracy = tp_value/y.shape[0]
        return accuracy

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              learning_rate: float = 1e-3,
              num_iters: int = 100,
              batch_size: int = 200,
              verbose: bool = False) -> list:
        """
        Train the classifier using stochastic gradient descent.

        This method samples minibatches, computes the loss and gradient,
        and performs weight updates.  It collects the loss value at each
        iteration in ``loss_history`` and returns it.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (N, D).

        y : np.ndarray
            Labels of shape (N,).

        learning_rate : float, optional
            Step size for gradient descent (default: 1e-3).

        num_iters : int, optional
            Number of iterations to run (default: 100).

        batch_size : int, optional
            Number of samples per minibatch (default: 200).

        verbose : bool, optional
            If True, prints loss every 100 iterations.

        Returns
        -------
        list
            A list containing the loss value at each iteration.
        """
        #########################################################################
        # TODO:                                                                 #
        # Sample batch_size elements from the training data and their           #
        # corresponding labels to use in every iteration.                       #
        # Store the data in X_batch and their corresponding labels in           #
        # y_batch                                                               #
        #                                                                       #
        # Hint: Use np.random.choice to generate indices. Sampling with         #
        # replacement is faster than sampling without replacement.              #
        #                                                                       #
        # Next, calculate the loss and gradient and update the weights using    #
        # the learning rate. Use the loss_history array to save the loss on     #
        # iteration to visualize the loss.                                      #
        #########################################################################
        num_instances = X.shape[0]
        loss_history = []
        loss = 0.0
        for i in range(num_iters):
            batch_id = np.random.choice(num_instances, batch_size, replace=False)
            X_batch = X[batch_id]
            y_batch = y[batch_id]
            ###########################################################################
            # TODO: Create X_batch and y_batch. Call the loss method to get the loss value  #
            # and grad (the loss function is being override, see the loss             #
            # function return values).                                                #
            # Finally, append each of the loss values created in each iteration       #
            # to loss_history.                                                        #
            loss, dW = self.loss(X_batch, y_batch)
            # TODO:                                                                   #
            # Perform parameter update                                                #
            # Update the weights using the gradient and the learning rate.            #
            self.W -= learning_rate * dW 
            loss_history.append(loss)

            if verbose and i % 100 == 0:
                print ('iteration %d / %d: loss %f' % (i, num_iters, loss))

        return loss_history

    def loss(self, X: np.ndarray, y: np.ndarray):
        """
        Compute the loss function and its gradient.

        Subclasses should override this method to compute the appropriate
        loss and gradient for their algorithm.

        Parameters
        ----------
        X : np.ndarray
            Minibatch of data of shape (N, D).

        y : np.ndarray
            Labels of shape (N,).

        Returns
        -------
        tuple
            A tuple (loss, dW) where ``loss`` is a scalar and ``dW`` is
            an array of the same shape as ``self.W``.
        """
        raise NotImplementedError("loss must be implemented in subclass")


class LinearPerceptron(LinearClassifier):
    """
    Linear classifier that uses the Perceptron loss.
    Students should implement the ``predict`` and ``loss`` methods.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the perceptron using the base class constructor.
        """
        # TODO: Initiate the parameters of your model.                            #
        # You can assume y takes values 0...K-1 where K is number of classes      #
        super().__init__(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using the perceptron rule.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (M, D).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (M,).
        """
        scores = X @ self.W
        # TODO: Implement this method.                                            #
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        Compute perceptron loss and gradient for a minibatch.

        Parameters
        ----------
        X_batch : np.ndarray
            Minibatch data of shape (N, D).

        y_batch : np.ndarray
            Minibatch labels of shape (N,).

        Returns
        -------
        tuple
            A tuple (loss, dW) where ``loss`` is a scalar and ``dW`` has
            the same shape as ``self.W``.
        """
        # Do Not change this function! the implementation of this function is in the `softmax_cross_entropy` function
        return softmax_cross_entropy(self.W, X_batch, y_batch)


class LogisticRegression(LinearClassifier):
    """
    Linear classifier that uses softmax and cross-entropy loss for multiclass
    classification.
    Students should implement the ``predict`` and ``loss`` methods.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the logistic regression model using the base class constructor.
        """
        self.W = None
        # TODO: Initialize the model via the base class constructor.              
        super().__init__(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using the softmax probabilities.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (M, D).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (M,).
        """
        y_pred = None
        scores = X @ self.W
        softmax_probs = softmax(scores)
        y_pred = np.argmax(softmax_probs, axis=1)

        return y_pred

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        Compute softmax cross-entropy loss and gradient for a minibatch.

        Parameters
        ----------
        X_batch : np.ndarray
            Minibatch data of shape (N, D).

        y_batch : np.ndarray
            Minibatch labels of shape (N,).

        Returns
        -------
        tuple
            A tuple (loss, dW) where ``loss`` is a scalar and ``dW`` has
            the same shape as ``self.W``.
        """
        # will be implemented later
        return softmax_cross_entropy_vectorized(self.W, X_batch, y_batch)

def perceptron_loss_naive(W: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Compute the multiclass perceptron loss using explicit loops.

    This function should compute the average number of classification mistakes
    over the batch and the gradient of the loss with respect to the weight
    matrix W.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (D, C).

    X : np.ndarray
        Data matrix of shape (N, D).

    y : np.ndarray
        Labels of shape (N,).

    Returns
    -------
    loss : float
        Average number of classification mistakes
    dW : (D, C)
        Gradient of the loss w.r.t. W
    """
    loss = 0.0
    # Align dimensions (drop extra bias columns if needed)
    D_w = W.shape[0]
    X_use = X[:, :D_w] if X.shape[1] != D_w else X

    N, D = X_use.shape
    _, C = W.shape

    # Initialize loss & gradient
    loss = 0.0
    dW = np.zeros_like(W)
    # TODO: Implement Perceptron loss with explicit loops                     
    #                                                                         
    # After looping over all samples:                                         
    #   - Average loss and gradient by N                                      
    print("Number of features", D) # also  X[i].shape
    print()
    for i in range(N):
      # Go through each data instance (containing all its features)
      # Then calculate the score on each class for THAT instance
      scores = X[i].dot(W)
      #print(f"The score of each class from the {i}th instance:  ", scores)
      #print("The class that should be picked (true class): ", y[i])
      #print()
      correct_class_score = scores[y[i]]
      
      for j in range(C):
        # This removes self comparitive cases 
        # We only care about cases where j != y[i]
        if j == y[i]:
          continue
        if scores[j] >= correct_class_score:
          loss += (scores[j] - correct_class_score)

          dW[:, j] += X[i]
          dW[:, y[i]] -= X[i]

      loss /= N
      dW /= N
    return loss, dW

def softmax_cross_entropy(W: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Compute the multiclass softmax cross-entropy loss and its gradient.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (D, C).

    X : np.ndarray
        Data matrix of shape (N, D).

    y : np.ndarray
        Labels of shape (N,).

    Returns
    -------
    loss : float
    dW   : (D, C) gradient of loss wrt W
    """
    # --- Align dimensions if X has extra columns (e.g., duplicate bias) ---
    D_w = W.shape[0]
    if X.shape[1] != D_w:
        X_use = X[:, :D_w]  # drop extra tail columns safely
    else:
        X_use = X

    N = X_use.shape[0]
    loss, dW = 0.0, np.zeros_like(W)
    # TODO: Implement the forward pass.                                         
    # This typically means: 
    # 1) Compute the scores
    # 2) Apply the activation function
    for i in range(N):
        scores = X_use[i].dot(W)

        stable_scores = scores - np.max(scores)


        exp_scores = np.exp(stable_scores)

        probs = exp_scores / np.sum(exp_scores)
        
        correct_class_prob = probs[y[i]]
        correct_log_likelihood = -np.log(correct_class_prob + 1e-10)
    # TODO: Compute the loss.                                                   
    # Use the average negative log-likelihood of the correct class.             
        loss += -np.log(correct_class_prob + 1e-10) 
    # TODO: Backward pass: compute gradient dW.                                                          
    #START OF YOUR CODE                              
        dscores = probs
        dscores[y[i]] -=1
        dW += np.outer(X_use[i], dscores)

    loss /= N
    dW /= N

    return loss, dW



def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax function for each row of the input x.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (N, C) where N is the number of samples and C
        is the number of classes.

    Returns
    -------
    probs : (N, C)
        Row-wise probabilities that sum to 1
    """
    probs = np.zeros_like(x)
    # STABILITY TRICK 
    # We create a vector of the same dimension as the number of samples to extract the max scores for stability
    # This helps reduce exponential explosion(thus either overfow or underflow)
    # we maintain dimensions: axis = 0 refers that we search down each column
    #                         axis = 1 refers that we search/apply an operation across a row
    max_scores = np.max(x, axis=1, keepdims=True)

    #broadcasting
    stable_scores = x - max_scores
    exp_scores = np.exp(stable_scores)

    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)

    probs = exp_scores / sum_exp_scores
    return probs

def softmax_cross_entropy_vectorized(W: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Compute the multiclass softmax cross-entropy loss and its gradient
    using a fully vectorized implementation.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (D, C).

    X : np.ndarray
        Data matrix of shape (N, D).

    y : np.ndarray
        Labels of shape (N,).

    Returns
    -------
    loss : float
    dW   : (D, C) gradient of loss wrt W
    """
    # --- Align dimensions if X has extra columns (e.g., duplicate bias) ---
    D_w = W.shape[0]
    if X.shape[1] != D_w:
        X_use = X[:, :D_w]  # drop extra tail columns safely
    else:
        X_use = X

    N = X_use.shape[0]
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Implement the forward pass in a fully vectorized way.               #
    # 1. Compute the scores (N, C) for all samples in X_use.                    #
    # 2. Stabilize the scores by subtracting the max score in each row.         #
    # 3. Compute the softmax probabilities (N, C) for all samples.              #
    #############################################################################
    # START OF YOUR CODE                                                        #
    #############################################################################
    raw_scores = X_use @ W 
    prob_mat = softmax(raw_scores)
   

    #############################################################################
    # END OF YOUR CODE                                                          #
    #############################################################################


    #############################################################################
    # TODO: Compute the loss.                                                   #
    # 1. Select the probabilities for the correct class for all samples.        #
    #    (Hint: Use advanced integer indexing with np.arange(N) and y)          #
    # 2. Compute the negative log-likelihood for these probabilities.           #
    # 3. Compute the average loss (a scalar) across all samples in the batch.   #
    #############################################################################
    # START OF YOUR CODE                                                        #
    #############################################################################
    row_indicies = np.arange(N)
    true_probs = prob_mat[row_indicies, y]
    log_likelihood = -np.log(true_probs + 1e-10)
    loss = np.sum(log_likelihood)/N
    

    #############################################################################
    # END OF YOUR CODE                                                          #
    #############################################################################


    #############################################################################
    # TODO: Backward pass: compute gradient dW.                                 #
    # 1. Compute the gradient of the loss with respect to the scores.           #
    # 2. Compute the gradient dW using the chain rule (X.T @ dscores).          #
    # 3. Average the gradient over the batch (divide by N).                     #
    #############################################################################
    # START OF YOUR CODE                                                        #
    #############################################################################
    # the ..._like allows you to choose from a matrix that exists
    bin_y = np.zeros_like(prob_mat)
    bin_y[row_indicies, y] = 1
    dW = (X_use.T @ (prob_mat - bin_y))/N


    #############################################################################
    # END OF YOUR CODE                                                          #
    #############################################################################

    return loss, dW


def tune_perceptron(
    ModelClass,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rates: Iterable[float],
    batch_sizes: Iterable[int],
    *,
    num_iters: int = 500,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[Dict[Tuple[float, int], Tuple[float, float]], Any, float]:
    """
    Hyperparameter sweep for a LinearPerceptron-like model.

    Parameters
    ----------
    ModelClass : class
        Class with API:
          - __init__(X_init, y_init, **kwargs)   # optional; ok if ignores data
          - train(X, y, learning_rate, num_iters, batch_size, verbose=False) -> loss_history
          - calc_accuracy(X, y) -> float
    X_train, y_train, X_val, y_val : arrays
        Training/validation data.
    learning_rates : iterable of float
    batch_sizes : iterable of int
    num_iters : int
        Iterations per configuration.
    model_kwargs : dict or None
        Extra kwargs to pass to ModelClass.
    verbose : bool
        If True, prints progress.

    Returns
    -------
    results : dict
        {(lr, batch): (train_acc, val_acc)} for each tried combo.
    best_model : object
        The fitted model instance with the highest validation accuracy.
    best_val : float
        The best validation accuracy achieved.
    """

    # Initialization
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    results: Dict[Tuple[float, int], Tuple[float, float]] = {}
    best_val = -1.0
    best_model = None

    ############################################################################
    # TODO: Iterate over all combinations of learning_rates and batch_sizes.   #
    #   For each (lr, batch_size):                                             #
    #     1. Create a new perceptron model using ModelClass.                   #
    #     2. Train it using the train() method with the given lr and batch.    #
    #     3. Compute training and validation accuracies using calc_accuracy(). #
    #     4. Store results[(lr, batch)] = (train_acc, val_acc).                #
    #     5. Track the best model based on validation accuracy.                #
    #                                                                          #
    # Hints:                                                                   #
    # - Use 'verbose' to optionally print current hyperparameters.             #
    # - Make sure to create a *new* model for each configuration.              #
    ############################################################################

            
    # TODO: Create a new model instance
    # model = ...
    best_val = 0.0
    for rate in learning_rates:
      for batch in batch_sizes:
        if verbose: 
            print(f"\nTrying lr={rate:.1e}, bs={batch}")
        model = ModelClass(X_train, y_train, **(model_kwargs or {}))
        loss_hist = model.train(X_train, y_train, rate, num_iters, batch, verbose=False)
            # TODO: Train the model using the provided hyperparameters
            # model.train(...)

            # TODO: Compute train and validation accuracy
            # train_acc = ...
            # val_acc = ...
        train_acc = model.calc_accuracy(X_train, y_train)
        val_acc = model.calc_accuracy(X_val, y_val)
            # TODO: Store the results in the 'results' dictionary
            # results[(lr, batch)] = ...
        results[(rate, batch)] = (train_acc, val_acc)
            # TODO: Update the best model and best_val if this is the best so far
            # if ... :
            #     best_val = ...
            #     best_model = ...
        if best_val < val_acc: 
          best_val = val_acc
          best_model = model

        if verbose:
                print(f"  Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}")

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    print("Best Model: ", best_model)
    print("Best validation accuracy: ", best_val)
    return results, best_model, best_val

