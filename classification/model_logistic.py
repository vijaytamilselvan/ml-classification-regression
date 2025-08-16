"""
Logistic Regression Implementation from Scratch
=================================================

This file implements Logistic Regression using only NumPy.  
It is not only runnable code, but also acts as a documentation 
of the mathematical derivations and project learnings.

-------------------------------------------------
1. Hypothesis Function (Sigmoid Activation)
-------------------------------------------------

We start with a linear function:

    z = X·w + b

But instead of predicting directly with z, we pass it through a
sigmoid function:

    y_hat = sigmoid(z) = 1 / (1 + exp(-z))

This ensures output is in range (0,1), interpreted as probability.

-------------------------------------------------
2. Loss Function (Binary Cross Entropy - BCE)
-------------------------------------------------

For a single training example:

    L = - [ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]

Why BCE?  
- If y=1 → loss = -log(y_hat). We want y_hat close to 1.  
- If y=0 → loss = -log(1 - y_hat). We want y_hat close to 0.  

For N samples, the cost function is:

    J(w, b) = (1/N) * Σ L_i

-------------------------------------------------
3. Gradient Derivation (Key Step!)
-------------------------------------------------

We need to minimize J(w, b) using gradient descent.

Step 1: Differentiate loss w.r.t z
    ∂L/∂z = (y_hat - y)       ← neat simplification!

Step 2: Chain rule
    z = X·w + b

So,  
    ∂L/∂w = ∂L/∂z * ∂z/∂w  
           = (y_hat - y) * X

    ∂L/∂b = ∂L/∂z * ∂z/∂b  
           = (y_hat - y) * 1

Step 3: For dataset (N samples), average across all examples:

    dw = (1/N) * X.T · (y_hat - y)
    db = (1/N) * Σ (y_hat - y)

This explains exactly **where dw and db come from**.

-------------------------------------------------
4. Gradient Descent Update Rule
-------------------------------------------------

We move parameters in the opposite direction of the gradient:

    w := w - lr * dw
    b := b - lr * db

where `lr` = learning rate.

-------------------------------------------------
5. Project Learning Notes
-------------------------------------------------

- Understood how sigmoid + binary cross entropy simplify to (y_hat - y).
- Derived gradients step-by-step with chain rule (dl/dw = dl/dz * dz/dw).
- Learned the importance of averaging gradients across samples.
- Gained intuition: dw points in the direction of correction for each feature,
  db adjusts the global bias (shifts decision boundary).
- Practiced GitHub workflow (Colab → push code → update README).
- Keeping both code + math notes in one file ensures reproducibility + clarity.

=================================================
"""

import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        """
        Initialize Logistic Regression model.
        lr      : learning rate
        n_iters : number of iterations
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train logistic regression using gradient descent
        X : feature matrix (num_samples x num_features)
        y : labels (0 or 1)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Forward pass: linear + sigmoid
            linear_output = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_output)

            # Gradients:
            # dl/dz = y_predicted - y
            # dw = (1/N) * X.T · (y_predicted - y)
            # db = (1/N) * Σ(y_predicted - y)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict class labels for input samples
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_output)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])
