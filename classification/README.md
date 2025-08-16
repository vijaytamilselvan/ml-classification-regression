**Sigmoid activation**
The sigmoid function is given by:
sigmoid(z) = 1 / (1 + e^(-z))
It maps any real-valued number to the range (0, 1), making it useful for binary classification problems where the output is interpreted as a probability.

**Binary Cross-Entropy Loss**
The loss function used is:
Loss = -mean( y * log(y_pred + 1e-15) + (1 - y) * log(1 - y_pred + 1e-15) )

The log function is used here because:

It heavily penalizes confident but wrong predictions, ensuring the model learns quickly from mistakes.

It transforms multiplicative relationships into additive ones, simplifying optimization.

**Gradient Derivation**
For logistic regression with sigmoid activation and binary cross-entropy loss, the derivative with respect to weights simplifies due to the chain rule:
dl/dw = dl/dz * dz/dw

Where:
dl/dz = y_pred - y
dz/dw = x

This simplification happens because when you combine the derivative of binary cross-entropy with the derivative of the sigmoid function, most terms cancel out, leaving (y_pred - y).

**Learning Summary**

Initially implemented binary cross-entropy loss directly.

Explored how sigmoid and binary cross-entropy combine to simplify gradients.

Understood that weight and bias updates depend on (y_pred - y) and input features.

Tried multiple versions to achieve desired output, which helped reinforce the theory even though the accuracy did not improve as expected.

Learned the importance of documenting both successful and unsuccessful attempts in GitHub to demonstrate learning progression.
