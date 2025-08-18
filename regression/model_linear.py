import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 100

# Features
#Used TV and radio as features here.

TV = np.random.uniform(0, 100, n_samples)   # TV ad spend
Radio = np.random.uniform(0, 50, n_samples) # Radio ad spend


# True relationship: Sales = 3*TV + 1.5*Radio + 5 + noise
noise = np.random.normal(0, 25, n_samples)
Sales = 3*TV + 1.5*Radio + 5 + noise

# Prepare features (n_samples x 2) and target (n_samples x 1)
X = np.column_stack((TV, Radio))       # shape (100,2) - Stacking the input column wise 
y = Sales.reshape(-1, 1)               # shape (100,1) 

print("Shape of X:", X.shape)  # (100,2)
print("Shape of y:", y.shape)  # (100,1)

# -------------------------------
# 2. Add bias term (X -> [1, TV, Radio])
# -------------------------------
X_b = np.c_[np.ones((n_samples, 1)), X]  # shape (100,3) - Adding boas column of 1's as the first column

# -------------------------------
# 3. Initialize Parameters
# -------------------------------
np.random.seed(42)
w = np.random.randn(3, 1)  # weights (bias + 2 features) 

# -------------------------------
# 4. Define Hypothesis, Loss, Gradients
# -------------------------------

# Hypothesis: y_pred = Xw
def predict(X, w):
    return X.dot(w)

# Mean Squared Error (MSE)
def compute_loss(y, y_pred):
    return np.mean((y_pred - y) ** 2)

# Gradients
def compute_gradients(X, y, y_pred):
    n = len(y)
    dw = (2/n) * X.T.dot(y_pred - y)   # derivative wrt weights
    return dw

# -------------------------------
# 5. Training with Gradient Descent
# -------------------------------
learning_rate = 1e-5
n_epochs = 1000

losses = []

for epoch in range(n_epochs):
    y_pred = predict(X_b, w)
    loss = compute_loss(y, y_pred)
    dw = compute_gradients(X_b, y, y_pred)
    
    # update weights
    w -= learning_rate * dw
    
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.2f}")

# -------------------------------
# 6. Results
# -------------------------------
print("Final Weights (Bias, TV, Radio):", w.ravel())

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.show()
