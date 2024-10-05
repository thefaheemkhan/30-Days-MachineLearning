## Machine Learning Equations

### 1. Linear Regression Model Prediction
$$
\[
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
\]
$$

### 2. Linear Regression Model Prediction (Vectorized Form)
$$
\[
\hat{y} = \theta^T \cdot x
\]
$$

### 3. Mean Squared Error (MSE) Cost Function
$$
\[
MSE(\theta) = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right)^2
\]
$$

### 4. Normal Equation
$$
\[
\theta = \left( X^T \cdot X \right)^{-1} \cdot X^T \cdot y
\]
$$

### 5. Partial Derivative of MSE
$$
\[
\frac{\partial MSE}{\partial \theta_j} = \frac{2}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}
\]
$$

### 6. Gradient Vector of MSE
$$
\[
\nabla_{\theta} MSE(\theta) = \frac{2}{m} X^T \cdot \left( X \cdot \theta - y \right)
\]
$$

### 7. Gradient Descent Update Rule
$$
\[
\theta = \theta - \eta \cdot \nabla_{\theta} MSE(\theta)
\]
$$

### 8. Ridge Regression Cost Function
$$
\[
J(\theta) = MSE(\theta) + \alpha \sum_{j=1}^n \theta_j^2
\]
$$

### 9. Ridge Regression Closed-Form Solution
$$
\[
\theta = \left( X^T \cdot X + \alpha A \right)^{-1} \cdot X^T \cdot y
\]
$$

### 10. Lasso Regression Cost Function
$$
\[
J(\theta) = MSE(\theta) + 2 \alpha \sum_{j=1}^n |\theta_j|
\]
$$

### 11. Subgradient Vector for Lasso Regression
$$
\[
g_j = \begin{cases} 
    \frac{\partial}{\partial \theta_j} MSE(\theta) + 2 \alpha \cdot \text{sign}(\theta_j) & \text{if } \theta_j \neq 0 \\
    \left[ -2 \alpha, 2 \alpha \right] & \text{if } \theta_j = 0
\end{cases}
\]
$$

### 12. Elastic Net Cost Function
$$
\[
J(\theta) = MSE(\theta) + r \alpha \sum_{j=1}^n |\theta_j| + \frac{1 - r}{2} \alpha \sum_{j=1}^n \theta_j^2
\]
$$

### 13. Logistic Regression Model Estimated Probability
$$
\[
p = h_{\theta}(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
\]
$$

### 14. Logistic (Sigmoid) Function
$$
\[
\sigma(t) = \frac{1}{1 + e^{-t}}
\]
$$

### 15. Logistic Regression Prediction with 50% Threshold
$$
\[
\hat{y} = \begin{cases} 
1 & \text{if } p \geq 0.5 \\
0 & \text{if } p < 0.5
\end{cases}
\]
$$

### 16. Logistic Regression Cost Function for a Single Instance
$$
\[
cost(h_{\theta}(x), y) = -y \log(h_{\theta}(x)) - (1 - y) \log(1 - h_{\theta}(x))
\]
$$

### 17. Logistic Regression Cost Function (Log Loss)
$$
\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta}(x^{(i)})) \right]
\]
$$

### 18. Partial Derivatives of the Logistic Cost Function
$$
\[
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\]
$$

### 19. Softmax Score for Class \( k \)
$$
\[
s_k(x) = \theta_k^T x
\]
$$

### 20. Softmax Function
$$
\[
\sigma(s(x))_k = \frac{e^{s_k(x)}}{\sum_{j=1}^K e^{s_j(x)}}
\]
$$

### 21. Softmax Regression Classifier Prediction
$$
\[
\hat{y} = \arg \max_k \sigma(s(x))_k
\]
$$

### 22. Cross Entropy Cost Function for Softmax Regression
$$
\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log(\hat{p}_k^{(i)})
\]
$$

### 23. Partial Derivatives of the Softmax Cost Function
$$
\[
\frac{\partial J(\theta)}{\partial \theta_{j,k}} = \frac{1}{m} \sum_{i=1}^m \left( \hat{p}_k^{(i)} - y_k^{(i)} \right) x_j^{(i)}
\]
$$
