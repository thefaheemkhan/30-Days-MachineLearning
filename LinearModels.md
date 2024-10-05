Here is a list of the equations found in Chapter 4 of the provided PDF:

1. **Equation 4-1**: Linear regression model prediction.

   $$\[
   \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
   \]$$

   (The prediction is a weighted sum of the input features.)

3. **Equation 4-2**: Linear regression model prediction (vectorized form).

   $$\[
   \hat{y} = \theta^T \cdot x
   \]$$
   
   (This is a dot product between the parameter vector and the feature vector.)

5. **Equation 4-3**: Mean Squared Error (MSE) cost function for a linear regression model.

   $$\[
   MSE(\theta) = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right)^2
   \]$$
   
   (Where \( \hat{y}^{(i)} \) is the predicted value and \( y^{(i)} \) is the actual value.)

7. **Equation 4-4**: Normal equation.

   $$\[
   \theta = \left( X^T \cdot X \right)^{-1} \cdot X^T \cdot y
   \]$$
   
   (Gives the optimal parameters for a linear regression model.)

9. **Equation 4-5**: Partial derivative of the MSE cost function with respect to \( \theta_j \).
   \[
   \frac{\partial MSE}{\partial \theta_j} = \frac{2}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}
   \]
   (Used for gradient descent optimization.)

10. **Equation 4-6**: Gradient vector of the cost function.
   \[
   \nabla_{\theta} MSE(\theta) = \frac{2}{m} X^T \cdot \left( X \cdot \theta - y \right)
   \]
   (Computes the gradient for all parameters at once.)

11. **Equation 4-7**: Gradient descent update rule.
   \[
   \theta = \theta - \eta \cdot \nabla_{\theta} MSE(\theta)
   \]
   (Where \( \eta \) is the learning rate.)

12. **Equation 4-8**: Ridge regression cost function.
   \[
   J(\theta) = MSE(\theta) + \alpha \sum_{j=1}^n \theta_j^2
   \]
   (Adds an L2 regularization term to linear regression.)

13. **Equation 4-9**: Ridge regression closed-form solution.
   \[
   \theta = \left( X^T \cdot X + \alpha A \right)^{-1} \cdot X^T \cdot y
   \]
   (Where \( A \) is the identity matrix, except the top-left corner is 0.)

14. **Equation 4-10**: Lasso regression cost function.
    \[
    J(\theta) = MSE(\theta) + 2 \alpha \sum_{j=1}^n |\theta_j|
    \]
    (Adds an L1 regularization term to linear regression.)

15. **Equation 4-11**: Subgradient vector for Lasso regression.
    \[
    g_j = \begin{cases} 
    \frac{\partial}{\partial \theta_j} MSE(\theta) + 2 \alpha \cdot \text{sign}(\theta_j) & \text{if } \theta_j \neq 0 \\
    \left[ -2 \alpha, 2 \alpha \right] & \text{if } \theta_j = 0
    \end{cases}
    \]
    (Used to handle non-differentiability of the cost function at \( \theta_j = 0 \).)

16. **Equation 4-12**: Elastic Net cost function.
    \[
    J(\theta) = MSE(\theta) + r \alpha \sum_{j=1}^n |\theta_j| + \frac{1 - r}{2} \alpha \sum_{j=1}^n \theta_j^2
    \]
    (Combines both L1 and L2 regularization terms.)

17. **Equation 4-13**: Logistic regression model estimated probability.
    \[
    p = h_{\theta}(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
    \]
    (The logistic (sigmoid) function for estimating probabilities.)

18. **Equation 4-14**: Logistic (sigmoid) function.
    \[
    \sigma(t) = \frac{1}{1 + e^{-t}}
    \]
    (Defines the logistic transformation.)

19. **Equation 4-15**: Logistic regression model prediction using a 50% threshold probability.
    \[
    \hat{y} = \begin{cases} 
    1 & \text{if } p \geq 0.5 \\
    0 & \text{if } p < 0.5
    \end{cases}
    \]
    (Classifies instances based on estimated probabilities.)

20. **Equation 4-16**: Cost function for a single training instance (logistic regression).
    \[
    cost(h_{\theta}(x), y) = -y \log(h_{\theta}(x)) - (1 - y) \log(1 - h_{\theta}(x))
    \]
    (Measures the error of a prediction for logistic regression.)

21. **Equation 4-17**: Logistic regression cost function (log loss).
    \[
    J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta}(x^{(i)})) \right]
    \]
    (Averaged cost over all training instances.)

22. **Equation 4-18**: Partial derivatives of the logistic cost function.
    \[
    \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
    \]
    (Used to update parameters during gradient descent.)

23. **Equation 4-19**: Softmax score for class k.
    \[
    s_k(x) = \theta_k^T x
    \]
    (The score for each class in softmax regression.)

24. **Equation 4-20**: Softmax function.
    \[
    \sigma(s(x))_k = \frac{e^{s_k(x)}}{\sum_{j=1}^K e^{s_j(x)}}
    \]
    (Computes the probability for each class in softmax regression.)

25. **Equation 4-21**: Softmax regression classifier prediction.
    \[
    \hat{y} = \arg \max_k \sigma(s(x))_k
    \]
    (Predicts the class with the highest estimated probability.)

26. **Equation 4-22**: Cross entropy cost function for softmax regression.
    \[
    J(\theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log(\hat{p}_k^{(i)})
    \]
    (Measures the error for a softmax classifier’s predictions.)

27. **Equation 4-23**: Partial derivatives of the softmax cost function.
    \[
    \frac{\partial J(\theta)}{\partial \theta_{j,k}} = \frac{1}{m} \sum_{i=1}^m \left( \hat{p}_k^{(i)} - y_k^{(i)} \right) x_j^{(i)}
    \]
    (Gradient for updating parameters in softmax regression.)

These equations play a fundamental role in understanding and implementing different machine learning algorithms discussed in this chapter.
