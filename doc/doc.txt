============
INTRODUCTION
============

"Intelligence and artificial intelligence is data compression through laws that predict based on data correlations."

Neural networks are part of machine learning which is part of AI which is part of computer science.

Neural networks become practical when data is not linearly separable and consists of lots of parameters,
practical for image recognition for example whereby at least each gray-scaled pixel forms a parameter input.
Can be used for both classification(choose a label) or regression(choose a quantity).

They consist of multiple node layers:
 - First layer takes parameter inputs and is also called input layer
 - Last layer making predictions also called output layer
 - Layers in between called deep layers and are optional. Once deep layers are introduced the data becomes non-linearly separable and the name "deep learning" can be used.

An AI will:
- Predict: Make a prediction based on input and given weights, in NN this process is called forward propagation
- Evaluate: Evaluate prediction compared to expected output, calculate total error and use cost function
- Adapt: Change weights of neural network to limit the total error, in NN this process is called backward propagation

===================
MATH IMPLEMENTATION
===================
Imagine a network with 4 layers, first is 0 and last 3

L layers
B bias, additional node in layer that is not connected to other nodes and always equals to one,
because it always equals to one we will use it to indicate its weight value that determines its final value
I inputs == L0
W equals to weights layer
Z equals layer/node output before activation function
A equals layer/node output after activation function
g equals activation function, output layer can have own activation function different from main activation function
g' equals activation function derivative
Y equals to expected output
Yhat or predicted output
D or delta is a measure of error for each layer's final activation value used in back-propagation
d indicates partial derivative which is same as gradient
@ equals dot product
TE is total error of NN output
.T is the vector/matrix transposed to allow for dot product calculations
C or cost function is used to compute total error
C' is used to compute derivative of cost function

Vectors representing complete layers can be used to make calculations more efficiently using numpy.

FORWARD PROPAGATION
-------------------

L0 = I

Z1 = (L0 @ W0) + B0
A1 = g(Z1)
L1 = A1

Z2 = (L1 @ W1) + B1
A2 = g(Z2)
L2 = A2

Z3 = (L2 @ W2) + B2
A3 = g(Z3)
L3 = A3

Yhat = L3
TE = C(Y, Yhat)

BACK PROPAGATION
----------------

D3 = C'(Y, Yhat) @ g'(Yhat)
dW2 = A2 @ D3
dB2 = D3

D2 = W2 @ D3 * g'(A2)
dW1 = A1 @ D2
dB1 = D2

D1 = W1 @ D2 * g'(A1)
dW0 = A0 @ D1
dB0 = D1

W0 -= dW0
D0 -= dD0
W1 -= dW1
D1 -= dD1
W2 -= dW2
D2 -= dD2

=========================
NEURAL NETWORK PARAMETERS
=========================

LAYERS & NODES
--------------
Contains at least an input layer and output layer. Deep layers sit in between. Each layer contains a certain amount of nodes.

If the data is linearly separable, you do not need any deep layers. Deep layers allow for non-linearity like polynomials would, when polynomials get too complicated neural networks come in. One layer is similar to linear/logistic regression without polynomials.

In general one hidden layer is sufficient for the majority of problems. The word deep learning a sub-field of machine learning refers to this.

More deep layers increase the complexity of the neural net which increases computational cost and slows down convergence, but they can improve precision, sometimes too much whereby they create overfitting if data is scarce.

For the number of nodes per layer a pyramid structure is used, whereby the number of nodes is highest at input each following deep layer is lower than the prior one and lowest at ouptut.

LEARNING RATE
-------------
Test to find out what learning rate is best, default learning rate used is 0,01.
Learning rate is denoted as alpha.

When alpha is too small algorithm needs to perform more steps until convergence and become slower.
When alpha is too big potentially no convergence or less precision as it will hover over the minima.

GRADIENT DESCEND
----------------
Gradient descend uses derivatives or slope of cost function to find the global minima in cost function to minimize the cost by going in opposite direction of gradient. Neural Networks uses partial derivatives for each weight and bias to minimize the error.

Stochastic:
Faster convergence on small datasets but slower on big datasets due to constant weight update
Can avoid local minimas or premature convergence but has higher variance in results due to randomness

Batch:
Slow but more computational efficient on big datasets
Stable convergence but risk of local minima or premature convergence

Mini-batch:
Mini-batch sits between stochastic and batch, trying to optimize benefits of both, and is the recommended variant of gradient descend.
b variable in NN holds size of batch, often 32 is used as default, some sources recommend number between 2 and 32...

ACTVATION FUNCTION
------------------
Linear: output -inf,inf
ReLU: rectified linear unit, output 0,+inf, less sensitive to vanishing gradient and non-relevant nodes, less computational cost, most used
Tanh: hyperbolic tangent function, output -1,1, could converge faster on larger dataset than sigmoid
Sigmoid: ouput 0,1
Softmax: vector total output = 1

~ OUTPUT LAYER ~
Regression -> Linear or ReLU
Binary classification or multiple classes with potential multiple correct answers -> sigmoid
Single answer possible from multiple classes -> softmax

~ DEEP LAYER ~
ReLU, Tanh or sigmoid
Can all be tried in following order: ReLu, Tanh, sigmoid

COST FUNCTION
-------------
Is used to calculate total error, total error is used to indicate NN performance and in back-propagation to adjust the weights and bias accordingly.
Regression -> mean square error (MSE)
classification -> cross entropy

WEIGHT & BIAS INIT
------------------
Weights  initialization is based on deep layer activation function:
ReLU -> He init
Tanh -> Xavier init
sigmoid -> random init (default init) -> between -1,1

Init to zero is also possible if bias is not equal to zero, but is not optimal.

Optimizing init is practical to fasten convergence by avoiding vanishing gradient problem.

Bias are usually init to 0, starting of neutral.

GRADIENT CHECKING
-----------------
Bugs can occur during your implementation of back-propagation, they can be subtle because the cost could properly descend but still the bug could lower the overall performance.

Gradient checking is used to debug back-propagation, by estimating them with numerical gradients(slope between two points around the one cost point) and comparing them with backpropagation gradients.

REGULARIZATION
--------------
Refers to all methods that limit over-fitting.
The most common ones are dropout method, L2-regularization and early stopping.

DROPOUT METHOD
--------------
Tries to reduce overfitting by temporarily removing (dropping out) certain nodes and all its associated connexions.
Can lead to the creation of situations whereby some nodes find themselves without the other ones and have to adapt, making the neural network more robust.

Can be implemented on all layers besides the output layer.
Dropout is only used during training.

Two hyper-parameters are used for drop-out:
- deep layers no-dropout -> range 0 - 1 -> thus one being no dropout and 0 being all dropout -> default between 0.5 - 0.8
- Input layer no-dropout -> range 0 - 1 -> thus one being no dropout and 0 being all dropout -> default is 0.8

L2-REGULARIZATION
-----------------
Works by reducing the weights on certain features, encouraging the model to use all of its input features equally.
Lambda is used to indicate regularization L2 strength, a value between 0 and 1 is used, 0 deactivates it.

EARLY STOPPING
--------------
Useful to avoid overtraining that can lead to overfitting.
While training neural network on training set cost of test/validation set is calculated too.
Once a trigger of diminishing performance for validation set (ex. cost of validation set starts to increase), the training stops.

Stopping too early can be bad as sometimes test set cost will increase for some time and decrease back afterward. Looking at graph without early stop can be interesting for this reason.
Validation hold outset, means waiting epochs until stopping with the goal of trying to capture costs that are descending back. Recommended is a validation hold outset of size 10% of training data set length.

Afterward right trigger must be used, this can be a cost function or validation function (also depending on goal of minimizing false negatives or positives).

MOMENTUM AND NESTEROV METHODS
-----------------------------
Momentum is an optimization method invented for reducing high variance in SGD, it does this through faster convergence, like a ball rolling down a hill.
Gamma/rho with default value 0.9 and value 0 to deactivate as momentum weight/neural network hyperparameter and makes use of EMAs (exponential moving average). 
It simply adds to weight updating for each weight -> + gamma * velocity
Velocity being a value starting  with zero and accumulating values that are re-used is equal to -> velocity - learning rate * gradient

Too high momentum can lead to the missing of local/global minima (which you do or do not want). If you do not want to miss it Nesterov method can be used who will slow down convergence when approaching a local minima.

ADAGRAD, ADADELTA, RMSPROP, ADAM, NADAM
---------------------------------------
AdaGrad adapts the learning rate for each parameter. Helps a lot when data is sparse and improves SGD robustness.
Main benefit is that it eliminates the need to manually tune the learning rate.
Main negative is that it ends up becoming very slow.
ADAdelta and RMSprop resolves this problem by limiting the learning rate smallness.

Adam (adaptive moment estimation) is another method that uses adaptive learning rates for each parameter.
Adam is a combination of momentum and ADAdelta.
Adam has been shown to work best compared to the other similar optimization algorithms.
Nadam (nesterov adaptive moment estimation) similar to Adam but uses nesterov momentum instead of simple momentum.

PARALLELIZING SGD
----------------
On large datasets SGD can be slow, running it asynchronously (multiple workers/threads) can speed it up.
Hogwild!, DownpourSGD, delay-tolerant algorithms, elastic averaging SGD are methods used to implement parallelized SGD.
Tensorflow also contain parallelized SGD.

GRADIENT NOSE
-------------
Adding nose to each gradient has been shown to make networks more robust towards poor initialization and increase the chance of escaping a local minima.

================
DATA PREPARATION
================

y or predicted values and x or features should be separated.

NORMALIZATION
------------
Normalization refers to reducing scale of data and leads to all features being on same scale, elimination of outliers and decreases computational costs.
min-max normalization: When we do not want impact of outliers
z-score normalization: When we do want impact of outliers, also avoid problem whereby different data has different max values

DATA SPLITING
-------------
Data can further be split into training data and test data (0.8 - 0.2 recommended ratio), to verify overfitting. Also possible training, test and validation set (0.6, 0.2, 0.2).

Relaunching fit function multiple times, to find good random splitting for data splitting but potentially weight init too, is possible.

DATA TEXT TO NUMBERS
--------------------
Features with textual data can be converted into numerical data, each label takes different number.

EXPECTED/Y DATA
---------------
If your NN has multiple output nodes, the y or expected values column, should be transformed not in single values but in vectors with same size as output nodes.

------------------
DATA VISUALIZATION
------------------

DESCRIBE
--------
Describe function goes over each feature in data and looks at different analytical parameters:
-See if data is correct in terms of numbers, are there missing values?
-See if data needs to be normalized? Big values or already small? Different features same scale? Alot of outliers?

*Skewness, gives normal distribution of values or measure of symmetry, 0 is symmetrical, +0.5 / -0.5 moderately skewed, +1/-1 is highly skewed, skewed data indicates outliers in tail region.
*Kurtosis result high number means the dataset has lots of outliers, outliers can be good or not, if not they can be removed or min-max normalization can be used

PAIRPLOT
--------
Pair-plots compares two features over the different classes, in a line plot and scatterplot:
-Scatterplots are useful to find correlations and homogeneity between two features.
If two features are homogenous, one of them has low predictive power and can be eliminated.
-Line plots are useful to find correlations between classes in one feature
Features that are homogenous or have low variation over the classes are not interesting for AI neither as they have low predictive power.

==============================
DEBUGGING A LEARNING ALGORITHM
==============================

EVALUATION
----------
There are two types of errors that occur in a classification algorithm
False positives, or predicting "yes", while the expected answer was "no"
False negatives, or predicting "no", while the expected answer was "yes"

In sigmoid probabilities to answer functions, you can for example change the "division point" to aim more at minimizing one error or the other.

Different measures are used:
Accuracy score: Gives % of correct answers
precision score: Appropriate when trying to minimize false positives
Recall score: Appropriate when trying to minimize false negatives
f1 score: Is combination of precision and recall, used when trying to maximize both false positives and negatives
Confusion matrix: Gives an overview of both false negatives and positives

OVERFITTING
-----------
HIGH VARIANCE: High variance between training sets, means very precise on each training sets, leads to overfitting.
-> Increasing regularization can lower high variance
-> Smaller sets of features can lower high variance
-> More training data

UNDERFITTING
------------
HIGH BIAS: Bias acts as strong suggestor, suggesting too much can lead to under-fitting on test sets
-> Decreasing regularization can lower high bias
-> Extra features can lower high bias
-> Adding polynomials or deep layers can lower high bias

OTHER PROBLEMS
--------------
Vanishing gradient problem -> small values are slow to change/learn, leading to no/slow convergence, problem when weights are initialized to zero for example
local/global minima -> Gradient descend weak point is to get stuck in the local minima instead of continuing towards the global minima as it can difficultly know when it arrived at the global minima or not. Local minima are low cost points whereby the cost increases afterward, but later on decrease even more to a potential global minima, global minima being the lowest cost point.
non-relevant nodes -> Some nodes that are not relevant should be deactivated by the activation function setting its value to 0. ReLU does this best. Proper data features selection helps.

=======
SOURCES
=======

Neural networks, a visual introduction for beginners - Michael Taylor

https://www.coursera.org/learn/machine-learning - Andrew Ng
https://course.elementsofai.com

https://towardsdatascience.com
https://machinelearningmastery.com
https://ruder.io
https://www.machinecurve.com
https://www.deeplearning.ai
http://neuralnetworksanddeeplearning.com
https://medium.com/@ODSC
https://mlfromscratch.com/
