# leong-building-makemore See https://github.com/RigidSeine/leong-building-makemore
## A Recap of the makemore Multilayer Perceptrons (MLP) (Following makemore_rnn)

### 1. Importing the libraries
  - Torch
  - Torch.nn.functional #For importing the loss function (cross entropy)
  - Matplotlib.pyplot #for making figures

### 2. Creating the dataset
  - Makemore uses a list of names and uses individual letters of the alphabet as the predicted output of the neural net.
  - To make things easier, the alphabet plus an “end-character” (in this case a full-stop) encoded as numbers (1-27)
  - We decide on length of our blocks which determines the length of the context - i.e. how many characters do we take to predict the next one.
  - To do the actual building a context window is built for each letter of every word and inserted into the input and output datasets. To let the neural net know when a name stops, we append the end-character to the end of each name.

#### E.g. with a block size of 3, for the name Emma:
  - E is the first letter of the name so we construct a context window of [0,0,0] aka [...] and append that to our input data/features and we append 5 (since E is encoded as 5) into our output data/labels.
  - Then we slide our context window along so that it becomes [0,0, 5]
  - Then we repeat this: the ‘a’ at the end of Emma has context window: [5, 13, 13] and output ‘a’. However, the last letter is actually the ‘end-character’ (‘.’) so we use 5 context windows for ‘E’, ‘m’, ‘m’, ‘a’, ‘.’ Eventually, our neural net learns to establish probabilities for when these characters appear (and in the order that they do).
  - The shape of our dataset then becomes ```number of letters across all names + number of words x block size```. The number of words is added to the first dimension because all words need to have an end-character added to the end of it (so that the neural network learns when a name finishes!)
  - Then we divide our dataset into the training set (80%), dev set and test/validation set.
### 3. Embedding our inputs
  -  Following the paper - A Neural Probabilistic Language Model 2003 Bengio et al., we create a matrix of random numbers that allows us to embed our input into vectors.
  - How this works is that we represent each letter as a high-dimension vector. The embedding matrix then has the dimensions of:
```
Number of characters X number of dimensions
```
  - In this case, we have (26 alphabetical characters + 1 end-character (‘.’)) X 10 dimensions (because we’ve arbitrarily chosen 10).
  - In 2003 Bengio et al. the authors use 17,000 words so the matrix dimensions will be 17000 X number of embed dimensions.
  - This is more or less the concept of word embedding.
### 4. Structure of the Neural Net
  - The embedding is important because it constitutes one part of our Neural Net, namely a layer.
  #### Quick side spiel: The structure of our Neural Net comes in the form of Multilayer Perceptron (MLP) - a network containing an input layer (which constitutes training data during neural net training, and random numbers during sampling post-training), one or more intermediate layers of neurons (where all the weights and bias, i.e. the know-how of the neural net, is stored) and an output layer of neurons. The intermediate layers are also called hidden layers. A neuron comprises a weight, bias, gradient and activation function. The activation function is just a calculation performed on the input its given (either the raw data or processed data from a previous neuron) - at its simplest, it’s a linear function: y = mx +b, but for a neuron this will be f(w,x) = wx + b, where w is the weight, x is the input and b is the bias. The gradient is calculated with respect to the loss (i.e. a differential) and is used to adjust the weights and bias. The size of the gradient indicates how much effect the neuron has on the calculated loss. The loss is the difference between the predicted value and expected value.
  - The structure of our neural net goes like this:
   Input Layer -> Embedding Layer -> Linear layer -> Batch Normalisation Layer -> Tanh layer  -> Output Layer
- The Tanh layer introduces non-linearity to our neural network - this is important because without the non-linearity (and only linearity) our neural network becomes a fancy trendline i.e. it’s only capable of predicting linear relationships. Therefore the non-linearity allows for our neural network to learn non-linear relationships between the input data and the expected output. Tanh(x) is the hyperbolic tan function - it’s convenient because it returns a value between -1 and 1 (inclusive); -1 for x <= -𝛑 and 1 for x >= 𝛑.
### 5. Parameters
- Given the structure of our neural, we need parameters to model them physically (i.e. in maths and code).
- Parameters are the trainable components of a neural net, that start with seemingly random values and are gradually adjusted (during training) with calculus to generalise well to unseen data. For this example of makemore we’re training the weights and bias on real names so that network can generate outputs that are “name-like”.
- The primary kind of parameters are:
  - Weights: Mathematically, they’re the coefficients for our inputs in ```f(w,x) = wx + b```, w stands for weight here. Conceptually, they represent the strength of a connection between neurons within a neural net. A high weight means a neuron contributes greatly towards shaping the result produced by the next neuron.
  - Bias: Mathematically, the b in ```f(w,x) = wx + b```. They’re an additional offset for helping shape a model.
  - For makemore we have a couple more parameters
- The first layer after our Input Layer is the Embedding Layer, C which is initialised as a matrix with dimensions (```vocab_size x n_embd```) or (27 possible outputs x 10 embedding dimensions) of random integers.
- The next layer is a Linear Layer of neurons that perform the ```f(w,x) = wx + b``` activation function.
- For this we have W1, a matrix of weights, typically as a matrix with dimensions (```n_embd * block size x n_hidden```). N_hidden is the number of neurons we’ve decided to put in our hidden layer. Our neural net is also fully connected (which means each neuron in a prior layer will affect each neuron in the subsequent layer), which requires matrix multiplication between the layers we represent with matrices. The n_embd dimension of our Embedding doesn’t match up with our Linear Layer’s n_embd * block size dimension right now, but it will match up because our training data already has the block size defined within its shape.
- The other part of the Linear Layer is the bias, b1 which is a simple column vector of dimension n_hidden. Matrix addition requires two matrices to be of the same order (A: n x m, B: n x m) but the bias gets added to the wx by using PyTorch’s built-in broadcasting. See https://stackoverflow.com/questions/51371070/how-does-pytorch-broadcasting-work for diagrams for what the broadcasting looks like.
- Batch normalisation layer comprises primarily low dimensional parameters that are applied to W1. For this we use bngain and bnbias, both of which have dimensions ```1 x n_hidden```. This means they make use of PyTorch’s broadcasting. More info on this in the batch normalisation section.
- One thing we have to talk about is Kaiming or He initialisation - it’s a weight initialisation technique that [Kaiming He (He is the surname) and his team came up with](https://arxiv.org/abs/1502.01852v1). Its purpose is to normalise weights and biases at initialisation in a Gaussian manner to speed up training and reduce the likelihood of gradients being too small (and having little impact on training) or being too big (and creating an unstable neural net). It does so in two parts by 1) by introducing a gain which takes into account the type of activation that is used for the weights/bias and 2) a normalisation component that uses the number of input neurons and number of output neurons (fan in and fan out, respectively). For this neural net, we use tanh as our nonlinear activation so the gain is a multiplier of 5/3 - see https://pytorch.org/docs/stable/nn.init.html for more optimal gains for each activation. 
- Now for the final layers, tanh and output layer. The tanh layer itself is just applying the tanh() function to the inputs from the batch normalisation layer. The tanh layer then feeds into the output layer which has its own set of weights and bias, W2 and b2. Since these parameters represent our output layer it then follows that W2’s shape is then ```n_hidden x vocab_size``` and b2’s shape is ```vocab_size x 1 ```. It makes sense that our final dimension is the vocab_size since that represents the number of possible outputs - in other words, our output layer represents a probability distribution over 27 characters which means our neural network is trying to make a prediction on which character should come next in generating a name. Also, if you check the shape of the activation function (after doing all the matrix multiplication) it turns out the final shape is ```number of inputs x number of possible outputs``` which reconciles really beautifully. The output layer is a linear layer so the activation function ```f(w,b) = xW2 + b2``` applies here. ```x``` being the output of tanh.

### 6. Training 

