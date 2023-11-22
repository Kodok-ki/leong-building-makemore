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
