# leong-building-makemore
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
### 3. Embedding our inputs and initialising the parameters
  -  Following the paper - A Neural Probabilistic Language Model 2003 Bengio et al., we create a matrix of random numbers that allows us to embed our input into vectors.
  - How this works is that we represent each letter as a high-dimension vector. The embedding matrix then has the dimensions of:
```
Number of characters X number of dimensions
```
  - In this case, we have (26 alphabetical characters + 1 end-character (‘.’)) X 10 dimensions (because we’ve arbitrarily chosen 10).
  - In 2003 Bengio et al. the authors use 17,000 words so the matrix dimensions will be 17000 X number of embed dimensions.
  - This is more or less the concept of word embedding.


