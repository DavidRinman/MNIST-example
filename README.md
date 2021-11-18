# MNIST-example
Example of classifying the MNIST dataset of handwritten digits using ML. For good overview and workflow when using the code, it is split into four different files:

**training.py**
Trains a model and saves the weights and training history to disk.

**evaluation.py**
Loads a trained model and evaluates it on the test dataset. This includes calculating loss and accuracy, as well as plotting the training history and some examples of misclassified digits. 

**models.py**
Contains models, in this case the single CNN model.

**utils.py**
Used for various helper functions. In this case, this only includes a function that converts a sparse category representation to a one-hot representation.  
