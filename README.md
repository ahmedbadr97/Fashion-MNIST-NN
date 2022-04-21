 # Fashion-MNIST-NN
- using neural networks via PyTorch to classify images of clothes


## contents
- [fashion_MNIST_nn notebook](notebooks/fashion_MNIST_nn.ipynb) contents
1. Dataset loading using custom pytorch custom dataset loader
2. view dataset
3. create the NN model
4. training and testing
5. inference and validation

 ## Dataset
![dataset-cover.png](dataset-cover.png)
- Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes
- dataset exists in [dataset directory](dataset)
  - two CSV files one for training and one for testing.
  - each file has rows of the images where col-1 represent the label of the image and the remaining 784 cols are the image pixels flattened 

### labels
- 0 T-shirt/top
- 1 Trouser
- 2 Pullover
- 3 Dress
- 4 Coat
- 5 Sandal
- 6 Shirt
- 7 Sneaker
- 8 Bag
- 9 Ankle boot