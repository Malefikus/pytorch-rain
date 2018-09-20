# pytorch-rain
Simple convnet + fcnet model to predict one year's monthly rainfalls from the preceding 5 years.

# Prerequisites
Python 3.x, Pytorch 0.4.0+, numpy and scipy.

# Training
To start the training, simply run:
```
python example/train-rain.py -a fc
```
The detailed configurations are presented in example/train-rain.py.

# Results
The network gives about 62% average accuracy for the monthly rainfalls (with a threshold error of 150mm). 
