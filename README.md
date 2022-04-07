# Identify plane from pictures 

Classify type of planes from random pictures. 
This would be visible online thanks to an application "Streamlit". 
## Data base 
```
https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-air 
```
## Imports 
```
import keras
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
#from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import datetime 
```

## Contributing 
Choose the dimensions (128,128) for each image, so then we are able to read the picture and constitute our learning base. 
We can write the learning code of the model and use TPU for the learning. 
Classify all test section (where we read the text/ picture and applied the learning model in order to measure the performance).
On Anaconda Prompt, we launch Jupyter and Streamlit. 

## License 
[MIT] (https://choosealicense.com/licenses/mit/#)
