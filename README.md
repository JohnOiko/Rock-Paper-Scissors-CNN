# Rock-Paper-Scissors-CNN
A jupyter notebook project where RGB images depicting rock, paper or scissors moves are detected using an SVC, RF and a CNN. An agent which plays a kind of rock paper scissors game using the CNN is built and the game is simulated against a random agent.

# Datasets
- **First dataset**: The main dataset is an image dataset named Rock-Paper-Scissors Images which contains 726 pictures depicting the rock move, 712 depicting the paper move and 750 depicting the scissors move. All images include a had on top of a green background. The dataset was sourced from kaggle and can be found [here](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors). This dataset is used to train the three classifiers as well as to simulate the rock paper scissors game.
- **Second dataset**: The other dataset is another image dataset, named rock-paper-scissors, which contains 726 pictures depicting the rock move, 712 depicting the paper move and 750 depicting the scissors move. This dataset was also sourced from kaggle and can be found [here](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors). This dataset is used to test the accuracy of the best classifier on images that are similar, but not the same, to the ones it was trained with.

# Models tested
It is noted that the first two models were tested to compare for performance comparison with the third and most complex model.
- **Suport Vector Machine**: The first mode fitted and tested is a simple Support Vector classifier, whose implementation is provided by the Sklearn library. It's parameters have not been tuned as it is only used for comparison purposes.
- **Random Forest**: The second model is a Random Forest classifier, again implemented by using the applicable Sklearn class. It's parameters have also not been tuned for the aforementioned reasons.
- **Convolutional Neural Network**: The third model is a Convolutional Neural Network built using the Keras framework provided by Tensorflow. It makes use of early stopping with a batch size of 256 and up to 100 epochs.

The architecture of the **Convolutional Neural Network** is the following:
1. Two random flip layers, one vertical and one horizontal.
2. One Gaussian noise layer.
3. Two convolutional layers with dropout and pooling using a 4x6 kernel.
4. One flattening layer.
5. Five hidden dense layers.
6. One one-hot-encode output layer.

# Resizing and scaling
Each image was resized to smaller dimensions so that there are fewer features and the training and predictions of the model take less time to compute. The aspect ratio of the images remains unchanged to preserve all the information of the image samples.

The method of scaling the dataset applied it min max scaling. This is chosen as all samples are RGB images, thus each feature value's range is [0, 255], in which case min max scaling is the best scaling.

# Implementation steps
The notebook is split into the eleven following sectors:
1. **Import libraries**: Imports all the necessary libraries. It is noted that the method cv2_imshow is used instead of cv2.imshow as the latter does not work in Google Colab.
2. **Project variables**: Sets the preprocessing and game simulation variables.
3. **Useful functions**: Defines two necessary functions. The first takes an image as input and returns an image that has been flipped vertically, horizontally and has had white noise added to it, based on the given parameters. The second simulates the game and returns a list of the profit after each round.
4. **Dataset reading**: Reads the dataset from the dataset_directory folder and saves both the images and their labels in numpy areas. Each image is resized to the dimensions specified in sector two while maintaining its aspect ratio.
5. **Stratified train test split**: Splits the dataset into a train and test set using a 20% stratified train test split.
6. **Train and test sets distortion**: Randomly flips and adds white noise to both the train and test sets so that they can be used to train the SVC and RF classifiers. Only the distorted samples are kept, the original ones are removed.
7. **Min max scaling**: Applies min max scaling to both sets using Sklearns's MinMaxScaler. The scaler is first fit to the train set and then tranforms both sets.
8. **Train different models to predict the move each image depicts**: Fits the three models and measures their performance using Sklearn's classification report.
9. **Game simulation**: Simulates a game and prints the final profit/score.
10. **Total profit plot**: Plots the total profit at the end of each round.

# Dataset class sample examples
The following is a distorted resized sample of the rock class (labeled as 0):  
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/756d4b9c-fd6e-4005-bc93-27a2e7601624)

The following is a distorted resized sample of the paper class (labeled as 1):  
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/095a2280-cc03-49bf-a713-b0054465dba8)

The following is a distorted resized sample of the scissors class (labeled as 2):  
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/6a97fd77-c62b-4e34-88e8-d4a63b324774)

# Model metrics
## Support Vector Classifier
The following are the measured classification metrics of the SVC:
| Set   | Accuracy | Precision | Recall | F1   |
| ----- |:--------:| ---------:|-------:|-----:|
| Train | 0.97     | 0.97      | 0.97   | 0.97 |
| Test  | 0.93     | 0.93      | 0.93   | 0.93 |

## Random Forest
The following are the measured classification metrics of the RF:
| Set   | Accuracy | Precision | Recall | F1   |
| ----- |:--------:| ---------:|-------:|-----:|
| Train | 1.00     | 1.00      | 1.00   | 1.00 |
| Test  | 0.91     | 0.91      | 0.91   | 0.93 |

## Convolutional Neural Network
The following are the measured classification metrics of the CNN:
| Set   | Accuracy | Precision | Recall | F1   |
| ----- |:--------:| ---------:|-------:|-----:|
| Train | 1.00     | 1.00      | 1.00   | 1.00 |
| Test  | 0.91     | 0.91      | 0.91   | 0.93 |

# Metrics conclusions
As evident in the metrics above, the Convolutional Neural Network achieves the absolute best performance on the test set. As such, it is the only model that was used for simulating the game and the results of that simulation are presented and analyzed next. The Support Vector Classifier achieves the next best performance, beating out the Random Forest in the test set. The Random Forest clearly is overfitting the dataset as it gets a measurement of 1.00 in every classification metric of the train set, while each metric of the test set is at least 0.07 lower, signifying overfitting.

Tuning the parameters of the Support Vector Classifier and Random Forest could potential improve their performance, especially potentially reduce the Random Forest's overfitting, however the Convolutional Neural Network is better than both other models for the current problem, that there is no point in tuning the other two models. These two models were chosen alongside the Convolutional Neural Netowork as the Support Vector Classifier is a relatively simple model, while the Random Forest is an ensemble method that often gives good performance.

The Convolutional Neural Network was chosen as it is known that this type of Neural Network performs extremely well on image recognition, which is essentially the category of this problem.

# Game simultation results
