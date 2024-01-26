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
The following is a distorted resized sample of the rock class:  
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/756d4b9c-fd6e-4005-bc93-27a2e7601624)

The following is a distorted resized sample of the paper class:  
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/095a2280-cc03-49bf-a713-b0054465dba8)

The following is a distorted resized sample of the scissors class:  
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/6a97fd77-c62b-4e34-88e8-d4a63b324774)



