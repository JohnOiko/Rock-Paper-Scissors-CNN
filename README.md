A jupyter notebook project where RGB images depicting rock, paper or scissors moves are detected using an SVC, RF and a CNN. An agent which plays a kind of rock paper scissors game using the CNN is built and the game is simulated against a random agent.

# Datasets
- **First dataset**: The main dataset is an image dataset named Rock-Paper-Scissors Images which contains 726 pictures depicting the rock move, 712 depicting the paper move and 750 depicting the scissors move. All images include a had on top of a green background. The dataset was sourced from kaggle and can be found [here](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors). This dataset is used to train the three classifiers as well as to simulate the rock paper scissors game.
- **Second dataset**: The other dataset is another image dataset, named rock-paper-scissors, which contains 500 pictures depicting the rock move, 500 depicting the paper move and 500 depicting the scissors move. The original dataset contains even more samples per class, but in order to reduce its size only the first 500 samples of each class were used. This dataset was also sourced from kaggle and can be found [here](https://www.kaggle.com/datasets/yash811/rockpaperscissors/data). This dataset is used to test the accuracy of the best classifier on images that are similar, but not the same, to the ones it was trained with.

# Models tested
It is noted that the first two models were tested for performance comparison with the third and most complex model.
- **Support Vector Machine**: The first mode fitted and tested is a simple Support Vector classifier, whose implementation is provided by the Sklearn library. It's parameters have not been tuned as it is only used for comparison purposes.
- **Random Forest**: The second model is a Random Forest classifier, again implemented by using the applicable Sklearn class. It's parameters have also not been tuned for the aforementioned reasons.
- **Convolutional Neural Network**: The third model is a Convolutional Neural Network built using the Keras framework provided by Tensorflow. It uses a batch size of 256 and is trained for 70 epochs.

The architecture of the **Convolutional Neural Network** is the following:
1. Two random flip layers, one vertical and one horizontal.
2. One Gaussian noise layer.
3. Two convolutional layers with dropout and pooling using a 4x6 kernel.
4. One flattening layer.
5. Five hidden dense layers.
6. One one-hot-encode output layer.

# Resizing and scaling
Each image was resized to smaller dimensions so that there are fewer features and the training and predictions of the model take less time to compute. The aspect ratio of the images remains unchanged to preserve all the information of the image samples.

The method of scaling applied to the dataset is min max scaling. This is chosen as all samples are RGB images (or grayscale if selected), thus each feature value's range is [0, 255], in which case min max scaling is the best scaling.

# Implementation steps
The notebook is split into the eleven following sectors:
1. **Import libraries**: Imports all the necessary libraries. It is noted that the method cv2_imshow is used instead of cv2.imshow as the latter does not work in Google Colab. If this notebook is run outside of Google Colab, the import of the cv2_imshow must be deleted and its calls must be replaced by cv2.imshow.
2. **Project variables**: Sets the preprocessing and game simulation variables.
3. **Useful functions**: Defines two necessary functions. The first takes an image as input and returns the given image after randomly flipping it vertically, then horizontally and adding white noise to it, based on the given parameters. The second simulates the game and returns a list of the total profit after each round.
4. **Dataset reading**: Reads the dataset from the dataset_directory folder and saves both the images and their labels in numpy arrays. Each image is resized to the dimensions specified in sector two while maintaining its aspect ratio.
5. **Stratified train test split**: Splits the dataset into a train and test set using a 20% stratified train test split.
6. **Train and test sets distortion**: Randomly flips and adds white noise to both the train and test sets so that they can be used to train the SVC and RF classifiers. Only the distorted samples are kept, the original ones are removed.
7. **Min max scaling**: Applies min max scaling to both sets using Sklearns's MinMaxScaler. The scaler is first fit to the train set and then tranforms both sets.
8. **Train different models to predict the move each image depicts**: Fits the three models and measures their performance using Sklearn's classification report.
9. **Game simulation**: Simulates a game and prints the final profit/score.
10. **Total profit plot**: Plots the total profit at the end of each round.
11. **Secondary dataset classification report**: Reads the secondary dataset and tests the Convolutional Neural Network's performance on its samples.

# Dataset class sample examples
The following is a distorted resized sample of the rock class (labeled as 0):  
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/756d4b9c-fd6e-4005-bc93-27a2e7601624)

The following is a distorted resized sample of the paper class (labeled as 1):  
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/095a2280-cc03-49bf-a713-b0054465dba8)

The following is a distorted resized sample of the scissors class (labeled as 2):  
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/6a97fd77-c62b-4e34-88e8-d4a63b324774)

# Model metrics
All the metrics mentioned are the macro variants, as each class is equally important in this problem.

## Support Vector Classifier
The following are the measured classification metrics of the SVC:
| Set   | Accuracy | Precision | Recall | F1   |
| ----- |:--------:| ---------:|-------:|-----:|
| Train | 0.97     | 0.98      | 0.97   | 0.97 |
| Test  | 0.92     | 0.93      | 0.92   | 0.92 |

## Random Forest
The following are the measured classification metrics of the RF:
| Set   | Accuracy | Precision | Recall | F1   |
| ----- |:--------:| ---------:|-------:|-----:|
| Train | 1.00     | 1.00      | 1.00   | 1.00 |
| Test  | 0.91     | 0.91      | 0.91   | 0.91 |

## Convolutional Neural Network
The following are the measured classification metrics of the CNN:
| Set   | Accuracy | Precision | Recall | F1   |
| ----- |:--------:| ---------:|-------:|-----:|
| Train | 1.00     | 1.00      | 1.00   | 1.00 |
| Test  | 0.98     | 0.98      | 0.98   | 0.98 |

# Metrics conclusions
As evident in the metrics above, the Convolutional Neural Network achieves the absolute best performance on the test set. As such, it is the only model that was used for simulating the game and the results of that simulation are presented and analyzed next. The Support Vector Classifier achieves the next best performance, beating out the Random Forest in the test set. The Random Forest clearly is overfitting the dataset as it gets a measurement of 1.00 in every classification metric of the train set, while each metric of the test set is 0.09 lower, signifying the existence of at least some overfitting.

Tuning the parameters of the Support Vector Classifier and Random Forest could potential improve their performance, especially potentially reduce the Random Forest's overfitting, however the Convolutional Neural Network is better than both other models for the current problem, that there is no point in tuning the other two models. These two models were chosen alongside the Convolutional Neural Netowork as the Support Vector Classifier is a relatively simple model, while the Random Forest is an ensemble method that often gives good performance.

The Convolutional Neural Network was chosen as it is known that this type of Neural Network performs extremely well on image recognition, which is essentially the category of this problem.

# Game simulation results
## Total profit
At the end of the 1000th round, the total profit was 979 using the Convolutional Neural Network as the player's model.

## Total profit plot
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/19b982bb-bf38-4752-9ca3-c64dbfd54392)  
The above plot showcases the total profit of the CNN player during the simulation of 1000 rounds.

## Samples resulting to losses
The following samples resulting in the agent losing to the random agent:  
![Untitled-1](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/b8438caf-17eb-4565-8a11-b7a125c91b3b)
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/46b59220-b0cb-4c85-ba5c-ebcd65c5006b)
![Untitled-1](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/b671802a-887c-4fed-b1d0-f7b88c0a9cd4)
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/0e8162fe-9d73-44e6-a128-46c2e1c12496)
![Untitled-1](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/225732b6-d222-47ed-af16-de26ba47e7ac)

## Samples resulting to ties
The following samples resulting in the agent tying with the random agent:  
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/80c8155d-b3f4-4fb2-8e7b-9f8260fb5df9)
![Untitled-1](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/3424a7e6-353d-45e7-92bc-289c6269e1b5)
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/2c24abe5-3a58-4d52-adb7-43cda8ff0511)
![Untitled-1](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/9df3edb8-2c1d-4a0b-9ded-889b6319763f)
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/815a5c39-bfd6-450a-b481-a15073b7ac28)
![Untitled-1](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/13bfba21-1a38-4247-a243-5f0babbcae83)
![Untitled](https://github.com/JohnOiko/Rock-Paper-Scissors-CNN/assets/72659858/7eb30eaf-f1be-4eab-9337-2a514296278f)

# Game simulation conclusions
As we can see in the previous samples, the Convolutional Neural Network player loses when the random agent plays the move rock and the fist of the hand is not fully closed. This happens because the model detects the move as paper, while the actual random agent's move is rock. I did not manage to find a way to improve this without increasing the training time of the model significantly. Similarly, when the hand in the image sample is dressed with dark clothing, or has a strong shadow around the wrist, the same mistake is made. One way to combat this could be to use the grayscale instead of the RGB color space, however I tester the same models with grayscale and the results were worse.

Additionally, the player ties with the random agent in two cases. The first is when the random agent plays the move scissor and the hand's ring and little fingers are not fully closed, resulting in the model detecting the move as paper instead of scissors. The second case is when the random agent plays the move paper, with the palm being in a vertical position instead of a horizontal one, but the Convolutional Neural Network detects it as rock. This presumably happens because the palm is shaped like a fist when in the vertical position and the model cannot detect the fingers. Again, I did not manage to find a way to improve these issues without increasing the training time of the model significantly.

# Secondary dataset CNN metrics
The following are the measured classification of the Convolutional Neural Network on the image samples of the secondary neural network.
| Set   | Accuracy | Precision | Recall | F1   |
| ----- |:--------:| ---------:|-------:|-----:|
| Train | 0.34     |    0.41   | 0.34   | 0.18 |

# Secondary dataset CNN metrics conclusion
As evident in the above measurements, the Convolutional Neural Network is not nearly as good at identifying the image samples of a different, yet similar dataset. The secondary dataset's images are a lot less uniform than the first dataset's, as they include many different hand positions and background colors. As such, the model struggles to correctly identify them.

Additionally, the scissors class is not identified at all, scoring 0 on every metric measurement, meaning the model cannot differentiate its samples from the samples of the other classes. This is reflected in the macro f1 score, as it is only 0.18, which is half or lower than the other three metric scores.












