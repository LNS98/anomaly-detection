# Anomaly App Detection

**K-means clustering classifier which identifies attacks on a robotic vehicle.** 

## Repo Guide

A brief explanation of the important files:

- `build_app.sh` - Shell scirpt command to build and start docker container for app. 
    - NOTE: If docker container is already built, run `docker start anomaly-detection-app` to restart the app.

- `/flask-app/api.py` - Server code written in flask which queries the model for predictions.

- `/flask-app/model.py` - Contains the model class used by the app to make predictions.

- `/flask-app/tests/` - Contains simple tests to make sure that the app works as expected.

- `/scripts/data_investigation.py` - Investigation of the data to visually and manually check for any noticable differences.

- `/scripts/feature_selection.py` - Investigation of the most important features, selected by validating a k-means (k=2) on different features in the dataset, using the ground truth label.
    - NOTE: The content of this file is similar to the object-oriented Model class in `/flask-app/model.py`.

- `/data/` - Location of any datasets used.
    - NOTE: Main dataset used for training has been removed.


## Using Repo

To activate App:
    
    - Run `sh build_app.sh` and wait for Docker container to start running.
    - The app can be tested by running `sh /flask-app/tests/test_app.sh` which runs a sample to the back-end. A prediction is returned when running this command. 

Both files in the scripts folder can be run by running `python name_of_file.py`. NOTE: for this to run one has to have all the modules from the `/flask-app/requirements.txt` file installed.


## Discussion 

### Data Investigation

 Before investigating any potential models which could be used, checks were performed on the data. It was found that the data contains 8 feature classes and a label class, all with only numerical features. Further analysis showed that no missing data is present and that the classes contain sensible values, for example, no negative values (or ones over 100) can be found for the CPU usage column, which implied no significant need for data cleaning. Finally, what can be noticed is that since the time-step parameter has no influence on the data and is just an index parameter this was dropped for this simple analysis. Perhaps more complex analysis could take account of a temporal fact to determine whether an attack has been made. However, for this model no temporal application was taken into account.

Secondly, class distributions of the labels were conducted to ensure that the classes are balanced, as this would have to be accounted for both in the training and evaluation procedures. Once again, a quick plot of the data suggests that the data is well distributed.

Finally, differences between the two class labels were investigated, to find any obvious variations between the classes. What was seen from this analysis is that there is significant difference between two main features: 'RxKBTot' and 'TRxKBTot', highlighting their likely importance. This make sense as for an intrusion to occur an unexpected amount of data must flow. Following this analysis, it was decided to investigate creating new features in conjunction with these two columns as they presented the most likely success for improving predictive power of any classifier built

### Model Selection

Whilst there is a variety of different unsupervised techniques, one of the most simple, yet effective is clustering. Several factors which shall now be discussed provide the reason for selecting a simple k-means (k=2) for classifying, in an unsupervised way, the data. Firstly, given that the data is fully numerical, a clustering approach such as k-means, would work successfully since distances can be easily computed. Secondly, given a binary label, 'k' is pre-determined, and as such, no validation method is necessary in order to try and find 'k'. Furthermore, such methods have the advantages that they are much interpretable compared to more complex deep learning models. Finally, another limitation of choosing a more complex deep learning method, would be the lack of a very big dataset, which would likely lead to similar performance compared to simpler methods. As such the final model was chosen to be k-means, with k=2. Given the nature of this algorithm, several pre-processing steps must be conducted to successfully implement this model. Most importantly, the features have to be normalised, such that each dimension is equally contributing to the clusters and no features over-dominates, just because of larger scales. In addition to this, feature selection was conducted and evaluated as described below. Finally, k-means, has some inherent stochasticity related to the initial cluster locations, as such the same data might lead to different clusters which should be taken into account when evaluating different models.

### Model Validation

When validating different model following an unsupervised technique, the dataset was divided only into validate and test. This is because training can be made on the validation set itself. The test set is still necessary to get an unbiased evaluation of the results such that one does not overfit to the validation result. As such the dataset is divided into validate and test (0.8/0.2 split) on which model selection was performed. With regard to what to vary for the selected model, no real hyper-parameters exist for the k-means model. Therefore, the model was validated against different types of features used. k-means was run with different combinations of features, then it was evaluated using the labels from the dataset. The algorithm was run several times to reduce any potential randomness, then the scores were averaged. The highest scoring model was selected ONLY if a standard deviation of under 0.2 was seen in the trials. This is done to decrease randomness due to the algorithm and to increase robustness on unseen data. Notably, 0.2 is chosen without a real criteria - simply from looking at different std values and as such a more meaningful methodology for choosing this number could be made. The best performing features and then saved.

In conclusion, the final model pipeline augments an incoming dataframe by adding the engineered features, selecting only the best features, normalising the data (as per a scaler fitted to the original training data) and finally clustered according to the previosuly trained k-means algorithm.

### Results

When running the evaluation model, what was confirmed was the original intuition that the 'RxKBTot', 'TxKBTot', 'WriteKBTot' and the 'CPU' contributing the most to the accuracy. Furthermore, the engineered features seem to add predictive power to the accuracy resulting in better results than when not added. This can be seen as the best model includes some of the new features in its final model. Encouragingly, there is no significant difference between the accuracy on the validation and test set, highlighting that the model is not overfitting to the validation set.

## Flask App

The final model was then refactored in a object-oriented way and made into a class. This makes the model much more usable for production purposes such that methods, for example for predicting, can be easily called from the class. Flask was used to create the back-end for the app as it is one of the most widely used frameworks built for the python language. In the main api file, a 'predict' function is used to POST predicitions made from the model after data, fed in JSON format, is posted to the app.
The app is built using docker, such that is can be easily deployed and built. To build the app one can simply run the shell comand found in the home of the directory.

### Testing 
The app is tested some sample data taken from the original dataset and converted into a JSON format. More tests should be done in the future to make sure the app performs robustly against different inputs.

