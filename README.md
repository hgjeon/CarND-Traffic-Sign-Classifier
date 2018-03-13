

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./images/graph_dataset.png "Visualization"
[image2]: ./images/preprocessor.png "Preprocessor"
[image3]: ./images/Augmented.png "Augmented"

[image4]: ./images/graph_aug_dataset.png "Graph Aug Dataset"
[image5]: ./images/sign_from_web.png "GermanSign"

[image6]: ./images/ProbSign1.png "GermanSign1"
[image7]: ./images/ProbSign2.png "GermanSign2"
[image8]: ./images/ProbSign3.png "GermanSign3"
[image9]: ./images/ProbSign4.png "GermanSign4"
[image10]: ./images/ProbSign5.png "GermanSign5"

[image11]: ./images/AccuracyPlot2.png "Accuracy Plot"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README


### Data Set Summary & Exploration

#### 1. Basic Summary of the Data Set

I used the numpy and panda library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data set has been used for each Training, Validation, and Test sets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing of the Image Data, and Techniques
<!--
####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
-->

As a first step, I decided to convert the images to grayscale, then applied histogram equalization technique. As a last step, I normalized the image data because this allows different features in the same scale, such that optimization can speed up.
1. Convert RGB image to grayscale
    - Grayscale conversion will preserve the salient features in a image while it can speed up the training speed significantly
2. Applied Histogram equalization
    - Typical image processing technique which can enhance the biased histogram.
    - However, I could not see the significant improvement in the training accuracy.
    - Possibly, I could use this to generate augmented training image instead.
3. Normalized the image data
    - Normalized scale from [0 : 255] to [0.0 : 1.0]

Here is an example of a traffic sign image. The first image is original image, the second is after grayscaling, and the third is after histogram equalization.

![alt text][image2]

#### 2. Augmented Training Data

Typically, number of feature sets will increase accuracy. The augmented image can be generated using additional noise using salt-and-pepper filter, blur filter, etc. Since the preprocessing already includes histogram equalization, contrast or histogram transformation may not increase the effective training data sets much.

Here is an example of an original image and an augmented image using median Blur filter.

![alt text][image3]

In this project, however, I could not see significant accuracy improvement using this particular augmented data set, while the training time increased significantly, then the augmented data set is not used in this project, and I still could achieve the >93% accuracy with given feature image set.

![alt text][image4]

#### 3. Final Model Architecture

<!--
####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
-->

My final Convolutional Neural Network model is based on LeNet-5 model. It is consisted of the following layers in detail:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x12 	|
| RELU, Dropout			| KeepDrop 67%									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x32 	|
| RELU, Dropout			| KeepDrop 67%									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 					|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x12 	|
| RELU, Dropout			| KeepDrop 67%									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Flatten               | etc.      									|
| Fully connected		| Input 800, Output 240        					|
| RELU, Dropout			| KeepDrop 67%									|
| Fully connected		| Input 240, Output 168        					|
| RELU, Dropout			| KeepDrop 67%									|
| Fully connected		| Input 168, Output 43 (n_classes) 				|
| Softmax				| etc.        									|


#### 4. Training Model
<!--
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
-->

The detail training specifications are described below to train the model:
- Optimizer: Adam Optimizer
- Batch: Mini-Batch using batch size of 128
- Number of Epochs: 20
- Training Rate: 0.001

#### 5. Approach for Better Accuracy
<!--
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
-->

My final model results were:
* Training set accuracy of 0.993%
* Validation set accuracy of 0.959%
* Test set accuracy of 0.936%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    - LeNet-5 model with RGB image has been tried first
* What were some problems with the initial architecture?
    - Slow Training Speed with RGB image
    - Overfit: While Training accuracy approaches to 99%, validation accuracy was below 90%
    - Other bug in the model: Sometimes, even with a bug, training still works, even though accuracy may not be accomplished higher enough
* How was the architecture adjusted and why was it adjusted?
    - Basic model LeNet-5 model: a well known powerful image classification
    - Convolution: increased (double) depth, keep the same width and hight
    - Activation function: RELU function
    - Dropout: 1/3 of dropout improved overfit issue, however, EPOCH has to be increased with slow training speed
    - Pooling: stride of 2 in width, and hight
    - Mini-Batch Training
* Which parameters were tuned? How were they adjusted and why?
    - EPOCH: increased from 10 to 20 due slower training speed caused by dropout
    - TRAINING_RATE: 0.001 (tried others but 0.001 works pretty well)
    - Batch Size: Not tuned
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    - Dropout intentionally skip the random nodes in the layer. This prevent the overfitting during training. However, once the training is done, this dropout should not be applied during validation procedure.
    - Observed that 1/3 of dropout (keep_drop = 0.67) in my model improves accuracy of about 5% in addition.  

If a well known architecture was chosen:
* What architecture was chosen?
    - Based on modified LeNet-5
* Why did you believe it would be relevant to the traffic sign application?
    - Convolutional Neural Network with multiple layer works pretty well for the image classifications
    - In general, 1st layer will recognize the simple patterns, such as line, circle, triangle, and so on. Then, it recognizes higher level traffic sign patterns toward deeper layers.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    - Validation set has been used to verify whether training has overfitting or underfitting
    - Validation set is not used for training itself
    - Finally, test set has been used for final accuracy measurement

The both training and validation accuracies have been plotted for each iteration.

![alt text][image11]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web
<!--
#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
-->

Here are five other German traffic signs that I found on the web:

![alt text][image5]

The first and third image might be difficult to classify because many other images with low resolution could be trained as these. Others are quite straightforward.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.
<!--
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
-->

Here are the results of the prediction:

| Image			        |     Prediction    |
|:---------------------:|:-----------------:|
| Pedestrians           | 70 km/h           |
| 30 km/h     			| 80 km/h 			|
| Road work			    | 30 km/h	  	    |
| 60 km/h	      		| 60 km/h     		|
| No entry			    | No entry         	|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This accuracy is pretty lower than test set, and the reasons could be image collection range, interpolation during resolution down grade, not legitimate sign image, etc.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

###### A. First Image:
For the first image, 'Pedestrians', the model predicted as a '70 km/h' with lower probability (probability of 0.54).
The top five soft max probabilities were

| Probability         	|     Prediction	        |
|:---------------------:|:-------------------------:|
| .536        			| 70 km/h   				|
| .387     				| 30 km/h 					|
| .059					| Road work					|
| .008	      			| Wild animals crossing		|
| .003				    | Pedestrians        		|

![alt text][image6]

###### B. Second Image:
For the second image, '30 km/h', it seems that the model is relatively sure, but the model predicted as a '80 km/h' (probability of 0.432), and the correct prediction was the second most probability case.
The model seems to be easily confused between '3' and '8' during the training with given training image quality.

The top five soft max probabilities were

| Probability         	|     Prediction	        |
|:---------------------:|:-------------------------:|
| .432        			| 80 km/h   				|
| .103     				| 30 km/h 					|
| .102					| Vehicles over 3.5 tons prohibited	|
| .085	      			| 20 km/h		            |
| .070				    | 60 km/h        		    |

![alt text][image7]

###### C. Third Image:
For the third image, 'Road work', the model predicted as '70 km/h' with very lower probability (probability of 0.15). This kind of image is often predicted as a 'wild animals crossing', or '70 km/h'.
The top five soft max probabilities were

| Probability         	|     Prediction	        |
|:---------------------:|:-------------------------:|
| .146        			| 70 km/h   				|
| .105     				| 30 km/h 					|
| .080					| Road work					|
| .067	      			| Wild animals crossing		|
| .053				    | Pedestrians        		|

![alt text][image8]

###### D. Fourth Image:
For the fourth image, the model is relatively sure, and it easily detected as '60 km/h' with probability of almost 1.0. The top five soft max probabilities were

| Probability         	|     Prediction	        |
|:---------------------:|:-------------------------:|
| 1.000        			| 60 km/h   				|
| .000     				| 80 km/h 					|
| .000					| 50 km/h					|
| .000	      			| Bicycles crossing		    |
| .000				    | No vehicle        		|

![alt text][image9]

###### E. Fifth Image:
For the fifth image, the model is relatively sure, and it easily detected as 'No entry' with probability of almost 1.0. The top five soft max probabilities were

| Probability         	|     Prediction	        |
|:---------------------:|:-------------------------:|
| 1.000        			| No entry   				|
| .000     				| Keep left 				|
| .000					| Priority road				|
| .000	      			| stop		                |
| .000				    | 120 km/h        		    |

![alt text][image10]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
