# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Dataset breakdown

The dataset used is a modified version of the original data set. All images are 32x32x3 in dimensions.

I used the numpy and matplotlib to get histogram data for the class distribution of the training data.

Following is the break down of the dataset

Number of training examples = 34799

Number of testing examples = 12630

Image data shape = (32, 32, 3)

Number of classes = 43

Number of validation =  4410

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many data samples are available for each traffic sign

[initial_samples]: ./writeup_dir/initial_images.png'

I implemented a TrafficDataInfo class to analyze the dataset


There are some classes that are significantly under represented.

I attempted to do histogram equalization. This involved generating new samples by clubbing the existing validation and testing data for the underrepresented classes.

But this to was not sufficient. So I decided to generate transformed versions of the existing samples.


The Transformations comprised the following operations

1. rotation within a range of 10 to -10

2. translation within a range of 5 to -5

3. random adjustment of brightness.

The transformation operations are implemented using tensorflow to leverage the parallel processing offered by TensorFlow.

The transformations are as per suggestions in the Yann Le Cunn [paper][http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf]

Here are samples of the original and transformed images.

![transformed images][./writeup_dir/transforms.png]

### Design and Test a Model Architecture

#### Preprocessing pipeline

As a first step, I decided to convert the images to grayscale because grayscale helped remove brightness related deviations.

Also to allow for better training of the model, I decided to zero center the image values and scale the pixel values to lie between -1 and 1. 

This prevents the problems related to exploding gradients.

Here are some samples after preprocessing step

![alt text][./writeup_dir/preprocess_samples]


#### 2.  Model Architecture

I decided to use the Lenet Model as is.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				|												| 
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout				|												| 
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Dropout				|												| 
| Fully connected		| input 400, output 120							|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 120, output 84							|
| RELU					|												|
| Fully connected		| input 84, output 43							|
| Softmax				|												|
|						|												|
 

#### 3. Hyperparameters, Optimizers

I set a learning rate of 0.0009. The low learning rate allows for progress towards high level of accuracy, without overfitting.

I set a batch size of 128. I experimented with different batch sizes, it seems a smaller batchsize, did not allow for 

sufficient representation of all classes in a given batch that led to more erractic learning.

For the optimizer I used the Adam optimizer which has an inbuilt momentum setting.

The loss function was a categorical cross entropy since we are dealing with multi class classification problem.

#### 4. Architecture Choice

I used the Lenet architecture.

However to push beyond the original accuracy of 93% , i had to add more data samples using transforms suggested in the original paper.

However with the increased samples , it was essential to add dropouts to prevent overfitting.

The validation accuracy saturated to 99% after 25 epochs, further epochs saw a decline in validation accuracy because of overfitting.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ![German traffic signs][./writeup_dir/test_images.png] that I found on the web:


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is a visualization of the feature maps.

These are the first layer convolutional filters.
![alt_text][./writeup_dir/conv1_fmap.png]

These are the second convolutional layer feature maps.
![alt_text][./writeup_dir/conv2_fmap.png]

It seems the first layer is responsible for detecting the outline of the sign, and disregarding the surrounding pixels.

The second layer looks to learn edges.


