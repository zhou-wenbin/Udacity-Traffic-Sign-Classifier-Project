# **Traffic Sign Recognition** 

## Writeup



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

[image1]: ./output_figures/train_set_class.png
[image2]: ./output_figures/test_set_class.png
[image3]: ./output_figures/validation_set_class.png
[image4]: ./output_figures/before_process.png
[image5]: ./output_figures/after_process.png
[image6]: ./output_figures/learning_curve.png
[image7]: ./germain_traffic_signs/online_images.png
[image8]: ./output_figures/online_test.png



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


You're reading it! and here is a link to my [project code](https://github.com/zhou-wenbin/Udacity-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_share.ipynb)

### Data Set Summary & Exploration

#### 2. A basic summary of the data set. 


| Data description         		|     number      					| 
|:---------------------:|:---------------------------------------------:| 
| size of training set         		| 					34799	| 
| size of the validation set    	|  4410 |
| size of test set					|			12630								|
| shape of a traffic sign image |   (32, 32, 3)   |
| number of unique labels  |  43  |


#### 1. The distribution of labels in train/test/validation data set.
  
train data                 | test data        |         validation data 
:-------------------------:|:-------------------------:|:-------------------------:
![alt text][image1]  |  ![alt text][image2]| ![alt text][image3]




### Design and Test a Model Architecture

I have tried different ways to process the images, for example, equalize the histogram of the Y channel of images, add weight to the gussian blurred version of the image and so on. But that did not increased the accuracy. For the record, I will post how I did the image process even though in the training process I did not use them. 

* After I checked the training set, I saw that there are some dark images in the set, as follows, 
![alt text][image4]

* After I preprocess the training set, I made the signs more standing out, as follows, 
![alt text][image5]


Before I train my model, the only thing I did to my data set is to shuffle the training data because after shuffling the data we cag get rid of the correlation between images.


#### 1. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Layer 1:conv1             	| filter=5x5x6, strides=[1,1,1,1], outputs=28x28x6 	|
| ReLU					|		activation function										|
|  Layer 2 :conv2	      	  |  filter=5x5x10, strides=[1,1,1,1], outputs=24x24x10			|
| ReLU		         |     						activation function					|
|Layer 3: conv3              	|  filter=5x5x16, strides=[1,1,1,1], outputs=20x20x16	      		|
| ReLU				| activation function       									|
|  Max pooling  |   strides=[1,2,2,1], Input = 20x20x16, Output = 10x10x16.    |
|	Layer 4:fully connected			| Input =	10x10x16, Output =120											|
|		Dropout				|												|
|	Layer 5:fully connected			| Input =	120, Output =100											|
| ReLU				| activation function       									|
|	Layer 6:fully connected			| Input =	100, Output =84											|
| ReLU				| activation function       									|
|	Layer 7:fully connected			| Input =	84, Output =43										|



#### 2. Hyper parameters


|Hyper parameters        		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| optimizer |      Adam    |
|  batch size |      128    |
| number of epochs |     50     |
|learning rate|   0.001     |

#### 3. My approach to get the above model. 

In the beginning, I implemeted the simple LeNet network with only 5 layers that was explained in the lecture and no matter how I tried to process the image I was not able to achieve test accuracy above 0.9. Then I started to increased the layer a bit, because I think the deeper the network, the better the performance will be.  In order to prevent overfitting I also added a dropout process in between layer 4 and layer 5 because layer 4 has many nurons so it might cause overfitting.  Surprisingly, after I increased 2 more layers the accuracy increased immediately. However, it will not increase the performance even if I did a preprocess of training set. I will leave this as following research to see why the preprocess of the image did not help. 

### 4. My final model results were:

* training set accuracy of  0.998
* validation set accuracy of  0.958
* test set accuracy of  0.950

#### The learning curve is shown as follows,
![alt text][image6]


###  5. Test a Model on New Images

* The online images are chosen as follows :

![alt text][image7]

I manully screenshot traffic signs from google images.

* Test result is (with top five guessing),
![alt text][image8]
with test accuray =0.800
###  6. Reflections
* w.r.t the testing on online images:

The only image that was not recongnized is due to the reason of resize, which makes the sign diffictult to recongnize either for out human eyes. You can that the input image of the unrecongnized one is very blur.

* w.r.t the model constructions:

I experimented that if I increased the NN layer from 5 to 7, the accuracy increased surprisely. However, there is no theory that can explain this. And also those parameters are tuned by trials and errors and I had no clue why should we decrease the learning rate or increase the EPOCH size or why should we do the image process. These will be left as future reseach.


```python

```
