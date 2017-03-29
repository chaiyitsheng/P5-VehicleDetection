##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/original.png
[image2]: ./output_images/ycrcb.png
[image3]: ./output_images/sliding1.png
[image4]: ./output_images/sliding2.png
[image5]: ./output_images/sliding3.png
[image6]: ./output_images/detected1.png
[image7]: ./output_images/detected2.png
[image8]: ./output_images/detected3.png
[image9]: ./output_images/detected4.png
[image10]: ./output_images/detected5.png
[image11]: ./output_images/detected6.png
[image12]: ./output_images/heat1.png
[image13]: ./output_images/heat2.png
[image14]: ./output_images/heat3.png
[image15]: ./output_images/heat4.png
[image16]: ./output_images/heat5.png
[image17]: ./output_images/heat6.png
[image18]: ./output_images/sliding4.png
[video1]: ./project_video_for_submission.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Starting Point

1. Code explanation

My code is divided into 5 sections in the ipython notebook:

1) Section 1 Import Libraries

2) Section 2 Define functions and classes

3) Section 3 Learn car and notcar images with LinearSVC

4) Section 4 Convert Video

5) Section 5 Test Visualizations for testing purposes

2. Preparation

I first imported an original image
![alt text][image1]


and showing example colorspaces, ![alt text][image2]

I also imported the car and not car images (not shown here because it's self explanatory)

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used the hog from "from skimage.feature import hog" which is a built in command (code in section 2)

####2. Explain how you settled on your final choice of HOG parameters.

For the HOG parameters, I used 9 orientations and the 8x8 cells_per_block. The reason for these are 1) 9 orientations captures as much detail as is probably needed for detection 2) the 8x8 cell makes the orientations less susceptible to noise and a larger cell say 32x32 cell would impose a higher computation cost. for the HOG channel, I chose to use "ALL" instead of the 0,1,2. Essentially choosing all color channels because it captures more details. Using a single channel is fast but based on my testing, "All" provided the best prediction

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the "import LinearSVC" (Section 3) During feature extraction, I used a spatial bin size and color of 16,16,16 respectively. A larger size may have improved the learning further but I am running an old PC without a GPU (budget constraints at the moment). The settings that I tried are about the max for my machine. During the learning process, my keyboard and mouse would stop working pending running of the learning job.

I also used the StandardScaler().fit so scale the data point so that the concatenated data has a more normalized magnitude.

Dividing the training and test sets into a typical 80/20 split resulted in an accuracy of 0.9887. This was done in the YCrCb colorspace. I tried all the common color spaces ie RGB, YUV HLS, etc but the YCrCb yielded the best result.

I then saved the results in a pickle file to avoid running the learning every time I had to restart the kernel.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search based largely on the methods provided in the lectures searching 3 scales using a 32x32 window. This small window was chosen due to the small size of the car/notcar images which was also 32x32. I used an overlap of 0.75 and 0.5 in the x and y scale respectively. The difference is to capture more details in the x direction. I derived these through trial and error starting from 0.5 and 32x32 respectively and enlarging it gradually. This combo yield the good best results given the other parameters chosen earlier. But this is largely a trial and error method.

![alt text][image3] ![alt text][image4] ![alt text][image5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline shows detections but resulted in some false positives as detailed in the two images below:

![alt text][image6] ![alt text][image7]

To improved performance, I implemented a heatmap method detailed below.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_for_submission.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I improved the performance by implementing the heatmap method as detailed in the lectures. The following are examples of the heatmap . If consecutive frames shows consecutive positive predictions, these are included in the heatmap.

![alt text][image6] ![alt text][image7]

I also reduced the span for one of the window sets to search within a more confined area of interest.
![alt text][image18]

---
### Here are six frames and their corresponding heatmap labels:

Image 1 ![alt text][image6]

Image 2 ![alt text][image7]

Image 3 ![alt text][image8]

Image 4 ![alt text][image9]

Image 5 ![alt text][image10]

Image 6 ![alt text][image11]

Image 1 Heatmap ![alt text][image12]

Image 2 Heatmap ![alt text][image13]

Image 3 Heatmap ![alt text][image14]

Image 4 Heatmap ![alt text][image15]

Image 5 Heatmap ![alt text][image16]

Image 6 Heatmap ![alt text][image17]

For a given sequence positive detection shown in the heatmap, we take in all the x and y values of the pixels with each positive detection and find the min and max. This is our bounding box. Using the cv2.rectangle method we then draw a bounding box

example bounding box ![alt text][image10]

---
###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The learning process using the LinearSVC was very slow. Accuracy was acceptable but perhaps using a neural network for classification would have yield a better result than HOG for this particular use case.

A color based thresholding technique for detect the white and black cars would have improved performance for this particular use case but the model would not be generalised enough for other car colours.

Doing machine learning on a PC without a GPU is a real pain! Will upgrade soon or subscribe for AWS!

To the reviewer:
Thanks for reviewing my project. It was a tiring project but I believe that reviewing code is even more tiring. Appreciate it very much.
