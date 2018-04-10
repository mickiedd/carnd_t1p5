**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./data/example_image.png
[image2]: ./data/hog_example.png
[image3]: ./data/sliding_windows.png
[image4]: ./data/car_with_sliding_windows.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./data/car_with_heatmap.png
[image9]: ./data/car_with_label.png
[image10]: ./data/car_with_rawlabel.png
[image11]: ./data/hog_example2.png
[image12]: ./data/hog_example3.png
[video1]: ./output_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook p5.ipynb.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

![alt text][image11]

![alt text][image12]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and at last decided to use orient = 9, pixel per cell equal to 8, cell per block equal to 2.

The images shows that the shape seems clear enough to identify cars of a image by these parmeters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the 2nd code cell of the IPython notebook p5_classifier.ipynb.  

I trained a linear SVM classifier using the combination of HOG feature vector, binned color features, as well as histograms of color.
Also I trained a neural network classifier to make the classify more robust, which provided in the p5_nn_classifier.ipynb.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the 1st code cell of the IPython notebook p5_sliding_windows.ipynb.  

I decided to search random window positions at the scales of 1, 2, 3 all over the image and came up with this:

![alt text][image3]

The overlap of the window is 0.5

It seems that the max scale of 3 is enough for this project.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using RGB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, and alow with the neural network features, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Also, in order to make the classifier more robust, I add a neural network classifier to the project, it seems that neural network work pretty well even without other classifier, I made the two classifiers to decide if the image has a car in it together while the sliding window is sliding.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are a test image the corresponding heatmap:

![alt text][image4]

![alt text][image8]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from the test image:
![alt text][image10]

### Here the resulting bounding boxes are drawn onto the test image:
![alt text][image9]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think sliding window is not a very good choice for this project, maybe yolo is a better choise.

Also I consider that neural network classifier work better than the SVM classifier when theres is enough training data, in this project, I augmentated the training data by flip each image, this operation double the training datas.
