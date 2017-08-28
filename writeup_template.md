##Writeup Template

###Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/spatial_binned.png
[image3]: ./output_images/car-color-hog.png
[image4]: ./output_images/notcar-color-hog.png
[image5]: ./output_images/sliding_1.png
[image6]: ./output_images/sliding_2.png
[image7]: ./output_images/sliding_3.png
[image8]: ./output_images/bbox_1.png
[image9]: ./output_images/bbox_2.png
[image10]: ./output_images/car_iden_1.png
[image11]: ./output_images/car_iden_2.png
[image12]: ./output_images/bboxes_and_heat_car-1.png
[image13]: ./output_images/bboxes_and_heat_car-2.png
[image14]: ./output_images/bboxes_and_heat_car-3.png
[image15]: ./output_images/bboxes_and_heat_car-4.png
[video1]: ./output_images/mapped-project_video-FINAL.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used `training.py` for training the images with a Classifier. 
In that file, `get_hog_features()` method extracts the HOG features. I read all the `cars` and `non-cars` images. Here is an example of a car and a non-car image in those classes.

![alt text][image1]

I experimented with different color spaces and settled on `YCrCb` color space due to better training convergence. I converted each car and non-car image to `YCrCb` color space and used `bin_spatial()` method that calculates a feature vector. Color space is useful to represent objects of the same class that can be of different colors. 
Here is a sample plot of a binned vector.

![alt text][image2]

I then took a color histogram of the featured vector, followed by extracting HOG features.
I experimented with `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 

Here is a sample image showing color converted and HOG extracted features of a car and a non-car. The HOG visualization shows the dominant gradient direction. 
![alt text][image3]
![alt text][image4]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters but finally settled on the values as-is that were used in the class. I used all channels to consider while extracting the HOG info.

Color Space = `YCrCb`,<br>
Hog Orientations = 9,<br>
Pixels Per Cell = 8,<br>
Cells Per Block = 2,<br>
Hog Channel = 'ALL',<br>

Orientations represents the number of orientations bins that the gradient information will be split up into the histogram.

Pixels Per Cell gives the cell size over which each gradient histogram is computed. 

Cells Per Block specifies the local area over which the histogram counts in a given cell will be normalized. 



####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After extrating features for an image using spatial bin, color histogram and HOG features from the training data, I normalized this data using `sklearn.preprocessing.StandardScaler`

I split up the training data into 2 sets for training and testing purposes. I used Linear SVM Classifier for training the images. Again, I used the same parameters and classifers from the class. It took 23.66 Seconds to traing the classifier on the training images with 98.7 accuracy.

The code is in training.py, lines: 207 to 235.

I saved the classifier and other data into a pickle file.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is in `vehicle_detect.py` that does the Sliding Window Search. In `find_cars()` method, the sliding window search is present, some code borrowed from the class.

I decided on 3 regions to focus on. One is the horizon, one in the middle and one in the bottom.

I limited the 3 regions to the bottom half of the image to avoid sky and tress. My 3 regions used scale of 1.5, 1.75 and 1.9. The scales and the regions are based on experimentation using the test_images as input. Some are tuned to identify cars in the horizon and near the camera.

Here are the 3 regions depicted on an image.

![alt text][image5]
![alt text][image6]
![alt text][image7]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on 3 regions using different scales using YCrCb ALL-channels HOG features, binned spatial and color histograms.

Here are some example images with sliding window detection + features:

![alt text][image8]
![alt text][image9]

Here are some example images with cars identified:
![alt text][image10]
![alt text][image11]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/mapped-project_video-FINAL.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

After I saved the positive identifications of bounding boxes in each frame, which may or may not include cars, I created a heatmap, with a threshold, to identify the *real* cars in a frame. I then used `scipy.ndimage.measurements.label()` to detect each bounding box in the heatmap.

In the method: `draw_labeled_bboxes`, I draw the bounding boxes of identified regions in the thresholded heatmap.

In the same method I employed several Filters to identify/reject the positive identifications earlier, by keep tracking of last 10 frames.

- Centroid filter: Calculate centroid of each bounding box, and reject the ones which are really far away from the current and previous frames detections.
- No cars detections: If a current frame has not car detections, use it from the previous frame.
- Smooth transitions - to create smooth transition of bounding boxes on cars, average the values of the cars from the last MAX_SAVED_FRAMES frames.
- Detecting overlapping cases of the bounding boxes, which helped in removing False positives, when all else above failed.



### Here are couple of frames with bounding boxes, heatmaps, and identified bounding boxes after filtering.

![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spent a lot of time in classifier identifying the regions and scale factors during training. I often ran into getting all cars in the test images detected positively, but fail in a corner case. Finally settled on one set of paramters, and moved on to the video frames.

There were false positives in the shadow regions, no cars detected in some frames, cars not being detected in some corner cases.

To fix those, I experimented with centroid detection which eliminated many false positives, but not completely. To fix those, I need to work on a single frame, and analyze the positions of the false positives. I then came up with overlapping detections, and range detections of the corrdinates which helped remove those.

Running the whole video frames, I saw that the bounding boxes frequently jumped around which are not smooth. So I tracked all the bounding boxes of the last 10 frames and using those values, acheived a smoother transition of the detections.

I feel the filtering of false positives should be handled mainly during the Training phase? Maybe I should use more data so, shadows won't be detected as false positives. My pipeline might still fail in corner cases. The filtering choices should probably more robust than analyzing a single frame which had false positives and fixing it. I am sure there are more corner cases that my pipeline is missing to handle.

All-in-all, I enjoyed this project.
