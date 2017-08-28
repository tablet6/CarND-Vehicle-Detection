import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from skimage.feature import hog
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# Write a function that takes in an image, and the resolution you would like to convert it to,
# and return a feature vector.
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bin_range=(0, 256)):
    # Compute the histogram of R, G, B, channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bin_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bin_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bin_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)

    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], (0, 0, 255), thick)

    return imcopy

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step


    # window_list = []
    # for xb in range(nxsteps):
    #     for yb in range(nysteps):
    #         ypos = yb * cells_per_step
    #         xpos = xb * cells_per_step
    #
    #         xleft = xpos * pix_per_cell
    #         ytop = ypos * pix_per_cell
    #
    #         xbox_left = np.int(xleft * scale)
    #         ytop_draw = np.int(ytop * scale)
    #         win_draw = np.int(window * scale)
    #
    #         window_list.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    # window_img = draw_boxes(img, window_list, color=(0, 0, 255), thick=6)
    # plt.imshow(window_img)
    # plt.show()


    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bbox = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                # cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                #               (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                # plt.imshow(draw_img)
                # plt.show()

                bbox.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    # return draw_img
    return bbox

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


prev_frames_centroids = []
last_frame_car_count = 0
last_frame_bbox = []
last_n_frames = []
CENTROID_THRESHOLD = 410
MAX_ENTRIES = 5
MAX_SAVED_FRAMES = 10

def is_valid_centroid(val):
    global prev_frames_centroids
    for cen in prev_frames_centroids:
        if abs(val - cen) > CENTROID_THRESHOLD:
            return False
    return True

def did_overlap(y1, y2):
    global last_frame_bbox

    if not last_frame_bbox:
        return True

    currY1 = last_frame_bbox[0][0][1]
    currY2 = last_frame_bbox[0][1][1]

    if (y1 >= currY1 and y1 <= currY2) or (y2 >= currY1 and y2 <= currY2):
         return True

    return False


def avg_box(car, box):
    avgx1 = []
    avgy1 = []
    avgx2 = []
    avgy2 = []

    avgx1.append(box[0][0])
    avgy1.append(box[0][1])
    avgx2.append(box[1][0])
    avgy2.append(box[1][1])

    for frame in last_n_frames:
        avgx1.append(frame[car][0][0])
        avgy1.append(frame[car][0][1])
        avgx2.append(frame[car][1][0])
        avgy2.append(frame[car][1][1])

    mean_x1 = int(np.mean(avgx1))
    mean_y1 = int(np.mean(avgy1))
    mean_x2 = int(np.mean(avgx2))
    mean_y2 = int(np.mean(avgy2))

    return mean_x1, mean_y1, mean_x2, mean_y2

def centroid_filter(bbox):
    centroidx = (bbox[1][0] + bbox[0][0]) // 2
    centroidy = (bbox[1][1] + bbox[0][1]) // 2

    if is_valid_centroid(centroidx):
        global prev_frames_centroids
        if len(prev_frames_centroids) > MAX_ENTRIES:
            prev_frames_centroids.pop(0)

        prev_frames_centroids.append(centroidx)

        if did_overlap(bbox[0][1], bbox[1][1]):
            return True

    return False

def draw_labeled_bboxes(img, labels):

    box_list = []

    global last_frame_bbox
    global last_frame_car_count

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Filter - Reject box if centroid doesn't meet threshold
        if centroid_filter(bbox):
            box_list.append((bbox[0], bbox[1]))

    # Filter - If current frame had no cars detected, use it from the last_frame
    if labels[1] == 0 and last_frame_car_count > 0:
        box_list = last_frame_bbox

    # Filter - Update the cars in last frame
    if labels[1] > 0:
        last_frame_car_count = labels[1]

    # Filter - Use last frame cars for a threshold
    if len(box_list) == len(last_frame_bbox):
        for box, last_box in zip(box_list, last_frame_bbox):
            if abs(abs(box[1][1]-box[0][1]) - abs(last_box[1][1]-last_box[0][1])) > 70:
                box_list = last_frame_bbox

    # Filter - Clear last_n_frames saved if the number of current identified cars are not same
    # Update current cars list into last saved frames.
    global last_n_frames
    if len(box_list) > 0:
        if last_n_frames and (len(last_n_frames[0]) != len(box_list)):
            del last_n_frames[:]

        last_n_frames.append(box_list)

    # Filter - Pop oldest entry
    if (len(last_n_frames) > MAX_SAVED_FRAMES):
        last_n_frames.pop(0)

    # Filter - if no cars detected by previous filters, use cars from last frame
    if (not box_list) and (last_frame_bbox):
        box_list = last_frame_bbox

    # Filter - to create smooth transisition of detected boxes on cars, average the values of the cars from the last MAX_SAVED_FRAMES frames
    car = 0
    for box in box_list:
        # Draw the box on the image
        x1, y1, x2, y2 = avg_box(car, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 6)
        car = car + 1

    # Filter - Save current detected boxes
    last_frame_bbox  = box_list

    # Return the image
    return img


dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

def pipeline(image):
    ystart = [400, 450, 400]
    ystop = [650, 680, 700]
    scale = [1.5, 1.75, 1.9]

    bounding_boxes = []

    for i in range(0, len(ystart)):
        boxes = find_cars(image, ystart[i], ystop[i], scale[i], svc, X_scaler, orient, pix_per_cell,
                          cell_per_block,
                          spatial_size, hist_bins)

        if len(boxes) > 0:
            bounding_boxes.extend(boxes)

    window_img = draw_boxes(image, bounding_boxes, color=(0, 0, 255), thick=6)

    # plt.imshow(window_img)
    # plt.show()

    # Add heat to each box in box list
    heat = np.zeros_like(window_img[:, :, 0]).astype(np.float)

    # plt.imshow(heat)
    # plt.show()

    heat = add_heat(heat, bounding_boxes)

    # plt.imshow(heat)
    # plt.show()

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # plt.imshow(heat)
    # plt.show()

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    # plt.imshow(labels[0], cmap='gray')
    # plt.show()

    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    # plt.imshow(window_img)
    # plt.show()
    #
    # plt.imshow(labels[0], cmap='gray')
    # plt.show()
    #


    # fig = plt.figure(figsize=(12,3))
    #
    # plt.subplot(131)
    # plt.imshow(window_img)
    # plt.title('Car Bounding boxes')
    #
    # plt.subplot(132)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    #
    # plt.subplot(133)
    # plt.imshow(draw_img)
    # plt.title('Car Positions')
    #
    #
    # fig.tight_layout()
    # plt.show()

    # plt.imshow(draw_img)
    # plt.show()

    return draw_img


# images_path = "test_images/"
# orig = mpimg.imread(images_path+'test3.jpg')
# draw_image = np.copy(orig)
# final = pipeline(draw_image)
#
# plt.imshow(final)
# plt.show()


# white_output = 'output_images/mapped-test_video.mp4'
# clip1 = VideoFileClip("test_video.mp4")

white_output = 'output_images/mapped-project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")

# white_output = 'output_images/project_video.subclip_41_42.mp4'
# clip1 = VideoFileClip("project_video.mp4").subclip(41,42)

white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(white_output, audio=False)

