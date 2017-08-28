import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pickle


# A function that takes in an image, and the resolution
# and return a feature vector.
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


# A function to compute color histogram features
def color_hist(img, nbins=32, bin_range=(0, 256)):
    # Compute the histogram of R, G, B, channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bin_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bin_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bin_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    # Generating bin centers
    # bin_edges = rhist[1]
    # bincen = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2

    # fig = plt.figure(figsize=(12,3))
    # plt.subplot(131)
    # plt.bar(bincen, rhist[0])
    # plt.xlim(0, 256)
    # plt.title('R Histogram')
    #
    # plt.subplot(132)
    # plt.bar(bincen, ghist[0])
    # plt.xlim(0, 256)
    # plt.title('G Histogram')
    #
    # plt.subplot(133)
    # plt.bar(bincen, bhist[0])
    # plt.xlim(0, 256)
    # plt.title('B Histogram')
    #
    # fig.tight_layout()
    # plt.show()

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


# A function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        file_features = []

        image = mpimg.imread(file)

        plt.imshow(image)
        plt.title('Original')
        plt.show()

        # Apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)


        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            # plt.plot(spatial_features)
            # plt.title('Spatially Binned Features')
            # plt.show()
            file_features.append(spatial_features)

        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    # features, hog_image = get_hog_features(feature_image[:, :, channel],
                    #                                         orient, pix_per_cell, cell_per_block,
                    #                                         vis=True, feature_vec=True)
                    # fig = plt.figure(figsize=(12, 4))
                    #
                    # plt.subplot(131)
                    # plt.imshow(image)
                    # plt.title('Car')
                    #
                    # plt.subplot(132)
                    # plt.imshow(feature_image)
                    # plt.title('Color_Conversion')
                    #
                    # plt.subplot(133)
                    # plt.imshow(hog_image)
                    # plt.title('Hog')
                    #
                    # fig.tight_layout()
                    # plt.show()

                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                                orient, pix_per_cell, cell_per_block,
                                                vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))

    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(image)
    # plt.title('Car')
    #
    # plt.subplot(122)
    # plt.imshow(feature_image)
    # plt.title('Car')
    #
    # plt.show()

    return features


cars=[]
images_path = 'training_images/vehicles/**/*.png'
for image in glob.iglob(images_path, recursive=True):
    cars.append(image)

notcars=[]
images_path = 'training_images/non-vehicles/**/*.png'
for image in glob.iglob(images_path, recursive=True):
    notcars.append(image)


color_space = 'YCrCb'
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL'
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = extract_features(notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)


X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X) # Normalization
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)


print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Training Feature vector length:', len(X_train[0]))
print('Training Labels vector length:', len(y_train))
print('Testing Feature vector length:', len(X_test[0]))
print('Testing Labels vector length:', len(y_test))


# Use a linear SVC
svc = LinearSVC()

# Check the training time for the SVC
t=time.time()

# Start the training
svc.fit(X_train, y_train)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


model = {'svc': svc, 'scaler': X_scaler, 'orient': orient,
         'pix_per_cell': pix_per_cell, 'cell_per_block': cell_per_block,
         'spatial_size': spatial_size, 'hist_bins': hist_bins}

with open('svc_pickle.p', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Display a car and not-car image
# car_ind = np.random.randint(0, len(cars))
# car_img = mpimg.imread(cars[car_ind])
#
# notcar_ind = np.random.randint(0, len(notcars))
# notcar_img = mpimg.imread(notcars[notcar_ind])
#
# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(car_img)
# plt.title('Car')
# plt.subplot(122)
# plt.imshow(notcar_img)
# plt.title('Not Car')
# plt.show()


