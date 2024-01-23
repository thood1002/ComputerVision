import numpy as np
import cv2

# Reading in my image
img = cv2.imread('testImage.png', cv2.IMREAD_GRAYSCALE)

# Center pixel coords
def spatialFunction(neighborX, neighborY, centerX, centerY, sigma_s):
    # Taking the distance between the two pixels and squares it
    distanceSquared = (neighborX - centerX)**2 + (neighborY - centerY)**2
    # Puts distanceSquared into the formula and then calculates the weight 
    weight = np.exp(-distanceSquared / (2 * sigma_s**2))
    return weight

def intensityFunction(pixel1, pixel2, sigma_r):
    # Take the absolute value of the two intensity values
    difference = np.abs(pixel1 - pixel2)
    # Square the distance between the two values
    distance = np.sum(difference**2)
    return np.exp(-distance**2 / (2 * sigma_r**2))

def bilateralFilter(img, sigma_s, sigma_r, windowSize):
    # Create an array of zeros with the same shape as the input image
    filtered = np.zeros_like(img)
    # Adding padding based on the window size
    pad = windowSize // 2
    # Creating a 2d kernel
    kernel = np.zeros((windowSize, windowSize))
    # Create the kernel based on the spatial function
    for i in range(-pad, pad + 1):
        for j in range(-pad, pad + 1):
            kernel[i + pad, j + pad] = np.exp(-((i ** 2 + j ** 2) / (2 * sigma_s ** 2)))
    # Looping through the image
    for i in range(pad, img.shape[0] - pad):
        for j in range(pad, img.shape[1] - pad):
            pixel = img[i, j]
            window = img[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # Create the weight based on the intensity function and the kernel
            intensity_diff = np.square(window - pixel)
            intensity_weight = np.exp(-intensity_diff / (2 * sigma_r ** 2))
            weight = intensity_weight * kernel
            # Calculate the filtered value for this pixel
            filtered[i, j] = np.sum(window * weight) / np.sum(weight)
    # Show the original and filtered images side by side
    cv2.imshow("original", img)
    cv2.imshow("filtered", filtered)
    return filtered
# Call the bilateralFilter function with the specified parameters
bilateralFilter(img, 1, 50, 9)
cv2.waitKey(0)
