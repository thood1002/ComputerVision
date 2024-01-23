import cv2
import numpy as np

def find_pears(img):
    # Convert to grayscale and blur the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 11)

    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological opening to remove small noise and fill holes in the objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Perform a distance transform and apply a threshold to create markers for the watershed algorithm
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    markers = cv2.connectedComponents(markers.astype(np.uint8))[1]

    # Apply the watershed algorithm to segment the objects
    markers = cv2.watershed(img, markers)

    # Find the center locations and sizes of the objects
    pear_x = []
    pear_y = []
    pear_size = []
    pear_masks = []
    for i in range(1, np.max(markers) + 1):
        area = np.sum(markers == i)
        if area > 2000:  # Only consider objects with area greater than 2000 pixels
            # Check if the current object overlaps with any previously detected pear
            mask = (markers == i).astype(np.uint8)
            overlap = False
            for j, pear_mask in enumerate(pear_masks):
                intersection = cv2.bitwise_and(mask, pear_mask)
                if np.sum(intersection) > 0.5 * np.sum(mask):  # If overlap area is greater than 50% of the current object
                    overlap = True
                    break
            if not overlap:
                pear_masks.append(mask)
                x, y = np.mean(np.argwhere(mask), axis=0)
                pear_x.append(x)
                pear_y.append(y)
                pear_size.append(area)

    # Draw red dots on the centers of the pears and highlight each object in a different color
    img_with_dots = img.copy()
    for i, (x, y) in enumerate(zip(pear_x, pear_y)):
        img_with_dots[pear_masks[i] > 0] = np.random.randint(0, 255, 3)
        cv2.circle(img_with_dots, (int(y), int(x)), 5, (0, 0, 255), -1)

    # Sort the detected pears by size in descending order
    pear_x = [x for _, x in sorted(zip(pear_size, pear_x), reverse=True)]
    pear_y = [y for _, y in sorted(zip(pear_size, pear_y), reverse=True)]
    pear_size = sorted(pear_size, reverse=True)

    return pear_x, pear_y, pear_size, img_with_dots

img = cv2.imread('pears.png')
pear_x, pear_y, pear_size, img_with_dots = find_pears(img)

# Display the original image and the colored image side by side
result = np.hstack((img, img_with_dots))
cv2.imshow('result', result)
cv2.waitKey(0)

print('Pear X:', pear_x)
print('Pear Y:', pear_y)
print('Pear Size:', pear_size)
print('Number of pears:', len(pear_size))
