import cv2
import numpy as np

def show(window_title,img_name):
    # Display the resulting edge image
    cv2.imshow(window_title, img_name)

    # Wait for a key press and then close the window
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

# Load the image
img = cv2.imread('TP2/monedas.jpg')

# Get the size of the image
height, width, channels = img.shape

# Print the size of the image
print(f'The size of the image is {width}x{height} pixels with {channels} channels.')
relation = width/height
print(f'The relation of the image is {relation}')

# Resize the image to a specific size
width = 1600
height = int(width/relation)

# Print the size of the image
print(f'The size of the image is {width}x{height} pixels with {channels} channels.')

#resize image
img = cv2.resize(img, (width, height))

# Convert the image to uint8 data type
img = img.astype(np.uint8)
show('Imagen original',img)

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show('Imagen en gris',img_gray)

# Apply the Canny edge detection algorithm
img_edge = cv2.Canny(img_gray, 80, 280)
show('Edges',img_edge)


#Define a kernel for the dilation
kernel = np.ones((3,3),np.uint8)
img_dil = cv2.dilate(img_edge,kernel,iterations = 3)

# Display the resulting dilate image
show('Dilation', img_dil)


### ACA VIENE EL IMFILL HAY QUE VER COMO HACER

# Define the seed point for flood filling
seed_point = (100, 100)

# Define the color for flood filling
fill_color = (255, 0, 0)

# Define the lower and upper difference thresholds for flood filling
lo_diff = (10, 10, 10)
up_diff = (10, 10, 10)

# Define the mask for flood filling
mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

# Perform flood filling on the image
cv2.floodFill(img, mask, seed_point, fill_color, lo_diff, up_diff)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to the grayscale image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

# Display the resulting binary image
show('Binary Image', thresh)





# Display the results
# show('Original Image', img)
# show('Segmented Image',)
