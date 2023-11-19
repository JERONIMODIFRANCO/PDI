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
width = 1440
height = int(width/relation)

# Print the size of the image
print(f'The size of the image is {width}x{height} pixels with {channels} channels.')

#resize image
img = cv2.resize(img, (width, height))

# Convert the image to uint8 data type
img = img.astype(np.uint8)
show('Imagen original',img)


# Apply Gaussian blur to the image
blurred_img = cv2.GaussianBlur(img, (11, 11), 0)
show('blurred_img',blurred_img)
# Subtract the blurred image from the original image
edges = cv2.absdiff(img, blurred_img)
show('bordes',edges)
img_edge = cv2.add(img,edges)
show('Imagen con bordes resaltados',img_edge)

# Convert the image to grayscale
img_gray = cv2.cvtColor(img_edge, cv2.COLOR_BGR2GRAY)
show('Imagen en gris',img_gray)

##################### IMAGEN EN GRIS Y ESCALADA ###########################################

# Apply the Canny edge detection algorithm
img_edge = cv2.Canny(img_gray, 100, 280)
show('Edges',img_edge)

#Define a kernel for the dilation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
img_dil = cv2.dilate(img_edge,kernel,iterations = 5)
show('Dilation', img_dil)

# Create a rectangular structuring element
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Perform closing on the image using the structuring element
closed_image = cv2.morphologyEx(img_dil, cv2.MORPH_CLOSE, kernel2)
show('Imagen con clausura', closed_image)

# Find all connected components in the binary image
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_image)

# Create a blank image to draw the filled components
img_filled = np.zeros_like(img)

# Loop through all components and fill them with a random color
for i in range(1, num_labels):
    mask = labels == i
    color = np.random.randint(0, 255, size=3).tolist()
    img_filled[mask] = color

show('Imagen final', img)




### ACA VIENE EL IMFILL HAY QUE VER COMO HACER

# Define the seed point for flood filling
seed_point = (550, 550)

# Define the color for flood filling
fill_color = (255, 255, 255)

# Define the lower and upper difference thresholds for flood filling
lo_diff = (10, 10, 10)
up_diff = (10, 10, 10)

# Define the mask for flood filling
mask = np.zeros((img_dil.shape[0] + 2, img_dil.shape[1] + 2), np.uint8)

# Perform flood filling on the image
cv2.floodFill(img_dil, mask, seed_point, fill_color, lo_diff, up_diff)

# Display the resulting binary image
show('Binary Image', img_dil)




blurred_image = cv2.GaussianBlur(img_gray, (13, 13), 1.7)
show('blurred', blurred_image)
# Apply adaptive thresholding to the blurred image
thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
show('thresh', thresholded_image)
# Apply morphological operations to remove noise and fill holes

# Find contours in the binary image
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
show('contours',contours)
# Draw contours on the original image
output_image = img.copy()
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
show('output',output_image)