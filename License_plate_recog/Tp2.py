import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(window_title,img_name):
    # Display the resulting edge image
    cv2.imshow(window_title, img_name)

    # Wait for a key press and then close the window
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

# Load the image
img = cv2.imread('TP2/monedas.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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
# img = cv2.resize(img, (width, height))

# Convert the image to uint8 data type
img = img.astype(np.uint8)
# show('Imagen original',img)
# imshow(img,title='Imagen original')

# Apply Gaussian blur to the image
blurred_img = cv2.GaussianBlur(img, (31, 31), 0)
# show('blurred_img',blurred_img)
# imshow(blurred_img,title='Imagen con filtro pasabajos')
# Subtract the blurred image from the original image
edges = cv2.absdiff(img, blurred_img)
# show('bordes',edges)
# imshow(edges,title='bordes')
img_edge = cv2.add(img,edges)
# show('Imagen con bordes resaltados',img_edge)
# imshow(img_edge,title='Imagen original con bordes resaltados')
img_gray = cv2.cvtColor(img_edge, cv2.COLOR_BGR2GRAY)
# show('Imagen en gris',img_gray)
# imshow(img_gray,'Imagen en gris')
# Apply the Canny edge detection algorithm
img_edge = cv2.Canny(img_gray, 70, 250)
# show('Edges',img_edge)
imshow(img_edge,title='Imagen con canny detection')
# Apply thresholding
retval, img_bw = cv2.threshold(img_edge, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 255, cv2.THRESH_BINARY)
imshow(img_bw,title='Imagen binarizada')
##################### IMAGEN EN GRIS Y ESCALADA ##########################################
#Define a kernel for the dilation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
img_dil = cv2.dilate(img_bw,kernel,iterations = 6)
imshow(img_dil,title='Dilation')

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

show('Imagen final', img_filled)




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