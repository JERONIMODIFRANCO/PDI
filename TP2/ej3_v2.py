
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

def distancia_euclidiana(punto1, punto2):
    return np.sqrt(np.sum((np.array(punto1) - np.array(punto2))**2))

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

# Load image
img = cv2.imread('TP2/Patentes/img06.png', cv2.IMREAD_COLOR)

# Get image properties
height, width, channels = img.shape
print('The size of the image is {width}x{height} pixels with {channels} channels.')

# Convert image from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, img_bw = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
imshow(img_bw,title="Binary Image")

# Find connected components
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bw, connectivity, cv2.CV_32S)
imshow(labels,title="Labels")

# Set area thresholds for filtering connected components
area_threshold_max = 140  
area_threshold_min = 20

# Create a dictionary to store centroids for each label
centroids_dict = np.zeros_like(centroids)
labels_posibles = np.zeros_like(labels)
K=0
# Loop through the connected components
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if (area > area_threshold_min) and (area < area_threshold_max):
        # Draw the bounding box and centroid for the current component
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        (cX, cY) = centroids[i]        
        # Draw a bounding box surrounding the connected component
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Draw a circle corresponding to the centroid
        cv2.circle(img, (int(cX), int(cY)), 1, (0, 0, 255), -1)    

        labels_posibles = np.where(labels == i, K, labels_posibles)
        # Get the centroid of the current component
        centroids_dict[K] = (cX, cY)
        K=K+1
imshow(img)

# Initialize an array to store the numbers
numeros = np.zeros_like(labels_posibles)
th_dist_max = 20
th_dist_min = 5
# Loop through the remaining components and compute distance        
for j in range(1, len(centroids_dict)):
    cX1, cY1 = centroids_dict[j]
    for m in range (1, len(centroids_dict)):
        if (j == m):
            continue
        # Get the centroid of the other component
        cX2, cY2 = centroids_dict[m]
        # Compute the Euclidean distance between the centroids
        dist = distancia_euclidiana((cX1, cY1), (cX2, cY2))
        # Print the distance between the centroids
        print(f"Distance between centroids {j} and {m}: {dist}")
        if (dist < th_dist_max) and (dist > th_dist_min):
            numeros =   np.where(labels_posibles == j, 1, numeros)   
  
imshow(img)        
imagen = (numeros == 1)
imagen = imagen.astype(np.uint8)
imshow(numeros == 1)
        
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,3))
Fd = cv2.dilate(imagen, kernel, iterations=5)
imshow(Fd,title="dilatada")

connectivity = 8
num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(Fd, connectivity, cv2.CV_32S)
imshow(labels2,title="Labels2")
patente = np.zeros_like(labels2)
th_area_patente_max=3000
th_area_patente_min=1500
for i in range(1,num_labels2):
    area_patente = stats2[i, cv2.CC_STAT_AREA]
    if (area_patente > th_area_patente_min) and (area < th_area_patente_max):
            patente = np.where(labels2 == i, 1, patente)
            imshow(patente,title="mascara patente")
patente = np.uint8(patente & patente & patente)*255
img_patente = cv2.bitwise_and(img, img, mask=patente)
imshow(img_patente,title="mascara patente")
