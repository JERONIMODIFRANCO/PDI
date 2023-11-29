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
img = cv2.imread('Patentes/img03.png', cv2.IMREAD_COLOR) 
#problema en imagen numero:  11(la patente la detecta, el tema es que hay varios rectangulos y al encontrar el primero ya lo toma como correcto)
# en la 8 tenemos problema si el th es menor a 124
#version 4 detectamos todas las patentes, caracteres: 60/72 = 0.833


# Get image properties
height, width, channels = img.shape
# print('The size of the image is {width}x{height} pixels with {channels} channels.')

# Convert image from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_2 = img.copy()
# Convert image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, img_bw = cv2.threshold(img_gray, 124, 255, cv2.THRESH_BINARY) # 124
imshow(img_bw,title="Binary Image")

# Find connected components
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bw, connectivity, cv2.CV_32S)
# imshow(labels,title="Labels")

# Set area thresholds for filtering connected components
area_threshold_max = 100    #100
area_threshold_min = 20     #20
#
#podemos fitrar por relacion de aspecto ademas de por area
#se me ocurre la altura que es siempre la misma 
altura_threshold_max = 20      #25
altura_threshold_min = 5        #5
ancho_threshold_max = 15        #15
ancho_threshold_min = 3         #3
#

# Create a dictionary to store centroids for each label
centroids_dict = np.zeros_like(centroids)
labels_posibles = np.zeros_like(labels)
K=0
# Loop through the connected components
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    altura = stats[i, cv2.CC_STAT_HEIGHT]
    ancho = stats[i, cv2.CC_STAT_WIDTH]
    if (area > area_threshold_min) and (area < area_threshold_max) and (altura > altura_threshold_min) and (altura < altura_threshold_max) and (ancho > ancho_threshold_min) and (ancho < ancho_threshold_max):
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
th_dist_max = 30
th_dist_min = 4
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
        # print(f"Distance between centroids {j} and {m}: {dist}")
        if (dist < th_dist_max) and (dist > th_dist_min):
            numeros =   np.where(labels_posibles == j, 1, numeros)   
  
# imshow(img)        
imagen = (numeros == 1)
imagen = imagen.astype(np.uint8)
# imshow(numeros == 1)
        
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,3))
Fd = cv2.dilate(imagen, kernel, iterations=5)
imshow(Fd,title="dilatada")

connectivity = 8
num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(Fd, connectivity, cv2.CV_32S)
# imshow(labels2,title="Labels2")

keypoints = cv2.findContours(Fd.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# img_cnt_v1 = cv2.drawContours(img.copy(),contours,-1, (0, 255, 0), 1)
# imshow(img_cnt_v1,title="Imagen con contornos")
contours = imutils.grab_contours(keypoints) #simplifica los contornos
contours = sorted(contours, key=cv2.contourArea,reverse=True)[:10] #devolvemos en orden descendente los 10 primeros contornos con mas area

location = None
for contour in contours:
    area = cv2.contourArea(contour)  # calculamos el area del contorno
    # print("area",area)
    aprox = cv2.approxPolyDP(contour, 10, True)  # aproximacion de contorno
    if (len(aprox) == 4) :  # restriccion de area (4 puntos)
        x, y, w, h = cv2.boundingRect(contour)
        if (h/w<1):# and (2.5<w/h):
            location = aprox
            # print("Se encontro un contorno con 4 puntos que lo aproximan y cumple con el área, y vale",area)
            break


mask = np.zeros(img_gray.shape,np.uint8)
new_img = cv2.drawContours(mask,[location],0,255,-1)
new_img = cv2.bitwise_and(img,img,mask=mask)
mask_fondo = ~ mask
img_fondo = cv2.bitwise_and(img_2,img_2,mask=mask_fondo)
img_patente = cv2.bitwise_and(img,img,mask=mask)
img_final = cv2.add(img_fondo,img_patente)
imshow(img_final,title="Ubicacion de la patente en la imagen original")

(x,y) = np.where(mask==255)
(x1,y1) = (np.min(x),np.min(y))
(x2,y2) = (np.max(x),np.max(y))
recorte = img_gray[x1:x2+1,y1:y2+1]
recorte = cv2.cvtColor(recorte,cv2.COLOR_BGR2RGB)

imshow(recorte,title="Patente")