# importacion de librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

# Funcion para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(punto1, punto2):
    return np.sqrt(np.sum((np.array(punto1) - np.array(punto2))**2))

# Funcion para mostrar las imagenes
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

# Cargar imagen
img = cv2.imread('Patentes/img04.png', cv2.IMREAD_COLOR) 
#problema en imagen numero:  11(la patente la detecta, el tema es que hay varios rectangulos y al encontrar el primero ya lo toma como correcto)
# en la 8 tenemos problema si el th es menor a 124
#version 4 detectamos todas las patentes, caracteres: 60/72 = 0.833

# Propiedades
height, width, channels = img.shape
# print('The size of the image is {width}x{height} pixels with {channels} channels.')

# Convertir de BGR a RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img_2 = img.copy() # auxiliar para imagen final
imshow(img,title="Imagen RGB")

# Convertir a escala de grises
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow(img_gray,title="Imagen en niveles de gris")

# Aplicar binarizacion por umbral
_, img_bw = cv2.threshold(img_gray, 124, 255, cv2.THRESH_BINARY) # 124
imshow(img_bw,title="Imagen binarizada")

# Encontrar componentes conectadas
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bw, connectivity, cv2.CV_32S)
imshow(labels,title="Labels")

# Umbrales de area para eliminar las componentes conectadas que no dispongan de las propiedas que disponen las letras
area_threshold_max = 100    #100
area_threshold_min = 20     #20

#podemos fitrar por relacion de aspecto ademas de por area
#se me ocurre la altura que es siempre la misma 
# Relaciones de aspecto de las letras
altura_threshold_max = 20      #25
altura_threshold_min = 5        #5
ancho_threshold_max = 15        #15
ancho_threshold_min = 3         #3
#

# Array para almacenar los centroides que cumplen con las relaciones de las letras
centroids_dict = np.zeros_like(centroids)
# Array para almacenar los labels que cumplen con las relaciones de las letras
labels_posibles = np.zeros_like(labels)

K=0
# Iteracion a traves de los labels encontrados
for i in range(1, num_labels):
    # Almacenar variables de interes para posterior busqueda
    area = stats[i, cv2.CC_STAT_AREA]
    altura = stats[i, cv2.CC_STAT_HEIGHT]
    ancho = stats[i, cv2.CC_STAT_WIDTH]
    # Busqueda de labels que cumplen con el area y relacion de aspecto 
    if (area > area_threshold_min) and (area < area_threshold_max) and (altura > altura_threshold_min) and (altura < altura_threshold_max) and (ancho > ancho_threshold_min) and (ancho < ancho_threshold_max):
        # bounding box del label en cuestion
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        (cX, cY) = centroids[i]        
        # Dibujo de bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Dibujo de centroide
        cv2.circle(img, (int(cX), int(cY)), 1, (0, 0, 255), -1)    
        # Almacenar los labels que cumplen con las especificaciones
        labels_posibles = np.where(labels == i, K, labels_posibles)
        # Almacenar el centroide del label en cuestion
        centroids_dict[K] = (cX, cY)
        K=K+1
imshow(img,title="Labels posibles según umbrales de área y relacion de aspecto")

# Initialize an array to store the numbers
numeros = np.zeros_like(labels_posibles)
# Umbrales de distancia entre 2 letras
th_dist_max = 30
th_dist_min = 4
# Iterar a traves de los componentes conectados comparando las distancias eucledianas entre centroides        
for j in range(1, len(centroids_dict)):
    # Centroide j
    cX1, cY1 = centroids_dict[j]
    for m in range (1, len(centroids_dict)):
        if (j == m):
            continue
        # Centroide m
        cX2, cY2 = centroids_dict[m]
        # Distancia eucledaina
        dist = distancia_euclidiana((cX1, cY1), (cX2, cY2))
        # Mostrar el valor obtenido
        # print(f"Distance between centroids {j} and {m}: {dist}")
        # Si la distancia euclediana esta entre los umbrales asignar 1 a las componentes que componen el label
        if (dist < th_dist_max) and (dist > th_dist_min):
            numeros =   np.where(labels_posibles == j, 1, numeros)   

# Nueva imagen con los labels que cumplen con las condiciones hasta el momento        
imagen = (numeros == 1)
imagen = imagen.astype(np.uint8)
imshow(imagen,title="Labels posibles según condiciones anteriores y distancia euclediana")

# Dilatar labels para encontrar la patente        
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,3))
imagen_dil = cv2.dilate(imagen, kernel, iterations=5)
imshow(imagen_dil,title="Labels dilatados en búsqueda de patente")

# connectivity = 8
# num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(imagen_dil, connectivity, cv2.CV_32S)
# imshow(labels2,title="Labels2")

# Busqueda de contornos de patente 
keypoints = cv2.findContours(imagen_dil.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints) #simplificar los contornos
contours = sorted(contours, key=cv2.contourArea,reverse=True)[:10] #devolvemos en orden descendente los 10 primeros contornos con mas area

# variable para posterior ubicacion de patente en la imagen original
location = None
# Iteracion a traves de contornos posibles de patente
for contour in contours:
    area = cv2.contourArea(contour)  # calculamos el area del contorno
    # print("area",area)
    aprox = cv2.approxPolyDP(contour, 10, True)  # aproximacion de contorno
    if (len(aprox) == 4) :  # restriccion de area (4 puntos)
        x, y, w, h = cv2.boundingRect(contour) # boundig box
        if (h/w<1): # relacion de aspecto de rectangulo 
            location = aprox 
            # print("Se encontro un contorno con 4 puntos que lo aproximan y cumple con el área, y vale",area)
            break

# generar mascara de patente para "unir" patente con caracteres detectados con la imagen original 
mask = np.zeros(img_gray.shape,np.uint8)
# new_img = cv2.drawContours(mask,[location],0,255,-1)
# new_img = cv2.bitwise_and(img,img,mask=mask)
mask_fondo = ~ mask # mascara para el fondo
img_fondo = cv2.bitwise_and(img_2,img_2,mask=mask_fondo)#componente fondo de la imagen
img_patente = cv2.bitwise_and(img,img,mask=mask) # componente patente de la imagen 
img_final = cv2.add(img_fondo,img_patente) # suma de fondo y patente
imshow(img_final,title="Ubicacion de la patente y caracteres en la imagen original")

# imagen recortada de patente
(x,y) = np.where(mask==255)
(x1,y1) = (np.min(x),np.min(y))
(x2,y2) = (np.max(x),np.max(y))
recorte = img_gray[x1:x2+1,y1:y2+1]
recorte = cv2.cvtColor(recorte,cv2.COLOR_BGR2RGB)
imshow(recorte,title="Patente")