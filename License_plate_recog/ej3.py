import cv2

import numpy as np
import matplotlib.pyplot as plt
import imutils

# Definimos función para mostrar imágenes
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

def show(window_title,img_name):
    # mostramos la imagen
    cv2.imshow(window_title, img_name)

    # cerramos con el esc
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        
# --- cargar imagen ---------------------------------------------------------------------------------
img = cv2.imread('TP2/Patentes/img02.png', cv2.IMREAD_COLOR)
# --- propiedades  ---------------------------------------------------------------------------------
height, width, channels = img.shape
print('The size of the image is {width}x{height} pixels with {channels} channels.')
# --- pasamos a rgb  ---------------------------------------------------------------------------------
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# imshow(img, title="Imagen Original")
# --- escala de grises  ------------------------------------------------------------------------
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imshow(img_gray, title="Imagen en escala de grises")
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
# img = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
# imshow(img,title="Gradiente morfologico")
# --- reduccion de ruido -----------------------------------------------------------------------------
img = cv2.bilateralFilter(img_gray,7,17,17) 
imshow(img, title="Imagen filtrada con filtro bilateral")
# imshow(img_bil_fil, title="Imagen filtrada con filtro bilateral")
# retval, img_bw = cv2.threshold(img_bil_fil, 140, 255, cv2.THRESH_BINARY)
# imshow(img_bw,title='Imagen binarizada')
#--------------------------------------------

# --- deteccion de bordes -----------------------------------------------------------------------------
img_edge = cv2.Canny(img, 20, 200, apertureSize=3, L2gradient=True)
imshow(img_edge,title="Deteccion de bordes con canny")
#--------------------------------------------------------
height, width = img_edge.shape
print(f'The size of the image is {width}x{height}.')
# ---  -----------------------------------------------------------------------------
keypoints = cv2.findContours(img_edge.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# img_cnt_v1 = cv2.drawContours(img.copy(),contours_v1,-1, (0, 255, 0), 1)
# imshow(img_cnt_v1,title="Imagen con contornos")
contours = imutils.grab_contours(keypoints) #simplifica los contornos
contours = sorted(contours, key=cv2.contourArea,reverse=True)[:10] #devolvemos en orden descendente los 10 primeros contornos con mas area
b = 0
location = None
for contour in contours:
    area = cv2.contourArea(contour)  # Calculate the area of the contour
    print("area",area)
    aprox = cv2.approxPolyDP(contour, 10, True)  # Approximate the contour
    if (len(aprox) == 4) :  # Add the area restriction condition
        location = aprox
        x, y, w, h = cv2.boundingRect(contour)
        print()
        if (b == 0) and (((w/h)>1.3) and (2>(w/h))):
            b = 1
            location = aprox
            print("Se encontro un contorno con 4 puntos que lo aproximan y cumple con el área, y vale",area)
        break

# print("ubicacion de la placa
# for row in location:
#     print(row)
    
mask = np.zeros(img_gray.shape,np.uint8)
new_img = cv2.drawContours(mask,[location],0,255,-1)
new_img = cv2.bitwise_and(img,img,mask=mask)
imshow(new_img,title="Ubicacion de la patente en la imagen original")

(x,y) = np.where(mask==255)
(x1,y1) = (np.min(x),np.min(y))
(x2,y2) = (np.max(x),np.max(y))
recorte = img_gray[x1:x2+1,y1:y2+1]
recorte = cv2.cvtColor(recorte,cv2.COLOR_BGR2RGB)
imshow(recorte,title="Patente")