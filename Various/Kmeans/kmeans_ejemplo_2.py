import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Cargo Imagen ----------------------------------------------------------------------------
img = cv2.imread('Kmeans/home.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # siempre se carga en bgr hay que pasarlo en rgb 
plt.figure()
plt.imshow(img)
plt.show()

#en este ejemplo vamos a usar kmeans para ver los colres predominantes de la imagen
# --- Preparo datos ---------------------------------------------------------------------------
Z = img.reshape((-1,3))   # Genero una matriz de Nx3 con todos los valores de los pixels de la imagen (N = ancho x alto) -1 es para autocompletar con lo que corresponda, lo hace automaticamente
Z = np.float32(Z)         # Paso a float
Ncolors = len(np.unique(Z, axis=0)) # obtengo las filas unicas, serian los colores 

# --- Aplico K-means para obtener la paleta de colores (Cuantización de colores) --------------
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
K = 4
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# --- Obtengo la nueva imagen con la nueva paleta de colores ----------------------------------
center = np.uint8(center) 
aux = center[label.flatten()] # siempre indexamos con flat, vamos a hacer un reshape con los colores 
imgQ = aux.reshape((img.shape))

plt.figure()
ax1 = plt.subplot(121)
plt.xticks([]), plt.yticks([])
plt.imshow(img), plt.title(f'Imagen Original. Ncolors = {Ncolors}') # f print es dinamico, lo dinamico esta entre corchetes
plt.subplot(122, sharex=ax1, sharey=ax1), plt.imshow(imgQ), plt.title(f'Imagen con colores cuantizados --> K = {K}')
plt.show()

