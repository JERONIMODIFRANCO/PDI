import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Cargo Imagen ---------------------------------------------------------------
img = cv2.imread("Ejercicios\sunflower.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convierto a RGB
plt.figure(), plt.imshow(img), plt.show(block=False)

# --- Segmento en HSV ------------------------------------------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)                   # convierto a HSV

mask_verde = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))     # Mascara de verde:     (36,0,0) ~ (70, 255,255)
mask_amarillo = cv2.inRange(hsv, (15,0,0), (36, 255, 255))   # Mascara de amarillo:  (15,0,0) ~ (36, 255, 255)
mask = cv2.bitwise_or(mask_verde, mask_amarillo)             # Mascara Final

salida = cv2.bitwise_and(img,img, mask=mask)
salida_verde = cv2.bitwise_and(img,img, mask=mask_verde)
salida_amarillo = cv2.bitwise_and(img,img, mask=mask_amarillo)

plt.figure(), plt.imshow(salida), plt.show(block=False)

plt.figure()
ax1 = plt.subplot(231); plt.xticks([]), plt.yticks([]), plt.imshow(mask_verde, cmap="gray"), plt.title('Mascara Verde')
plt.subplot(232,sharex=ax1,sharey=ax1), plt.imshow(mask_amarillo, cmap="gray"), plt.title('Mascara Amarillo')
plt.subplot(233,sharex=ax1,sharey=ax1), plt.imshow(mask, cmap="gray"), plt.title('Mascara')
plt.subplot(234,sharex=ax1,sharey=ax1), plt.imshow(salida_verde), plt.title('Salida - Verde')
plt.subplot(235,sharex=ax1,sharey=ax1), plt.imshow(salida_amarillo), plt.title('Salida - Amarillo')
plt.subplot(236,sharex=ax1,sharey=ax1), plt.imshow(salida), plt.title('Salida')
plt.show(block=False)