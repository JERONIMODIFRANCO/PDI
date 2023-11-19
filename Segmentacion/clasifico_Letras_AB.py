"""
Clasificador de letras "A" y "B"
Uso:
    python clasifico_Letras_AB.py --src <image_path>

Ejemplo:
    python clasifico_Letras_AB.py --src letrasAB.png
"""

import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Proceso argumentos de entrada
# parser = argparse.ArgumentParser()              # Creo el parser
# parser.add_argument('--src', required=True)     # Agrego un parámetro de entrada
# args = parser.parse_args()                      # Proceso entradas
# src_path = args.src                             # Asigno el parámetro de entrada a la variable

src_path = 'letrasAB.png'

# Cargo imagen
source_img = cv2.imread(src_path, cv2.IMREAD_COLOR)
plt.figure()
plt.imshow(source_img, cmap='gray')
# plt.show()

# Filtro
filtered_img = cv2.medianBlur(source_img, 5) # filtro de mediana
plt.figure()
plt.imshow(filtered_img, cmap='gray')
# plt.show()

filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
plt.figure()
plt.imshow(filtered_img, cmap='gray')
# plt.show()

# Binarizo
_, binary_img = cv2.threshold(filtered_img, 125, 1, cv2.THRESH_BINARY)

# Operaciones morfológicas para mejorar la segmentación obtenida
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, se)   # Apertura para remover elementos pequeños
binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, se)  # Clausura para rellenar huecos.

plt.figure()
plt.imshow(binary_img, cmap='gray')
# plt.show()

num_labels, labels = cv2.connectedComponents(binary_img)
labeled_shapes = np.zeros((labels.shape[0], labels.shape[1], 3))
# Clasifico
for i in range(1, num_labels):
    obj = (labels == i).astype(np.uint8)
    contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
    if len(contours) == 3:
        labeled_shapes[obj == 1, 2] = 255  # Si tiene 3 contornos --> B 
    else:
        labeled_shapes[obj == 1, 0] = 255  # Caso contrario --> A

plt.figure()
plt.imshow(labeled_shapes)
# LOS CONTORNOS NO SE VINCULAN EN ORDEN CON LOS OBJETOS SEGMENTADOS, ES DECIR CONTORNO 1 NO NECESARIAMENTE CORRESPONDE  CON OBJETO 1
plt.show()
