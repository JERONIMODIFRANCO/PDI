import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Proceso argumentos de entrada
parser = argparse.ArgumentParser()              # Creo el parser
parser.add_argument('--src', required=True)     # Agrego un parámetro de entrada
args = parser.parse_args()                      # Proceso entradas
src_path = args.src                             # Asigno el parámetro de entrada a la variable

# src_path = 'celulas.tiff'

# Load image
source_img = np.float32(cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)/255.0)
plt.figure()
plt.imshow(source_img, cmap='gray')
# plt.show()

# Binarize
_, binary_img = cv2.threshold(source_img, 0.5, 1, cv2.THRESH_BINARY)
print(binary_img.shape)
binary_img = binary_img.astype(np.uint8)
plt.figure()
plt.imshow(binary_img, cmap='gray')
# plt.show()

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

# Factor de forma (rho)
RHO_TH = 0.8

# Umbral de area
AREA_TH = 500

H, W = source_img.shape[:2]

aux = np.zeros_like(labels)
labeled_image = cv2.merge([aux, aux, aux])

#
# Clasifico en base al factor de forma
for i in range(1, len(stats)):
    
    # plt.figure()
    # plt.imshow(labels==i, cmap="gray")
    # plt.show()  

    # Remuevo las celulas que tocan el borde ded la imagen
    if (stats[i, cv2.CC_STAT_LEFT] == 0 or stats[i, cv2.CC_STAT_TOP] == 0 or stats[i, cv2.CC_STAT_HEIGHT] + stats[i, cv2.CC_STAT_TOP] == H
            or stats[i, cv2.CC_STAT_WIDTH] + stats[i, cv2.CC_STAT_LEFT] == W):
        labels[labels == i] = 0
        continue

    # Remuevo celulas con area chica
    if (stats[i, cv2.CC_STAT_AREA] < AREA_TH):
        labels[labels == i] = 0
        continue

    obj = (labels == i).astype(np.uint8)
    contour_img = np.zeros_like(obj)

    # Calculo Rho
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_contours[0])
    perimeter = cv2.arcLength(ext_contours[0], True)
    rho = 4 * np.pi * area/(perimeter**2)

    flag_circular = rho > RHO_TH

    # Calculo cantidad de huecos
    all_contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = len(all_contours) - 1
    print(str(i) + " Circular: " + str(flag_circular) + " Huecos: " + str(holes))

    # Clasifico
    if flag_circular:
        if holes == 1:
            labeled_image[obj == 1, 0] = 255    # Circular con 1 hueco
        else:
            labeled_image[obj == 1, 1] = 255    # Circular con mas de 1 hueco
    else:
        labeled_image[obj == 1, 2] = 255        # No circular

plt.figure()
plt.imshow(labeled_image)

plt.show()
