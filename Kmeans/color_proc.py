import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image   # Pillow --> https://pillow.readthedocs.io/en/stable/

img = cv2.imread('peppers.png')
plt.figure(1)
plt.imshow(img)
plt.show()

# --- Acomodamos canales ----------------------------------------------------------------
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(2)
plt.imshow(img_RGB)
plt.show()

# --- Separar canales -------------------------------------------------------------------
B, G, R = cv2.split(img)
plt.figure(3)
plt.imshow(R, cmap='gray')
plt.title("Canal R")
plt.show()

ax1 = plt.subplot(221)
plt.xticks([]), plt.yticks([]), plt.imshow(img_RGB), plt.title('Imagen RGB')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(R,cmap='gray'), plt.title('Canal R')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(G,cmap='gray'), plt.title('Canal G')
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(B,cmap='gray'), plt.title('Canal B')
plt.show()


# --- Modifico un canal ---------------------------------------------------------------
# img2 = img_RGB    # No! así crea una referencia: si se modifica una, se modifica la otra también.  
img2 = img_RGB.copy()  # Así crea una copia. Otra forma sería "img2 = np.array(img_RGB)"
img2[:,:,0] = 0
plt.figure, plt.imshow(img2), plt.title('Canal R anulado')
plt.show()

R2 = R.copy()
R2 = R2*0.5
R2 = R2.astype(np.uint8)
img3 = cv2.merge((R2,G,B))
plt.figure, plt.imshow(img3), plt.title('Canal R escalado')
plt.show()

ax1 = plt.subplot(221)
plt.xticks([]), plt.yticks([]), plt.imshow(img_RGB), plt.title('Imagen RGB')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img2,cmap='gray'), plt.title('Canal R anulado')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img3), plt.title('Canal R escalado')
plt.show()


# --- Dithering ------------------------------------------------------------------------
img_PIL = Image.open('cameraman.tif')
image_dithering = img_PIL.convert(mode='1', dither=Image.FLOYDSTEINBERG)   # https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=convert#PIL.Image.Image.convert
plt.figure(4)
ax1 = plt.subplot(121)
plt.xticks([]), plt.yticks([])
plt.imshow(img_PIL, cmap='gray'), plt.title('Imagen original')
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(image_dithering), plt.title('Imagen con dithering')
plt.show()
# -----------------------------------
# --- Analisis ----------------------
# -----------------------------------
# -- Imagen Original --------
img_PIL.size
x = img_PIL.getdata()
list_of_pixels = list(x)
len(list_of_pixels)
list_of_pixels[:5]
# print(list_of_pixels)
Ncolors = len(list(set(list_of_pixels)))
# -- Imagen Procesada --------
list_of_pixels_out = list(image_dithering.getdata())
len(list_of_pixels_out)
list_of_pixels_out[:5]
Ncolors_out = len(list(set(list_of_pixels_out)))
# -----------------------------------
# -----------------------------------
# -----------------------------------


# Color
img_PIL = Image.open('peppers.png')
# image_dithering = img_PIL.convert(mode='P', palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG)  # https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
image_dithering = img_PIL.convert(mode='P', palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG, colors=4)
plt.figure(5)
ax1 = plt.subplot(121)
plt.xticks([]), plt.yticks([])
plt.imshow(img_PIL), plt.title('Imagen original')
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(image_dithering), plt.title('Imagen con dithering')
plt.show()
# -----------------------------------
# --- Analisis ----------------------
# -----------------------------------
# -- Imagen Original --------
img_PIL.size
x = img_PIL.getdata()
list_of_pixels = list(x)
len(list_of_pixels)
list_of_pixels[:5]
# print(list_of_pixels)
Ncolors = len(list(set(list_of_pixels)))
# -- Imagen Procesada --------
list_of_pixels_out = list(image_dithering.getdata())
len(list_of_pixels_out)
list_of_pixels_out[:5]
Ncolors_out = len(list(set(list_of_pixels_out)))

image_dithering.getcolors() # [ ( count, index ), ( count, index ), ... ]
palette = np.array(image_dithering.getpalette(),dtype=np.uint8).reshape((256,3))
palette[0:4,]

# Paso a RGB
image_dithering_RGB = np.array(image_dithering.convert('RGB'))  # Paso a RGB
colours, counts = np.unique(image_dithering_RGB.reshape(-1,3), axis=0, return_counts=1)    # Obtengo colores y cuentas

# Obtengo índices
image_dithering_indexs = np.array(image_dithering.convert('L'))  # Matriz de índices
indexs, counts = np.unique(image_dithering_indexs, return_counts=1)
plt.figure, plt.imshow(image_dithering_indexs, cmap='gray'), plt.colorbar(), plt.show()
# -----------------------------------
# -----------------------------------
# -----------------------------------

# --- Espacio de color HSV ----------------------------------------------
img = cv2.imread('peppers.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
h, s, v = cv2.split(img_hsv)
plt.figure(6)
plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.imshow(h, cmap='gray'), plt.title('Canal H')
plt.subplot(223), plt.imshow(s, cmap='gray'), plt.title('Canal S')
plt.subplot(224), plt.imshow(v, cmap='gray'), plt.title('Canal V')
plt.show()

# Segmentacion en color - Detectar solo el rojo
ix_h1 = np.logical_and(h > 180 * .9, h < 180)
ix_h2 = h < 180 * 0.04
ix_s = np.logical_and(s > 255 * 0.3, s < 255)
ix = np.logical_and(np.logical_or(ix_h1, ix_h2), ix_s)
# ix2 = (ix_h1 | ix_h2) & ix_s   # Otra opcion que da igual...

r, g, b = cv2.split(img)
r[ix != True] = 0
g[ix != True] = 0
b[ix != True] = 0
rojo_img = cv2.merge((r, g, b))
plt.figure(7)
plt.imshow(rojo_img)
plt.show()

# --- Filtrado espacial ----------------------------------------------------------------
img = cv2.imread('peppers.png')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Usando kernel y filter 2D
kernel = np.ones((5, 5), np.float32)/25
img_filt = cv2.filter2D(img_RGB, -1, kernel)
plt.figure(8)
plt.imshow(img_filt)
plt.show()

# Funciones filtrado
gblur = cv2.GaussianBlur(img_RGB, (55, 55), 0)
median = cv2.medianBlur(img_RGB, 5)
blur = cv2.blur(img_RGB, (55, 55))
plt.figure(9)
plt.subplot(221), plt.imshow(img_RGB), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(blur), plt.title('Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(gblur), plt.title('Gaussian blur'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(median), plt.title('Median blur'), plt.xticks([]), plt.yticks([])
plt.show()

# Filtrado Espacial - High Boost
img = cv2.imread('Fig0604(a).tif')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
w1 = np.ones((3, 3), np.float32)/9
w2 = np.ones((3, 3), np.float32)  # Laplaciano  
w2[1,1] = -8                      #
# -----------------
def im2double(im):
    info = np.iinfo(im.dtype) 
    return im.astype(np.float) / info.max 

img_RGB = im2double(img_RGB)
# ------------------
img_pb = cv2.filter2D(img_RGB, -1, w1)
img_en = img_pb - cv2.filter2D(img_pb, -1, w2)
plt.figure(10)
ax1 = plt.subplot(221)
plt.imshow(img_RGB)
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_pb), plt.title('Filtro Pasa-Bajos en todos los canales'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_en), plt.title('Mejorada utilizando Laplaciano'), plt.xticks([]), plt.yticks([])
plt.show()

