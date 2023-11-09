import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, title=None, color_img=False, show=True, blocking=False):
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    if show:
        plt.show(block=blocking)

def imadjust(x, vin=[None,None], vout=[0,255], gamma=1):
    # x      : Imagen de entrada en escalas de grises (2D), formato uint8.
    # vin    : Límites de los valores de intensidad de la imagen de entrada
    # vout   : Límites de los valores de intensidad de la imagen de salida
    # y      : Imagen de salida
    if vin[0]==None:
        vin[0] = x.min()
    if vin[1]==None:
        vin[1] = x.max()
    y = (((x - vin[0]) / (vin[1] - vin[0])) ** gamma) * (vout[1] - vout[0]) + vout[0]
    y[x<vin[0]] = vout[0]   # Valores menores que low_in se mapean a low_out
    y[x>vin[1]] = vout[1]   # Valores mayores que high_in se mapean a high_out
    if x.dtype==np.uint8:
        y = np.uint8(np.clip(y+0.5,0,255))   # Numpy underflows/overflows para valores fuera de rango, se debe utilizar clip.
    return y

# --- Cargo Imagen ---------------------------------------------------------
f = cv2.imread('arte.jpg')
f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
imshow(f, title='Imagen Original')

# --- Filtro Ruido Salt-and-Pepper -----------------------------------------
f2 = cv2.medianBlur(f,3)
imshow(f2, title='Filtro de mediana')

# --- Sharping --------------------------------------------------------------
# https://en.wikipedia.org/wiki/Unsharp_masking
w = np.array([[-0.5, 0, -0.5], [0, 3.0, 0], [-0.5, 0, -0.5]])
img_fil = cv2.filter2D(f2, -1, w, borderType=cv2.BORDER_DEFAULT) 
imshow(img_fil, 'Sharping')

# --- Realce de contraste --------------------------------------------------------------
# I_con = imadjust(I_sharp, [0 0 0; .5 .5 .5], []);
# figure, imshow(I_con)
R,G,B = cv2.split(img_fil)
Radj = imadjust(R, vin=[0,127], vout=[0,255])
Gadj = imadjust(G, vin=[0,127], vout=[0,255])
Badj = imadjust(B, vin=[0,127], vout=[0,255])
f3 = cv2.merge((Radj,Gadj,Badj))
imshow(f3, 'Realce de contraste')

# --- Binarización por planos ------------------------------------------------------------
ix1 = (Radj>254) | (Radj<1)
ix2 = (Gadj>254) | (Gadj<1)
ix3 = (Badj>254) | (Badj<1)
ix = ix1 | ix2 | ix3
imshow(ix, title='Binarización')

# --- Eliminar objetos pequeños -----------------------------------------------------------
f5 = np.uint8(ix*255)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(f5, 8)       # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
labels_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)  # Muestro objetos conectados detectados
imshow(labels_color, title=f'Componentes Conectados ({num_labels})')                 #
imshow(labels, title=f'Componentes Conectados ({num_labels})', color_img=True)                 
# Elimino objetos chicos
f6 = labels.copy()
for ii in range(1,num_labels):
    if stats[ii][4] < 30:
        f6[f6==ii] = 0
f6_bin = np.uint8( (f6>0 )*255)
N = len(np.unique(f6))
imshow(f6_bin, title=f'Filtrado de elementos pequeños - Elementos detectados: {N}')
# Muestro resultados con boundingbox y label
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(f6_bin, 8)   # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
labels_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)
imshow(labels_color, title=f'Filtrado de elementos pequeños - Elementos detectados: {N}')
#
aux = cv2.merge((f6_bin.copy(), f6_bin.copy(), f6_bin.copy()))
for ii in range(1,num_labels):
    cv2.rectangle(aux, (stats[ii,0], stats[ii,1]), (stats[ii,0]+stats[ii,2], stats[ii,1]+stats[ii,3]), color=(0,255,0), thickness=1)
    cv2.putText(aux, f'{ii}', (centroids[ii,0].astype(int), centroids[ii,1].astype(int)) , cv2.FONT_HERSHEY_COMPLEX, 0.3, (255,0,0))
imshow(aux, color_img=True, title=f'Filtrado de elementos pequeños - Elementos detectados: {N}')

# --- Relleno huecos -----------------------------------------------------------
def fillhole(input_image):
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)  # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out 

tt = fillhole(f6_bin)
imshow(tt, title='Relleno huecos')

# --- Erosión -------------------------------------------------------------------
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
ff = cv2.erode(tt, se, iterations=1)
imshow(ff, title='Erosión')

# --- OBJETOS DETECTADOS - FINAL -------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ff, 8)
labels_color = cv2.applyColorMap(np.uint8(255/num_labels*labels), cv2.COLORMAP_JET)
imshow(labels_color, color_img=True, title=f'Elementos detectados: {num_labels}')
# Contornos
contours, hierarchy = cv2.findContours(ff, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
mm = cv2.merge((ff,ff,ff))
cv2.drawContours(mm, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)  # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
imshow(mm, color_img=True, title='Elementos detectados + Contornos')


# -----------------------------------------------------------------------------------------------------------
# --- Obtengo features --------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
X = [[]]*num_labels

# ---------------------------
# --- Rho -----------------------------------------------------------------------------------
# ---------------------------
#     rho = perim**2 / area
# Cuadrado = (4.L)**2 / L**2 = 16.L**2 / L**2 = 16
# Circulo  = (pi.2.R)**2 / (pi.R**2) = 4.pi**2.R**2 / pi.R**2 = 4.pi = 12.56
# Triangulo = (b.h/2) / (L1+L2+L3)
# Rectangulo = (2.L1 + 2.L2)**2 / (L1.L2)**2  =  4.(L1**2 + 2L1L2 + L2**2) / L1**2.L2**2
for ii in range(1,num_labels):
    # -- Obtengo el objeto y lo acondiciono ---------------------------------------
    obj = labels==ii
    obj8 = np.uint8((labels==ii)*255)
    contours, hierarchy = cv2.findContours(obj8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # -- Calculo características --------------------------------------------------
    area = np.sum(obj8>0)
    perimetro = cv2.arcLength(contours[0],True) # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga8d26483c636be6b35c3ec6335798a47c  /  https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    rho = (perimetro**2)/area
    # --- Guardo para analizar -------------------------------------------------------------------------------
    X[ii] = {'rho':rho, 'centroid':centroids[ii]}

aux = ((labels>0)*255).astype(np.uint8)
img = cv2.merge((aux, aux, aux))
for ii in range(1,num_labels):
    cv2.putText(img, f'{X[ii]["rho"]:5.2f}', (X[ii]['centroid'][0].astype(int), X[ii]['centroid'][1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,0,0), 0)
imshow(img, title='rho')
# *** Conclusion ***************************************
# rho < 20  --> Son circulos o cuadrados
# rho >=20  --> Son triángulos o rectangulos
# ******************************************************

# ---------------------------
# ---  Relacion entre el area del objeto y el area de un círculo de radio igual a la mitad de la maxima diagonal del objeto ----
# ---------------------------
for ii in range(1,num_labels):
    # -- Obtengo el objeto y lo acondiciono ---------------------------------------
    obj = labels==ii
    obj8 = np.uint8((labels==ii)*255)
    contours, hierarchy = cv2.findContours(obj8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # -- Calculo características --------------------------------------------------
    area = np.sum(obj8>0)
    bottom_left = (stats[ii,0], stats[ii,1])
    top_rigth = (stats[ii,0]+stats[ii,2], stats[ii,1]+stats[ii,3])
    diag = np.sqrt( (top_rigth[0] - bottom_left[0])**2 +  (top_rigth[1] - bottom_left[1])**2)
    pp = (np.pi * ((diag/2)**2)) / area
    # --- Guardo para analizar -------------------------------------------------------------------------------
    X[ii]["pp"] = pp

aux = ((labels>0)*255).astype(np.uint8)
img = cv2.merge((aux, aux, aux))
for ii in range(1,num_labels):
    if X[ii]['rho']<20:
        cv2.putText(img, f'{X[ii]["pp"]:5.2f}', (X[ii]['centroid'][0].astype(int), X[ii]['centroid'][1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,0,0), 0)
imshow(img, title='pp')
# *** Conclusion ***************************************
# pp > 2.6  --> Son cuadrados
# pp <=2.6  --> Son circulos
# ******************************************************

# ---------------------------
# --- Relacion entre el area del bounding box y el bounding box orientado -----------------------------------
# ---------------------------
for ii in range(1,num_labels):
    # -- Obtengo el objeto y lo acondiciono ---------------------------------------
    obj = labels==ii
    obj8 = np.uint8((labels==ii)*255)
    contours, hierarchy = cv2.findContours(obj8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # --- Bounding box orientado --------------------------------------------------
    rect = cv2.minAreaRect(contours[0])  # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    # -- Calculo características --------------------------------------------------
    area = np.sum(obj8>0)
    L1 = np.sum((box[0,:]-box[1,:])**2)**0.5
    L2 = np.sum((box[1,:]-box[2,:])**2)**0.5
    box_area = L1*L2
    rel = area/box_area
    # --- Guardo para analizar -------------------------------------------------------------------------------
    X[ii]["rel"] = rel
    X[ii]["box"] = box

aux = ((labels>0)*255).astype(np.uint8)
img = cv2.merge((aux, aux, aux))
for ii in range(1,num_labels):
    if X[ii]['rho']>=20:
        cv2.drawContours(img,[X[ii]["box"]],0,(0,0,255),1)
        cv2.putText(img, f'{X[ii]["rel"]:5.2f}', (X[ii]['centroid'][0].astype(int), X[ii]['centroid'][1].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255,0,0), 0)
imshow(img, title='rel')
# *** Conclusion ***************************************
# rel <  0.6  --> Son triángulos
# rel >= 0.6  --> Son rectángulos
# ******************************************************



# -----------------------------------------------------------------------------------------------------------
# --- Clasifico ---------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
Lc = np.zeros(labels.shape, dtype="uint8")  # Imagen donde estarán las etiquetas de los diferentes objetos
etiquetas = np.zeros(num_labels)            # Etiquetas de cada objeto conectado
for ii in range(1,num_labels):
    if X[ii]['rho'] < 20:           # --> Son circulos o cuadrados
        if X[ii]['pp'] > 2.6:  
            Lc[labels==ii] = 1      # Son cuadrados
            etiquetas[ii] = 1
        else:
            Lc[labels==ii] = 2      # Son círculos
            etiquetas[ii] = 2
    else:                           # --> Son rectángulos o triángulos
        if X[ii]['rel'] < 0.6:
            Lc[labels==ii] = 3      # Son Triangulos
            etiquetas[ii] = 3
        else:
            Lc[labels==ii] = 4      # Son Rectángulos
            etiquetas[ii] = 4


# --- Muestro en escala de grises ------------------------------------
imshow(Lc, title='Clasificación - escala de grises')
Lc_gray = ((Lc/Lc.max())*255).astype('uint8')
imshow(Lc_gray, title='Clasificación - escala de grises')

# --- Muestro en colores - version 1 ---------------------------------
Lc_color = cv2.applyColorMap(Lc_gray, cv2.COLORMAP_JET)
imshow(Lc_color, title='Clasificación - Colores')

# --- Muestro en colores - version 2 ---------------------------------
texto = ['FONDO', 'CUADRADOS', 'CÍRCULOS', 'TRIÁNGULOS', 'RECTÁNGULOS']
paleta = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])
Lc_colors = np.zeros((Lc.shape[0], Lc.shape[1], 3))
for ii in range(0,5):
    Lc_colors[Lc==ii,:] = paleta[ii]

plt.figure()
plt.imshow(Lc_colors)
plt.xticks([]), plt.yticks([])
D=25
for ii in range(1,5):
    plt.text(Lc_colors.shape[0]+10, D*ii, texto[ii] + f' ({np.sum(etiquetas==ii)})', fontsize=12, color=paleta[ii]/255, bbox=dict(facecolor=[0.6, 0.6, 0.6], edgecolor=paleta[ii]/255, pad=0.3, boxstyle='round'))
plt.title('Clasificación - Colores')
plt.show(block=False)

        


