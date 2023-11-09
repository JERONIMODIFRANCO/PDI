import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos fuinción para mostrar imágenes
def imshow(img, title=None, color_img=False, blocking=False):
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show(block=blocking)


# --- Cargo Imagen --------------------------------------------------------------
f = cv2.imread('Fig1006(a).tif')            # Leemos imagen
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises
imshow(gray)

# --- SOBEL ---------------------------------------------------------------------
ddepth = cv2.CV_16S  # Formato salida
grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3) # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3) # Tutorial: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html

# Pasamos a 8 bit
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
cv2.imshow("Sobel dx", abs_grad_x)
cv2.imshow("Sobel dy", abs_grad_y)

# Sumamos los gradientes en una nueva imagen
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
cv2.imshow("Sobel", grad)

# --- Umbralizamos los gradientes -------------------------------
abs_grad_x_th = np.zeros(abs_grad_x.shape) 
abs_grad_x_th[abs_grad_x >= abs_grad_x.max()] = 255
cv2.imshow("sobel x + umbral", abs_grad_x_th)

abs_grad_y_th = np.zeros(abs_grad_y.shape) 
abs_grad_y_th[abs_grad_y == abs_grad_y.max()] = 255
cv2.imshow("sobel y + umbral", abs_grad_y_th)

grad_th = np.zeros(grad.shape) 
grad_th[grad >= 0.5*grad.max()] = 255
cv2.imshow("sobel x+y + umbral", grad_th)

# --- Mas parecido al resultado de Matlab... ----------------------------------------------------
x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
xa = np.abs(x)
# imshow(xa)
xa1 = np.zeros(x.shape) 
xa1[xa  >= 0.33*xa.max()] = 255
imshow(xa1)


# --- CANNY ---------------------------------------------------------------------------------------
f_blur = cv2.GaussianBlur(f, ksize=(3, 3), sigmaX=1.5)
gcan = cv2.Canny(f_blur, threshold1=0.04*255, threshold2=0.1*255)
gcan = cv2.Canny(f_blur, threshold1=0.4*255, threshold2=0.75*255)
cv2.imshow("Canny", gcan)


# --- Contornos -----------------------------------------------------------------------------------
f = cv2.imread('contornos.png')             # Leemos imagen
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises
cv2.imshow('contornos', gray)

umbral, thresh_img = cv2.threshold(gray, thresh=128, maxval=255, type=cv2.THRESH_OTSU)  # Umbralamos
cv2.imshow('Umbral', thresh_img)

# Tutorial: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga95f5b48d01abc7c2e0732db24689837b

# Dibujamos
f = cv2.imread('contornos.png')
cv2.drawContours(f, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)  # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
cv2.imshow('Contornos', f)

# Contornos externos
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# Dibujamos
f = cv2.imread('contornos.png')
cv2.drawContours(f, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
cv2.imshow('contours externos', f)

# Contornos por jerarquía
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(hierarchy)    # hierarchy: [Next, Previous, First_Child, Parent]

# --- Contornos que no tienen padres ----------------------------------------------
f = cv2.imread('contornos.png')
for ii in range(len(contours)):
    if hierarchy[0][ii][3]==-1:
        cv2.drawContours(f, contours, contourIdx=ii, color=(0, 255, 0), thickness=2)
cv2.imshow('Contornos sin padre', f)

# --- Contornos que no tienen hijos ----------------------------------------------
f = cv2.imread('contornos.png')
for ii in range(len(contours)):
    if hierarchy[0][ii][2]==-1:
        cv2.drawContours(f, contours, contourIdx=ii, color=(0, 255, 0), thickness=2)
cv2.imshow('Contorno sin hijos', f)

# --- Ejemplo particular ---------------------------------------------------------
k = 4 
f = cv2.imread('contornos.png')
cv2.drawContours(f, contours, contourIdx=k, color=(0, 255, 0), thickness=2)
cv2.imshow('Contorno particular', f)
print(hierarchy[0][k])

# Dibujo al padre en azul
if hierarchy[0][k][3] != -1:
    cv2.drawContours(f, contours, contourIdx=hierarchy[0][k][3], color=(255, 0, 0), thickness=2)
cv2.imshow('Contorno particular', f)

# Dibujo los hijos en rojo
for ii in range(len(contours)):
    if hierarchy[0][ii][3]==k:
        cv2.drawContours(f, contours, contourIdx=ii, color=(0, 0, 255), thickness=2)
cv2.imshow('Contorno particular', f)

# Dibujo todos los que están en su mismo nivel
for ii in range(len(contours)):
    if hierarchy[0][ii][3]==hierarchy[0][k][3]:
        cv2.drawContours(f, contours, contourIdx=ii, color=(0, 255, 255), thickness=2)
cv2.imshow('Contorno particular', f)


# --- Ordeno según los contornos mas grandes -------------------------------------
contours_area = sorted(contours, key=cv2.contourArea, reverse=True)
f = cv2.imread('contornos.png')
cv2.drawContours(f, contours_area, contourIdx=0, color=(255, 0, 0), thickness=2)
cv2.drawContours(f, contours_area, contourIdx=1, color=(0, 255, 0), thickness=2)
cv2.drawContours(f, contours_area, contourIdx=2, color=(0, 0, 255), thickness=2)
cv2.imshow('Contorno ordenados por area', f)


# -- Aproximación de contornos con polinomios ----------------------------------
# cnt = contours[2] # Rectángulo
cnt = contours[12]  # Círculo
f = cv2.imread('contornos.png')
cv2.drawContours(f, cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
cv2.imshow('Aproximacion de contorno', f)

approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)   # https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
len(cnt)
len(approx) 

cv2.drawContours(f, approx, contourIdx=-1, color=(0, 0, 255), thickness=2)
imshow(f)

# Bounding Box
x,y,w,h = cv2.boundingRect(cnt)
f = cv2.imread('contornos.png')
cv2.drawContours(f, cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
cv2.rectangle(f, (x,y), (x+w,y+h), color=(255, 0, 255), thickness=2)
cv2.imshow('boundingRect', f)


# Momentos del contorno
M=cv2.moments(cnt)
huMoments = cv2.HuMoments(M)  # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gab001db45c1f1af6cbdbe64df04c4e944


# --- Hough Lineas --------------------------------------------------------------------------------
# Tutorial: 
#   https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
#   https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
f = cv2.imread('contornos.png')             # Leemos imagen
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises
cv2.imshow('imagen', gray)

edges = cv2.Canny(gray, 100, 170, apertureSize=3)
cv2.imshow('imagen', edges)

lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=250)   # https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
# lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)   # https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]        
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*(a))
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*(a))
    cv2.line(f,(x1,y1),(x2,y2),(0,255,0),2)

# cv2.imshow('hough lines', f)
imshow(f)



for i in range(0, len(lines)):
    f = cv2.imread('contornos.png')             # Leemos imagen
    rho = lines[i][0][0]
    theta = lines[i][0][1]        
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*(a))
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*(a))
    cv2.line(f,(x1,y1),(x2,y2),(0,255,0),2)
    imshow(f)
