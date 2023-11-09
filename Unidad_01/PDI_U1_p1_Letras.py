import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Cargo imagen ------------------------------------------------------------
img = cv2.imread('letras.png',cv2.IMREAD_GRAYSCALE) 
img.shape
plt.imshow(img, cmap='gray')
plt.show(block=False)                    

# *** Ejemplo de como dibujar lineas sobre una imagen *************
plt.imshow(img, cmap='gray')
plt.plot([0,100], [0, 50])
plt.plot([0, img.shape[1]-1], [0, img.shape[0]-1])
plt.show(block=False)                    
# *****************************************************************

# **** Ejemplo - búsqueda de columas con elementos != 0 ***********
a = np.array([[0,1,0],
              [0,1,0],
              [0,1,1],
              [0,1,0]])

# Busco columnas que poseen algún valor distinto de cero
a.any(axis=0)
b = np.argwhere(a.any(axis=0))

# Busco columnas que poseen algún valor particular
val = 0
b = a==val
b.any(axis=0)
np.argwhere(b.any(axis=0))
# *****************************************************************

# --- Analizo filas y columnas ------------------------------------------------
img_zeros = img==0
plt.imshow(img_zeros, cmap='gray'), plt.show()

# Columnas
img_col_zeros = img_zeros.any(axis=0)
img_col_zeros_idxs = np.argwhere(img_zeros.any(axis=0))
plt.imshow(img, cmap='gray')
xc = np.arange(img.shape[1])
yc = img_col_zeros*(img.shape[0]-1)
plt.plot(xc, yc, c='b')
plt.show(block=False)                    

# Filas
img_row_zeros = img_zeros.any(axis=1)
img_row_zeros_idxs = np.argwhere(img_zeros.any(axis=1))
xr = np.arange(img.shape[0])
yr = img_row_zeros*(img.shape[1]-1)
plt.plot(yr, xr, c='r')
plt.show(block=False)                    


# --- Separo en renglones -----------------------------------------------------

# *** Ejemplo *************************************************************************************
a = np.array([False,False,True,True,True,False,False])  # El resultado debería ser [2, 4]
ad = np.diff(a)         # [False,  True, False, False,  True, False]
ind = np.argwhere(ad)   # [1,4]
ind[0] += 1             # Modifico el indice inicial para que coincida con el resultado esperado
# *************************************************************************************************

x = np.diff(img_row_zeros)
renglones_indxs = np.argwhere(x) 
len(renglones_indxs)
# *** Modifico índices ***********
ii = np.arange(0, len(renglones_indxs),2)
renglones_indxs[ii]+=1
# ********************************
xx = np.arange(img.shape[0])
yy = np.zeros(img.shape[0])
yy[renglones_indxs] = (img.shape[1]-1)
plt.imshow(img, cmap='gray')
plt.plot(yy, xx, c='r')
plt.show(block=False)         

# Re-ordeno los índices en grupos de a 2 (inicio-final)
x_indxs = renglones_indxs[:(len(renglones_indxs)//2)*2]
x_indxs = x_indxs.reshape((-1,2))

# Obtengo renglones 
renglones = []
for ir, idxs in enumerate(x_indxs):
    # renglones.append(img[idxs[0]:idxs[1],:])
    renglones.append({
        "ir": ir+1,
        "cord": idxs,
        "img": img[idxs[0]:idxs[1],:]
    })

plt.figure()
for ii, renglon in enumerate(renglones):
    plt.subplot(2,2,ii+1)
    plt.imshow(renglon["img"], cmap='gray')
    plt.title(f"Renglón {ii+1}")
plt.show(block=False)       

# --- Analizo en renglones -----------------------------------------------------
letras = []
for ir, renglon in enumerate(renglones):
    renglon_zeros = renglon["img"]<=20  # Acondiciono imagen...

    # --- Analizo columnas del renglón ------------------------------
    ren_col_zeros = renglon_zeros.any(axis=0)
    ren_col_zeros_idxs = np.argwhere(renglon_zeros.any(axis=0))
    # *** Show *************************************
    plt.figure()
    plt.imshow(renglon_zeros, cmap='gray')
    xc = np.arange(renglon_zeros.shape[1])
    yc = ren_col_zeros*(renglon_zeros.shape[0]-1)
    plt.plot(xc, yc, c='b')
    plt.title(f"Renglón {ir+1}")
    plt.show()        
    # **********************************************
    
    # --- Separo en letras ------------------------------------------
    x = np.diff(ren_col_zeros)
    letras_indxs = np.argwhere(x) 
    # *** Modifico índices ***********
    ii = np.arange(0,len(letras_indxs),2)
    letras_indxs[ii]+=1
    # ********************************

    # Re-ordeno los índices en grupos de a 2 (inicio-final)
    letras_indxs = letras_indxs[:(len(letras_indxs)//2)*2]
    letras_indxs = letras_indxs.reshape((-1,2))

    # Obtengo letras del renglon
    letras_ren = []
    for idxs in letras_indxs:
        letras_ren.append(renglon["img"][: , idxs[0]:idxs[1]])

    # *** Show ********************************************
    plt.figure()
    Nrows = len(letras_ren)//4 + len(letras_ren)%4
    plt.suptitle(f"Renglón {ir+1}")
    for ii, letra in enumerate(letras_ren):
        plt.subplot(Nrows, 4, ii+1)
        plt.imshow(letra, cmap='gray')
        plt.title(f"letra {ii+1}")
    plt.show()
    # ******************************************************

    # --- Guardo letras ---------------------------------------------
    for il, idxs in enumerate(letras_indxs):
        letras.append({
            "ir":ir+1,
            "ir_l":il+1,
            "cord": [renglon["cord"][0], idxs[0], renglon["cord"][1], idxs[1]],
            "img": renglon["img"][:, idxs[0]:idxs[1]]
        })


# --- Agrego índice de letras independiente del renglón ---------------------
for il, letra in enumerate(letras):
    letra.update({"il":il+1})

# --- Imagen final -----------------------------------------------------------
from matplotlib.patches import Rectangle
plt.figure()
plt.imshow(img, cmap='gray')
for il, letra in enumerate(letras):
    yi = letra["cord"][0]
    xi = letra["cord"][1]
    W = letra["cord"][2] -letra["cord"][0]
    H = letra["cord"][3] -letra["cord"][1]
    rect = Rectangle((xi,yi),H,W,linewidth=1,edgecolor='r',facecolor='none')    # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
    ax = plt.gca()
    ax.add_patch(rect) 

plt.show()
