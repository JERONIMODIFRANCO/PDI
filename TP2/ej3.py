import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    # Display the resulting edge image
    cv2.imshow(window_title, img_name)

    # Wait for a key press and then close the window
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        
# --- Load image ---------------------------------------------------------------------------------
img = cv2.imread('TP2/Patentes/img01.png', cv2.IMREAD_COLOR)
# Get the size of the image
height, width, channels = img.shape
# Print the size of the image
print(f'The size of the image is {width}x{height} pixels with {channels} channels.')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# imshow(img, title="Imagen Original", ticks=True)
# --- Convert to grayscale ------------------------------------------------------------------------
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imshow(img_gray, title="Imagen en escala de grises", ticks=True)
# Apply adaptive thresholding
img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
imshow(img_bw, title="Imagen binarizada")
# img_bw=~img_bw
# --- Find contours ------------------------------------------------------------------------------
contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# --- Find contours with an area greater than 1200 and smaller than a specified max area ------------
min_area = 1000
max_area = 2000
selected_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) ]
cv2.drawContours(img, selected_contours, -1, (0, 0, 255), 1)
imshow(img, title="Contornos con área mayor a {}".format(min_area))

