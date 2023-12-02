import numpy as np
import cv2
from PIL import Image, ImageDraw

# --- Input ---------------------------------------------------------------------------------
file_name = "ruta_2.mp4"

# --- Proceso video -------------------------------------------------------------------------
capture = cv2.VideoCapture(file_name)
while True:
    # --- Obtengo frame -----------------------------------------------------------------
    ret, img = capture.read()
    if not ret: 
        break
    cv2.imshow("Original", img)
    # cv2.line(img,(50,50),(500,500),(0,0,255),2)
    # cv2.imshow("linea", img)
    # --- ROI --------------------------------------------------------------------------
    dims = img.shape
    left_bottom = [0, dims[0]-1]
    left_top = [450, 320]
    right_top = [520, 320]
    right_bottom = [dims[1]-1, dims[0]-1]
    roi_vertices = np.array([left_bottom, left_top, right_top, right_bottom])
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # edges
    img_edg = cv2.Canny(img_gray,150,200)
    mask = np.zeros(img.shape[:-1], dtype=np.uint8) # con el -1 nos quedamos con todo menos el numero de canales
    cv2.fillPoly(mask, [roi_vertices], 255)
    img_roi = cv2.bitwise_and(img_edg, img_edg, mask=mask)

    # bw
    # _, img_bw = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY) # 124

    cv2.imshow("Mask", mask)
    cv2.imshow("ROI", img_roi)
    # cv2.imshow("Canny",img_edg)
    # cv2.imshow("ROI", img_bw)

    # Apply Hough transform
    lines = cv2.HoughLinesP(img_roi, rho=2, theta=np.pi/180*2, threshold=100, minLineLength=10, maxLineGap=100)

    # Draw the detected lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 > 480:
            alfa = np.arctan(x2/y2)
            x2 = x2 + int(np.tan(alfa)*(539-y2))
            y2 = 539
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.imshow("Imagen con lineas de carril", img)

    
    # --- Para salir del algoritmo ----------------------------------------------------------- 
    if cv2.waitKey(25) & 0xFF==ord('q'):     # Si la 'q' es pulsada, salimos del programa
        break
capture.release()
# out.release()
cv2.destroyAllWindows()






