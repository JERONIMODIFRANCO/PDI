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
    # cv2.imshow("Original", img)
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

    # cv2.imshow("Mask", mask)
    cv2.imshow("ROI", img_roi)
    # cv2.imshow("Canny",img_edg)
    # cv2.imshow("ROI", img_bw)

    # Apply Hough transform
    lines = cv2.HoughLinesP(img_roi, rho=1.5, theta=np.pi/180, threshold=80, minLineLength=5, maxLineGap=450)
    # lines = cv2.HoughLinesP(img_roi, rho=1.5, theta=np.pi/180, threshold=70, minLineLength=5, maxLineGap=450) mejor resultado
    # Draw the detected lines on the original image

    # linea derecha
    x1_der_sum=0
    x2_der_sum=0
    y1_der_sum=0
    y2_der_sum=0

    x1_d_med=0
    x2_d_med=0
    y1_d_med=0
    y2_d_med=0

    # linea izquierda
    x1_izq_sum=0
    x2_izq_sum=0
    y1_izq_sum=0
    y2_izq_sum=0

    x1_i_med=0
    x2_i_med=0
    y1_i_med=0
    y2_i_med=0
    k=0
    j=0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 > 480:
            x2 = x2 + int((x2/y2)*(539-y2))
            if x2 >= 960:
                x2=959
            y2 = 539
            x1_der_sum=x1_der_sum+x1  
            x2_der_sum=x2_der_sum+x2               #append(x) agrega el elemento x al final de la lista x2_der  
            y1_der_sum=y1_der_sum+y1              
            y2_der_sum=y2_der_sum+y2 
            k+=1
        else:
            x1 = x1 - int((x2/y2)*(539-y1))
            if x1 <= 0:
                x1=1
            y1 = 539
            x1_izq_sum=x1_izq_sum+x1  
            x2_izq_sum=x2_izq_sum+x2               #append(x) agrega el elemento x al final de la lista x2_der  
            y1_izq_sum=y1_izq_sum+y1              
            y2_izq_sum=y2_izq_sum+y2 
            j+=1

    if k==0:
        k=1
    if j==0:
        j=1

    # media linea derecha
    x1_d_med=int(x1_der_sum/k)
    x2_d_med=int(x2_der_sum/k)
    y1_d_med=int(y1_der_sum/k)
    y2_d_med=int(y2_der_sum/k)

    # media linea izquierda
    x1_i_med=int(x1_izq_sum/j)
    x2_i_med=int(x2_izq_sum/j)
    y1_i_med=int(y1_izq_sum/j)
    y2_i_med=int(y2_izq_sum/j)

    # grafico lineas
    cv2.line(img, (x1_d_med, y1_d_med), (x2_d_med, y2_d_med), (255, 0, 0), 10)   # linea derecha
    cv2.line(img, (x1_i_med, y1_i_med), (x2_i_med, y2_i_med), (255, 0, 0), 10)   # linea izquierda

    cv2.imshow("Imagen con lineas de carril", img)
        
    # --- Para salir del algoritmo ----------------------------------------------------------- 
    if cv2.waitKey(25) & 0xFF==ord('q'):     # Si la 'q' es pulsada, salimos del programa
        break
capture.release()
# out.release()
cv2.destroyAllWindows()

