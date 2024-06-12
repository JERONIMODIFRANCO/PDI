import numpy as np
import cv2
from PIL import Image, ImageDraw

# --- Input ---------------------------------------------------------------------------------
file_name = "ruta_1.mp4"

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

    # cv2.imshow("Mask", mask)
    cv2.imshow("ROI", img_roi)
    # cv2.imshow("Canny",img_edg)
    # cv2.imshow("ROI", img_bw)

    # Apply Hough transform
    lines = cv2.HoughLinesP(img_roi, rho=1.5, theta=np.pi/180, threshold=75, minLineLength=5, maxLineGap=450)
    # lines = cv2.HoughLinesP(img_roi, rho=1.5, theta=np.pi/180, threshold=70, minLineLength=5, maxLineGap=450) mejor resultado
    # Draw the detected lines on the original image

    # linea derecha
    x1_der=[]
    x2_der=[]
    y1_der=[]
    y2_der=[]

    x1_d_med=0
    x2_d_med=0
    y1_d_med=0
    y2_d_med=0

    # linea izquierda
    x1_izq=[]
    x2_izq=[]
    y1_izq=[]
    y2_izq=[]

    x1_i_med=0
    x2_i_med=0
    y1_i_med=0
    y2_i_med=0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 > 480:
            # x2 = x2 + int((x2/y2)*(539-y2))
            if x2 >= 960:
                x2=959
            y2 = 539
            print('derecha')
            print(x1,y1,x2,y2)
            x2_der.append(x2)                    #append(x) agrega el elemento x al final de la lista x2_der
            x1_der.append(x1)
            y1_der.append(y1)
            y2_der.append(y2)
            
        else:
            # x1 = x1 - int((x2/y2)*(539-y1))
            if x1 <= 0:
                x1=2
            y1 = 539
            print('izq gato')
            print(x1,y1,x2,y2)
            x2_izq.append(x2)
            x1_izq.append(x1)
            y1_izq.append(y1)
            y2_izq.append(y2)


    # media linea derecha
    x1_d_med=int(np.mean(np.array(x1_der)))
    x2_d_med=int(np.mean(np.array(x2_der)))
    y1_d_med=int(np.mean(np.array(y1_der)))
    y2_d_med=int(np.mean(np.array(y2_der)))

    # media linea izquierda
    x1_i_med=int(np.mean(np.array(x1_izq)))
    x2_i_med=int(np.mean(np.array(x2_izq)))
    y1_i_med=int(np.mean(np.array(y1_izq)))
    y2_i_med=int(np.mean(np.array(y2_izq)))
               
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