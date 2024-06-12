### Documentación del Código
### Descripción del Código
### Funciones Auxiliares
Se define una función `imshow` para mostrar imágenes utilizando `matplotlib`.

### Operaciones Lógicas
Se cargan dos imágenes `A` y `B`, se realiza la negación de `A`, la unión lógica (`AuB`), la intersección lógica (`AiB`) y la diferencia lógica (`AmB`). Cada resultado se muestra utilizando `imshow`.

### Operaciones Morfológicas
### Dilatación
Se carga una imagen `F`, se define un kernel y se aplica dilatación utilizando `cv2.dilate`. Se muestran la imagen original y la imagen dilatada.

### Erosión
Se carga una imagen `A`, se define un kernel elíptico y se aplica erosión utilizando `cv2.erode`. Se muestran la imagen original y la imagen erosionada.

### Apertura
Se carga una imagen `A`, se define un kernel rectangular y se aplica apertura utilizando `cv2.morphologyEx`. Se muestran la imagen original y la imagen con apertura.

### Cierre
Se carga una imagen `A`, se define un kernel rectangular y se aplica cierre utilizando `cv2.morphologyEx`. Se muestran la imagen original y la imagen con cierre.

### Apertura y Cierre
Se carga una imagen `f`, se definen dos kernels y se aplica apertura seguida de cierre utilizando `cv2.morphologyEx`. Se muestran la imagen original, la imagen con apertura y la imagen con apertura y cierre.

### Hit or Miss
Se carga una imagen `f` en escala de grises, se define un kernel y se aplica la transformación Hit or Miss utilizando `cv2.morphologyEx`. Se muestran la imagen original y el resultado de la transformación.

### Componentes Conectadas
Se carga una imagen `img` en escala de grises, se etiquetan los componentes conectados utilizando `cv2.connectedComponentsWithStats`. Se colorean los componentes y se muestran los centroides y rectángulos delimitadores.

### Reconstrucción por Dilatación
Se carga una imagen `f`, se erosiona y se abre utilizando `cv2.erode` y `cv2.morphologyEx`. Se implementa la reconstrucción por dilatación utilizando una función auxiliar `imreconstruct`. Se muestran la imagen original, la erosionada, la abierta y el resultado de la reconstrucción.

### Morfología en Escala de Grises
Se carga una imagen `f` en escala de grises, se umbraliza utilizando `cv2.threshold`, se abre utilizando `cv2.morphologyEx`, se calcula la diferencia absoluta entre la imagen original y la abierta, y se aplica la transformación Top-Hat utilizando `cv2.morphologyEx`. Se muestran la imagen original, la umbralizada, la diferencia y el resultado de la transformación Top-Hat.

### Entrada
El script carga las imágenes desde archivos especificados en el código.

### Salida
La salida del script consiste en múltiples gráficos de dispersión que muestran las imágenes originales, procesadas y los resultados de los diferentes pasos del código.