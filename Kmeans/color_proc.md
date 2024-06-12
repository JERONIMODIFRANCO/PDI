### Documentación del Código
### Descripción del Código
### Carga de Imagen y Preprocesamiento
### Carga de Imagen
Este script utiliza OpenCV para cargar una imagen y mostrarla utilizando `matplotlib`.

### Acomodado de Canales
La imagen se convierte de espacio de color BGR a RGB utilizando `cv2.cvtColor` y se muestra nuevamente.

### Separación de Canales
Los canales de la imagen se separan utilizando `cv2.split` y se muestran individualmente en escala de grises.

### Modificación de Canales
Se crea una copia de la imagen RGB y se modifica el canal rojo. Luego, se muestra la imagen modificada.

### Dithering
Se carga una imagen en escala de grises utilizando `Image.open` y se aplica dithering utilizando `Image.convert`. Se muestran la imagen original y la imagen con dithering.

### Análisis de Imagen
Se realiza un análisis de la imagen original y la imagen procesada, incluyendo el número de colores únicos.

### Dithering en Color
Se carga una imagen en color utilizando `Image.open` y se aplica dithering utilizando `Image.convert`. Se muestran la imagen original y la imagen con dithering.

### Análisis de Imagen en Color
Se realiza un análisis de la imagen original y la imagen procesada en color, incluyendo el número de colores únicos y la paleta de colores.

### Espacio de Color HSV
La imagen se convierte del espacio de color RGB a HSV utilizando `cv2.cvtColor`. Los canales H, S y V se separan y se muestran individualmente en escala de grises.

### Segmentación en Color
Se detecta el color rojo en la imagen utilizando condiciones en los canales H y S. Se muestra la imagen con solo el rojo resaltado.

### Filtrado Espacial
Se aplican diferentes filtros espaciales a la imagen utilizando `cv2.filter2D`, `cv2.GaussianBlur`, `cv2.medianBlur` y `cv2.blur`. Se muestran la imagen original y las imágenes filtradas.

### Filtrado Espacial - High Boost
Se aplica un filtro pasa-altos a la imagen utilizando un kernel Laplaciano. Se muestran la imagen original, la imagen filtrada pasa-bajos y la imagen mejorada utilizando el filtro High Boost.

### Entrada
El script carga las imágenes desde archivos especificados en el código.

### Salida
La salida del script consiste en múltiples gráficos de dispersión que muestran las imágenes originales, procesadas y los resultados de los diferentes pasos del código.