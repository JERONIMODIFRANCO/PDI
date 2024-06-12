### Documentación del Código
### Descripción del Código
### Carga de Imagen y Preprocesamiento
### Carga de Imagen
Este script utiliza OpenCV para cargar una imagen y convertirla a color RGB.

### Preprocesamiento
La imagen se carga con `cv2.imread` y se convierte a color RGB utilizando `cv2.cvtColor`. Luego, se muestra la imagen utilizando `matplotlib`.

### Uso de K-means para Obtener la Paleta de Colores
### Preparación de Datos
Se prepara una matriz `Z` con todos los valores de los píxeles de la imagen, que se convierte a tipo `float32`.

### Aplicación de K-means
Se aplica el algoritmo de K-means con los siguientes parámetros:
- **criteria**: Criterio de finalización que combina el número máximo de iteraciones (10) y la precisión (1.0).
- **K**: Número de clusters a formar, en este caso, 4.
- **attempts**: Número de veces que el algoritmo se ejecutará con diferentes inicializaciones de centroides, aquí 10.

### Obtención de la Nueva Imagen con la Nueva Paleta de Colores
### Reshape y Convertir a Tipo Entero
Se convierte `center` a tipo `uint8` y se utiliza para obtener la nueva imagen con la nueva paleta de colores.

### Visualización
Se muestran dos gráficos de dispersión:
- **Gráfico inicial**: Muestra la imagen original.
- **Gráfico final**: Muestra la imagen con colores cuantizados.

### Entrada
El script no toma entradas externas, ya que la imagen se carga dentro del propio código.

### Salida
La salida del script consiste en dos gráficos de dispersión:
- **Gráfico inicial**: Muestra la imagen original.
- **Gráfico final**: Muestra la imagen con colores cuantizados.