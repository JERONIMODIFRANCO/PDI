### Generación de Datos y Agrupación con K-means
### Descripción del Código
### Generación de Datos
Este script utiliza numpy para generar dos conjuntos de datos aleatorios bidimensionales y combinarlos en un solo array. Luego, aplica el algoritmo de K-means para agrupar estos datos en dos clusters y visualiza los resultados utilizando gráficos de dispersión.

### Generación de Datos
El código comienza generando dos conjuntos de datos aleatorios bidimensionales con numpy y los combina en un solo array. Los conjuntos de datos se generan con las siguientes características:
- **X**: Matriz de tamaño (25, 2) con valores enteros aleatorios en el rango [25, 50).
- **Y**: Matriz de tamaño (25, 2) con valores enteros aleatorios en el rango [60, 85).
- **Z**: Se apila verticalmente **X** y **Y** en una sola matriz y se convierte a tipo float32.

### K-means
El algoritmo de K-means se aplica a los datos combinados **Z** con los siguientes parámetros:
- **criteria**: Criterio de finalización que combina el número máximo de iteraciones (10) y la precisión (1.0).
- **k**: Número de clusters a formar, en este caso, 2.
- **attempts**: Número de veces que el algoritmo se ejecutará con diferentes inicializaciones de centroides, aquí 10.

### Separación de Clusters y Visualización
Los puntos de datos se separan en dos clusters **A** y **B** utilizando las etiquetas generadas por K-means. Luego, se visualizan:
- **Puntos de cada cluster**: Se muestran en colores diferentes.
- **Centroides**: Se muestran como cuadrados amarillos.

### Entrada
El script no toma entradas externas, ya que los datos se generan aleatoriamente dentro del propio código.

### Salida
La salida del script consiste en dos gráficos de dispersión:
- **Gráfico inicial**: Muestra todos los datos generados aleatoriamente.
- **Gráfico final**: Muestra los dos clusters separados con sus respectivos centroides.