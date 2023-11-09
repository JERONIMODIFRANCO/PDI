# --- Listas  -----------------------------------------------------------
# Creacion, manipulación, etc.
x = [1,2,4,6,10,8]
type(x)
len(x)
x.sort()
xs = sorted(x)
x.extend([3,5,7])

# Comportamiento al copiar
x = [1,2,3]
y=x
# y=x.copy()

y[0]=10
x
y

# Ejemplo: Elevar al cuadrado
x = list(range(10))

y = [0]*len(x)              # Version 1
for ii in range(len(x)):
    y[ii] = x[ii]**2

y = []                      # Version 2
for item in x:
    y.append(item**2)

y = [item**2 for item in x] # Version 3

def pow2(item):
    return item**2

x = [1,2,3,-1,-10,4,5,-123]
y = list(map(pow2, x))
y = list(map(lambda i: i**2, x))    # Otra forma...


zz = list(filter(lambda item: item>2, x))   # Filtrar
zzz = [item for item in x if item>2]        #

# Listas anidadas
m = [[1,2,3],[4,5,6],[7,8,9]]
m[0][1] # ~ matriz

# --- Tuplas -----------------------------------------------
x = (1,2,3,4,5)
x[0]
x[1:4]

x[0] = 22
x = list(x)
x[0] = 22
x = tuple(x)

# --- Diccionarios -----------------------------------------
x = {"indice":1, "descripcion":"Producto_1", "precio":125.4, "tamaño":[10,20], "info":None}
type(x)
x["indice"]
x["tamaño"]
x["tamaño"][0] = 5
x["info"] = "Discontinuado"

for key,val in x.items():
    print(f"{key}: {val}")

list(x.keys())
x = {}    # Inicialización

# --- Numpy -----------------------------------------------
# https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
import numpy as np

x = np.array([1,2,3])
x
type(x)
x.dtype
len(x)
x.shape
x.ndim

x = np.array([1,2,3],dtype=np.float64)
x
x.dtype

z = np.array([[1,2,3],[4,5,6],[7,8,9]])
z.ndim
z.shape
z[0,0]
z[0:2,1:2]
z[0:2,1]

z[1,:] = np.array([8,8,9])
z[:,1] = np.array([-1,-1,-1])
z[:,2] = [9,9,9]
z[0,:] = 99
z[0:2,0:2] = 77

z.max()
z.min()
np.unique(z)
len(np.unique(z))

# --- Operaciones basicas ----------------
z = np.array([[1,2,3],[4,5,6],[7,8,9]])
zt = z.T        # Transpuesta
np.sum(z, 0)    # Sumo sobre las filas
np.sum(z, 1)    # Sumo sobre las colmnas
np.sum(z)       # Sumo todos los elementos

# Ordenar
x = [1,2,4,6,10,8]
x = np.array(x)
x.sort()
np.sort(x)
x.argsort()
np.argsort(x)

x = np.array([[6,5,4],[3,2,1],[9,8,7]])
x.sort()
np.sort(x)
x.sort(0)
np.sort(x,0)
x.argsort()
x.argsort(0)

x = np.array([[1,50,0.1],
              [3,60,0.4],
              [5,15,0.9],
              [2,35,1.5],
              [0,22,7.2],
              [4,18,9.6]])

x[:,0].argsort()
x[x[:,0].argsort(),:]
x[x[:,0].argsort()]     # Analizar

# Rangos
np.arange(10)
np.arange(2,10)
np.arange(0,10,2)
np.arange(1,10,2)

# Extras
np.zeros(6)
np.zeros((3,6))
np.zeros((3,6), dtype=np.uint8)
np.ones((3,3))
np.diag([1,2,3,4])
np.diag(np.arange(1,5))
np.diag([1,2,3,4],-1)
