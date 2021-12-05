import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar las diferentes imágenes.
def muestra_imagen(titulo,imagen):
    plt.figure()
    plt.title(titulo)
    f = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) # Tendremos que cambiarlo a RGB para poder mostrar la imagen por consola con pyplot
    plt.imshow(f)
    plt.show


f=cv2.imread('first.PNG',0)
contornosP2=f # La utilizaremos después como plantilla para trazar los contornos
muestra_imagen("Imagen original",f)

#################### Histograma de la imagen original #################################

hist2 = cv2.calcHist([f], [0], None, [256], [0, 256]) # Con el 0 indicamos que queremos calcular un histograma de una imagen en escala de grises. No usamos máscara. histSize de 256. Píxeles en rango 0 a 256.
plt.figure()
plt.title("Histograma imagen original")
plt.xlabel("Rango")
plt.ylabel("Número de píxeles")
plt.plot(hist2)


# Paso 1. Segmentación de Bordes - Canny.

minVal=np.min(f)
maxVal=np.max(f)

canny=cv2.Canny(f,minVal,maxVal) # Primer parámetro la imagen. Los otros dos son los umbrales (nose q valor hay q poner).

muestra_imagen("Imagen original Binarizada con Bordes",canny)

# Paso 2. Selección de contornos

#Metodo solo aplicable a imagenes binarizadas -> Obtención de contornos

contornos=cv2.findContours(canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# Para dibujar los contornos lo haremos sobre una plantilla de ceros (imagen negra)
# -> Dibujado de contornos sobre plantilla

plantilla = np.zeros_like(f)
cv2.drawContours(plantilla , contornos[0], -1, (255, 255, 0), 1)

#### Lo de arriba no está saliendo bien

muestra_imagen("Imagen tras dibujar Contornos",plantilla)





