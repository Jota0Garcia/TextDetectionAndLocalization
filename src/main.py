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

original=cv2.imread('first.png')
f=cv2.imread('first.png',0)
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

######################### Para obtención de valores aut.
# (no está explicado en la memoria)
# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
v = np.median(f)
# apply automatic Canny edge detection using the computed median
sigma=0.33
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))


#########################
canny=cv2.Canny(f,lower,upper) # Primer parámetro la imagen. Los otros dos son los umbrales (nose q valor hay q poner).

muestra_imagen("Imagen original Binarizada con Bordes",canny)

# Paso 2. Selección de contornos

#Metodo solo aplicable a imagenes binarizadas -> Obtención de contornos

contours, hierarchy = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
hierarchy = hierarchy[0]

# Paso 2.2.: Dibujar los contornos cerrados.

# Si ambos valores son negativos (-1), estamos ante un contorno abierto.
# " hierarchy[i][2] = denotes its first child contour "
# " hierarchy[i][3] = denotes index of its parent contour "


# Para dibujar los contornos lo haremos sobre una plantilla de ceros (imagen negra)
# -> Dibujado de contornos sobre plantilla

plantilla = np.zeros_like(f)



for i, c in enumerate(contours):
    if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
        pass
    else:
        cv2.drawContours(plantilla, contours, i, (255, 255,255), 1)


muestra_imagen("Imagen tras dibujar Contornos",plantilla)


# Paso 3. Text Localization.


# Cambiar esto
result = plantilla.copy()
contours, hierarchy = cv2.findContours(result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

original2=original.copy()

ROI_num = 0
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    ROI = original[y:y+h, x:x+w]
    cv2.rectangle(original2, (x, y), (x + w, y + h), (0,0,255), 1) # Dibujar los ROI
    ROI_num += 1


muestra_imagen("Imagen tras dibujar Contornos",original2)