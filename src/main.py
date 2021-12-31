from typing import final
import cv2
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar las diferentes imágenes.
def muestra_imagen(titulo,imagen):
    plt.figure()
    plt.title(titulo)
    f = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) # Tendremos que cambiarlo a RGB para poder mostrar la imagen por consola con pyplot
    plt.imshow(f)
    plt.show

imagen='first.PNG'
original=cv2.imread(imagen) # La utilizaremos para sacar el filtro de sobel sobre la imagen original.
f=cv2.imread(imagen,0)

muestra_imagen("Imagen original",f)


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
canny=cv2.Canny(f,20,100)
#canny=cv2.Canny(f,lower,upper) # Primer parámetro la imagen. Los otros dos son los umbrales (nose q valor hay q poner).

muestra_imagen("Imagen original Binarizada con Bordes",canny)

# Paso 2. Selección de contornos

#Metodo solo aplicable a imagenes binarizadas -> Obtención de contornos

contours1, hierarchy = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
hierarchy = hierarchy[0]

# Paso 2.2.: Dibujar los contornos cerrados.

# Si ambos valores son negativos (-1), estamos ante un contorno abierto.
# " hierarchy[i][2] = denotes its first child contour "
# " hierarchy[i][3] = denotes index of its parent contour "


# Para dibujar los contornos lo haremos sobre una plantilla de ceros (imagen negra)
# -> Dibujado de contornos sobre plantilla

plantilla = np.zeros_like(f)



for i, c in enumerate(contours1):
    if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
        pass
    else:
        cv2.drawContours(plantilla, contours1, i, (255, 255,255), 1)


muestra_imagen("Imagen tras dibujar Contornos",plantilla)


# Paso 3. Text Localization.


#En primer lugar cogemos la imagen original y la pasamos a escala de grises


original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)


#Hacemos el filtro de sobel a la x
#El tamaño del kernel es 5 porque la documentación lo considera adecuado
x_sobel = cv2.Sobel(original_gray, cv2.CV_64F, 1, 0, ksize=3) 


#Le hacemos el valor absoluto para aproximar a la fórmula del artículo
x_sobel = cv2.convertScaleAbs(x_sobel)

#Hamemos lo mismo sobre el filtro de la Y
y_sobel = cv2.Sobel(original_gray, cv2.CV_64F, 0, 1, ksize=3)

#Volvemos a hacer el valor absoluto
y_sobel = cv2.convertScaleAbs(y_sobel)

#Fusionamos
xy_sobel = x_sobel * 0.5 + y_sobel * 0.5

#Convertimos la imagen en uint-8
xy_sobel = (xy_sobel).astype('uint8')



# Obtencion jerarquia contornos
contours2, hierarchy = cv2.findContours(plantilla, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
hierarchy = hierarchy[0]



##########################


## Paso final comprobacion 

original2=original.copy()

for c in contours2:
    x,y,w,h = cv2.boundingRect(c)
    ROI = original[y:y+h, x:x+w]
    cv2.rectangle(original2, (x, y), (x + w, y + h), (0,0,255), 1) # Dibujar los ROI


muestra_imagen("Resultado inicial",original2) 




##########################



muestra_imagen("Imagen tras realizar el filtro de sobel", xy_sobel)

print("Numero contornos ",len(hierarchy))
print("Jerarquia: ",len(contours2))
print(len(f),len(f[0]))
#print("sobel ",xy_sobel," fin sobel")
#print("inicio contornos", contours)

# Recorremos las coordenadas de los píxeles de los contornos cerrados y obtenermos su correspondiente valor sobel
# Vamos sumando los valores 
edge_intensity_region_list=[] # lista de EI_rt 
for contorno in contours2: # (  [  [[j,i]], [[j,i]], [[j,i]]  ] ,   [ ...... ] ) ----> CONTORNO1 , CONTORNO-N
    edge_intensity_region=0 
    for coordenadas in contorno:
        coordenada=coordenadas[0]
        i,j=coordenada[1],coordenada[0]
        edge_intensity_region+=xy_sobel[i][j]
    edge_intensity_region_list.append(edge_intensity_region)


print("Numero EIrts",len(edge_intensity_region_list))

# Para que una región sea considerada de texto, su 
# EI_rt debe ser mayor al umbral T

# El umbral T se calcula de la siguiente forma:
# T = α * sum EI_rt/N ; N = Número de regiones cerradas

num_regions=len(contours1)
suma=0
for ei_rt in edge_intensity_region_list:
    suma+= (ei_rt/num_regions)

T=0.8*suma

print(T)


filtered_regions=[]
# Filtramos aquellas regiones que no superen el umbral (FILTRO DE REGIONES DE TEXTO)
for region in enumerate(edge_intensity_region_list): # Lo hacemos con un enumerate para tras filtrar las regiones, saber a cual nos referimos.
    if(region[1]>T):
        filtered_regions.append(region)
    
print("Filtro 1: Regiones finales que superan el umbral:", len(filtered_regions),".\n")



# Filtramos aquellas componentes NO consideradas carácteres (Las eliminamos).

filtered_characters=[] # Con la imagen de los chicles se me queda en 2 regiones. wtf
for contorno_filtrado in filtered_regions:
    contornoId=contorno_filtrado[0]
    x,y,width,height=cv2.boundingRect(contours2[contornoId])
    value=width/height

    if(0.1<value and value<2): # Regla 1 caracteres.

        if(50<width*height): # Regla 2 caracteres.

          filtered_characters.append(contours2[contornoId])


print("Filtro 2: Lista de regiones detectadas como caracter potencial:", len(filtered_characters),".\n")


# Comprobaremos las reglas de vecindad
filtered_characters_enum=[]
for charcater in enumerate(filtered_characters): # Para identificar unicamente los ya seleccionados
    filtered_characters_enum.append(charcater) 



id_regiones=[]
final_regions=[]
for i in range(len(filtered_characters_enum)-1):
    for j in range(i+1,len(filtered_characters_enum)):

        # Obtención de momentos
        moments_i=cv2.moments(filtered_characters_enum[i][1])
        moments_j=cv2.moments(filtered_characters_enum[j][1])

        # Obtención anchuras y alturas
        _,_,w_i,h_i=cv2.boundingRect(filtered_characters_enum[i][1])
        _,_,w_j,h_j=cv2.boundingRect(filtered_characters_enum[j][1])

        # Calculo coordenadas mass center
        mass_center_i=(moments_i['m10']/moments_i['m00'],moments_i['m01']/moments_i['m00'])
        mass_center_j=(moments_j['m10']/moments_j['m00'],moments_j['m01']/moments_j['m00'])

        dist_x = abs(mass_center_i[0]-mass_center_j[0])
        dist_y = abs(mass_center_i[1]-mass_center_j[1])

        # Comprobamos la R1 de vecindad
        if(dist_x<=0.2*max(h_i,h_j)):            
            if(dist_y<=2*max(w_i,w_j)):  
                value=h_i/h_j              
                if(0.5<=value and value<=2):
                
                    if(filtered_characters_enum[i][0] not in id_regiones):
                        final_regions.append(filtered_characters_enum[i][1])
                        id_regiones.append(filtered_characters_enum[i][0])

                    if(filtered_characters_enum[j][0] not in id_regiones):
                        final_regions.append(filtered_characters_enum[j][1])
                        id_regiones.append(filtered_characters_enum[j][0])


print("Filtro 3: Regiones que han pasado filtros de vecindad", len(final_regions))


## Paso final comprobacion 

original2=original.copy()

for c in final_regions:
    x,y,w,h = cv2.boundingRect(c)
    ROI = original[y:y+h, x:x+w]
    cv2.rectangle(original2, (x, y), (x + w, y + h), (0,0,255), 3) # Dibujar los ROI


muestra_imagen("Resultado final",original2) 