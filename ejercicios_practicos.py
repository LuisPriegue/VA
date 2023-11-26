import skimage as skimage
from skimage import io, color
import numpy as npy
import matplotlib.pyplot as plot

def mostrarHistograma(Imagen_entrada, Imagen_salida):
   # Mostramos 4 elementos en una cuadrícula:
   # Imagen entrada y salida y sus histogramas
	plot.figure(figsize=(10, 12))
	inImage = plot.subplot(421)
	inImage.imshow(Imagen_entrada, cmap='gray', vmax=1, vmin=0)
	inImage.set_title("Imanen de entrada")
	inImage.set_axis_off()
	outImage = plot.subplot(422, sharex=inImage, sharey=inImage)
	outImage.imshow(Imagen_salida, cmap='gray', vmax=1, vmin=0)
	outImage.set_title("Imagen de salida")
	outImage.set_axis_off()
	inImage_hist = plot.subplot(423)
	inImage_hist.hist(Imagen_entrada.ravel() * 255, bins=256, range=(0,255), color='blue')
	inImage_hist.set_xlim(0, 255)
	inImage_hist.autoscale(enable=True, axis='y', tight=True)	
	inImage_hist.set_ylabel("Frecuencia")
	inImage_hist.set_xlabel("Intensidad")
	inImage_hist.legend(["Imagen de Entrada"])
	outImage_hist = plot.subplot(424, sharex=inImage_hist, sharey=inImage_hist)
	outImage_hist.hist(Imagen_salida.ravel() * 255, bins=256, range=(0,255), color='orange')
	outImage_hist.set_xlim(0, 255)
	outImage_hist.set_ylabel("Frecuencia")
	outImage_hist.set_xlabel("Intensidad")
	outImage_hist.legend(["Imagen de Salida"])
	plot.tight_layout()
	plot.suptitle("Comparación entre Imagen de Entrada y Salida")
	plot.subplots_adjust(top=0.93)
	plot.show()

def mostrarImagenes(Imagen_entrada, Imagen_salida):

   inImage = plot.subplot(121)
   inImage.imshow(Imagen_entrada, cmap='gray', vmax=1, vmin=0)
   inImage.set_title("Imanen de entrada")
   inImage.set_axis_off()
   outImage = plot.subplot(122, sharex=inImage, sharey=inImage)
   outImage.imshow(Imagen_salida, cmap='gray', vmax=1, vmin=0)
   outImage.set_title("Imagen de salida")
   outImage.set_axis_off()
   plot.tight_layout()
   plot.show()

def mostrarGradientes(Imagen_entrada, gx, gy, magn):
    fig, axs = plot.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(Imagen_entrada, cmap='gray', vmax=1, vmin=0)
    axs[0, 0].set_title("Imagen de entrada")
    axs[0, 0].axis('off')

    im = axs[0, 1].imshow(magn, cmap='gray', vmax=1, vmin=0)
    axs[0, 1].set_title("Magnitud de gradientes")
    axs[0, 1].axis('off')
    fig.colorbar(im, ax=axs[0, 1], orientation='vertical')

    im = axs[1, 0].imshow(gx, cmap='gray', vmax=1, vmin=0)
    axs[1, 0].set_title("Gradiente en dirección X")
    axs[1, 0].axis('off')
    fig.colorbar(im, ax=axs[1, 0], orientation='vertical')

    im = axs[1, 1].imshow(gy, cmap='gray', vmax=1, vmin=0)
    axs[1, 1].set_title("Gradiente en dirección Y")
    axs[1, 1].axis('off')
    fig.colorbar(im, ax=axs[1, 1], orientation='vertical')

    plot.tight_layout()
    plot.show()

   
def adjustIntensity(inImage, inRange=[], outRange=[0,1]):
	if inRange:
			imin = inRange[0]
	else: 
			imin = npy.min(inImage)	
	
	if inRange:
			imax = inRange[1]
	else: 
			imax = npy.max(inImage)	

	omin = outRange[0]
	omax = outRange[1]	
    
	imagen_salida = omin+(((omax-omin)*(inImage-imin))/(imax-imin))

	return imagen_salida


# OBTENER IMAGEN ENTRADA
"""
#Del paquete data de imágenes de scikit-images 
inImage =skimage.data.camera()

#Del directorio de imágenes de prueba
ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/prueba1.png'
inImage = io.imread(ruta_imagen_test)
"""

# PRUEBA AJUSTAR INTENSIDAD
"""
La fórmula realiza una transformación lineal para normalizar el valor de 
entrada desde el rango de entrada al rango de salida deseado.

#COMPRESIÓN del histograma de la imagen.
#Se aprecia la pérdida de contraste. Al contraer el histograma se disminuye
#el rango dinámico de la distribución de nivel de gris de la imagen.

inImage =skimage.data.camera()
outImage = adjustIntensity(inImage, outRange=[0.2, 0.7])
mostrarHistograma(inImage, outImage)

#EXPANSIÓN del histograma de la imagen.
#La imagen prueba1 por defecto no aprovecha todo el rango [0,1] de intensidad.
#Lo solucionaremos ajustando el rango de salida.

ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/prueba1.png'
inImage = io.imread(ruta_imagen_test)
outImage = adjustIntensity(inImage,outRange=[0,1])
mostrarHistograma(inImage, outImage)

#DESPLAZAMIENTO DEL HISTOGRAMA
ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/prueba1.png'
inImage = io.imread(ruta_imagen_test)
outImage = adjustIntensity(inImage,inRange=[0.1,1], outRange=[0.0,0.9])
#Aclaramos la imagen
outImage = adjustIntensity(inImage,inRange=[0,1], outRange=[0.6,1])
#Oscurezemos la imagen
outImage = adjustIntensity(inImage,inRange=[0,1], outRange=[0,0.6])

"""

def equalizeIntensity (inImage, nBins=256):
	
	# Se calculan el número de celdas de salida
    bins=npy.linspace(npy.min(inImage),npy.max(inImage),nBins+1)

    # Calcular el histograma acumulado normalizado
    hist,_=npy.histogram(inImage.flatten(),bins=bins)
	 # Este calculo asegura que la última entrada suma 1 y tenga la forma
    # necesaria de función de densidad que garantiza que funcione la ecualización 
    hist_acumulado=hist.cumsum()/hist.sum()

    # Aplicar la ecualización de histograma utilizando interpolación
    # Left y right protegen los valores que se caen fuera de rango.
    resultado=npy.interp(inImage,bins[:-1],hist_acumulado,left=0,right=1)

    # Devolver la imagen ecualizada
    return resultado

# PRUEBA ECUALIZACÓN
"""
#CONCEPTO:
Donde la pendiente del histograma acumulado es mayor separo mucho los niveles
de gris, pues es dónde mas información hay. Es una transformación con pérdidas
las partes comprimidas no se pueden volver a separar.

La interpolación calcula para cada pixel de la imagen de entrada, su correspon-
diente valor usando como referencia la función acumulativa normalizada que es 
nuestro histograma acumulado, utilizando los bins como referencia. Uso bins[:-1]
para que coincida con la longitud del histograma_acum. facilitando la interpolacion.

En el ejemplo de data.camera se ve muy bien el histograma ecualizado.

inImage = skimage.data.camera()
outImage = equalizeIntensity(inImage)
mostrarHistograma(inImage, outImage)

Un ejemplo mas liviano:
ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/circles.png'
inImage = io.imread(ruta_imagen_test)
"""

def filterImage(inImage, kernel):
    kernel = npy.array(kernel)

    ancho, alto = npy.shape(inImage)
    p, q = npy.shape(kernel)
    outImage = npy.zeros((ancho, alto), dtype='float32')

    centrox = p // 2
    centroy = q // 2

    for x in range(ancho):
        for y in range(alto):
            
            arriba = x-centrox
            izquierda = y-centroy
            derecha = y+q-centroy
            abajo = x+p-centrox

            lim_superior = max(0, -arriba)
            lim_izquierdo = max(0, -izquierda)
            lim_derecho = max(0, derecha - alto)
            lim_inferior = max(0, abajo - ancho)
            
            ri = inImage[max(0, arriba):min(ancho, abajo), max(0, izquierda):min(alto, derecha)]
            kernel_ri = kernel[lim_superior:p - lim_inferior, lim_izquierdo:q - lim_derecho]

            outImage[x, y] = (ri * kernel_ri).sum()

    return outImage

#PRUEBAS SUAVIZADO
"""
Con la imagen Prueba1, se aprecia muy bien el efecto en el histograma
#EJEMPLO 1
kernel = npy.ones((3, 3)) / 9
ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/prueba1.png'
inImage = io.imread(ruta_imagen_test)
#Elimina el canal alpha
if inImage.shape[2] == 4:
    inImage = inImage[:, :, :3]
inImage = color.rgb2gray(inImage)
outImage = filterImage(inImage, kernel)
#outImage = adjustIntensity(outImage)
mostrarHistograma(inImage, outImage)

#EJEMPLO 2
kernel = [[0,0.2,0],
          [0.2,0.3,0.2],
          [0,0.2,0]
          ]
inImage = skimage.data.camera()
outImage = filterImage(inImage, kernel)
mostrarHistograma(inImage, outImage)

	kernel_suavizado = npy.ones((3, 3)) / 9
	
	kernel_sobel_x = npy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
   kernel_sobel_y = npy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

   kernel_laplaciano = npy.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

"""
def gaussKernel1D(sigma):
    # Calcular el tamaño del kernel
    N = 2 * int(npy.ceil(3 * sigma)) + 1

    # Calcular el centro del kernel
    centro = (N - 1) // 2

    # Calcular el kernel Gaussiano 1D
    x = npy.arange(N) - centro
    kernel = npy.exp(-0.5 * (x / sigma)**2)
    # Normalizar para que la suma sea 1
    kernel /= npy.sum(kernel)  
    return kernel

# PRUEBAS KERNEL GAUSSIANO 1D
"""
sigma = 1.5
kernel = gaussKernel1D(sigma)
print("Kernel Gaussiano 1D con sigma =", sigma, "y tamaño N =", len(kernel), ":\n", kernel)
"""

def gaussianFilter(inImage, sigma):
    # Crear kernel Gaussiano 1D
    kernel = npy.atleast_2d(gaussKernel1D(sigma))

    # Aplicar filtro Gaussiano en la dimensión vertical
    filtered_image = filterImage(inImage, kernel)

    # Transponer el kernel
    kernel_transpuesto = kernel.reshape(-1, 1)

    # Aplicar filtro Gaussiano en la dimensión horizontal
    outImage = filterImage(filtered_image, kernel_transpuesto)

    return outImage

#PRUEBAS FILTRO GAUSSIANO
"""
ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/prueba1.png'
inImage = io.imread(ruta_imagen_test)
#Elimina el canal alpha
if inImage.shape[2] == 4:
    inImage = inImage[:, :, :3]
inImage = color.rgb2gray(inImage)
outImage = gaussianFilter(inImage,1.5)
mostrarHistograma(inImage, outImage)
"""

def medianFilter (inImage, filterSize):
   ancho,alto = npy.shape(inImage)
   outImage = npy.zeros((ancho,alto), dtype='float32')

   centro = filterSize//2 #División entera 

   for x in range(ancho):
      for y in range(alto):
         
         arriba = x-centro
         izquierda = y-centro
         derecha = y+filterSize-centro
         abajo = x+filterSize-centro
         
         convolucion = inImage[max(0, arriba):min(ancho, abajo), max(0, izquierda):min(alto, derecha)]

         outImage[x,y] = npy.median(convolucion)

   return outImage

#PRUEBAS FILTRO MEDIAS
"""
ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/prueba1.png'
inImage = io.imread(ruta_imagen_test)
#Elimina el canal alpha
if inImage.shape[2] == 4:
    inImage = inImage[:, :, :3]
inImage = color.rgb2gray(inImage)
outImage = medianFilter(inImage, 4)
mostrarHistograma(inImage, outImage)
"""

"""
OPERADORES MORFOLOGICOS
"""
def __canErode(SE, ri):
    p, q = npy.shape(SE)

    for x in range(p):
        for y in range(q):
            if SE[x, y] == 1 and ri[x, y] == 0:
                return 0
    return 1

def erode(inImage, SE, center=[]):
    SE = npy.array(SE)
    ancho, alto = npy.shape(inImage)
    p, q = npy.shape(SE)
    outImage = npy.zeros((ancho, alto), dtype='float32')

    if not center:
        center = [p // 2, q // 2]

    for x in range(ancho):
        for y in range(alto):

            arriba = x - center[0]
            izquierda = y - center[1]
            derecha = x + p - center[0]
            abajo = y + q - center[1]

            lim_superior = max(0, -arriba)
            lim_izquierdo = max(0, -izquierda)
            lim_derecho = max(0, derecha - alto)
            lim_inferior = max(0, abajo - ancho)

            ri = inImage[x - center[0] + lim_superior:x + p - center[0] - lim_inferior, y - center[1] + lim_izquierdo:y + q - center[1] - lim_derecho]
            SEutil = SE[lim_superior: p - lim_inferior, lim_izquierdo:q - lim_derecho]

            if not (npy.any(SEutil)):
                outImage[x, y] = inImage[x, y]
            else:
                outImage[x, y] = __canErode(SEutil, ri)

    return outImage


#PRUEBAS ERODE
"""
SE = [
		[0,0,0],
		[1,1,1],
		[0,0,0]
	]
ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/image.png'
inImage = io.imread(ruta_imagen_test)
outImage = erode(inImage, SE)
outImage = erode(inImage, SE, [0,0])
mostrarImagenes(inImage, outImage)
"""

def __canDilate (SE, ri):
   p, q = npy.shape(SE)

   for x in range(p):
      for y in range(q):
         if SE[x,y] == 1 and ri[x,y] == 1:
            return 1
   return 0

def dilate (inImage, SE, center=[]):

   SE = npy.array(SE)	
   ancho,alto = npy.shape(inImage)
   p,q = npy.shape(SE)
   outImage = npy.zeros((ancho,alto), dtype='float32')

   if not center:
      center = [p//2, q//2]

   for x in range(ancho):
      for y in range(alto):
         arriba = x-center[0]
         izquierda = y-center[1]
         derecha = x+p-center[0]
         abajo = y+q-center[1]

         lim_superior = max(0, -arriba)
         lim_izquierdo = max(0, -izquierda)
         lim_derecho = max(0, derecha - alto)
         lim_inferior = max(0, abajo - ancho)

         ri = inImage[x - center[0] + lim_superior:x + p - center[0] - lim_inferior, y - center[1] + lim_izquierdo:y + q - center[1] - lim_derecho]

         outImage[x, y] = __canDilate(SE[lim_superior:p-lim_inferior, lim_izquierdo:q-lim_derecho], ri)

   return outImage	

#PRUEBAS DILATE
"""
SE = [
		[0,0,0],
		[1,1,1],
		[0,0,0]
	]

ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/image.png'
inImage = io.imread(ruta_imagen_test)
outImage = dilate(inImage, SE)
mostrarImagenes(inImage, outImage)
"""
def opening (inImage, SE, center=[]):
    return dilate(erode(inImage, SE), SE)

def closing (inImage, SE, center=[]):
    return erode(dilate(inImage, SE), SE)

#PRUEBAS OPENING Y CLOSING
"""
SE = [
		[0,0,0],
		[1,1,1],
		[0,0,0]
	]

ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/image.png'
inImage = io.imread(ruta_imagen_test)
outImage = closing(inImage, SE)
mostrarImagenes(inImage, outImage)

outImage = opening(inImage, SE)
mostrarImagenes(inImage, outImage)
"""

def gradientImage (inImage, operator):

   gx = []
   gy = []

   if operator == 'CentralDiff':
      gx = filterImage(inImage, [[-1,0,1]])
      gy = filterImage(inImage, [[-1],[0],[1]])
   
   elif operator == 'Roberts':
      gx = filterImage(inImage, [[-1,0],[0,1]])
      gy = filterImage(inImage, [[0,-1],[1,0]])    
   
   elif operator == 'Sobel':
      gx = filterImage(filterImage(inImage, [[1],[2],[1]]), [[-1,0,1]])
      gy = filterImage(filterImage(inImage, [[-1],[0],[1]]), [[1,2,1]])

   elif operator == 'Prewitt':
      gx = filterImage(filterImage(inImage, [[1],[1],[1]]), [[-1,0,1]])
      gy = filterImage(filterImage(inImage, [[-1],[0],[1]]), [[1,1,1]])
   
   else:
      print("Los operadores disponibles son: 'Prewitt', 'Sobel', 'Roberts' o 'CentralDiff'")
      return [gx, gy]

   return [gx, gy]

#PRUEBAS GRADIENTES
"""
ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/prueba1.png'
inImage = io.imread(ruta_imagen_test)
#Elimina el canal alpha
if inImage.shape[2] == 4:
    inImage = inImage[:, :, :3]
inImage = color.rgb2gray(inImage)
[gx,gy] = gradientImage(inImage, 'Roberts')
magnitud = npy.sqrt((gx**2) + (gy**2))
gx, gy, magnitud = adjustIntensity(gx), adjustIntensity(gy), adjustIntensity(magnitud)
mostrarGradientes(inImage, gx, gy, magnitud)

[gx,gy] = gradientImage(inImage, 'CentralDiff')
[gx,gy] = gradientImage(inImage, 'Sobel')
[gx,gy] = gradientImage(inImage, 'Prewitt')

"""

"""
Implementar el filtro Laplaciano de Gaussiano que permita especificar el par´ametro σ
de la Gaussiana utilizada.
outImage = LoG (inImage, sigma)
inImage, outImage: ...
sigma: Par´ametro σ de la Gaussiana
"""
def LoG(inImage, sigma):

   kernel_laplaciano = npy.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
   
   suavizado_laplaciano = adjustIntensity(filterImage(inImage,kernel_laplaciano))

   outImage = gaussianFilter(suavizado_laplaciano, sigma)
   
   return outImage

#PRUEBAS LAPLACIANO
"""
ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/prueba1.png'
inImage = io.imread(ruta_imagen_test)
#Elimina el canal alpha
if inImage.shape[2] == 4:
    inImage = inImage[:, :, :3]
inImage = color.rgb2gray(inImage)
outImage = equalizeIntensity(LoG(inImage, 3))
mostrarHistograma(inImage, outImage)

""" 
#ZONA DE PRUEBAS
ruta_imagen_test = '/home/luis/Escritorio/VA/P1/testimages/prueba1.png'
inImage = io.imread(ruta_imagen_test)
#Elimina el canal alpha
if inImage.shape[2] == 4:
    inImage = inImage[:, :, :3]
inImage = color.rgb2gray(inImage)
outImage = equalizeIntensity(LoG(inImage, 3))
mostrarHistograma(inImage, outImage)
