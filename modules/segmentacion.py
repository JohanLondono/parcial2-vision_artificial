#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Segmentación para el Sistema de Detección Vehicular
===========================================================

Contiene todas las técnicas de segmentación necesarias para el 
preprocesamiento en el sistema de detección vehicular.

Autor: Sistema de Detección Vehicular
Fecha: Octubre 2025
"""

import cv2
import numpy as np

class Segmentacion:
    """
    Clase que implementa diferentes técnicas de segmentación de imágenes.
    Incluye umbralización simple, umbralización adaptativa, Canny, contornos,
    K-means, Watershed y crecimiento de regiones.
    """
    
    @staticmethod
    def umbral_simple(imagen, umbral=127):
        """
        Aplica umbralización simple a una imagen.
        
        Args:
            imagen: Imagen en escala de grises
            umbral: Valor de umbral (0-255)
            
        Returns:
            Imagen binarizada
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
        _, imagen_binarizada = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
        return imagen_binarizada
    
    @staticmethod
    def umbral_adaptativo(imagen, tam_bloque=11, constante=2, tipo='MEAN'):
        """
        Aplica umbralización adaptativa a una imagen.
        
        Args:
            imagen: Imagen en escala de grises
            tam_bloque: Tamaño del bloque para el cálculo del umbral
            constante: Valor constante que se resta
            tipo: Tipo de umbral adaptativo ('MEAN' o 'GAUSSIAN')
            
        Returns:
            Imagen con umbral adaptativo aplicado
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
        if tipo.upper() == 'GAUSSIAN':
            metodo = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            metodo = cv2.ADAPTIVE_THRESH_MEAN_C
            
        umbral_adaptativo = cv2.adaptiveThreshold(
            imagen, 255, metodo, cv2.THRESH_BINARY, tam_bloque, constante)
        return umbral_adaptativo
    
    @staticmethod
    def umbral_otsu(imagen):
        """
        Aplica umbralización usando el método de Otsu.
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Imagen binarizada con Otsu
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        _, imagen_otsu = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return imagen_otsu
    
    @staticmethod
    def detector_canny(imagen, umbral1=100, umbral2=200):
        """
        Aplica el detector de bordes Canny.
        
        Args:
            imagen: Imagen de entrada
            umbral1: Umbral inferior para la histéresis
            umbral2: Umbral superior para la histéresis
            
        Returns:
            Imagen con bordes detectados
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
        bordes = cv2.Canny(imagen, umbral1, umbral2)
        return bordes
    
    @staticmethod
    def deteccion_contornos(imagen, umbral=127):
        """
        Aplica segmentación por detección de contornos.
        
        Args:
            imagen: Imagen de entrada
            umbral: Umbral para la binarización previa
            
        Returns:
            Imagen con contornos dibujados
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gray = imagen.copy()
            
        # Binarizar imagen
        _, thresh = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen para mostrar contornos
        imagen_contornos = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(imagen.shape) <= 2 else imagen.copy()
        
        # Dibujar contornos
        cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)
        
        return imagen_contornos
    
    @staticmethod
    def kmeans_segmentacion(imagen, k=2):
        """
        Aplica segmentación mediante K-means.
        
        Args:
            imagen: Imagen de entrada en color
            k: Número de clusters
            
        Returns:
            Imagen segmentada por K-means
        """
        # Asegurar que la imagen esté en formato BGR
        if len(imagen.shape) <= 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        
        # Convertir imagen en una matriz 2D de valores de píxeles
        datos = imagen.reshape((-1, 3))
        datos = np.float32(datos)
        
        # Definir criterios de parada
        criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Aplicar K-means
        _, etiquetas, centros = cv2.kmeans(datos, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convertir los centros a uint8
        centros = np.uint8(centros)
        
        # Reemplazar cada píxel con su respectivo centro
        resultado = centros[etiquetas.flatten()]
        
        # Remodelar resultado a la forma de la imagen original
        imagen_kmeans = resultado.reshape(imagen.shape)
        
        return imagen_kmeans
    
    @staticmethod
    def watershed_segmentacion(imagen):
        """
        Aplica segmentación mediante Watershed.
        
        Args:
            imagen: Imagen de entrada en color
            
        Returns:
            Imagen segmentada con Watershed
        """
        # Asegurar que la imagen esté en formato BGR
        if len(imagen.shape) <= 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        
        # Crear copia de la imagen original
        imagen_resultado = imagen.copy()
        
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbralización
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Eliminar ruido con apertura morfológica
        kernel = np.ones((3,3), np.uint8)
        apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Identificar área de fondo segura
        fondo_seguro = cv2.dilate(apertura, kernel, iterations=3)
        
        # Calcular la transformada de distancia
        dist_transform = cv2.distanceTransform(apertura, cv2.DIST_L2, 5)
        
        # Obtener objetos seguros
        _, objetos_seguros = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Convertir a enteros
        objetos_seguros = np.uint8(objetos_seguros)
        
        # Región desconocida
        desconocido = cv2.subtract(fondo_seguro, objetos_seguros)
        
        # Etiquetar marcadores
        _, marcadores = cv2.connectedComponents(objetos_seguros)
        
        # Marcar la región desconocida con cero
        marcadores = marcadores + 1
        marcadores[desconocido == 255] = 0
        
        # Aplicar watershed
        cv2.watershed(imagen_resultado, marcadores)
        
        # Marcar los bordes de watershed con rojo
        imagen_resultado[marcadores == -1] = [0, 0, 255]
        
        return imagen_resultado
    
    @staticmethod
    def segmentar_color_hsv(imagen, hue_min, hue_max, sat_min=50, val_min=50, sat_max=255, val_max=255):
        """
        Segmenta una imagen en color usando umbrales en el espacio HSV.
        
        Args:
            imagen: Imagen en formato BGR
            hue_min: Valor mínimo de tono (0-179)
            hue_max: Valor máximo de tono (0-179)
            sat_min: Valor mínimo de saturación (0-255)
            sat_max: Valor máximo de saturación (0-255)
            val_min: Valor mínimo de valor (0-255)
            val_max: Valor máximo de valor (0-255)
            
        Returns:
            Imagen segmentada con los colores en el rango especificado
        """
        # Asegurar que la imagen esté en formato BGR
        if len(imagen.shape) <= 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
            
        # Convertir a HSV
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Definir rango de color
        rango_inferior = np.array([hue_min, sat_min, val_min])
        rango_superior = np.array([hue_max, sat_max, val_max])
        
        # Crear máscara
        mascara = cv2.inRange(hsv, rango_inferior, rango_superior)
        
        # Aplicar máscara a la imagen original
        resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)
        
        return resultado
    
    @staticmethod
    def crecimiento_regiones(imagen, semillas=None, umbral=20):
        """
        Aplica segmentación por crecimiento de regiones desde puntos semilla.
        Versión simplificada.
        
        Args:
            imagen: Imagen de entrada
            semillas: Lista de puntos semilla (si es None, se generan automáticamente)
            umbral: Umbral de similitud para agregar píxeles a la región
            
        Returns:
            Imagen segmentada por crecimiento de regiones
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gris = imagen.copy()
        
        # Crear una máscara vacía
        mascara = np.zeros_like(gris)
        
        # Generar semillas automáticamente si no se proporcionan
        if semillas is None:
            h, w = gris.shape
            semillas = [
                (w//4, h//4),
                (3*w//4, h//4),
                (w//4, 3*h//4),
                (3*w//4, 3*h//4)
            ]
        
        # Procesar cada semilla
        for i, semilla in enumerate(semillas):
            # Usar flood fill para crecimiento de región
            mask = np.zeros((gris.shape[0]+2, gris.shape[1]+2), np.uint8)
            valor_semilla = gris[semilla[1], semilla[0]]
            
            # Aplicar flood fill
            cv2.floodFill(gris.copy(), mask, semilla, 255, 
                         loDiff=umbral, upDiff=umbral, flags=cv2.FLOODFILL_MASK_ONLY)
            
            # Combinar con la máscara principal
            mascara = cv2.bitwise_or(mascara, mask[1:-1, 1:-1])
        
        # Mostrar resultado: si es imagen original en color, mostrar con máscara aplicada
        if len(imagen.shape) > 2:
            resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)
        else:
            resultado = cv2.bitwise_and(gris, gris, mask=mascara)
            
        return resultado
    
    @staticmethod
    def segmentacion_por_textura(imagen, tamano_ventana=15):
        """
        Segmentación básica por textura usando varianza local.
        
        Args:
            imagen: Imagen en escala de grises
            tamano_ventana: Tamaño de la ventana para análisis de textura
            
        Returns:
            Imagen segmentada por textura
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Calcular varianza local usando filtro de desenfoque
        imagen_float = imagen.astype(np.float32)
        
        # Media local
        media_local = cv2.blur(imagen_float, (tamano_ventana, tamano_ventana))
        
        # Media de cuadrados
        cuadrados = imagen_float * imagen_float
        media_cuadrados = cv2.blur(cuadrados, (tamano_ventana, tamano_ventana))
        
        # Varianza local
        varianza = media_cuadrados - (media_local * media_local)
        
        # Normalizar varianza
        varianza_norm = cv2.normalize(varianza, None, 0, 255, cv2.NORM_MINMAX)
        
        # Umbralizar para obtener regiones de alta textura
        _, resultado = cv2.threshold(varianza_norm.astype(np.uint8), 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return resultado
    
    @staticmethod
    def grabcut_segmentacion(imagen, rect=None, iteraciones=5):
        """
        Aplica segmentación GrabCut.
        
        Args:
            imagen: Imagen en color
            rect: Rectángulo inicial (x, y, w, h). Si None, usa imagen completa
            iteraciones: Número de iteraciones
            
        Returns:
            Máscara de segmentación y imagen segmentada
        """
        # Asegurar que la imagen esté en formato BGR
        if len(imagen.shape) <= 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        
        # Crear máscara inicial
        mascara = np.zeros(imagen.shape[:2], np.uint8)
        
        # Modelos de fondo y primer plano
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Si no se proporciona rectángulo, usar toda la imagen menos un borde
        if rect is None:
            h, w = imagen.shape[:2]
            rect = (10, 10, w-20, h-20)
        
        # Aplicar GrabCut
        cv2.grabCut(imagen, mascara, rect, bgdModel, fgdModel, iteraciones, cv2.GC_INIT_WITH_RECT)
        
        # Crear máscara final (0 y 2 son fondo, 1 y 3 son primer plano)
        mascara_final = np.where((mascara == 2) | (mascara == 0), 0, 1).astype('uint8')
        
        # Aplicar máscara a la imagen
        resultado = imagen * mascara_final[:, :, np.newaxis]
        
        return mascara_final, resultado