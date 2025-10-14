#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Operaciones Morfológicas para el Sistema de Detección Vehicular
========================================================================

Contiene todas las operaciones morfológicas necesarias para el 
preprocesamiento en el sistema de detección vehicular.

Autor: Sistema de Detección Vehicular
Fecha: Octubre 2025
"""

import cv2
import numpy as np

class OperacionesMorfologicas:
    """
    Clase para operaciones morfológicas en imágenes binarias.
    """
    
    @staticmethod
    def crear_kernel(forma='rectangulo', tamano=5):
        """
        Crea un kernel para operaciones morfológicas.
        
        Args:
            forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            tamano: Tamaño del kernel
            
        Returns:
            Kernel para operaciones morfológicas
        """
        if forma == 'rectangulo':
            return np.ones((tamano, tamano), np.uint8)
        elif forma == 'elipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano, tamano))
        elif forma == 'cruz':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (tamano, tamano))
        else:
            return np.ones((tamano, tamano), np.uint8)
    
    @staticmethod
    def erosion(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
        """
        Aplica erosión a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            kernel_size: Tamaño del kernel
            iteraciones: Número de iteraciones
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen erosionada
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.erode(imagen, kernel, iterations=iteraciones)
    
    @staticmethod
    def dilatacion(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
        """
        Aplica dilatación a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            kernel_size: Tamaño del kernel
            iteraciones: Número de iteraciones
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen dilatada
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.dilate(imagen, kernel, iterations=iteraciones)
    
    @staticmethod
    def apertura(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
        """
        Aplica apertura (erosión seguida de dilatación) a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            kernel_size: Tamaño del kernel
            iteraciones: Número de iteraciones
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con apertura aplicada
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel, iterations=iteraciones)
    
    @staticmethod
    def cierre(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
        """
        Aplica cierre (dilatación seguida de erosión) a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            kernel_size: Tamaño del kernel
            iteraciones: Número de iteraciones
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con cierre aplicado
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)
    
    @staticmethod
    def gradiente_morfologico(imagen, kernel_size=5, kernel_forma='rectangulo'):
        """
        Aplica gradiente morfológico (dilatación - erosión) a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            kernel_size: Tamaño del kernel
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con gradiente morfológico aplicado
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.morphologyEx(imagen, cv2.MORPH_GRADIENT, kernel)
    
    @staticmethod
    def top_hat(imagen, kernel_size=5, kernel_forma='rectangulo'):
        """
        Aplica transformación Top Hat (original - apertura) a una imagen.
        
        Args:
            imagen: Imagen de entrada
            kernel_size: Tamaño del kernel
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con transformación Top Hat aplicada
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)
    
    @staticmethod
    def black_hat(imagen, kernel_size=5, kernel_forma='rectangulo'):
        """
        Aplica transformación Black Hat (cierre - original) a una imagen.
        
        Args:
            imagen: Imagen de entrada
            kernel_size: Tamaño del kernel
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con transformación Black Hat aplicada
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)
    
    @staticmethod
    def eliminar_ruido_binaria(imagen, metodo='apertura', kernel_size=5, kernel_forma='rectangulo'):
        """
        Elimina ruido en una imagen binaria usando operaciones morfológicas.
        
        Args:
            imagen: Imagen binaria
            metodo: Método a utilizar ('apertura' o 'cierre')
            kernel_size: Tamaño del kernel
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen binaria con ruido eliminado
        """
        # Asegurar que la imagen sea binaria
        if len(imagen.shape) > 2:
            imagen_bin = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            _, imagen_bin = cv2.threshold(imagen_bin, 127, 255, cv2.THRESH_BINARY)
        else:
            imagen_bin = imagen.copy()
        
        # Aplicar operación morfológica según el método elegido
        if metodo.lower() == 'apertura':
            return OperacionesMorfologicas.apertura(imagen_bin, kernel_size, 1, kernel_forma)
        elif metodo.lower() == 'cierre':
            return OperacionesMorfologicas.cierre(imagen_bin, kernel_size, 1, kernel_forma)
        else:
            return imagen_bin
    
    @staticmethod
    def extraer_contornos_morfologicos(imagen, kernel_size=3, kernel_forma='rectangulo'):
        """
        Extrae contornos de una imagen binaria usando operaciones morfológicas.
        
        Args:
            imagen: Imagen binaria o en escala de grises
            kernel_size: Tamaño del kernel
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con los contornos extraídos
        """
        # Asegurar que la imagen sea binaria
        if len(imagen.shape) > 2:
            imagen_bin = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            _, imagen_bin = cv2.threshold(imagen_bin, 127, 255, cv2.THRESH_BINARY)
        else:
            imagen_bin = imagen.copy()
        
        # Crear kernel
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        
        # Aplicar dilatación
        dilatada = cv2.dilate(imagen_bin, kernel, iterations=1)
        
        # Extraer contornos (dilatación - original)
        contornos = cv2.subtract(dilatada, imagen_bin)
        
        return contornos
    
    @staticmethod
    def esqueletizacion(imagen):
        """
        Aplica esqueletización a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            
        Returns:
            Imagen esqueletizada
        """
        # Asegurar que la imagen sea binaria
        if len(imagen.shape) > 2:
            imagen_bin = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            _, imagen_bin = cv2.threshold(imagen_bin, 127, 255, cv2.THRESH_BINARY)
        else:
            imagen_bin = imagen.copy()
        
        # Crear una copia para el resultado
        esqueleto = np.zeros(imagen_bin.shape, np.uint8)
        
        # Implementación de esqueletización mediante sucesivas erosiones y aperturas
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = imagen_bin.copy()
        
        while True:
            eroded = cv2.erode(img, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, temp)
            esqueleto = cv2.bitwise_or(esqueleto, temp)
            img = eroded.copy()
            
            if cv2.countNonZero(img) == 0:
                break
        
        return esqueleto
    
    @staticmethod
    def rellenar_huecos(imagen):
        """
        Rellena huecos internos en una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            
        Returns:
            Imagen con huecos internos rellenados
        """
        # Asegurar que la imagen sea binaria
        if len(imagen.shape) > 2:
            imagen_bin = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            _, imagen_bin = cv2.threshold(imagen_bin, 127, 255, cv2.THRESH_BINARY)
        else:
            imagen_bin = imagen.copy()
        
        # Crear una máscara de tamaño mayor para asegurar que el punto semilla esté fuera de la imagen
        h, w = imagen_bin.shape
        mascara = np.zeros((h+2, w+2), np.uint8)
        
        # Crear una copia de la imagen
        relleno = imagen_bin.copy()
        
        # Punto semilla en la esquina (asumiendo que el fondo es negro)
        cv2.floodFill(relleno, mascara, (0, 0), 255)
        
        # Invertir para obtener solo los huecos
        relleno = cv2.bitwise_not(relleno)
        
        # Combinar la imagen original con los huecos rellenados
        resultado = cv2.bitwise_or(imagen_bin, relleno)
        
        return resultado
    
    @staticmethod
    def limpiar_bordes(imagen, conectividad=4):
        """
        Elimina objetos que tocan los bordes de la imagen.
        
        Args:
            imagen: Imagen binaria
            conectividad: Conectividad para el análisis (4 u 8)
            
        Returns:
            Imagen con objetos de borde eliminados
        """
        # Asegurar que la imagen sea binaria
        if len(imagen.shape) > 2:
            imagen_bin = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            _, imagen_bin = cv2.threshold(imagen_bin, 127, 255, cv2.THRESH_BINARY)
        else:
            imagen_bin = imagen.copy()
        
        h, w = imagen_bin.shape
        resultado = imagen_bin.copy()
        
        # Marcar píxeles en los bordes
        bordes = np.zeros_like(imagen_bin)
        bordes[0, :] = imagen_bin[0, :]  # Borde superior
        bordes[-1, :] = imagen_bin[-1, :]  # Borde inferior
        bordes[:, 0] = imagen_bin[:, 0]  # Borde izquierdo
        bordes[:, -1] = imagen_bin[:, -1]  # Borde derecho
        
        # Encontrar componentes conectados que tocan los bordes
        num_labels, labels = cv2.connectedComponents(bordes, connectivity=conectividad)
        
        for label in range(1, num_labels):
            # Eliminar todos los píxeles de este componente de la imagen original
            resultado[labels == label] = 0
        
        return resultado