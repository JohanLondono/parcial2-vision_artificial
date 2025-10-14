#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Filtros para el Sistema de Detección Vehicular
=======================================================

Contiene todas las operaciones de filtrado de imágenes necesarias
para el preprocesamiento en el sistema de detección vehicular.

Autor: Sistema de Detección Vehicular
Fecha: Octubre 2025
"""

import cv2
import numpy as np

class Filtros:
    """
    Clase para operaciones de filtrado de imágenes.
    """
    
    @staticmethod
    def aplicar_filtro_desenfoque(imagen, kernel_size=(5, 5)):
        """
        Aplica un filtro de desenfoque promedio a la imagen.
        
        Args:
            imagen: Imagen de entrada
            kernel_size: Tamaño del kernel para el desenfoque
            
        Returns:
            Imagen con filtro de desenfoque aplicado
        """
        return cv2.blur(imagen, kernel_size)
    
    @staticmethod
    def aplicar_filtro_gaussiano(imagen, kernel_size=(5, 5), sigma=0):
        """
        Aplica un filtro gaussiano a la imagen.
        
        Args:
            imagen: Imagen de entrada
            kernel_size: Tamaño del kernel para el filtro
            sigma: Desviación estándar en X e Y
            
        Returns:
            Imagen con filtro gaussiano aplicado
        """
        return cv2.GaussianBlur(imagen, kernel_size, sigma)
    
    @staticmethod
    def aplicar_filtro_mediana(imagen, kernel_size=5):
        """
        Aplica un filtro de mediana a la imagen.
        
        Args:
            imagen: Imagen de entrada
            kernel_size: Tamaño del kernel para el filtro
            
        Returns:
            Imagen con filtro de mediana aplicado
        """
        return cv2.medianBlur(imagen, kernel_size)
    
    @staticmethod
    def aplicar_filtro_nitidez(imagen):
        """
        Aplica un filtro de nitidez a la imagen.
        
        Args:
            imagen: Imagen de entrada
            
        Returns:
            Imagen con filtro de nitidez aplicado
        """
        # Kernel para nitidez
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        
        return cv2.filter2D(imagen, -1, kernel)
    
    @staticmethod
    def detectar_bordes_canny(imagen, umbral1=100, umbral2=200):
        """
        Detecta bordes usando el algoritmo Canny.
        
        Args:
            imagen: Imagen en escala de grises
            umbral1: Primer umbral para la detección
            umbral2: Segundo umbral para la detección
            
        Returns:
            Imagen con bordes detectados
        """
        return cv2.Canny(imagen, umbral1, umbral2)
    
    @staticmethod
    def ecualizar_histograma(imagen):
        """
        Ecualiza el histograma de una imagen en escala de grises.
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Imagen con histograma ecualizado
        """
        return cv2.equalizeHist(imagen)
    
    @staticmethod
    def aplicar_filtro_bilateral(imagen, d=9, sigma_color=75, sigma_space=75):
        """
        Aplica un filtro bilateral para reducir ruido preservando bordes.
        
        Args:
            imagen: Imagen de entrada
            d: Diámetro de cada vecindad de píxeles
            sigma_color: Valor sigma en el espacio de color
            sigma_space: Valor sigma en el espacio de coordenadas
            
        Returns:
            Imagen con filtro bilateral aplicado
        """
        return cv2.bilateralFilter(imagen, d, sigma_color, sigma_space)
    
    @staticmethod
    def aplicar_filtro_sobel(imagen, direccion='ambas'):
        """
        Aplica el filtro Sobel para detección de bordes.
        
        Args:
            imagen: Imagen en escala de grises
            direccion: 'x', 'y' o 'ambas'
            
        Returns:
            Imagen con filtro Sobel aplicado
        """
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        if direccion == 'x':
            return cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
        elif direccion == 'y':
            return cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
        else:  # ambas
            sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.magnitude(sobel_x, sobel_y)
    
    @staticmethod
    def aplicar_filtro_laplaciano(imagen):
        """
        Aplica el filtro Laplaciano para detección de bordes.
        
        Args:
            imagen: Imagen de entrada
            
        Returns:
            Imagen con filtro Laplaciano aplicado
        """
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        return cv2.Laplacian(imagen, cv2.CV_64F)