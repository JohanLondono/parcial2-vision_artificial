#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Operaciones Aritméticas para el Sistema de Detección Vehicular
=======================================================================

Contiene todas las operaciones aritméticas necesarias para el 
preprocesamiento en el sistema de detección vehicular.

Autor: Sistema de Detección Vehicular
Fecha: Octubre 2025
"""

import cv2
import numpy as np

class OperacionesAritmeticas:
    """
    Clase para operaciones aritméticas en imágenes.
    """
    
    @staticmethod
    def suma_imagenes(imagen1, imagen2):
        """
        Realiza la suma de dos imágenes.
        
        Args:
            imagen1: Primera imagen
            imagen2: Segunda imagen
            
        Returns:
            Imagen resultante de la suma
        """
        # Verificar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        return cv2.add(imagen1, imagen2)
    
    @staticmethod
    def resta_imagenes(imagen1, imagen2):
        """
        Realiza la resta de dos imágenes.
        
        Args:
            imagen1: Primera imagen (minuendo)
            imagen2: Segunda imagen (sustraendo)
            
        Returns:
            Imagen resultante de la resta
        """
        # Verificar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        return cv2.subtract(imagen1, imagen2)
    
    @staticmethod
    def multiplicacion_imagenes(imagen1, imagen2):
        """
        Realiza la multiplicación de dos imágenes.
        
        Args:
            imagen1: Primera imagen
            imagen2: Segunda imagen
            
        Returns:
            Imagen resultante de la multiplicación
        """
        # Verificar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        return cv2.multiply(imagen1, imagen2)
    
    @staticmethod
    def division_imagenes(imagen1, imagen2):
        """
        Realiza la división de dos imágenes.
        
        Args:
            imagen1: Primera imagen (numerador)
            imagen2: Segunda imagen (denominador)
            
        Returns:
            Imagen resultante de la división
        """
        # Verificar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        # Evitar división por cero
        imagen2_safe = np.where(imagen2 == 0, 1, imagen2)
        
        return cv2.divide(imagen1, imagen2_safe)
    
    @staticmethod
    def ajustar_brillo(imagen, factor):
        """
        Ajusta el brillo de una imagen multiplicando sus valores por un factor.
        
        Args:
            imagen: Imagen de entrada
            factor: Factor de ajuste (>1 para aumentar brillo, <1 para disminuir)
            
        Returns:
            Imagen con brillo ajustado
        """
        # Convertir a tipo de dato adecuado para la operación
        imagen_float = imagen.astype(np.float32)
        
        # Aplicar el factor de brillo
        imagen_ajustada = imagen_float * factor
        
        # Asegurar que los valores estén en el rango correcto [0, 255]
        imagen_ajustada = np.clip(imagen_ajustada, 0, 255)
        
        # Convertir de vuelta al tipo de dato original
        return imagen_ajustada.astype(imagen.dtype)
    
    @staticmethod
    def ajustar_contraste(imagen, factor):
        """
        Ajusta el contraste de una imagen.
        
        Args:
            imagen: Imagen de entrada
            factor: Factor de ajuste (>1 para aumentar contraste, <1 para disminuir)
            
        Returns:
            Imagen con contraste ajustado
        """
        # Convertir a tipo de dato adecuado para la operación
        imagen_float = imagen.astype(np.float32)
        
        # Calcular el valor medio de la imagen
        media = np.mean(imagen_float)
        
        # Aplicar el ajuste de contraste: (valor - media) * factor + media
        imagen_ajustada = (imagen_float - media) * factor + media
        
        # Asegurar que los valores estén en el rango correcto [0, 255]
        imagen_ajustada = np.clip(imagen_ajustada, 0, 255)
        
        # Convertir de vuelta al tipo de dato original
        return imagen_ajustada.astype(imagen.dtype)
    
    @staticmethod
    def ajustar_gamma(imagen, gamma=1.0):
        """
        Aplica corrección gamma a una imagen.
        
        Args:
            imagen: Imagen de entrada
            gamma: Valor gamma (>1 oscurece, <1 aclara, 1.0 sin cambios)
            
        Returns:
            Imagen con corrección gamma aplicada
        """
        # Crear tabla de lookup para la corrección gamma
        tabla_gamma = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                               for i in np.arange(0, 256)]).astype("uint8")
        
        # Aplicar corrección gamma usando la tabla de lookup
        return cv2.LUT(imagen, tabla_gamma)
    
    @staticmethod
    def mezclar_imagenes(imagen1, imagen2, alpha=0.5):
        """
        Mezcla dos imágenes con un factor de transparencia.
        
        Args:
            imagen1: Primera imagen
            imagen2: Segunda imagen
            alpha: Factor de transparencia (0.0 a 1.0)
            
        Returns:
            Imagen mezclada
        """
        # Verificar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        return cv2.addWeighted(imagen1, alpha, imagen2, 1 - alpha, 0)