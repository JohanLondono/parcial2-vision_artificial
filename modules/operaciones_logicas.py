#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Operaciones Lógicas para el Sistema de Detección Vehicular
===================================================================

Contiene todas las operaciones lógicas necesarias para el 
preprocesamiento en el sistema de detección vehicular.

Autor: Sistema de Detección Vehicular
Fecha: Octubre 2025
"""

import cv2
import numpy as np

class OperacionesLogicas:
    """
    Clase para operaciones lógicas en imágenes binarias.
    """
    
    @staticmethod
    def operacion_and(imagen1, imagen2):
        """
        Realiza la operación lógica AND entre dos imágenes binarias.
        
        Args:
            imagen1: Primera imagen binaria
            imagen2: Segunda imagen binaria
            
        Returns:
            Imagen resultante de la operación AND
        """
        # Asegurar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        return cv2.bitwise_and(imagen1, imagen2)
    
    @staticmethod
    def operacion_or(imagen1, imagen2):
        """
        Realiza la operación lógica OR entre dos imágenes binarias.
        
        Args:
            imagen1: Primera imagen binaria
            imagen2: Segunda imagen binaria
            
        Returns:
            Imagen resultante de la operación OR
        """
        # Asegurar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        return cv2.bitwise_or(imagen1, imagen2)
    
    @staticmethod
    def operacion_not(imagen):
        """
        Realiza la operación lógica NOT en una imagen binaria.
        
        Args:
            imagen: Imagen binaria de entrada
            
        Returns:
            Imagen resultante de la operación NOT (inversión)
        """
        return cv2.bitwise_not(imagen)
    
    @staticmethod
    def operacion_xor(imagen1, imagen2):
        """
        Realiza la operación lógica XOR entre dos imágenes binarias.
        
        Args:
            imagen1: Primera imagen binaria
            imagen2: Segunda imagen binaria
            
        Returns:
            Imagen resultante de la operación XOR
        """
        # Asegurar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        return cv2.bitwise_xor(imagen1, imagen2)
    
    @staticmethod
    def crear_mascara_rectangular(imagen, x, y, ancho, alto):
        """
        Crea una máscara rectangular para aplicar operaciones lógicas.
        
        Args:
            imagen: Imagen de referencia para obtener dimensiones
            x, y: Coordenadas de la esquina superior izquierda
            ancho, alto: Dimensiones del rectángulo
            
        Returns:
            Máscara binaria rectangular
        """
        h, w = imagen.shape[:2]
        mascara = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mascara, (x, y), (x + ancho, y + alto), 255, -1)
        return mascara
    
    @staticmethod
    def crear_mascara_circular(imagen, centro_x, centro_y, radio):
        """
        Crea una máscara circular para aplicar operaciones lógicas.
        
        Args:
            imagen: Imagen de referencia para obtener dimensiones
            centro_x, centro_y: Coordenadas del centro del círculo
            radio: Radio del círculo
            
        Returns:
            Máscara binaria circular
        """
        h, w = imagen.shape[:2]
        mascara = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mascara, (centro_x, centro_y), radio, 255, -1)
        return mascara
    
    @staticmethod
    def aplicar_mascara(imagen, mascara):
        """
        Aplica una máscara a una imagen usando operación AND.
        
        Args:
            imagen: Imagen de entrada
            mascara: Máscara binaria
            
        Returns:
            Imagen con máscara aplicada
        """
        # Asegurar que la máscara tenga el mismo tamaño que la imagen
        if imagen.shape[:2] != mascara.shape[:2]:
            mascara = cv2.resize(mascara, (imagen.shape[1], imagen.shape[0]))
        
        # Si la imagen es en color, convertir máscara a 3 canales
        if len(imagen.shape) == 3:
            mascara = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
        
        return cv2.bitwise_and(imagen, mascara)
    
    @staticmethod
    def crear_mascara_por_rango_color(imagen, limite_inferior, limite_superior, espacio_color='BGR'):
        """
        Crea una máscara basada en un rango de colores.
        
        Args:
            imagen: Imagen en color
            limite_inferior: Límite inferior del rango de color
            limite_superior: Límite superior del rango de color
            espacio_color: Espacio de color ('BGR', 'HSV', 'LAB')
            
        Returns:
            Máscara binaria del rango de color
        """
        if espacio_color == 'HSV':
            imagen_convertida = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        elif espacio_color == 'LAB':
            imagen_convertida = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        else:  # BGR
            imagen_convertida = imagen.copy()
        
        # Crear máscara usando el rango de colores
        mascara = cv2.inRange(imagen_convertida, limite_inferior, limite_superior)
        return mascara
    
    @staticmethod
    def combinar_mascaras(mascaras, operacion='OR'):
        """
        Combina múltiples máscaras usando una operación lógica.
        
        Args:
            mascaras: Lista de máscaras binarias
            operacion: Operación a aplicar ('OR', 'AND', 'XOR')
            
        Returns:
            Máscara combinada
        """
        if not mascaras:
            return None
        
        resultado = mascaras[0].copy()
        
        for mascara in mascaras[1:]:
            # Asegurar mismo tamaño
            if resultado.shape != mascara.shape:
                mascara = cv2.resize(mascara, (resultado.shape[1], resultado.shape[0]))
            
            if operacion.upper() == 'AND':
                resultado = cv2.bitwise_and(resultado, mascara)
            elif operacion.upper() == 'XOR':
                resultado = cv2.bitwise_xor(resultado, mascara)
            else:  # OR
                resultado = cv2.bitwise_or(resultado, mascara)
        
        return resultado