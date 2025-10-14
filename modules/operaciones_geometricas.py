#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Operaciones Geométricas para el Sistema de Detección Vehicular
=======================================================================

Contiene todas las operaciones geométricas necesarias para el 
preprocesamiento en el sistema de detección vehicular.

Autor: Sistema de Detección Vehicular
Fecha: Octubre 2025
"""

import cv2
import numpy as np

class OperacionesGeometricas:
    """
    Clase para operaciones geométricas en imágenes.
    """
    
    @staticmethod
    def redimensionar_imagen(imagen, ancho=None, alto=None):
        """
        Redimensiona una imagen al tamaño especificado.
        Si solo se proporciona una dimensión, mantiene la proporción.
        
        Args:
            imagen: Imagen de entrada
            ancho: Nuevo ancho (None para mantener proporción)
            alto: Nuevo alto (None para mantener proporción)
            
        Returns:
            Imagen redimensionada
        """
        h, w = imagen.shape[:2]
        
        if ancho is None and alto is None:
            return imagen
        
        if ancho is None:
            # Calcular ancho manteniendo la proporción
            r = alto / float(h)
            ancho = int(w * r)
        elif alto is None:
            # Calcular alto manteniendo la proporción
            r = ancho / float(w)
            alto = int(h * r)
        
        return cv2.resize(imagen, (ancho, alto), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def rotar_imagen(imagen, angulo):
        """
        Rota una imagen el ángulo especificado.
        
        Args:
            imagen: Imagen de entrada
            angulo: Ángulo de rotación en grados
            
        Returns:
            Imagen rotada
        """
        (h, w) = imagen.shape[:2]
        centro = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        return cv2.warpAffine(imagen, M, (w, h))
    
    @staticmethod
    def voltear_imagen(imagen, modo):
        """
        Voltea una imagen horizontal, vertical o ambos.
        
        Args:
            imagen: Imagen de entrada
            modo: Modo de volteo (0=vertical, 1=horizontal, -1=ambos)
            
        Returns:
            Imagen volteada
        """
        return cv2.flip(imagen, modo)
    
    @staticmethod
    def trasladar_imagen(imagen, dx, dy):
        """
        Traslada una imagen en las direcciones x e y.
        
        Args:
            imagen: Imagen de entrada
            dx: Desplazamiento en eje x
            dy: Desplazamiento en eje y
            
        Returns:
            Imagen trasladada
        """
        h, w = imagen.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(imagen, M, (w, h))
    
    @staticmethod
    def recortar_imagen(imagen, x1, y1, x2, y2):
        """
        Recorta una región de la imagen.
        
        Args:
            imagen: Imagen de entrada
            x1, y1: Coordenadas de la esquina superior izquierda
            x2, y2: Coordenadas de la esquina inferior derecha
            
        Returns:
            Imagen recortada
        """
        return imagen[y1:y2, x1:x2]
    
    @staticmethod
    def aplicar_transformacion_perspectiva(imagen, pts1, pts2):
        """
        Aplica una transformación de perspectiva a una imagen.
        
        Args:
            imagen: Imagen de entrada
            pts1: Cuatro puntos en la imagen original
            pts2: Cuatro puntos correspondientes en la imagen de salida
            
        Returns:
            Imagen con perspectiva transformada
        """
        M = cv2.getPerspectiveTransform(pts1, pts2)
        h, w = imagen.shape[:2]
        return cv2.warpPerspective(imagen, M, (w, h))
    
    @staticmethod
    def escalar_imagen(imagen, factor_x, factor_y=None):
        """
        Escala una imagen por factores específicos.
        
        Args:
            imagen: Imagen de entrada
            factor_x: Factor de escala en X
            factor_y: Factor de escala en Y (si es None, usa factor_x)
            
        Returns:
            Imagen escalada
        """
        if factor_y is None:
            factor_y = factor_x
        
        h, w = imagen.shape[:2]
        nuevo_ancho = int(w * factor_x)
        nuevo_alto = int(h * factor_y)
        
        return cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def aplicar_transformacion_afin(imagen, puntos_origen, puntos_destino):
        """
        Aplica una transformación afín a una imagen.
        
        Args:
            imagen: Imagen de entrada
            puntos_origen: Tres puntos en la imagen original
            puntos_destino: Tres puntos correspondientes en la imagen de salida
            
        Returns:
            Imagen con transformación afín aplicada
        """
        M = cv2.getAffineTransform(puntos_origen, puntos_destino)
        h, w = imagen.shape[:2]
        return cv2.warpAffine(imagen, M, (w, h))
    
    @staticmethod
    def corregir_distorsion_barril(imagen, k1=-0.2, k2=0.0, k3=0.0):
        """
        Corrige la distorsión de barril en una imagen.
        
        Args:
            imagen: Imagen de entrada
            k1, k2, k3: Coeficientes de distorsión radial
            
        Returns:
            Imagen con distorsión corregida
        """
        h, w = imagen.shape[:2]
        
        # Matriz de la cámara (simplificada)
        camera_matrix = np.array([[w, 0, w/2],
                                 [0, h, h/2],
                                 [0, 0, 1]], dtype=np.float32)
        
        # Coeficientes de distorsión
        dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float32)
        
        # Corregir distorsión
        return cv2.undistort(imagen, camera_matrix, dist_coeffs)