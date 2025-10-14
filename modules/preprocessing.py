#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Preprocesamiento de Imágenes
======================================

Implementación de técnicas de preprocesamiento para mejorar
la calidad de las imágenes antes del análisis de tráfico vehicular.

Funcionalidades:
- Filtrado Gaussiano
- Normalización de histograma
- Redimensionamiento
- Ecualización de histograma
- Corrección de iluminación
- Reducción de ruido
"""

import cv2
import numpy as np
from skimage import exposure, filters
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle


class ImagePreprocessor:
    """Preprocesador de imágenes para análisis de tráfico vehicular."""
    
    def __init__(self):
        """Inicializar el preprocesador."""
        self.preprocessing_history = []
        
    def apply_gaussian_blur(self, imagen, kernel_size=5, sigma=0):
        """
        Aplica filtro Gaussiano para suavizar la imagen.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            kernel_size (int): Tamaño del kernel (debe ser impar)
            sigma (float): Desviación estándar
            
        Returns:
            np.ndarray: Imagen suavizada
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Asegurar que sea impar
        
        resultado = cv2.GaussianBlur(imagen, (kernel_size, kernel_size), sigma)
        self.preprocessing_history.append(f"Gaussian Blur (kernel={kernel_size}, sigma={sigma})")
        return resultado
    
    def normalize_image(self, imagen):
        """
        Normaliza la imagen al rango [0, 255].
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            
        Returns:
            np.ndarray: Imagen normalizada
        """
        normalized = cv2.normalize(imagen, None, 0, 255, cv2.NORM_MINMAX)
        self.preprocessing_history.append("Normalización [0, 255]")
        return normalized.astype(np.uint8)
    
    def equalize_histogram(self, imagen):
        """
        Ecualiza el histograma de la imagen.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            
        Returns:
            np.ndarray: Imagen con histograma ecualizado
        """
        if len(imagen.shape) == 3:
            # Convertir a YCrCb y ecualizar solo el canal Y
            ycrcb = cv2.cvtColor(imagen, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            resultado = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            resultado = cv2.equalizeHist(imagen)
        
        self.preprocessing_history.append("Ecualización de histograma")
        return resultado
    
    def apply_clahe(self, imagen, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            clip_limit (float): Límite de contraste
            tile_grid_size (tuple): Tamaño de la cuadrícula de tiles
            
        Returns:
            np.ndarray: Imagen procesada
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if len(imagen.shape) == 3:
            lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            resultado = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            resultado = clahe.apply(imagen)
        
        self.preprocessing_history.append(f"CLAHE (clip={clip_limit}, grid={tile_grid_size})")
        return resultado
    
    def resize_image(self, imagen, target_size=(512, 512), keep_aspect_ratio=True):
        """
        Redimensiona la imagen.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            target_size (tuple): Tamaño objetivo (width, height)
            keep_aspect_ratio (bool): Mantener proporción de aspecto
            
        Returns:
            np.ndarray: Imagen redimensionada
        """
        if keep_aspect_ratio:
            h, w = imagen.shape[:2]
            target_w, target_h = target_size
            
            # Calcular escala manteniendo aspecto
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Redimensionar
            resized = cv2.resize(imagen, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Crear canvas del tamaño objetivo
            if len(imagen.shape) == 3:
                canvas = np.zeros((target_h, target_w, imagen.shape[2]), dtype=imagen.dtype)
            else:
                canvas = np.zeros((target_h, target_w), dtype=imagen.dtype)
            
            # Centrar imagen en canvas
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            resultado = canvas
        else:
            resultado = cv2.resize(imagen, target_size, interpolation=cv2.INTER_AREA)
        
        self.preprocessing_history.append(f"Resize a {target_size} (aspect_ratio={keep_aspect_ratio})")
        return resultado
    
    def reduce_noise_bilateral(self, imagen, d=9, sigma_color=75, sigma_space=75):
        """
        Reduce ruido usando filtro bilateral (preserva bordes).
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            d (int): Diámetro del vecindario
            sigma_color (float): Filtro en espacio de color
            sigma_space (float): Filtro en espacio de coordenadas
            
        Returns:
            np.ndarray: Imagen con ruido reducido
        """
        resultado = cv2.bilateralFilter(imagen, d, sigma_color, sigma_space)
        self.preprocessing_history.append(f"Filtro Bilateral (d={d}, σ_color={sigma_color})")
        return resultado
    
    def correct_illumination(self, imagen):
        """
        Corrige iluminación no uniforme.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            
        Returns:
            np.ndarray: Imagen con iluminación corregida
        """
        if len(imagen.shape) == 3:
            # Convertir a LAB
            lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Estimar fondo con filtro gaussiano grande
            background = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=50, sigmaY=50)
            
            # Corregir
            l_corrected = cv2.subtract(l_channel, background)
            l_corrected = cv2.normalize(l_corrected, None, 0, 255, cv2.NORM_MINMAX)
            
            # Reconstruir imagen
            lab[:, :, 0] = l_corrected
            resultado = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            background = cv2.GaussianBlur(imagen, (0, 0), sigmaX=50, sigmaY=50)
            resultado = cv2.subtract(imagen, background)
            resultado = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
        
        self.preprocessing_history.append("Corrección de iluminación")
        return resultado
    
    def sharpen_image(self, imagen, amount=1.0):
        """
        Aumenta la nitidez de la imagen.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            amount (float): Cantidad de sharpening (0-2)
            
        Returns:
            np.ndarray: Imagen con mayor nitidez
        """
        # Kernel de sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]]) * amount / 9.0
        kernel[1, 1] += (1.0 - amount)
        
        resultado = cv2.filter2D(imagen, -1, kernel)
        self.preprocessing_history.append(f"Sharpening (amount={amount})")
        return resultado
    
    def auto_preprocessing(self, imagen, config=None):
        """
        Aplica preprocesamiento automático según configuración.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            config (dict): Configuración de preprocesamiento
            
        Returns:
            np.ndarray: Imagen preprocesada
        """
        if config is None:
            config = {
                'gaussian_blur': True,
                'gaussian_kernel': 5,
                'clahe': True,
                'normalize': True,
                'reduce_noise': False,
                'resize': False,
                'target_size': (512, 512)
            }
        
        resultado = imagen.copy()
        self.preprocessing_history = []
        
        # Reducción de ruido
        if config.get('reduce_noise', False):
            resultado = self.reduce_noise_bilateral(resultado)
        
        # Filtro Gaussiano
        if config.get('gaussian_blur', False):
            kernel = config.get('gaussian_kernel', 5)
            resultado = self.apply_gaussian_blur(resultado, kernel)
        
        # CLAHE para mejorar contraste
        if config.get('clahe', False):
            resultado = self.apply_clahe(resultado)
        
        # Normalización
        if config.get('normalize', False):
            resultado = self.normalize_image(resultado)
        
        # Redimensionar
        if config.get('resize', False):
            target_size = config.get('target_size', (512, 512))
            resultado = self.resize_image(resultado, target_size)
        
        return resultado
    
    def get_preprocessing_history(self):
        """
        Obtiene el historial de operaciones de preprocesamiento.
        
        Returns:
            list: Lista de operaciones aplicadas
        """
        return self.preprocessing_history


# Funciones de utilidad
def preprocess_for_vehicle_detection(imagen):
    """
    Preprocesamiento específico para detección de vehículos.
    
    Args:
        imagen (np.ndarray): Imagen de entrada
        
    Returns:
        np.ndarray: Imagen preprocesada
    """
    preprocessor = ImagePreprocessor()
    config = {
        'gaussian_blur': True,
        'gaussian_kernel': 3,
        'clahe': True,
        'normalize': True,
        'reduce_noise': True
    }
    return preprocessor.auto_preprocessing(imagen, config)


def preprocess_for_lane_detection(imagen):
    """
    Preprocesamiento específico para detección de carriles.
    
    Args:
        imagen (np.ndarray): Imagen de entrada
        
    Returns:
        np.ndarray: Imagen preprocesada
    """
    preprocessor = ImagePreprocessor()
    config = {
        'gaussian_blur': True,
        'gaussian_kernel': 5,
        'clahe': True,
        'normalize': False,
        'reduce_noise': False
    }
    return preprocessor.auto_preprocessing(imagen, config)


def preprocess_for_sign_detection(imagen):
    """
    Preprocesamiento específico para detección de señales.
    
    Args:
        imagen (np.ndarray): Imagen de entrada
        
    Returns:
        np.ndarray: Imagen preprocesada
    """
    preprocessor = ImagePreprocessor()
    config = {
        'gaussian_blur': True,
        'gaussian_kernel': 3,
        'clahe': False,
        'normalize': True,
        'reduce_noise': True
    }
    return preprocessor.auto_preprocessing(imagen, config)
