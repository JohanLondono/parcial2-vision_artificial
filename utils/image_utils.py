#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades para Manejo de Imágenes
==================================

Funciones básicas para cargar, procesar y guardar imágenes
para el sistema de detección de objetos vehiculares.
"""

import cv2
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt

class ImageHandler:
    """Clase para manejar operaciones básicas con imágenes."""
    
    @staticmethod
    def cargar_imagen(ruta_imagen):
        """
        Carga una imagen desde archivo.
        
        Args:
            ruta_imagen (str): Ruta a la imagen
            
        Returns:
            np.ndarray: Imagen cargada o None si hay error
        """
        try:
            if not os.path.exists(ruta_imagen):
                print(f"❌ Error: La imagen no existe: {ruta_imagen}")
                return None
                
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
                return None
                
            return imagen
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            return None
    
    @staticmethod
    def obtener_imagenes_carpeta(carpeta, extensiones=None):
        """
        Obtiene lista de imágenes en una carpeta.
        
        Args:
            carpeta (str): Ruta de la carpeta
            extensiones (list): Lista de extensiones permitidas
            
        Returns:
            list: Lista de rutas de imágenes
        """
        if extensiones is None:
            extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        
        imagenes = set()  # Usar set para evitar duplicados
        for ext in extensiones:
            # Buscar tanto minúsculas como mayúsculas en un solo paso
            patron_minus = os.path.join(carpeta, ext.lower())
            patron_mayus = os.path.join(carpeta, ext.upper())
            
            imagenes.update(glob.glob(patron_minus))
            imagenes.update(glob.glob(patron_mayus))
        
        return sorted(imagenes)
    
    @staticmethod
    def convertir_a_gris(imagen):
        """
        Convierte imagen a escala de grises.
        
        Args:
            imagen (np.ndarray): Imagen en color
            
        Returns:
            np.ndarray: Imagen en escala de grises
        """
        if len(imagen.shape) == 3:
            return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return imagen
    
    @staticmethod
    def redimensionar_imagen(imagen, ancho_max=800):
        """
        Redimensiona imagen manteniendo proporción.
        
        Args:
            imagen (np.ndarray): Imagen original
            ancho_max (int): Ancho máximo deseado
            
        Returns:
            np.ndarray: Imagen redimensionada
        """
        alto, ancho = imagen.shape[:2]
        
        if ancho <= ancho_max:
            return imagen
        
        factor = ancho_max / ancho
        nuevo_ancho = ancho_max
        nuevo_alto = int(alto * factor)
        
        return cv2.resize(imagen, (nuevo_ancho, nuevo_alto))
    
    @staticmethod
    def guardar_imagen(imagen, ruta_salida, crear_directorio=True):
        """
        Guarda imagen en archivo.
        
        Args:
            imagen (np.ndarray): Imagen a guardar
            ruta_salida (str): Ruta donde guardar
            crear_directorio (bool): Si crear directorio si no existe
            
        Returns:
            bool: True si se guardó exitosamente
        """
        try:
            if crear_directorio:
                directorio = os.path.dirname(ruta_salida)
                os.makedirs(directorio, exist_ok=True)
            
            success = cv2.imwrite(ruta_salida, imagen)
            if success:
                print(f"Imagen guardada: {ruta_salida}")
                return True
            else:
                print(f"Error guardando imagen: {ruta_salida}")
                return False
        except Exception as e:
            print(f"Error guardando imagen: {e}")
            return False
    
    @staticmethod
    def dibujar_circulos(imagen, circulos, color=(0, 255, 0), grosor=2):
        """
        Dibuja círculos en la imagen.
        
        Args:
            imagen (np.ndarray): Imagen donde dibujar
            circulos (list): Lista de círculos (x, y, radio)
            color (tuple): Color BGR
            grosor (int): Grosor de la línea
            
        Returns:
            np.ndarray: Imagen con círculos dibujados
        """
        imagen_resultado = imagen.copy()
        
        for circulo in circulos:
            x, y, r = int(circulo[0]), int(circulo[1]), int(circulo[2])
            # Dibujar círculo
            cv2.circle(imagen_resultado, (x, y), r, color, grosor)
            # Dibujar centro
            cv2.circle(imagen_resultado, (x, y), 2, color, -1)
        
        return imagen_resultado
    
    @staticmethod
    def dibujar_rectangulos(imagen, rectangulos, color=(255, 0, 0), grosor=2):
        """
        Dibuja rectángulos en la imagen.
        
        Args:
            imagen (np.ndarray): Imagen donde dibujar
            rectangulos (list): Lista de rectángulos (x, y, w, h)
            color (tuple): Color BGR
            grosor (int): Grosor de la línea
            
        Returns:
            np.ndarray: Imagen con rectángulos dibujados
        """
        imagen_resultado = imagen.copy()
        
        for rect in rectangulos:
            x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
            cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), color, grosor)
        
        return imagen_resultado
    
    @staticmethod
    def mostrar_imagen(imagen, titulo="Imagen", esperar_tecla=True):
        """
        Muestra imagen en ventana.
        
        Args:
            imagen (np.ndarray): Imagen a mostrar
            titulo (str): Título de la ventana
            esperar_tecla (bool): Si esperar tecla para cerrar
        """
        cv2.imshow(titulo, imagen)
        if esperar_tecla:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    @staticmethod
    def obtener_timestamp():
        """
        Obtiene timestamp actual.
        
        Returns:
            str: Timestamp formateado
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

def preprocesar_imagen(imagen, aplicar_blur=True, aplicar_clahe=True):
    """
    Aplica preprocesamiento básico a la imagen.
    
    Args:
        imagen (np.ndarray): Imagen original
        aplicar_blur (bool): Si aplicar desenfoque Gaussiano
        aplicar_clahe (bool): Si aplicar ecualización adaptativa
        
    Returns:
        np.ndarray: Imagen preprocesada
    """
    resultado = imagen.copy()
    
    # Aplicar desenfoque para reducir ruido
    if aplicar_blur:
        resultado = cv2.GaussianBlur(resultado, (5, 5), 0)
    
    # Aplicar CLAHE para mejorar contraste
    if aplicar_clahe and len(resultado.shape) == 3:
        # Convertir a LAB para aplicar CLAHE solo en luminancia
        lab = cv2.cvtColor(resultado, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        resultado = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif aplicar_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        resultado = clahe.apply(resultado)
    
    return resultado