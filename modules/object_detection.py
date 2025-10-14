#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Identificación Específica de Objetos en Tráfico Vehicular
===================================================================

Sistema especializado para identificar diferentes tipos de objetos en imágenes
de tráfico vehicular utilizando los algoritmos más apropiados para cada tipo.

Objetos detectables:
- Vehículos (automóviles, camiones, autobuses)
- Motocicletas y bicicletas  
- Peatones
- Señales de tráfico circulares y rectangulares
- Carriles y líneas viales
- Semáforos
- Elementos de infraestructura

Algoritmos utilizados según el objeto:
- HOG + SVM: Excelente para vehículos y peatones
- Hough: Ideal para señales circulares y carriles
- SURF/ORB: Para matching y seguimiento de objetos
- GrabCut: Segmentación precisa de vehículos
- Edge detection + morphology: Para elementos estructurales
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from skimage import morphology, measure, segmentation
from skimage.filters import gaussian, sobel
from skimage.feature import canny

# Importar analizadores existentes
from .texture_analysis import TextureAnalyzer
from .hough_analysis import HoughAnalyzer  
from .hog_kaze import HOGKAZEAnalyzer
from .surf_orb import SURFORBAnalyzer
from .advanced_algorithms import AdvancedAnalyzer

class ObjectDetectionSystem:
    """Sistema de identificación específica de objetos en tráfico vehicular."""
    
    def __init__(self, output_dir="./resultados"):
        """
        Inicializar el sistema de detección.
        
        Args:
            output_dir (str): Directorio de salida para resultados
        """
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "object_detection")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Inicializar analizadores
        self.texture_analyzer = TextureAnalyzer(output_dir)
        self.hough_analyzer = HoughAnalyzer(output_dir)
        self.hog_kaze_analyzer = HOGKAZEAnalyzer(output_dir)
        self.surf_orb_analyzer = SURFORBAnalyzer(output_dir)
        self.advanced_analyzer = AdvancedAnalyzer(output_dir)
        
        self.detection_results = []
        
        # Configuraciones específicas por tipo de objeto
        self.object_configs = {
            'senales_circulares': {
                'algorithms': ['hough_circles', 'akaze', 'color_analysis', 'glcm', 'log'],
                'circle_params': {
                    'dp': 1.2,
                    'min_dist': 30,
                    'param1': 50,
                    'param2': 30,
                    'min_radius': 15,
                    'max_radius': 100
                },
                'preprocessing': {
                    'clahe_clip': 2.0,
                    'gaussian_kernel': (9, 9),
                    'gaussian_sigma': 2
                }
            },
            'llantas': {
                'algorithms': ['hough_circles', 'glcm', 'sobel', 'orb', 'akaze'],
                'circle_params': {
                    'dp': 1.5,
                    'min_dist': 40,
                    'param1': 50,
                    'param2': 35,
                    'min_radius': 20,
                    'max_radius': 150
                },
                'preprocessing': {
                    'clahe_clip': 3.0,
                    'bilateral_d': 9,
                    'bilateral_sigma': 75
                },
                'color_analysis': {
                    'dark_threshold': 80,
                    'use_hsv': True
                }
            },
            'semaforos': {
                'algorithms': ['hough_circles', 'color_analysis', 'akaze', 'glcm', 'structure_analysis'],
                'circle_params': {
                    'dp': 1.5,
                    'min_dist': 20,
                    'param1': 50,
                    'param2': 30,
                    'min_radius': 8,
                    'max_radius': 50
                },
                'color_ranges': {
                    'red': [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
                    'yellow': [(15, 120, 120), (35, 255, 255)],
                    'green': [(40, 50, 50), (90, 255, 255)]
                },
                'preprocessing': {
                    'clahe_clip': 3.0,
                    'gaussian_kernel': (5, 5),
                    'min_circles_for_traffic_light': 3
                }
            }
        }
    
    # ============================================================================
    # MÉTODOS ESPECIALIZADOS CON PREPROCESAMIENTO + ALGORITMOS INTEGRADOS
    # ============================================================================
    
    def detectar_senales_con_preprocesamiento_avanzado(self, imagen, visualizar=True, tipo_preprocesamiento='adaptativo'):
        """
        Método especializado para señales circulares con preprocesamiento optimizado.
        
        Args:
            imagen: Imagen de entrada
            visualizar: Si mostrar y guardar imagen procesada
            tipo_preprocesamiento: 'adaptativo', 'morfologico', 'multiescala'
        """
        print(f"🔴 Detectando señales con preprocesamiento {tipo_preprocesamiento.upper()}...")
        
        imagen_gris = self._convertir_a_gris(imagen)
        
        if tipo_preprocesamiento == 'adaptativo':
            # Preprocesamiento adaptativo
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            imagen_eq = clahe.apply(imagen_gris)
            
            # Filtro bilateral para preservar bordes
            filtered = cv2.bilateralFilter(imagen_eq, 11, 17, 17)
            
            # Detección con parámetros adaptativos
            circles = cv2.HoughCircles(
                filtered, cv2.HOUGH_GRADIENT, dp=1.1, minDist=25,
                param1=60, param2=25, minRadius=12, maxRadius=120
            )
            
        elif tipo_preprocesamiento == 'morfologico':
            # Para imágenes con alto contraste
            normalized = cv2.normalize(imagen_gris, None, 0, 255, cv2.NORM_MINMAX)
            blur = cv2.GaussianBlur(normalized, (7, 7), 1.5)
            
            circles = cv2.HoughCircles(
                blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=30,
                param1=50, param2=30, minRadius=15, maxRadius=100
            )
            
        elif tipo_preprocesamiento == 'multiescala':
            # Para condiciones de iluminación variable
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12,12))
            imagen_eq = clahe.apply(imagen_gris)
            
            # Filtro morfológico para reducir ruido
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened = cv2.morphologyEx(imagen_eq, cv2.MORPH_OPEN, kernel)
            
            circles = cv2.HoughCircles(
                opened, cv2.HOUGH_GRADIENT, dp=1.3, minDist=35,
                param1=45, param2=35, minRadius=10, maxRadius=110
            )
        
        # Procesar resultados
        senales_detectadas = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Validación adicional basada en color y forma
                confidence = self._validar_senal_circular(imagen, x, y, r)
                
                if confidence > 0.3:  # Umbral de confianza
                    senales_detectadas.append({
                        'centro': (int(x), int(y)),
                        'radio': int(r),
                        'confidence': float(confidence),
                        'metodo_preprocesamiento': tipo_preprocesamiento,
                        'area': int(np.pi * r * r),
                        'bbox': (x-r, y-r, 2*r, 2*r)
                    })
        
        resultado = {
            'num_senales': len(senales_detectadas),
            'senales_detectadas': senales_detectadas,
            'metodo_usado': f'preprocesamiento_{tipo_preprocesamiento}',
            'tipo_objeto': 'senales_circulares',
            'estadisticas': {
                'senales_confirmadas': len(senales_detectadas),
                'confianza_promedio': np.mean([s['confidence'] for s in senales_detectadas]) if senales_detectadas else 0.0,
                'radio_promedio': np.mean([s['radio'] for s in senales_detectadas]) if senales_detectadas else 0.0,
                'area_total': sum([s['area'] for s in senales_detectadas]) if senales_detectadas else 0
            },
            'imagen_procesada': {
                'ecualizacion': imagen_gris,
                'original': imagen
            }
        }
        
        # Guardar imagen procesada automáticamente
        if visualizar:
            self._visualizar_deteccion_senales_circulares(imagen, resultado, guardar=True, mostrar=True)
        
        return resultado
    
    def detectar_llantas_con_analisis_textura(self, imagen, visualizar=True, tipo_analisis='textura_avanzada'):
        """
        Método especializado para llantas con análisis de textura integrado.
        
        Args:
            imagen: Imagen de entrada
            visualizar: Si mostrar y guardar imagen procesada
            tipo_analisis: 'textura_avanzada', 'morfologico', 'multinivel'
        """
        print(f"🔘 Detectando llantas con análisis {tipo_analisis.upper()}...")
        
        imagen_gris = self._convertir_a_gris(imagen)
        
        if tipo_analisis == 'textura_avanzada':
            # Combina GLCM + gradientes + color
            from skimage.feature import graycomatrix, graycoprops
            
            # Preprocesamiento para texturas
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            imagen_eq = clahe.apply(imagen_gris)
            
            # Filtro bilateral para preservar texturas
            filtered = cv2.bilateralFilter(imagen_eq, 9, 75, 75)
            
            # Detección inicial de círculos
            circles = cv2.HoughCircles(
                filtered, cv2.HOUGH_GRADIENT, dp=1.4, minDist=50,
                param1=45, param2=40, minRadius=25, maxRadius=180
            )
            
        elif tipo_analisis == 'morfologico':
            # Enfoque en gradientes radiales
            blur = cv2.GaussianBlur(imagen_gris, (5, 5), 1)
            
            # Gradientes direccionales
            sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Normalizar y convertir
            magnitude = np.uint8(magnitude / magnitude.max() * 255)
            
            circles = cv2.HoughCircles(
                magnitude, cv2.HOUGH_GRADIENT, dp=1.2, minDist=45,
                param1=40, param2=35, minRadius=20, maxRadius=150
            )
            
        elif tipo_analisis == 'multinivel':
            # Específico para patrones de banda de rodadura
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
            imagen_eq = clahe.apply(imagen_gris)
            
            # Filtro para realzar patrones repetitivos
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(imagen_eq, -1, kernel)
            
            circles = cv2.HoughCircles(
                sharpened, cv2.HOUGH_GRADIENT, dp=1.6, minDist=55,
                param1=50, param2=45, minRadius=30, maxRadius=160
            )
        
        # Procesar y validar llantas
        llantas_detectadas = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                confidence = self._validar_llanta(imagen, x, y, r, tipo_analisis)
                
                if confidence > 0.25:  # Umbral para llantas
                    llantas_detectadas.append({
                        'centro': (int(x), int(y)),
                        'radio': int(r),
                        'confidence': float(confidence),
                        'metodo_analisis': tipo_analisis,
                        'area': int(np.pi * r * r),
                        'color_dominante': self._analizar_color_llanta(imagen, x, y, r),
                        'bbox': (x-r, y-r, 2*r, 2*r)
                    })
        
        resultado = {
            'num_llantas': len(llantas_detectadas),
            'llantas_detectadas': llantas_detectadas,
            'metodo_usado': f'analisis_{tipo_analisis}',
            'tipo_objeto': 'llantas',
            'estadisticas': {
                'llantas_confirmadas': len(llantas_detectadas),
                'confianza_promedio': np.mean([l['confidence'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'radio_promedio': np.mean([l['radio'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'area_total': sum([l['area'] for l in llantas_detectadas]) if llantas_detectadas else 0
            },
            'imagen_procesada': {
                'ecualizacion': imagen_gris,
                'original': imagen,
                'dark_mask': cv2.inRange(cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV), (0, 0, 0), (180, 255, 80))
            }
        }
        
        # Guardar imagen procesada automáticamente
        if visualizar:
            self._visualizar_deteccion_llantas(imagen, resultado, guardar=True, mostrar=True)
        
        return resultado
    
    def detectar_semaforos_con_estructura_completa(self, imagen, visualizar=True, tipo_deteccion='estructura_completa'):
        """
        Método especializado para semáforos con análisis de estructura vertical.
        
        Args:
            imagen: Imagen de entrada
            visualizar: Si mostrar y guardar imagen procesada
            tipo_deteccion: 'estructura_completa', 'hsv_avanzado', 'geometrico'
        """
        print(f"🚦 Detectando semáforos con {tipo_deteccion.upper()}...")
        
        imagen_gris = self._convertir_a_gris(imagen)
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        if tipo_deteccion == 'estructura_completa':
            # Análisis completo: color + estructura + geometría
            
            # 1. Preprocesamiento optimizado
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
            imagen_eq = clahe.apply(imagen_gris)
            blur = cv2.GaussianBlur(imagen_eq, (3, 3), 0)
            
            # 2. Detección de círculos (luces)
            circles = cv2.HoughCircles(
                blur, cv2.HOUGH_GRADIENT, dp=1.3, minDist=15,
                param1=50, param2=28, minRadius=6, maxRadius=45
            )
            
            # 3. Análisis de colores específicos
            red_mask = self._crear_mascara_rojo(hsv)
            yellow_mask = self._crear_mascara_amarillo(hsv)
            green_mask = self._crear_mascara_verde(hsv)
            
        elif tipo_deteccion == 'hsv_avanzado':
            # Enfoque en detección por color primero
            
            # Máscaras de color más amplias
            red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
            red_mask = red_mask | red_mask2
            
            yellow_mask = cv2.inRange(hsv, (10, 150, 150), (40, 255, 255))
            green_mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
            
            # Detectar círculos en regiones de color
            combined_mask = red_mask | yellow_mask | green_mask
            masked_gray = cv2.bitwise_and(imagen_gris, imagen_gris, mask=combined_mask)
            
            circles = cv2.HoughCircles(
                masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                param1=40, param2=25, minRadius=8, maxRadius=50
            )
            
        elif tipo_deteccion == 'geometrico':
            # Usando keypoints para validación adicional
            
            # AKAZE para detectar características
            try:
                akaze = cv2.AKAZE_create()
                keypoints, descriptors = akaze.detectAndCompute(imagen_gris, None)
            except:
                keypoints = []
            
            # Detección de círculos estándar
            blur = cv2.GaussianBlur(imagen_gris, (5, 5), 1)
            circles = cv2.HoughCircles(
                blur, cv2.HOUGH_GRADIENT, dp=1.4, minDist=18,
                param1=45, param2=30, minRadius=7, maxRadius=48
            )
        
        # Analizar grupos verticales de círculos
        semaforos_detectados = []
        if circles is not None and len(circles[0]) >= 2:
            circles = np.round(circles[0, :]).astype("int")
            
            # Agrupar círculos verticalmente
            grupos_verticales = self._agrupar_circulos_verticalmente(circles, tolerancia=50)
            
            for grupo in grupos_verticales:
                if len(grupo) >= 2:  # Al menos 2 luces para considerar semáforo
                    confidence = self._validar_semaforo(imagen, grupo, tipo_deteccion)
                    
                    if confidence > 0.4:  # Umbral para semáforos
                        semaforos_detectados.append({
                            'luces': [{'centro': (int(x), int(y)), 'radio': int(r)} for x, y, r in grupo],
                            'num_luces': len(grupo),
                            'confidence': float(confidence),
                            'metodo_estructura': tipo_deteccion,
                            'bbox': self._calcular_bbox_semaforo(grupo),
                            'colores_detectados': self._analizar_colores_semaforo(imagen, grupo)
                        })
        
        resultado = {
            'num_semaforos': len(semaforos_detectados),
            'semaforos_detectados': semaforos_detectados,
            'metodo_usado': f'estructura_{tipo_deteccion}',
            'tipo_objeto': 'semaforos',
            'estadisticas': {
                'semaforos_confirmados': len(semaforos_detectados),
                'confianza_promedio': np.mean([s['confidence'] for s in semaforos_detectados]) if semaforos_detectados else 0.0,
                'luces_total': sum([s['num_luces'] for s in semaforos_detectados]) if semaforos_detectados else 0,
                'luces_promedio': np.mean([s['num_luces'] for s in semaforos_detectados]) if semaforos_detectados else 0.0
            },
            'imagen_procesada': {
                'red_mask': cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)),
                'yellow_mask': cv2.inRange(hsv, (10, 150, 150), (40, 255, 255)),
                'green_mask': cv2.inRange(hsv, (35, 100, 100), (85, 255, 255)),
                'original': imagen
            }
        }
        
        # Guardar imagen procesada automáticamente
        if visualizar:
            self._visualizar_deteccion_semaforos(imagen, resultado, guardar=True, mostrar=True)
        
        return resultado
    
    
    def detectar_senales_circulares(self, imagen, visualizar=True, guardar=True, mostrar=True, metodo='hough'):
        """
        Detecta señales de tráfico circulares usando DIFERENTES MÉTODOS.
        
        El usuario puede elegir el método específico a utilizar.
        
        MÉTODOS DISPONIBLES:
        - 'hough': HoughCircles (Geométrico, rápido) ⚡
        - 'akaze': AKAZE + Análisis de keypoints (Preciso) 🎯
        - 'color': Análisis HSV de colores (Robusto a iluminación) 🌈
        - 'glcm': Texturas GLCM (Análisis de patrones) 📐
        - 'log': Laplaciano de Gauss (Detección de bordes circulares) ⭕
        - 'combinado': Fusión de múltiples métodos (Más preciso pero lento) 🔥
        
        Args:
            imagen (np.ndarray): Imagen de entrada  
            visualizar (bool): Si mostrar visualización
            guardar (bool): Si guardar resultado
            mostrar (bool): Si mostrar ventana plt.show()
            metodo (str): Método a utilizar
            
        Returns:
            dict: Resultados de detección con estructura consistente
        """
        print(f"🔴🔵🟡 Detectando señales circulares con método: {metodo.upper()}...")
        
        # Llamar al método específico según selección
        if metodo == 'hough':
            resultados = self._detectar_senales_hough(imagen)
        elif metodo == 'akaze':
            resultados = self._detectar_senales_akaze(imagen)
        elif metodo == 'color':
            resultados = self._detectar_senales_color(imagen)
        elif metodo == 'glcm':
            resultados = self._detectar_senales_glcm(imagen)
        elif metodo == 'log':
            resultados = self._detectar_senales_log(imagen)
        elif metodo == 'combinado':
            resultados = self._detectar_senales_combinado(imagen)
        else:
            print(f"⚠️  Método '{metodo}' no reconocido. Usando 'hough' por defecto.")
            resultados = self._detectar_senales_hough(imagen)
        
        # Agregar metadatos
        resultados['metodo_usado'] = metodo
        resultados['tipo_objeto'] = 'senales_circulares'
        
        print(f"  ✓ Detección completada: {resultados['num_senales']} señales encontradas")
        
        if visualizar:
            self._visualizar_deteccion_senales_circulares(imagen, resultados, guardar=guardar, mostrar=mostrar)
        
        return resultados
    
    # ============================================================================
    # MÉTODOS INDIVIDUALES PARA SEÑALES CIRCULARES
    # ============================================================================
    
    def _detectar_senales_hough(self, imagen):
        """Método 1: HoughCircles - Detección geométrica pura."""
        print("  🔵 Método: HoughCircles (Geométrico)")
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # Preprocesamiento
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        blur = cv2.GaussianBlur(imagen_eq, (9, 9), 2)
        
        # HoughCircles
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=15,
            maxRadius=min(ancho, altura) // 4
        )
        
        senales_detectadas = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"  ✓ {len(circles)} círculos detectados")
            
            for (x, y, r) in circles:
                # Validar bounds
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # ROI para análisis básico
                roi = imagen[max(0, y-r):min(altura, y+r), max(0, x-r):min(ancho, x+r)]
                
                if roi.size == 0:
                    continue
                
                # Análisis simple de confianza
                roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                std_dev = np.std(roi_gris)
                variance_score = min(std_dev / 50.0, 1.0)
                
                # Bordes
                edges = cv2.Canny(roi_gris, 50, 150)
                edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
                edge_score = min(edge_density * 5, 1.0)
                
                confidence = (variance_score * 0.5 + edge_score * 0.5)
                
                if confidence > 0.25:
                    senales_detectadas.append({
                        'bbox': (x-r, y-r, 2*r, 2*r),
                        'centro': (x, y),
                        'radio': r,
                        'confidence': confidence,
                        'metricas': {
                            'variance_score': variance_score,
                            'edge_score': edge_score
                        }
                    })
        
        return {
            'num_senales': len(senales_detectadas),
            'senales_detectadas': senales_detectadas,
            'imagen_procesada': {
                'ecualizacion': imagen_eq,
                'blur': blur
            },
            'estadisticas': {
                'circulos_totales': len(circles) if circles is not None else 0,
                'senales_confirmadas': len(senales_detectadas),
                'confianza_promedio': np.mean([s['confidence'] for s in senales_detectadas]) if senales_detectadas else 0.0,
                'algoritmo': 'HoughCircles'
            }
        }
    
    def _detectar_senales_akaze(self, imagen):
        """Método 2: AKAZE - Detección por keypoints."""
        print("  🎯 Método: AKAZE + Análisis de Forma")
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # Preprocesamiento
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        
        # AKAZE
        try:
            akaze = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                threshold=0.001,
                nOctaves=4
            )
            keypoints, descriptors = akaze.detectAndCompute(imagen_eq, None)
            print(f"  ✓ {len(keypoints)} keypoints AKAZE detectados")
        except Exception as e:
            print(f"  ⚠️  Error AKAZE: {e}")
            keypoints = []
            descriptors = None
        
        # HoughCircles para candidatos iniciales
        blur = cv2.GaussianBlur(imagen_eq, (9, 9), 2)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=30, minRadius=15, maxRadius=min(ancho, altura) // 4
        )
        
        senales_detectadas = []
        
        if circles is not None and len(keypoints) > 0:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # Contar keypoints en ROI
                points_in_roi = sum(1 for kp in keypoints 
                                   if (x-r) <= kp.pt[0] <= (x+r) and (y-r) <= kp.pt[1] <= (y+r))
                
                roi_area = np.pi * r * r
                keypoint_density = points_in_roi / roi_area * 100
                keypoint_score = min(keypoint_density, 1.0)
                
                # Análisis de circularidad
                roi = imagen[max(0, y-r):min(altura, y+r), max(0, x-r):min(ancho, x+r)]
                roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(roi_gris, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                circularity_score = 0.0
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        circularity_score = min(circularity, 1.0)
                
                confidence = keypoint_score * 0.6 + circularity_score * 0.4
                
                if confidence > 0.30:
                    senales_detectadas.append({
                        'bbox': (x-r, y-r, 2*r, 2*r),
                        'centro': (x, y),
                        'radio': r,
                        'confidence': confidence,
                        'metricas': {
                            'keypoint_density': keypoint_score,
                            'circularity': circularity_score,
                            'keypoints_in_roi': points_in_roi
                        }
                    })
        
        return {
            'num_senales': len(senales_detectadas),
            'senales_detectadas': senales_detectadas,
            'imagen_procesada': {
                'ecualizacion': imagen_eq,
                'keypoints_totales': len(keypoints)
            },
            'estadisticas': {
                'keypoints_akaze': len(keypoints),
                'senales_confirmadas': len(senales_detectadas),
                'confianza_promedio': np.mean([s['confidence'] for s in senales_detectadas]) if senales_detectadas else 0.0,
                'algoritmo': 'AKAZE + Circularidad'
            }
        }
    
    def _detectar_senales_color(self, imagen):
        """Método 3: Análisis HSV - Detección por color."""
        print("  🌈 Método: Análisis de Color HSV")
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Preprocesamiento
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        blur = cv2.GaussianBlur(imagen_eq, (9, 9), 2)
        
        # HoughCircles para candidatos
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=30, minRadius=15, maxRadius=min(ancho, altura) // 4
        )
        
        senales_detectadas = []
        analisis_color = {'red_pct': 0, 'blue_pct': 0, 'yellow_pct': 0, 'white_pct': 0}
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # ROI para análisis de color
                roi = imagen[max(0, y-r):min(altura, y+r), max(0, x-r):min(ancho, x+r)]
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Máscaras de colores
                red_mask1 = cv2.inRange(roi_hsv, (0, 100, 100), (10, 255, 255))
                red_mask2 = cv2.inRange(roi_hsv, (170, 100, 100), (180, 255, 255))
                red_mask = red_mask1 | red_mask2
                
                blue_mask = cv2.inRange(roi_hsv, (100, 100, 50), (130, 255, 255))
                yellow_mask = cv2.inRange(roi_hsv, (20, 100, 100), (30, 255, 255))
                white_mask = cv2.inRange(roi_hsv, (0, 0, 200), (180, 30, 255))
                
                total_pixels = roi.shape[0] * roi.shape[1]
                red_pct = np.sum(red_mask > 0) / total_pixels
                blue_pct = np.sum(blue_mask > 0) / total_pixels
                yellow_pct = np.sum(yellow_mask > 0) / total_pixels
                white_pct = np.sum(white_mask > 0) / total_pixels
                
                # Determinar color dominante
                max_color_pct = max(red_pct, blue_pct, yellow_pct)
                
                # Bonus si tiene blanco + color (típico de señales)
                has_sign_pattern = white_pct > 0.15 and max_color_pct > 0.10
                
                confidence = max_color_pct * 0.7 + white_pct * 0.3
                if has_sign_pattern:
                    confidence *= 1.2
                
                confidence = min(confidence, 1.0)
                
                if confidence > 0.25:
                    senales_detectadas.append({
                        'bbox': (x-r, y-r, 2*r, 2*r),
                        'centro': (x, y),
                        'radio': r,
                        'confidence': confidence,
                        'metricas': {
                            'red_percentage': red_pct,
                            'blue_percentage': blue_pct,
                            'yellow_percentage': yellow_pct,
                            'white_percentage': white_pct,
                            'max_color_pct': max_color_pct
                        }
                    })
                
                # Acumular para estadísticas globales
                analisis_color['red_pct'] += red_pct
                analisis_color['blue_pct'] += blue_pct
                analisis_color['yellow_pct'] += yellow_pct
                analisis_color['white_pct'] += white_pct
        
        return {
            'num_senales': len(senales_detectadas),
            'senales_detectadas': senales_detectadas,
            'imagen_procesada': {
                'ecualizacion': imagen_eq,
                'hsv': hsv
            },
            'analisis_color': analisis_color,
            'estadisticas': {
                'circulos_totales': len(circles) if circles is not None else 0,
                'senales_confirmadas': len(senales_detectadas),
                'confianza_promedio': np.mean([s['confidence'] for s in senales_detectadas]) if senales_detectadas else 0.0,
                'algoritmo': 'Análisis HSV Color'
            }
        }
    
    def _detectar_senales_glcm(self, imagen):
        """Método 4: GLCM - Detección por texturas."""
        print("  📐 Método: Texturas GLCM")
        
        from skimage.feature import graycomatrix, graycoprops
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # Preprocesamiento
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        blur = cv2.GaussianBlur(imagen_eq, (9, 9), 2)
        
        # HoughCircles para candidatos
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=30, minRadius=15, maxRadius=min(ancho, altura) // 4
        )
        
        senales_detectadas = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # ROI para análisis de textura
                roi_gris = imagen_gris[max(0, y-r):min(altura, y+r), max(0, x-r):min(ancho, x+r)]
                
                if roi_gris.size == 0:
                    continue
                
                try:
                    # Cuantizar para GLCM
                    roi_glcm = ((roi_gris / 255.0) * 15).astype(np.uint8)
                    glcm = graycomatrix(roi_glcm, distances=[1], 
                                       angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                       levels=16, symmetric=True, normed=True)
                    
                    contraste = graycoprops(glcm, 'contrast').mean()
                    homogeneidad = graycoprops(glcm, 'homogeneity').mean()
                    energia = graycoprops(glcm, 'energy').mean()
                    correlacion = graycoprops(glcm, 'correlation').mean()
                    
                    # Las señales tienen texturas distintivas
                    # Alto contraste (texto/símbolos vs fondo)
                    # Energía moderada (patrones repetitivos)
                    texture_score = (
                        min(contraste / 10.0, 1.0) * 0.4 +
                        energia * 0.3 +
                        (1 - homogeneidad) * 0.3
                    )
                    
                    confidence = min(texture_score, 1.0)
                    
                    if confidence > 0.30:
                        senales_detectadas.append({
                            'bbox': (x-r, y-r, 2*r, 2*r),
                            'centro': (x, y),
                            'radio': r,
                            'confidence': confidence,
                            'metricas': {
                                'contraste': float(contraste),
                                'homogeneidad': float(homogeneidad),
                                'energia': float(energia),
                                'correlacion': float(correlacion),
                                'texture_score': texture_score
                            }
                        })
                
                except Exception as e:
                    print(f"    ⚠️  Error GLCM en círculo ({x},{y}): {e}")
                    continue
        
        return {
            'num_senales': len(senales_detectadas),
            'senales_detectadas': senales_detectadas,
            'imagen_procesada': {
                'ecualizacion': imagen_eq
            },
            'estadisticas': {
                'circulos_totales': len(circles) if circles is not None else 0,
                'senales_confirmadas': len(senales_detectadas),
                'confianza_promedio': np.mean([s['confidence'] for s in senales_detectadas]) if senales_detectadas else 0.0,
                'algoritmo': 'Texturas GLCM'
            }
        }
    
    def _detectar_senales_log(self, imagen):
        """Método 5: LoG - Detección por Laplaciano de Gauss."""
        print("  ⭕ Método: Laplaciano de Gauss (LoG)")
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # Preprocesamiento
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        
        # Laplaciano de Gauss
        gaussian = cv2.GaussianBlur(imagen_eq, (5, 5), 1.4)
        laplacian = cv2.Laplacian(gaussian, cv2.CV_64F, ksize=3)
        laplacian_abs = np.uint8(np.absolute(laplacian))
        
        # HoughCircles para candidatos
        blur = cv2.GaussianBlur(imagen_eq, (9, 9), 2)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=30, minRadius=15, maxRadius=min(ancho, altura) // 4
        )
        
        senales_detectadas = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # ROI para análisis LoG
                roi_log = laplacian_abs[max(0, y-r):min(altura, y+r), max(0, x-r):min(ancho, x+r)]
                
                if roi_log.size == 0:
                    continue
                
                total_pixels = roi_log.shape[0] * roi_log.shape[1]
                
                # Bordes circulares fuertes
                log_density = np.sum(roi_log > 30) / total_pixels
                log_mean = np.mean(roi_log) / 255.0
                log_std = np.std(roi_log) / 255.0
                
                # Los círculos de señales tienen bordes bien definidos
                log_score = min(log_density * 3, 1.0) * 0.5 + log_std * 0.5
                
                confidence = log_score
                
                if confidence > 0.25:
                    senales_detectadas.append({
                        'bbox': (x-r, y-r, 2*r, 2*r),
                        'centro': (x, y),
                        'radio': r,
                        'confidence': confidence,
                        'metricas': {
                            'log_density': log_density,
                            'log_mean': float(log_mean),
                            'log_std': float(log_std),
                            'log_score': log_score
                        }
                    })
        
        return {
            'num_senales': len(senales_detectadas),
            'senales_detectadas': senales_detectadas,
            'imagen_procesada': {
                'ecualizacion': imagen_eq,
                'laplacian': laplacian_abs
            },
            'estadisticas': {
                'circulos_totales': len(circles) if circles is not None else 0,
                'senales_confirmadas': len(senales_detectadas),
                'confianza_promedio': np.mean([s['confidence'] for s in senales_detectadas]) if senales_detectadas else 0.0,
                'algoritmo': 'Laplaciano de Gauss'
            }
        }
    
    def _detectar_senales_combinado(self, imagen):
        """Método 6: Combinado - Fusión de múltiples métodos."""
        print("  🔥 Método: COMBINADO (Fusión Multi-Algoritmo)")
        print("     Ejecutando: Hough + AKAZE + Color + GLCM + LoG...")
        
        # Ejecutar todos los métodos
        resultados_hough = self._detectar_senales_hough(imagen)
        resultados_akaze = self._detectar_senales_akaze(imagen)
        resultados_color = self._detectar_senales_color(imagen)
        resultados_glcm = self._detectar_senales_glcm(imagen)
        resultados_log = self._detectar_senales_log(imagen)
        
        # Fusionar detecciones usando IOU (Intersection over Union)
        todos_resultados = [
            resultados_hough, resultados_akaze, resultados_color,
            resultados_glcm, resultados_log
        ]
        
        senales_fusionadas = self._fusionar_detecciones(todos_resultados)
        
        print(f"  ✓ Fusión completada: {len(senales_fusionadas)} señales únicas")
        
        return {
            'num_senales': len(senales_fusionadas),
            'senales_detectadas': senales_fusionadas,
            'imagen_procesada': resultados_hough['imagen_procesada'],  # Usar una de referencia
            'estadisticas': {
                'detecciones_hough': resultados_hough['num_senales'],
                'detecciones_akaze': resultados_akaze['num_senales'],
                'detecciones_color': resultados_color['num_senales'],
                'detecciones_glcm': resultados_glcm['num_senales'],
                'detecciones_log': resultados_log['num_senales'],
                'senales_confirmadas': len(senales_fusionadas),
                'confianza_promedio': np.mean([s['confidence'] for s in senales_fusionadas]) if senales_fusionadas else 0.0,
                'algoritmo': 'Fusión Multi-Método'
            }
        }
    
    def _fusionar_detecciones(self, lista_resultados, iou_threshold=0.5):
        """Fusiona detecciones de múltiples métodos usando IOU."""
        todas_detecciones = []
        
        # Recolectar todas las detecciones con su fuente
        for idx, resultado in enumerate(lista_resultados):
            for senal in resultado['senales_detectadas']:
                senal_copy = senal.copy()
                senal_copy['metodo_idx'] = idx
                todas_detecciones.append(senal_copy)
        
        if not todas_detecciones:
            return []
        
        # Ordenar por confianza (mayor primero)
        todas_detecciones.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Non-Maximum Suppression (NMS) usando IOU
        detecciones_finales = []
        usadas = set()
        
        for i, det1 in enumerate(todas_detecciones):
            if i in usadas:
                continue
            
            # Buscar duplicados
            duplicados = [det1]
            x1, y1, w1, h1 = det1['bbox']
            
            for j, det2 in enumerate(todas_detecciones[i+1:], start=i+1):
                if j in usadas:
                    continue
                
                x2, y2, w2, h2 = det2['bbox']
                
                # Calcular IOU
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                
                if x_right < x_left or y_bottom < y_top:
                    intersection = 0.0
                else:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    duplicados.append(det2)
                    usadas.add(j)
            
            # Promediar confianzas de duplicados
            confidence_promedio = np.mean([d['confidence'] for d in duplicados])
            det1['confidence'] = confidence_promedio
            det1['num_detectores'] = len(duplicados)
            
            detecciones_finales.append(det1)
            usadas.add(i)
        
        return detecciones_finales
    
    # ============================================================================
    # FIN MÉTODOS SEÑALES CIRCULARES
    # ============================================================================
    
    
    def detectar_semaforos(self, imagen, visualizar=True):
        """
        Detecta semáforos usando MÚLTIPLES ALGORITMOS.
        
        ALGORITMOS UTILIZADOS:
        1. HoughCircles - Detectar luces circulares
        2. HSV Color - Detectar rojo, amarillo, verde
        3. Análisis de Estructura Vertical - 3 luces apiladas
        4. AKAZE - Puntos clave para validar textura
        5. Contornos - Detectar caja rectangular del semáforo
        6. GLCM - Análisis de texturas
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si mostrar visualización
            
        Returns:
            dict: Resultados de detección de semáforos
        """
        print("🚦🟢🟡🔴 Detectando semáforos (MULTI-ALGORITMO)...")
        
        from skimage.feature import graycomatrix, graycoprops
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # ============================================
        # PASO 1: Preprocesamiento
        # ============================================
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        blur = cv2.GaussianBlur(imagen_eq, (5, 5), 0)
        
        # ============================================
        # PASO 2: Detección de círculos (luces del semáforo)
        # ============================================
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=8,
            maxRadius=50
        )
        
        if circles is None:
            print("  ⚠️  No se detectaron círculos (luces de semáforo)")
            circles = np.array([])
        else:
            circles = np.round(circles[0, :]).astype("int")
            print(f"  ✓ Círculos detectados: {len(circles)}")
        
        # ============================================
        # PASO 3: Análisis de color HSV
        # ============================================
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Máscaras para colores de semáforo
        red_mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        red_mask = red_mask1 | red_mask2
        
        yellow_mask = cv2.inRange(hsv, (15, 120, 120), (35, 255, 255))
        green_mask = cv2.inRange(hsv, (40, 50, 50), (90, 255, 255))
        
        # ============================================
        # PASO 4: AKAZE para validación de textura
        # ============================================
        try:
            akaze = cv2.AKAZE_create(threshold=0.001)
            kp_akaze, _ = akaze.detectAndCompute(imagen_eq, None)
            print(f"  ✓ AKAZE: {len(kp_akaze)} keypoints")
        except Exception as e:
            print(f"  ⚠️  AKAZE no disponible: {e}")
            kp_akaze = []
        
        # ============================================
        # PASO 5: Agrupar círculos verticalmente
        # ============================================
        semaforos_detectados = []
        
        if len(circles) >= 3:
            # Ordenar círculos por posición Y (vertical)
            circles_sorted = sorted(circles, key=lambda c: c[1])
            
            # Buscar grupos de 3 círculos verticales
            i = 0
            while i < len(circles_sorted) - 2:
                c1, c2, c3 = circles_sorted[i:i+3]
                x1, y1, r1 = c1
                x2, y2, r2 = c2
                x3, y3, r3 = c3
                
                # Verificar alineación vertical
                diff_x12 = abs(x2 - x1)
                diff_x23 = abs(x3 - x2)
                
                # Verificar espaciado vertical
                diff_y12 = abs(y2 - y1)
                diff_y23 = abs(y3 - y2)
                
                # Verificar radios similares
                diff_r = max(abs(r2-r1), abs(r3-r2))
                
                # Criterios de alineación
                max_x_diff = 30  # Máxima diferencia horizontal
                min_y_spacing = r1 * 1.5  # Espaciado mínimo vertical
                max_y_spacing = r1 * 4  # Espaciado máximo vertical
                max_r_diff = r1 * 0.5  # Diferencia máxima en radios
                
                if (diff_x12 < max_x_diff and diff_x23 < max_x_diff and
                    min_y_spacing < diff_y12 < max_y_spacing and
                    min_y_spacing < diff_y23 < max_y_spacing and
                    diff_r < max_r_diff):
                    
                    # Posible semáforo encontrado
                    # Calcular bounding box del semáforo
                    x_min = min(x1-r1, x2-r2, x3-r3)
                    x_max = max(x1+r1, x2+r2, x3+r3)
                    y_min = y1 - r1
                    y_max = y3 + r3
                    
                    # Validar que está dentro de la imagen
                    if x_min >= 0 and y_min >= 0 and x_max < ancho and y_max < altura:
                        
                        # Extraer ROI
                        roi = imagen[y_min:y_max, x_min:x_max]
                        roi_gris = imagen_gris[y_min:y_max, x_min:x_max]
                        
                        if roi.size == 0:
                            i += 1
                            continue
                        
                        # ========================================
                        # ANÁLISIS 1: Color de cada luz
                        # ========================================
                        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        
                        # Analizar color de cada círculo
                        colores_detectados = []
                        
                        for (cx, cy, cr) in [c1, c2, c3]:
                            # Coordenadas relativas al ROI
                            cx_rel = cx - x_min
                            cy_rel = cy - y_min
                            
                            # Crear máscara circular
                            mask_circle = np.zeros(roi_gris.shape, dtype=np.uint8)
                            cv2.circle(mask_circle, (cx_rel, cy_rel), cr, 255, -1)
                            
                            # Aplicar máscaras de color
                            red_in_circle = cv2.bitwise_and(red_mask[y_min:y_max, x_min:x_max], mask_circle)
                            yellow_in_circle = cv2.bitwise_and(yellow_mask[y_min:y_max, x_min:x_max], mask_circle)
                            green_in_circle = cv2.bitwise_and(green_mask[y_min:y_max, x_min:x_max], mask_circle)
                            
                            red_pct = np.sum(red_in_circle > 0) / (np.pi * cr * cr)
                            yellow_pct = np.sum(yellow_in_circle > 0) / (np.pi * cr * cr)
                            green_pct = np.sum(green_in_circle > 0) / (np.pi * cr * cr)
                            
                            # Determinar color dominante
                            color_scores = {'rojo': red_pct, 'amarillo': yellow_pct, 'verde': green_pct}
                            color_dominante = max(color_scores, key=color_scores.get)
                            
                            colores_detectados.append({
                                'color': color_dominante,
                                'scores': color_scores
                            })
                        
                        # ========================================
                        # ANÁLISIS 2: Texturas GLCM
                        # ========================================
                        try:
                            roi_glcm = ((roi_gris / 255.0) * 15).astype(np.uint8)
                            glcm = graycomatrix(roi_glcm, distances=[1], angles=[0],
                                               levels=16, symmetric=True, normed=True)
                            
                            contraste = graycoprops(glcm, 'contrast')[0, 0]
                            homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
                            texture_score = min(contraste / 5.0, 1.0)
                        except:
                            texture_score = 0.5
                        
                        # ========================================
                        # ANÁLISIS 3: Densidad de keypoints
                        # ========================================
                        akaze_density = 0.0
                        if len(kp_akaze) > 0:
                            points_in_roi = sum(1 for kp in kp_akaze
                                               if x_min <= kp.pt[0] <= x_max and y_min <= kp.pt[1] <= y_max)
                            roi_area = (x_max - x_min) * (y_max - y_min)
                            akaze_density = points_in_roi / roi_area * 1000
                            akaze_density = min(akaze_density, 1.0)
                        
                        # ========================================
                        # ANÁLISIS 4: Estructura rectangular
                        # ========================================
                        edges = cv2.Canny(roi_gris, 50, 150)
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        rect_score = 0.0
                        if contours:
                            # Buscar contorno rectangular
                            for cnt in contours:
                                area = cv2.contourArea(cnt)
                                if area > 100:
                                    peri = cv2.arcLength(cnt, True)
                                    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                                    if len(approx) == 4:  # Rectángulo
                                        rect_score = 1.0
                                        break
                        
                        # ========================================
                        # CÁLCULO DE CONFIANZA
                        # ========================================
                        # Verificar que tiene los 3 colores
                        colores_unicos = {c['color'] for c in colores_detectados}
                        tiene_3_colores = len(colores_unicos) == 3
                        
                        color_score = 1.0 if tiene_3_colores else 0.3
                        
                        confidence = (
                            color_score * 0.50 +         # Colores correctos es crucial
                            texture_score * 0.15 +       # Textura
                            akaze_density * 0.15 +       # Puntos clave
                            rect_score * 0.20            # Estructura rectangular
                        )
                        
                        if confidence > 0.30:
                            semaforos_detectados.append({
                                'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                                'luces': [
                                    {'posicion': (x1, y1), 'radio': r1, 'color': colores_detectados[0]},
                                    {'posicion': (x2, y2), 'radio': r2, 'color': colores_detectados[1]},
                                    {'posicion': (x3, y3), 'radio': r3, 'color': colores_detectados[2]}
                                ],
                                'confidence': confidence,
                                'tiene_3_colores': tiene_3_colores,
                                'metricas': {
                                    'texture_score': texture_score,
                                    'akaze_density': akaze_density,
                                    'rect_score': rect_score
                                }
                            })
                            
                            # Saltar los círculos usados
                            i += 3
                            continue
                
                i += 1
        
        print(f"  ✓ Semáforos detectados: {len(semaforos_detectados)}")
        
        resultados = {
            'tipo_objeto': 'semaforos',
            'num_semaforos': len(semaforos_detectados),
            'semaforos_detectados': semaforos_detectados,
            'imagen_procesada': {
                'ecualizacion': imagen_eq,
                'circulos_totales': len(circles),
                'red_mask': red_mask,
                'yellow_mask': yellow_mask,
                'green_mask': green_mask
            },
            'estadisticas': {
                'circulos_detectados': len(circles),
                'semaforos_confirmados': len(semaforos_detectados),
                'confianza_promedio': np.mean([s['confidence'] for s in semaforos_detectados]) if semaforos_detectados else 0.0,
                'algoritmos_usados': 'HoughCircles + HSV + AKAZE + GLCM + Contornos'
            }
        }
        
        if visualizar:
            self._visualizar_deteccion_semaforos(imagen, resultados)
        
        return resultados
    
    def detectar_llantas(self, imagen, visualizar=True, guardar=True, mostrar=True, metodo='hough'):
        """
        Detecta llantas/neumáticos de vehículos usando DIFERENTES MÉTODOS.
        
        El usuario puede elegir el método específico a utilizar.
        
        MÉTODOS DISPONIBLES:
        - 'hough': HoughCircles + Color oscuro (Rápido) ⚡
        - 'glcm': Texturas GLCM de banda de rodadura (Textura) 📐
        - 'sobel': Gradientes radiales (Estructura) ⭕
        - 'orb': ORB Keypoints (Puntos clave) 🎯
        - 'akaze': AKAZE Keypoints (Multiscale) 🔍
        - 'combinado': Fusión de múltiples métodos (Más preciso pero lento) 🔥
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si mostrar visualización
            guardar (bool): Si guardar resultado
            mostrar (bool): Si mostrar ventana plt.show()
            metodo (str): Método a utilizar
            
        Returns:
            dict: Resultados de detección de llantas
        """
        print(f"🔘⚫🔵 Detectando llantas/neumáticos con método: {metodo.upper()}...")
        
        # Llamar al método específico según selección
        if metodo == 'hough':
            resultados = self._detectar_llantas_hough(imagen)
        elif metodo == 'glcm':
            resultados = self._detectar_llantas_glcm(imagen)
        elif metodo == 'sobel':
            resultados = self._detectar_llantas_sobel(imagen)
        elif metodo == 'orb':
            resultados = self._detectar_llantas_orb(imagen)
        elif metodo == 'akaze':
            resultados = self._detectar_llantas_akaze(imagen)
        elif metodo == 'combinado':
            resultados = self._detectar_llantas_combinado(imagen)
        else:
            print(f"⚠️  Método '{metodo}' no reconocido. Usando 'hough' por defecto.")
            resultados = self._detectar_llantas_hough(imagen)
        
        # Agregar metadatos
        resultados['metodo_usado'] = metodo
        resultados['tipo_objeto'] = 'llantas'
        
        print(f"  ✓ Detección completada: {resultados['num_llantas']} llantas encontradas")
        
        if visualizar:
            self._visualizar_deteccion_llantas(imagen, resultados, guardar=guardar, mostrar=mostrar)
        
        return resultados
    
    # ============================================================================
    # MÉTODOS INDIVIDUALES PARA LLANTAS
    # ============================================================================
    
    def _detectar_llantas_hough(self, imagen):
        """Método 1: HoughCircles + HSV Color - Detección geométrica y color."""
        print("  🔵 Método: HoughCircles + Color Oscuro")
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # Preprocesamiento
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        blur = cv2.bilateralFilter(imagen_eq, 9, 75, 75)
        
        # HoughCircles (parámetros para llantas)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=40,
            param1=50, param2=35, minRadius=20, maxRadius=min(ancho, altura) // 3
        )
        
        # Máscara de colores oscuros
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
        
        llantas_detectadas = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"  ✓ {len(circles)} círculos detectados")
            
            for (x, y, r) in circles:
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # Extraer ROI
                y1 = max(0, y-r)
                y2 = min(altura, y+r)
                x1 = max(0, x-r)
                x2 = min(ancho, x+r)
                
                roi_mask = dark_mask[y1:y2, x1:x2]
                roi_gris = imagen_gris[y1:y2, x1:x2]
                
                if roi_mask.size == 0:
                    continue
                
                # Análisis de color oscuro
                dark_pct = np.sum(roi_mask > 0) / (roi_mask.shape[0] * roi_mask.shape[1])
                color_score = min(dark_pct * 2, 1.0)
                
                # Análisis de intensidad
                mean_intensity = np.mean(roi_gris)
                darkness_score = 1 - (mean_intensity / 255.0)
                
                confidence = color_score * 0.6 + darkness_score * 0.4
                
                if confidence > 0.30:
                    llantas_detectadas.append({
                        'bbox': (x-r, y-r, 2*r, 2*r),
                        'centro': (x, y),
                        'radio': r,
                        'confidence': confidence,
                        'metricas': {
                            'dark_percentage': dark_pct,
                            'color_score': color_score,
                            'mean_intensity': mean_intensity
                        }
                    })
        
        return {
            'num_llantas': len(llantas_detectadas),
            'llantas_detectadas': llantas_detectadas,
            'imagen_procesada': {
                'ecualizacion': imagen_eq,
                'dark_mask': dark_mask
            },
            'estadisticas': {
                'circulos_totales': len(circles) if circles is not None else 0,
                'llantas_confirmadas': len(llantas_detectadas),
                'confianza_promedio': np.mean([l['confidence'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'radio_promedio': np.mean([l['radio'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'algoritmo': 'HoughCircles + HSV Color'
            }
        }
    
    def _detectar_llantas_glcm(self, imagen):
        """Método 2: GLCM - Detección por texturas de banda de rodadura."""
        print("  📐 Método: Texturas GLCM")
        
        from skimage.feature import graycomatrix, graycoprops
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # Preprocesamiento
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        blur = cv2.GaussianBlur(imagen_eq, (9, 9), 2)
        
        # HoughCircles para candidatos
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=40,
            param1=50, param2=35, minRadius=20, maxRadius=min(ancho, altura) // 3
        )
        
        llantas_detectadas = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # ROI para análisis
                roi_gris = imagen_gris[max(0, y-r):min(altura, y+r), max(0, x-r):min(ancho, x+r)]
                
                if roi_gris.size == 0:
                    continue
                
                try:
                    # Análisis GLCM (textura de banda de rodamiento)
                    roi_glcm = ((roi_gris / 255.0) * 15).astype(np.uint8)
                    glcm = graycomatrix(roi_glcm, distances=[1, 2], 
                                       angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                       levels=16, symmetric=True, normed=True)
                    
                    contraste = graycoprops(glcm, 'contrast').mean()
                    homogeneidad = graycoprops(glcm, 'homogeneity').mean()
                    disimilitud = graycoprops(glcm, 'dissimilarity').mean()
                    
                    # Llantas tienen alto contraste y baja homogeneidad (surcos)
                    texture_score = (contraste / 15.0 * 0.5 + 
                                   (1 - homogeneidad) * 0.3 + 
                                   disimilitud / 5.0 * 0.2)
                    texture_score = min(texture_score, 1.0)
                    
                    confidence = texture_score
                    
                    if confidence > 0.35:
                        llantas_detectadas.append({
                            'bbox': (x-r, y-r, 2*r, 2*r),
                            'centro': (x, y),
                            'radio': r,
                            'confidence': confidence,
                            'metricas': {
                                'texture_score': texture_score,
                                'contraste': float(contraste),
                                'homogeneidad': float(homogeneidad),
                                'disimilitud': float(disimilitud)
                            }
                        })
                
                except Exception as e:
                    print(f"    ⚠️  Error GLCM: {e}")
                    continue
        
        return {
            'num_llantas': len(llantas_detectadas),
            'llantas_detectadas': llantas_detectadas,
            'imagen_procesada': {
                'ecualizacion': imagen_eq
            },
            'estadisticas': {
                'circulos_totales': len(circles) if circles is not None else 0,
                'llantas_confirmadas': len(llantas_detectadas),
                'confianza_promedio': np.mean([l['confidence'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'radio_promedio': np.mean([l['radio'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'algoritmo': 'Texturas GLCM'
            }
        }
    
    def _detectar_llantas_sobel(self, imagen):
        """Método 3: Gradientes Sobel - Detección por gradientes radiales."""
        print("  ⭕ Método: Gradientes Radiales (Sobel)")
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # Preprocesamiento
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        blur = cv2.bilateralFilter(imagen_eq, 9, 75, 75)
        
        # Gradientes Sobel
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)
        
        # HoughCircles para candidatos
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=40,
            param1=50, param2=35, minRadius=20, maxRadius=min(ancho, altura) // 3
        )
        
        llantas_detectadas = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # ROI de gradientes
                roi_sobel = sobel_magnitude[max(0, y-r):min(altura, y+r), max(0, x-r):min(ancho, x+r)]
                roi_gris = imagen_gris[max(0, y-r):min(altura, y+r), max(0, x-r):min(ancho, x+r)]
                
                if roi_sobel.size == 0:
                    continue
                
                # Crear máscara circular
                mask_circle = np.zeros(roi_sobel.shape, dtype=np.uint8)
                center_roi = (roi_sobel.shape[1]//2, roi_sobel.shape[0]//2)
                cv2.circle(mask_circle, center_roi, min(roi_sobel.shape)//2, 255, -1)
                
                # Densidad de gradientes en círculo
                gradients_in_circle = cv2.bitwise_and(roi_sobel, roi_sobel, mask=mask_circle)
                gradient_density = np.sum(gradients_in_circle > 30) / np.sum(mask_circle > 0)
                gradient_score = min(gradient_density * 3, 1.0)
                
                # Varianza (debe tener estructura)
                variance_score = min(np.std(roi_gris) / 50.0, 1.0)
                
                confidence = gradient_score * 0.7 + variance_score * 0.3
                
                if confidence > 0.30:
                    llantas_detectadas.append({
                        'bbox': (x-r, y-r, 2*r, 2*r),
                        'centro': (x, y),
                        'radio': r,
                        'confidence': confidence,
                        'metricas': {
                            'gradient_score': gradient_score,
                            'gradient_density': gradient_density,
                            'variance_score': variance_score
                        }
                    })
        
        return {
            'num_llantas': len(llantas_detectadas),
            'llantas_detectadas': llantas_detectadas,
            'imagen_procesada': {
                'sobel_magnitude': sobel_magnitude
            },
            'estadisticas': {
                'circulos_totales': len(circles) if circles is not None else 0,
                'llantas_confirmadas': len(llantas_detectadas),
                'confianza_promedio': np.mean([l['confidence'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'radio_promedio': np.mean([l['radio'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'algoritmo': 'Gradientes Sobel Radiales'
            }
        }
    
    def _detectar_llantas_orb(self, imagen):
        """Método 4: ORB - Detección por keypoints."""
        print("  🎯 Método: ORB Keypoints")
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # Preprocesamiento
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        
        # ORB
        try:
            orb = cv2.ORB_create(nfeatures=500)
            keypoints, descriptors = orb.detectAndCompute(imagen_eq, None)
            print(f"  ✓ {len(keypoints)} keypoints ORB")
        except Exception as e:
            print(f"  ⚠️  Error ORB: {e}")
            keypoints = []
        
        # HoughCircles para candidatos
        blur = cv2.GaussianBlur(imagen_eq, (9, 9), 2)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=40,
            param1=50, param2=35, minRadius=20, maxRadius=min(ancho, altura) // 3
        )
        
        llantas_detectadas = []
        
        if circles is not None and len(keypoints) > 0:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # Contar keypoints en ROI
                points_in_roi = sum(1 for kp in keypoints
                                   if (x-r) <= kp.pt[0] <= (x+r) and (y-r) <= kp.pt[1] <= (y+r))
                
                roi_area = np.pi * r * r
                keypoint_density = points_in_roi / roi_area * 1000
                keypoint_score = min(keypoint_density, 1.0)
                
                confidence = keypoint_score
                
                if confidence > 0.25:
                    llantas_detectadas.append({
                        'bbox': (x-r, y-r, 2*r, 2*r),
                        'centro': (x, y),
                        'radio': r,
                        'confidence': confidence,
                        'metricas': {
                            'keypoint_density': keypoint_score,
                            'keypoints_in_roi': points_in_roi
                        }
                    })
        
        return {
            'num_llantas': len(llantas_detectadas),
            'llantas_detectadas': llantas_detectadas,
            'imagen_procesada': {
                'keypoints_totales': len(keypoints)
            },
            'estadisticas': {
                'circulos_totales': len(circles) if circles is not None else 0,
                'llantas_confirmadas': len(llantas_detectadas),
                'confianza_promedio': np.mean([l['confidence'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'radio_promedio': np.mean([l['radio'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'keypoints_orb': len(keypoints),
                'algoritmo': 'ORB Keypoints'
            }
        }
    
    def _detectar_llantas_akaze(self, imagen):
        """Método 5: AKAZE - Detección por keypoints multiscale."""
        print("  🔍 Método: AKAZE Multiscale")
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # Preprocesamiento
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        
        # AKAZE
        try:
            akaze = cv2.AKAZE_create(threshold=0.001, nOctaves=4)
            keypoints, descriptors = akaze.detectAndCompute(imagen_eq, None)
            print(f"  ✓ {len(keypoints)} keypoints AKAZE")
        except Exception as e:
            print(f"  ⚠️  Error AKAZE: {e}")
            keypoints = []
        
        # HoughCircles para candidatos
        blur = cv2.GaussianBlur(imagen_eq, (9, 9), 2)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=40,
            param1=50, param2=35, minRadius=20, maxRadius=min(ancho, altura) // 3
        )
        
        llantas_detectadas = []
        
        if circles is not None and len(keypoints) > 0:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # Contar keypoints en ROI
                points_in_roi = sum(1 for kp in keypoints
                                   if (x-r) <= kp.pt[0] <= (x+r) and (y-r) <= kp.pt[1] <= (y+r))
                
                roi_area = np.pi * r * r
                keypoint_density = points_in_roi / roi_area * 1000
                keypoint_score = min(keypoint_density, 1.0)
                
                confidence = keypoint_score
                
                if confidence > 0.25:
                    llantas_detectadas.append({
                        'bbox': (x-r, y-r, 2*r, 2*r),
                        'centro': (x, y),
                        'radio': r,
                        'confidence': confidence,
                        'metricas': {
                            'keypoint_density': keypoint_score,
                            'keypoints_in_roi': points_in_roi
                        }
                    })
        
        return {
            'num_llantas': len(llantas_detectadas),
            'llantas_detectadas': llantas_detectadas,
            'imagen_procesada': {
                'keypoints_totales': len(keypoints)
            },
            'estadisticas': {
                'circulos_totales': len(circles) if circles is not None else 0,
                'llantas_confirmadas': len(llantas_detectadas),
                'confianza_promedio': np.mean([l['confidence'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'radio_promedio': np.mean([l['radio'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'keypoints_akaze': len(keypoints),
                'algoritmo': 'AKAZE Multiscale'
            }
        }
    
    def _detectar_llantas_combinado(self, imagen):
        """Método 6: Combinado - Fusión de múltiples métodos."""
        print("  🔥 Método: COMBINADO (Fusión Multi-Algoritmo)")
        print("     Ejecutando: Hough + GLCM + Sobel + ORB + AKAZE...")
        
        # Ejecutar todos los métodos
        resultados_hough = self._detectar_llantas_hough(imagen)
        resultados_glcm = self._detectar_llantas_glcm(imagen)
        resultados_sobel = self._detectar_llantas_sobel(imagen)
        resultados_orb = self._detectar_llantas_orb(imagen)
        resultados_akaze = self._detectar_llantas_akaze(imagen)
        
        # Fusionar detecciones
        todos_resultados = [
            resultados_hough, resultados_glcm, resultados_sobel,
            resultados_orb, resultados_akaze
        ]
        
        llantas_fusionadas = self._fusionar_detecciones(todos_resultados, iou_threshold=0.6)
        
        print(f"  ✓ Fusión completada: {len(llantas_fusionadas)} llantas únicas")
        
        return {
            'num_llantas': len(llantas_fusionadas),
            'llantas_detectadas': llantas_fusionadas,
            'imagen_procesada': resultados_hough['imagen_procesada'],
            'estadisticas': {
                'detecciones_hough': resultados_hough['num_llantas'],
                'detecciones_glcm': resultados_glcm['num_llantas'],
                'detecciones_sobel': resultados_sobel['num_llantas'],
                'detecciones_orb': resultados_orb['num_llantas'],
                'detecciones_akaze': resultados_akaze['num_llantas'],
                'llantas_confirmadas': len(llantas_fusionadas),
                'confianza_promedio': np.mean([l['confidence'] for l in llantas_fusionadas]) if llantas_fusionadas else 0.0,
                'radio_promedio': np.mean([l['radio'] for l in llantas_fusionadas]) if llantas_fusionadas else 0.0,
                'algoritmo': 'Fusión Multi-Método'
            }
        }
    
    # ============================================================================
    # FIN MÉTODOS LLANTAS
    # ============================================================================
    
    # Métodos auxiliares
    def _convertir_a_gris(self, imagen):
        """Convertir imagen a escala de grises."""
        if len(imagen.shape) == 3:
            return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        """
        Detecta llantas/neumáticos de vehículos usando MÚLTIPLES ALGORITMOS.
        
        CARACTERÍSTICAS DE LLANTAS:
        - Forma circular/elíptica
        - Color oscuro (negro, gris oscuro)
        - Textura distintiva (banda de rodadura)
        - Aro metálico en el centro (opcional)
        - Gradientes radiales desde el centro
        
        ALGORITMOS UTILIZADOS:
        1. HoughCircles - Detectar forma circular
        2. HSV Color Analysis - Detectar tonos oscuros (negro/gris)
        3. GLCM Texturas - Analizar textura de banda de rodadura
        4. Gradientes Radiales - Verificar patrón radial típico
        5. ORB/AKAZE - Puntos clave para validación
        6. Sobel - Detectar bordes y estructura
        7. Contornos - Análisis de forma circular
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si mostrar visualización
            
        Returns:
            dict: Resultados de detección de llantas
        """
        print("🔘⚫🔵 Detectando llantas/neumáticos (MULTI-ALGORITMO)...")
        
        from skimage.feature import graycomatrix, graycoprops
        
        imagen_gris = self._convertir_a_gris(imagen)
        altura, ancho = imagen_gris.shape
        
        # ============================================
        # PASO 1: Preprocesamiento
        # ============================================
        # Ecualización adaptativa para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        imagen_eq = clahe.apply(imagen_gris)
        
        # Suavizado bilateral para preservar bordes
        blur = cv2.bilateralFilter(imagen_eq, 9, 75, 75)
        
        # ============================================
        # PASO 2: Detección de círculos (llantas)
        # ============================================
        # Parámetros ajustados para llantas (más grandes que luces)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=40,  # Llantas separadas entre sí
            param1=50,
            param2=35,
            minRadius=20,  # Llantas más grandes que luces
            maxRadius=min(ancho, altura) // 3
        )
        
        if circles is None:
            print("  ⚠️  No se detectaron círculos (llantas)")
            circles = np.array([])
        else:
            circles = np.round(circles[0, :]).astype("int")
            print(f"  ✓ Círculos detectados: {len(circles)}")
        
        # ============================================
        # PASO 3: ORB/AKAZE para puntos clave
        # ============================================
        try:
            orb = cv2.ORB_create(nfeatures=500)
            kp_orb, desc_orb = orb.detectAndCompute(imagen_eq, None)
            print(f"  ✓ ORB: {len(kp_orb)} keypoints")
        except Exception as e:
            print(f"  ⚠️  ORB error: {e}")
            kp_orb = []
        
        try:
            akaze = cv2.AKAZE_create(threshold=0.001)
            kp_akaze, desc_akaze = akaze.detectAndCompute(imagen_eq, None)
            print(f"  ✓ AKAZE: {len(kp_akaze)} keypoints")
        except Exception as e:
            print(f"  ⚠️  AKAZE error: {e}")
            kp_akaze = []
        
        # ============================================
        # PASO 4: Análisis de HSV para colores oscuros
        # ============================================
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Máscara para tonos oscuros (negro, gris oscuro)
        # Las llantas son típicamente negras o muy oscuras
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))  # V < 80 = oscuro
        
        # Máscara para grises (neumáticos desgastados)
        gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 50, 150))  # Baja saturación
        
        # Combinar máscaras
        tire_color_mask = cv2.bitwise_or(dark_mask, gray_mask)
        
        # ============================================
        # PASO 5: Gradientes con Sobel
        # ============================================
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)
        
        # ============================================
        # PASO 6: Análisis detallado de cada círculo
        # ============================================
        llantas_detectadas = []
        
        for idx, (x, y, r) in enumerate(circles):
            # Validar que el círculo está dentro de la imagen
            if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                continue
            
            # Extraer ROI
            margin = 10
            y1 = max(0, y-r-margin)
            y2 = min(altura, y+r+margin)
            x1 = max(0, x-r-margin)
            x2 = min(ancho, x+r+margin)
            
            roi = imagen[y1:y2, x1:x2]
            roi_gris = imagen_gris[y1:y2, x1:x2]
            roi_eq = imagen_eq[y1:y2, x1:x2]
            
            if roi.size == 0 or roi_gris.size == 0:
                continue
            
            # ========================================
            # ANÁLISIS 1: Color Oscuro (HSV)
            # ========================================
            roi_color_mask = tire_color_mask[y1:y2, x1:x2]
            roi_area = roi_gris.shape[0] * roi_gris.shape[1]
            dark_percentage = np.sum(roi_color_mask > 0) / roi_area
            
            # Las llantas deben ser >40% oscuras
            color_score = min(dark_percentage * 2, 1.0)
            
            # ========================================
            # ANÁLISIS 2: Texturas GLCM (banda de rodadura)
            # ========================================
            try:
                # Las llantas tienen textura repetitiva (banda de rodadura)
                roi_glcm = ((roi_eq / 255.0) * 15).astype(np.uint8)
                glcm = graycomatrix(roi_glcm, distances=[1, 2], 
                                   angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                   levels=16, symmetric=True, normed=True)
                
                contraste = graycoprops(glcm, 'contrast').mean()
                homogeneidad = graycoprops(glcm, 'homogeneity').mean()
                energia = graycoprops(glcm, 'energy').mean()
                disimilitud = graycoprops(glcm, 'dissimilarity').mean()
                
                # Las llantas tienen alto contraste (surcos) y baja homogeneidad
                texture_score = (contraste / 15.0 * 0.4 + 
                               (1 - homogeneidad) * 0.3 + 
                               disimilitud / 5.0 * 0.3)
                texture_score = min(texture_score, 1.0)
                
            except Exception as e:
                texture_score = 0.5
                contraste = homogeneidad = energia = disimilitud = 0
            
            # ========================================
            # ANÁLISIS 3: Gradientes Radiales
            # ========================================
            # Las llantas tienen gradientes que apuntan hacia el centro
            roi_sobel = sobel_magnitude[y1:y2, x1:x2]
            
            # Crear máscara circular para el ROI
            mask_circle = np.zeros(roi_gris.shape, dtype=np.uint8)
            center_roi = (roi_gris.shape[1]//2, roi_gris.shape[0]//2)
            cv2.circle(mask_circle, center_roi, r, 255, -1)
            
            # Calcular densidad de gradientes en el círculo
            gradients_in_circle = cv2.bitwise_and(roi_sobel, roi_sobel, mask=mask_circle)
            gradient_density = np.sum(gradients_in_circle > 30) / np.sum(mask_circle > 0)
            gradient_score = min(gradient_density * 3, 1.0)
            
            # ========================================
            # ANÁLISIS 4: Densidad de Puntos Clave (ORB + AKAZE)
            # ========================================
            orb_density = 0.0
            if len(kp_orb) > 0:
                points_in_roi = sum(1 for kp in kp_orb
                                   if x1 <= kp.pt[0] <= x2 and y1 <= kp.pt[1] <= y2)
                orb_density = points_in_roi / roi_area * 1000
                orb_density = min(orb_density, 1.0)
            
            akaze_density = 0.0
            if len(kp_akaze) > 0:
                points_in_roi = sum(1 for kp in kp_akaze
                                   if x1 <= kp.pt[0] <= x2 and y1 <= kp.pt[1] <= y2)
                akaze_density = points_in_roi / roi_area * 1000
                akaze_density = min(akaze_density, 1.0)
            
            keypoint_score = (orb_density + akaze_density) / 2
            
            # ========================================
            # ANÁLISIS 5: Análisis de Estructura Circular
            # ========================================
            # Detectar bordes en el ROI
            edges = cv2.Canny(roi_eq, 50, 150)
            
            # Buscar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Evaluar circularidad de los contornos
            circularity_score = 0.0
            if contours:
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 100:
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter ** 2)
                            circularity_score = max(circularity_score, circularity)
            
            # ========================================
            # ANÁLISIS 6: Intensidad y Contraste Local
            # ========================================
            mean_intensity = np.mean(roi_gris)
            std_intensity = np.std(roi_gris)
            
            # Las llantas son oscuras con buena varianza (textura)
            darkness_score = 1 - (mean_intensity / 255.0)  # Más oscuro = mejor
            variance_score = min(std_intensity / 50.0, 1.0)  # Debe tener textura
            
            # ========================================
            # CÁLCULO DE CONFIANZA MULTI-ALGORITMO
            # ========================================
            confidence = (
                color_score * 0.25 +         # Color oscuro es crucial
                texture_score * 0.20 +       # Textura de banda de rodadura
                gradient_score * 0.15 +      # Gradientes radiales
                keypoint_score * 0.15 +      # Puntos clave ORB/AKAZE
                circularity_score * 0.10 +   # Forma circular
                darkness_score * 0.10 +      # Intensidad oscura
                variance_score * 0.05        # Varianza de textura
            )
            
            # Bonus si es muy oscuro Y tiene buena textura
            if dark_percentage > 0.5 and texture_score > 0.5:
                confidence *= 1.2
            
            # Penalización si es muy pequeño (probable falso positivo)
            if r < 25:
                confidence *= 0.8
            
            # Umbral de confianza
            if confidence > 0.30:
                llantas_detectadas.append({
                    'bbox': (x-r, y-r, 2*r, 2*r),
                    'centro': (x, y),
                    'radio': r,
                    'confidence': confidence,
                    'tipo': 'llanta',
                    'metricas': {
                        'color_score': color_score,
                        'dark_percentage': dark_percentage,
                        'texture_score': texture_score,
                        'gradient_score': gradient_score,
                        'orb_density': orb_density,
                        'akaze_density': akaze_density,
                        'circularity': circularity_score,
                        'mean_intensity': mean_intensity,
                        'std_intensity': std_intensity,
                        'contraste_glcm': contraste,
                        'homogeneidad_glcm': homogeneidad
                    }
                })
        
        print(f"  ✓ Llantas detectadas: {len(llantas_detectadas)}")
        
        # Ordenar por confianza
        llantas_detectadas.sort(key=lambda x: x['confidence'], reverse=True)
        
        resultados = {
            'tipo_objeto': 'llantas',
            'num_llantas': len(llantas_detectadas),
            'llantas_detectadas': llantas_detectadas,
            'imagen_procesada': {
                'ecualizacion': imagen_eq,
                'blur': blur,
                'dark_mask': dark_mask,
                'sobel_magnitude': sobel_magnitude,
                'circulos_totales': len(circles)
            },
            'estadisticas': {
                'circulos_hough': len(circles),
                'llantas_confirmadas': len(llantas_detectadas),
                'confianza_promedio': np.mean([l['confidence'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'confianza_maxima': np.max([l['confidence'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'radio_promedio': np.mean([l['radio'] for l in llantas_detectadas]) if llantas_detectadas else 0.0,
                'keypoints_orb': len(kp_orb),
                'keypoints_akaze': len(kp_akaze),
                'algoritmos_usados': 'HoughCircles + HSV + GLCM + Sobel + ORB + AKAZE + Contornos'
            }
        }
        
        if visualizar:
            self._visualizar_deteccion_llantas(imagen, resultados)
        
        return resultados
    
    # Métodos auxiliares
    def _convertir_a_gris(self, imagen):
        """Convertir imagen a escala de grises."""
        if len(imagen.shape) == 3:
            return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return imagen
    
    # ============================================================================
    # MÉTODOS DE VALIDACIÓN PARA LOS MÉTODOS ESPECIALIZADOS
    # ============================================================================
    
    def _validar_senal_circular(self, imagen, x, y, r):
        """Validar si un círculo detectado es realmente una señal de tráfico."""
        try:
            # Extraer región de interés
            x1, y1 = max(0, x-r), max(0, y-r)
            x2, y2 = min(imagen.shape[1], x+r), min(imagen.shape[0], y+r)
            roi = imagen[y1:y2, x1:x2]
            
            if roi.size == 0:
                return 0.0
            
            # Análisis de color en HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Verificar colores típicos de señales (rojo, azul, amarillo, blanco)
            red_mask1 = cv2.inRange(hsv_roi, (0, 120, 120), (10, 255, 255))
            red_mask2 = cv2.inRange(hsv_roi, (170, 120, 120), (180, 255, 255))
            red_mask = red_mask1 | red_mask2
            
            blue_mask = cv2.inRange(hsv_roi, (100, 120, 120), (130, 255, 255))
            yellow_mask = cv2.inRange(hsv_roi, (20, 120, 120), (30, 255, 255))
            white_mask = cv2.inRange(hsv_roi, (0, 0, 200), (180, 30, 255))
            
            # Calcular porcentajes de color
            total_pixels = roi.shape[0] * roi.shape[1]
            red_pct = np.sum(red_mask) / 255 / total_pixels
            blue_pct = np.sum(blue_mask) / 255 / total_pixels
            yellow_pct = np.sum(yellow_mask) / 255 / total_pixels
            white_pct = np.sum(white_mask) / 255 / total_pixels
            
            # Puntuación basada en colores típicos de señales
            color_score = max(red_pct, blue_pct, yellow_pct, white_pct) * 2
            
            # Análisis de circularidad
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            circularity_score = 0.0
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    circularity_score = min(circularity, 1.0)
            
            # Puntuación combinada
            confidence = (color_score * 0.6 + circularity_score * 0.4)
            return min(confidence, 1.0)
            
        except Exception as e:
            return 0.0
    
    def _validar_llanta(self, imagen, x, y, r, metodo):
        """Validar si un círculo detectado es realmente una llanta."""
        try:
            # Extraer región de interés
            x1, y1 = max(0, x-r), max(0, y-r)
            x2, y2 = min(imagen.shape[1], x+r), min(imagen.shape[0], y+r)
            roi = imagen[y1:y2, x1:x2]
            
            if roi.size == 0:
                return 0.0
            
            # Análisis de color (llantas suelen ser oscuras)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            dark_mask = cv2.inRange(hsv_roi, (0, 0, 0), (180, 255, 100))
            
            total_pixels = roi.shape[0] * roi.shape[1]
            dark_pct = np.sum(dark_mask) / 255 / total_pixels
            
            # Análisis de textura específico para el método
            texture_score = 0.0
            
            if metodo == 'textura_avanzada':
                # Usar GLCM para analizar textura de banda de rodadura
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                try:
                    from skimage.feature import graycomatrix, graycoprops
                    
                    # Reducir resolución para GLCM
                    small_roi = cv2.resize(gray_roi, (32, 32))
                    glcm = graycomatrix(small_roi, [1], [0], 256, symmetric=True, normed=True)
                    
                    # Propiedades de textura
                    contrast = graycoprops(glcm, 'contrast')[0, 0]
                    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                    
                    # Las llantas tienen alta textura/contraste
                    texture_score = min((contrast + dissimilarity) / 100.0, 1.0)
                except:
                    texture_score = 0.5
                    
            elif metodo == 'bordes_radiales':
                # Analizar gradientes radiales
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Calcular gradientes
                sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                
                # Promedio de gradientes (llantas tienen muchos bordes)
                avg_gradient = np.mean(magnitude) / 255.0
                texture_score = min(avg_gradient * 2, 1.0)
                
            else:
                texture_score = 0.5
            
            # Puntuación combinada
            confidence = (dark_pct * 0.4 + texture_score * 0.6)
            return min(confidence, 1.0)
            
        except Exception as e:
            return 0.0
    
    def _analizar_color_llanta(self, imagen, x, y, r):
        """Analizar el color dominante de una llanta."""
        try:
            x1, y1 = max(0, x-r), max(0, y-r)
            x2, y2 = min(imagen.shape[1], x+r), min(imagen.shape[0], y+r)
            roi = imagen[y1:y2, x1:x2]
            
            # Calcular color promedio
            mean_color = np.mean(roi, axis=(0, 1))
            
            # Determinar si es oscuro (típico de llantas)
            brightness = np.mean(mean_color)
            
            if brightness < 80:
                return 'negro'
            elif brightness < 120:
                return 'gris_oscuro'
            else:
                return 'claro'
                
        except Exception as e:
            return 'desconocido'
    
    def _validar_semaforo(self, imagen, grupo_circulos, metodo):
        """Validar si un grupo de círculos forma un semáforo."""
        try:
            if len(grupo_circulos) < 2:
                return 0.0
            
            # Verificar alineación vertical
            centers = [(x, y) for x, y, r in grupo_circulos]
            x_coords = [x for x, y in centers]
            y_coords = [y for x, y in centers]
            
            # Calcular desviación estándar en X (debe ser pequeña para semáforos)
            x_std = np.std(x_coords)
            max_radius = max([r for x, y, r in grupo_circulos])
            
            # Puntuación de alineación (menor desviación = mejor)
            alignment_score = max(0, 1 - (x_std / max_radius))
            
            # Verificar espaciado vertical regular
            y_sorted = sorted(y_coords)
            if len(y_sorted) >= 3:
                spacing1 = y_sorted[1] - y_sorted[0]
                spacing2 = y_sorted[2] - y_sorted[1]
                spacing_consistency = 1 - abs(spacing1 - spacing2) / max(spacing1, spacing2, 1)
            else:
                spacing_consistency = 0.8  # Por defecto para 2 luces
            
            # Análisis de colores en las regiones
            color_score = 0.0
            hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
            
            traffic_colors_detected = 0
            for x, y, r in grupo_circulos:
                # Extraer región
                x1, y1 = max(0, x-r), max(0, y-r)
                x2, y2 = min(imagen.shape[1], x+r), min(imagen.shape[0], y+r)
                roi_hsv = hsv[y1:y2, x1:x2]
                
                # Verificar colores de semáforo
                red_mask = cv2.inRange(roi_hsv, (0, 100, 100), (10, 255, 255))
                red_mask2 = cv2.inRange(roi_hsv, (170, 100, 100), (180, 255, 255))
                red_mask = red_mask | red_mask2
                
                yellow_mask = cv2.inRange(roi_hsv, (15, 100, 100), (35, 255, 255))
                green_mask = cv2.inRange(roi_hsv, (40, 100, 100), (80, 255, 255))
                
                total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
                
                red_pct = np.sum(red_mask) / 255 / total_pixels if total_pixels > 0 else 0
                yellow_pct = np.sum(yellow_mask) / 255 / total_pixels if total_pixels > 0 else 0
                green_pct = np.sum(green_mask) / 255 / total_pixels if total_pixels > 0 else 0
                
                # Si hay un color significativo de semáforo
                if max(red_pct, yellow_pct, green_pct) > 0.1:
                    traffic_colors_detected += 1
            
            color_score = traffic_colors_detected / len(grupo_circulos)
            
            # Puntuación final combinada
            confidence = (alignment_score * 0.4 + spacing_consistency * 0.3 + color_score * 0.3)
            return min(confidence, 1.0)
            
        except Exception as e:
            return 0.0
    
    def _agrupar_circulos_verticalmente(self, circles, tolerancia=50):
        """Agrupar círculos que están alineados verticalmente."""
        if len(circles) < 2:
            return []
        
        grupos = []
        circles_usados = set()
        
        for i, (x1, y1, r1) in enumerate(circles):
            if i in circles_usados:
                continue
                
            grupo_actual = [(x1, y1, r1)]
            circles_usados.add(i)
            
            # Buscar círculos cercanos verticalmente
            for j, (x2, y2, r2) in enumerate(circles):
                if j in circles_usados:
                    continue
                
                # Verificar si están alineados verticalmente
                distancia_x = abs(x1 - x2)
                distancia_y = abs(y1 - y2)
                
                if distancia_x <= tolerancia and distancia_y > r1:  # Alineados verticalmente
                    grupo_actual.append((x2, y2, r2))
                    circles_usados.add(j)
            
            # Solo considerar grupos con al menos 2 círculos
            if len(grupo_actual) >= 2:
                # Ordenar por posición Y (de arriba a abajo)
                grupo_actual.sort(key=lambda circle: circle[1])
                grupos.append(grupo_actual)
        
        return grupos
    
    def _calcular_bbox_semaforo(self, grupo_circulos):
        """Calcular bounding box de un grupo de círculos."""
        if not grupo_circulos:
            return (0, 0, 0, 0)
        
        x_coords = [x for x, y, r in grupo_circulos]
        y_coords = [y for x, y, r in grupo_circulos]
        radios = [r for x, y, r in grupo_circulos]
        
        x_min = min(x_coords) - max(radios)
        x_max = max(x_coords) + max(radios)
        y_min = min(y_coords) - max(radios)
        y_max = max(y_coords) + max(radios)
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
    def _analizar_colores_semaforo(self, imagen, grupo_circulos):
        """Analizar qué colores de semáforo están presentes."""
        colores_detectados = {'rojo': False, 'amarillo': False, 'verde': False}
        
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        for x, y, r in grupo_circulos:
            # Extraer región
            x1, y1 = max(0, x-r), max(0, y-r)
            x2, y2 = min(imagen.shape[1], x+r), min(imagen.shape[0], y+r)
            roi_hsv = hsv[y1:y2, x1:x2]
            
            if roi_hsv.size == 0:
                continue
            
            # Verificar cada color
            red_mask = cv2.inRange(roi_hsv, (0, 120, 120), (10, 255, 255))
            red_mask2 = cv2.inRange(roi_hsv, (170, 120, 120), (180, 255, 255))
            red_mask = red_mask | red_mask2
            
            yellow_mask = cv2.inRange(roi_hsv, (15, 150, 150), (35, 255, 255))
            green_mask = cv2.inRange(roi_hsv, (40, 120, 120), (80, 255, 255))
            
            total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
            
            if np.sum(red_mask) / 255 / total_pixels > 0.2:
                colores_detectados['rojo'] = True
            if np.sum(yellow_mask) / 255 / total_pixels > 0.2:
                colores_detectados['amarillo'] = True
            if np.sum(green_mask) / 255 / total_pixels > 0.2:
                colores_detectados['verde'] = True
        
        return colores_detectados
    
    def _crear_mascara_rojo(self, hsv):
        """Crear máscara para color rojo."""
        red_mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        return red_mask1 | red_mask2
    
    def _crear_mascara_amarillo(self, hsv):
        """Crear máscara para color amarillo."""
        return cv2.inRange(hsv, (15, 120, 120), (35, 255, 255))
    
    def _crear_mascara_verde(self, hsv):
        """Crear máscara para color verde."""
        return cv2.inRange(hsv, (40, 50, 50), (90, 255, 255))
    
    
    def _calcular_puntuacion_textura_metalica(self, texture_results):
        """Calcula puntuación basada en texturas típicas de vehículos."""
        # Vehículos suelen tener alta homogeneidad y baja entropía en superficies pintadas
        homogeneidad = texture_results.get('Homogeneidad', 0)
        energia = texture_results.get('Energia', 0)
        contraste = texture_results.get('Contraste', 0)
        
        # Combinar métricas (alta homogeneidad y energía, contraste moderado)
        score = (homogeneidad * 0.5 + energia * 0.3 + (1.0 - min(1.0, contraste/100.0)) * 0.2)
        return min(1.0, score)
    
    def _analizar_gradientes_verticales(self, imagen_gris):
        """Analiza gradientes verticales característicos de peatones."""
        # Filtro Sobel vertical
        sobel_v = sobel(imagen_gris, axis=0)
        
        # Estadísticas de gradientes verticales
        return {
            'mean_vertical_gradient': np.mean(np.abs(sobel_v)),
            'std_vertical_gradient': np.std(sobel_v),
            'max_vertical_gradient': np.max(np.abs(sobel_v))
        }
    
    def _analizar_colores_senales(self, imagen):
        """Analiza colores típicos de señales de tráfico."""
        # Convertir a HSV para mejor análisis de color
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Rangos de colores para señales de tráfico
        color_ranges = {
            'red': [(0, 120, 70), (10, 255, 255)],
            'blue': [(100, 150, 0), (140, 255, 255)],
            'yellow': [(20, 100, 100), (30, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)]
        }
        
        color_analysis = {}
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pixel_count = np.sum(mask > 0)
            color_analysis[f'{color_name}_pixels'] = pixel_count
            color_analysis[f'{color_name}_percentage'] = pixel_count / (imagen.shape[0] * imagen.shape[1])
        
        return color_analysis
    
    def _verificar_caracteristicas_senales_circulares(self, imagen, hough_results, color_analysis):
        """Verifica características específicas de señales circulares."""
        confirmed_signals = []
        
        # Obtener círculos detectados por Hough
        circles_opencv = hough_results.get('opencv_circles')
        if circles_opencv is not None and len(circles_opencv) > 0:
            for circle in circles_opencv[0]:
                x, y, r = circle
                
                # Extraer región circular
                mask = np.zeros(imagen.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                
                # Analizar colores en la región
                region_color_score = self._calcular_puntuacion_color_senal(imagen, mask, color_analysis)
                
                if region_color_score > 0.3:  # Umbral de confianza
                    confirmed_signals.append({
                        'center': (int(x), int(y)),
                        'radius': int(r),
                        'color_score': region_color_score,
                        'type': 'circular_sign'
                    })
        
        return {
            'confirmed_signals': confirmed_signals,
            'confidence': len(confirmed_signals) / max(1, len(circles_opencv[0]) if circles_opencv is not None else 1)
        }
    
    def _calcular_puntuacion_color_senal(self, imagen, mask, color_analysis):
        """Calcula puntuación basada en colores típicos de señales."""
        # Combinar porcentajes de colores típicos de señales
        red_score = color_analysis.get('red_percentage', 0)
        blue_score = color_analysis.get('blue_percentage', 0)
        yellow_score = color_analysis.get('yellow_percentage', 0)
        white_score = color_analysis.get('white_percentage', 0)
        
        # Las señales suelen tener combinaciones específicas de colores
        total_signal_colors = red_score + blue_score + yellow_score + white_score
        return min(1.0, total_signal_colors * 10)  # Amplificar señal débil
    
    def _analizar_textura_asfalto(self, imagen):
        """Analiza textura típica del asfalto."""
        imagen_gris = self._convertir_a_gris(imagen)
        
        # Análisis de textura en la parte inferior de la imagen (donde suele estar la carretera)
        h = imagen_gris.shape[0]
        road_region = imagen_gris[int(0.6 * h):h, :]  # 40% inferior
        
        # Estadísticas de textura del asfalto
        return {
            'road_mean_intensity': np.mean(road_region),
            'road_texture_variance': np.var(road_region),
            'road_texture_uniformity': 1.0 / (1.0 + np.var(road_region))
        }
    

    
    def _detectar_colores_semaforo(self, imagen):
        """Detecta colores específicos de semáforos."""
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Rangos más específicos para luces de semáforo
        semaforo_colors = {
            'red_light': [(0, 150, 150), (10, 255, 255)],
            'yellow_light': [(25, 150, 150), (35, 255, 255)],
            'green_light': [(50, 150, 100), (70, 255, 255)]
        }
        
        color_detection = {}
        for color_name, (lower, upper) in semaforo_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Encontrar contornos circulares de las luces
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            lights = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 2000:  # Tamaño típico de luces de semáforo
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if radius > 5:  # Radio mínimo
                        lights.append({
                            'center': (int(x), int(y)),
                            'radius': int(radius),
                            'area': area
                        })
            
            color_detection[color_name] = lights
        
        return color_detection
    
    def _detectar_estructura_semaforo(self, imagen):
        """Detecta la estructura rectangular típica de semáforos."""
        imagen_gris = self._convertir_a_gris(imagen)
        
        # Detectar rectángulos usando contornos
        edges = canny(imagen_gris, sigma=1.0)
        contours, _ = cv2.findContours(
            (edges * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        traffic_light_structures = []
        for contour in contours:
            # Aproximar contorno a polígono
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Buscar rectángulos (4 vértices)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if 500 < area < 10000:  # Tamaño típico de semáforos
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w  # Semáforos suelen ser más altos que anchos
                    
                    if 1.5 < aspect_ratio < 4.0:
                        traffic_light_structures.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'contour': contour
                        })
        
        return traffic_light_structures
    
    def _verificar_patrones_semaforo(self, imagen, color_detection, structure_detection):
        """Verifica patrones típicos de semáforos."""
        confirmed_lights = []
        
        # Buscar combinación de estructura rectangular + luces de colores
        for structure in structure_detection:
            bbox = structure['bbox']
            x, y, w, h = bbox
            
            # Contar luces dentro de la estructura
            lights_in_structure = 0
            light_types = []
            
            for color_type, lights in color_detection.items():
                for light in lights:
                    lx, ly = light['center']
                    # Verificar si la luz está dentro de la estructura
                    if x <= lx <= x + w and y <= ly <= y + h:
                        lights_in_structure += 1
                        light_types.append(color_type)
            
            # Un semáforo típico debe tener al menos 2-3 luces
            if lights_in_structure >= 2:
                confidence = min(1.0, lights_in_structure / 3.0)
                confirmed_lights.append({
                    'bbox': bbox,
                    'lights_count': lights_in_structure,
                    'light_types': light_types,
                    'confidence': confidence,
                    'structure': structure
                })
        
        avg_confidence = np.mean([light['confidence'] for light in confirmed_lights]) if confirmed_lights else 0
        
        return {
            'confirmed_lights': confirmed_lights,
            'confidence': avg_confidence
        }
    
    def _visualizar_deteccion_senales_circulares(self, imagen, resultados, guardar=True, mostrar=True):
        """Visualiza detección de señales circulares (MEJORADA - SIN CLASIFICACIONES)."""
        num_senales = len(resultados.get('senales_detectadas', []))
        metodo = resultados.get('metodo_usado', 'desconocido')
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # Panel 1-2: Imagen original con detecciones MEJORADA
        ax = fig.add_subplot(gs[0, 0:2])
        ax.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        
        titulo = f'🔴 Señales Detectadas: {num_senales}'
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        
        # Dibujar círculos con MEJOR VISIBILIDAD
        for i, senal in enumerate(resultados['senales_detectadas']):
            cx, cy = senal['centro']
            r = senal['radio']
            conf = senal['confidence']
            
            # Color según confianza
            if conf > 0.6:
                color = 'lime'
                color_borde = 'darkgreen'
            elif conf > 0.4:
                color = 'yellow'
                color_borde = 'orange'
            else:
                color = 'orange'
                color_borde = 'red'
            
            # Círculo doble para mejor visibilidad
            circle_outer = plt.Circle((cx, cy), r+2, color=color_borde, fill=False, linewidth=4)
            circle_inner = plt.Circle((cx, cy), r, color=color, fill=False, linewidth=3)
            ax.add_patch(circle_outer)
            ax.add_patch(circle_inner)
            
            # Cruz en el centro
            ax.plot(cx, cy, 'r+', markersize=12, markeredgewidth=3)
            
            # Etiqueta SIMPLE
            label_text = f'#{i+1}\n{conf:.2f}'
            ax.text(cx, cy - r - 15, label_text,
                   bbox={'boxstyle': 'round,pad=0.5', 'facecolor': color, 'alpha': 0.9, 'edgecolor': color_borde, 'linewidth': 2},
                   fontsize=11, fontweight='bold', color='black', ha='center')
        
        ax.axis('off')
        
        # Panel 3: Preprocesamiento
        ax = fig.add_subplot(gs[0, 2])
        if 'imagen_procesada' in resultados and 'ecualizacion' in resultados['imagen_procesada']:
            ax.imshow(resultados['imagen_procesada']['ecualizacion'], cmap='gray')
            ax.set_title('Ecualización CLAHE', fontsize=11)
        else:
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            ax.imshow(gray, cmap='gray')
            ax.set_title('Escala de Grises', fontsize=11)
        ax.axis('off')
        
        # Panel 4: Bordes
        ax = fig.add_subplot(gs[0, 3])
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        edges = cv2.Canny(blur, 50, 150)
        ax.imshow(edges, cmap='gray')
        ax.set_title('Bordes Canny', fontsize=11)
        ax.axis('off')
        
        # Panel 5: Métricas de confianza
        ax = fig.add_subplot(gs[1, 0])
        if num_senales > 0:
            confianzas = [s['confidence'] for s in resultados['senales_detectadas']]
            indices = range(1, num_senales + 1)
            colors_bar = ['green' if c > 0.6 else 'yellow' if c > 0.4 else 'orange' for c in confianzas]
            
            bars = ax.barh(indices, confianzas, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_yticks(indices)
            ax.set_yticklabels([f'Señal {i}' for i in indices], fontsize=9)
            ax.set_xlabel('Confianza', fontsize=10)
            ax.set_xlim(0, 1.0)
            ax.set_title('Confianza por Señal', fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Valores en barras
            for bar, val in zip(bars, confianzas):
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                       f'{val:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Sin señales\ndetectadas', ha='center', va='center', 
                   fontsize=12, color='red', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.axis('on')
        
        # Paneles 6-8: ROIs de señales (sin clasificación, solo datos técnicos)
        for idx in range(3):
            ax = fig.add_subplot(gs[1, idx+1])
            
            if idx < num_senales:
                senal = resultados['senales_detectadas'][idx]
                x, y, w, h = senal['bbox']
                
                # Extraer ROI
                margin = 10
                y1 = max(0, y - margin)
                y2 = min(imagen.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(imagen.shape[1], x + w + margin)
                
                roi = imagen[y1:y2, x1:x2]
                
                if roi.size > 0:
                    ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    
                    # Info técnica SOLO
                    conf = senal['confidence']
                    r = senal['radio']
                    
                    title_text = f'Señal #{idx+1}\n'
                    title_text += f'Confianza: {conf:.3f}\n'
                    title_text += f'Radio: {r}px'
                    
                    ax.set_title(title_text, fontsize=9, fontweight='bold')
            else:
                ax.text(0.5, 0.5, '-', ha='center', va='center', fontsize=20, color='gray')
            
            ax.axis('off')
        
        # Título general con método usado
        stats = resultados['estadisticas']
        metodo_nombre = stats.get('algoritmo', metodo.upper())
        stats_text = f"Método: {metodo_nombre} | "
        stats_text += f"Confianza promedio: {stats['confianza_promedio']:.3f}"
        
        fig.suptitle(f'Detección de Señales Circulares\n{stats_text}',
                    fontsize=13, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Guardar imagen
        if guardar:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.results_dir, f"senales_circulares_{metodo}_{timestamp}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Imagen guardada: {output_path}")
        
        if mostrar:
            plt.show()
        else:
            plt.close(fig)
    
    def _visualizar_deteccion_semaforos(self, imagen, resultados, guardar=True, mostrar=True):
        """Visualiza detección de semáforos (MULTI-ALGORITMO)."""
        num_semaforos = len(resultados['semaforos_detectados'])
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # Panel 1: Imagen original con detecciones
        ax = fig.add_subplot(gs[0, 0:2])
        ax.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        
        titulo = f'🚦 Semáforos Detectados: {num_semaforos}'
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        
        for idx, semaforo in enumerate(resultados['semaforos_detectados']):
            x, y, w, h = semaforo['bbox']
            conf = semaforo['confidence']
            
            # Color del bounding box según confianza
            if conf > 0.6:
                color = 'green'
            elif conf > 0.4:
                color = 'yellow'
            else:
                color = 'orange'
            
            # Dibujar bounding box del semáforo
            rect = plt.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Dibujar cada luz individual
            for luz_idx, luz in enumerate(semaforo['luces']):
                lx, ly = luz['posicion']
                lr = luz['radio']
                
                # Color de la luz
                luz_color_map = {
                    'rojo': 'red',
                    'amarillo': 'yellow',
                    'verde': 'green'
                }
                luz_color = luz_color_map.get(luz['color']['color'], 'gray')
                
                # Dibujar círculo de la luz
                circle = plt.Circle((lx, ly), lr, color=luz_color, fill=False, linewidth=2, alpha=0.7)
                ax.add_patch(circle)
            
            # Etiqueta
            label_text = f'S{idx+1}: {conf:.2f}'
            ax.text(x, y-10, label_text,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8),
                   fontsize=10, fontweight='bold', color='black')
        
        ax.axis('off')
        
        # Panel 2: Máscaras de colores (Rojo)
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(resultados['imagen_procesada']['red_mask'], cmap='Reds')
        ax.set_title('Máscara Roja', fontsize=11)
        ax.axis('off')
        
        # Panel 3: Máscaras de colores (Amarillo)
        ax = fig.add_subplot(gs[0, 3])
        ax.imshow(resultados['imagen_procesada']['yellow_mask'], cmap='YlOrBr')
        ax.set_title('Máscara Amarilla', fontsize=11)
        ax.axis('off')
        
        # Panel 4: Máscaras de colores (Verde)
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(resultados['imagen_procesada']['green_mask'], cmap='Greens')
        ax.set_title('Máscara Verde', fontsize=11)
        ax.axis('off')
        
        # Panel 5-7: ROIs de hasta 3 semáforos
        for idx in range(3):
            col_idx = idx + 1
            ax = fig.add_subplot(gs[1, col_idx])
            
            if idx < num_semaforos:
                semaforo = resultados['semaforos_detectados'][idx]
                x, y, w, h = semaforo['bbox']
                
                # Extraer ROI
                margin = 5
                y1 = max(0, y - margin)
                y2 = min(imagen.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(imagen.shape[1], x + w + margin)
                
                roi = imagen[y1:y2, x1:x2]
                
                if roi.size > 0:
                    ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    
                    # Información
                    conf = semaforo['confidence']
                    tiene_3 = semaforo['tiene_3_colores']
                    colores = [luz['color']['color'] for luz in semaforo['luces']]
                    
                    title_text = f'Semáforo {idx+1}\n'
                    title_text += f'Conf: {conf:.2f}\n'
                    title_text += f'3 Colores: {"✓" if tiene_3 else "✗"}\n'
                    title_text += f'{"/".join(colores)}'
                    
                    ax.set_title(title_text, fontsize=9)
            else:
                ax.text(0.5, 0.5, '-', ha='center', va='center', fontsize=20, color='gray')
            
            ax.axis('off')
        
        # Título general
        stats = resultados['estadisticas']
        stats_text = f"Algoritmos: {stats.get('algoritmos_usados', 'Multi-método')}\n"
        stats_text += f"Círculos: {stats['circulos_detectados']} → Semáforos: {stats['semaforos_confirmados']}"
        
        fig.suptitle(f'Detección de Semáforos\n{stats_text}',
                    fontsize=13, fontweight='bold', y=0.98)
        
        # Guardar imagen
        if guardar:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.results_dir, f"semaforos_deteccion_{timestamp}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Imagen guardada: {output_path}")
        
        if mostrar:
            plt.show()
        else:
            plt.close(fig)
    
    def analizar_imagen_completa(self, imagen_path, tipos_objetos=None):
        """
        Analiza una imagen buscando todos los tipos de objetos especificados.
        
        Args:
            imagen_path (str): Ruta a la imagen
            tipos_objetos (list): Lista de tipos de objetos a detectar
            
        Returns:
            dict: Resultados completos de detección
        """
        if tipos_objetos is None:
            tipos_objetos = ['vehiculos', 'peatones', 'senales_circulares', 'carriles']
        
        # Cargar imagen
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            print(f"❌ Error al cargar la imagen: {imagen_path}")
            return None
        
        print(f"\n🔍 Analizando imagen: {os.path.basename(imagen_path)}")
        print(f"📋 Objetos a detectar: {', '.join(tipos_objetos)}")
        
        resultados_completos = {
            'imagen_path': imagen_path,
            'imagen_shape': imagen.shape,
            'timestamp': datetime.now().isoformat(),
            'detecciones': {}
        }
        
        # Ejecutar detecciones según los tipos especificados
        if 'vehiculos' in tipos_objetos:
            resultados_completos['detecciones']['vehiculos'] = self.detectar_vehiculos(imagen, visualizar=False)
        
        if 'peatones' in tipos_objetos:
            resultados_completos['detecciones']['peatones'] = self.detectar_peatones(imagen, visualizar=False)
        
        if 'senales_circulares' in tipos_objetos:
            resultados_completos['detecciones']['senales_circulares'] = self.detectar_senales_circulares(imagen, visualizar=False)
        
        if 'carriles' in tipos_objetos:
            resultados_completos['detecciones']['carriles'] = self.detectar_carriles(imagen, visualizar=False)
        
        if 'motocicletas' in tipos_objetos:
            resultados_completos['detecciones']['motocicletas'] = self.detectar_motocicletas(imagen, visualizar=False)
        
        if 'semaforos' in tipos_objetos:
            resultados_completos['detecciones']['semaforos'] = self.detectar_semaforos(imagen, visualizar=False)
        
        # Guardar resultados
        self.detection_results.append(resultados_completos)
        
        # Mostrar resumen
        self._mostrar_resumen_deteccion(resultados_completos)
        
        return resultados_completos
    
    def _mostrar_resumen_deteccion(self, resultados):
        """Muestra un resumen de los resultados de detección."""
        print("\n" + "="*60)
        print("📊 RESUMEN DE DETECCIÓN")
        print("="*60)
        
        for tipo_objeto, deteccion in resultados['detecciones'].items():
            print(f"\n🎯 {tipo_objeto.upper().replace('_', ' ')}")
            print("-" * 30)
            
            if tipo_objeto == 'vehiculos':
                num_candidatos = deteccion.get('num_candidatos', 0)
                confianza = deteccion.get('confianza_promedio', 0)
                print(f"   Candidatos detectados: {num_candidatos}")
                print(f"   Confianza promedio: {confianza:.2f}")
            
            elif tipo_objeto == 'peatones':
                num_candidatos = deteccion.get('num_candidatos', 0)
                print(f"   Contornos humanos: {num_candidatos}")
            
            elif tipo_objeto == 'senales_circulares':
                circulos = deteccion.get('circulos_detectados', 0)
                confirmadas = len(deteccion.get('senales_confirmadas', []))
                print(f"   Círculos detectados: {circulos}")
                print(f"   Señales confirmadas: {confirmadas}")
            
            elif tipo_objeto == 'carriles':
                lineas = deteccion.get('lineas_detectadas', 0)
                carriles = deteccion.get('carriles_identificados', 0)
                marcas = len(deteccion.get('marcas_viales', []))
                print(f"   Líneas detectadas: {lineas}")
                print(f"   Carriles identificados: {carriles}")
                print(f"   Marcas viales: {marcas}")
            
            elif tipo_objeto == 'motocicletas':
                candidatos = deteccion.get('candidatos_detectados', 0)
                ruedas = len(deteccion.get('partes_detectadas', {}).get('wheels', []))
                print(f"   Candidatos detectados: {candidatos}")
                print(f"   Ruedas detectadas: {ruedas}")
            
            elif tipo_objeto == 'semaforos':
                confirmados = len(deteccion.get('semaforos_confirmados', []))
                confianza = deteccion.get('confianza', 0)
                print(f"   Semáforos detectados: {confirmados}")
                print(f"   Confianza: {confianza:.2f}")
        
        print("\n" + "="*60)
    
    
    def _non_maximum_suppression(self, candidatos, overlap_threshold=0.3):
        """Elimina detecciones duplicadas usando Non-Maximum Suppression."""
        if len(candidatos) == 0:
            return []
        
        boxes = np.array([c['bbox'] for c in candidatos])
        scores = np.array([c['confidence'] for c in candidatos])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = boxes[:, 2] * boxes[:, 3]
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= overlap_threshold)[0]
            order = order[inds + 1]
        
        return [candidatos[i] for i in keep]
    
    
    def _visualizar_deteccion_llantas(self, imagen, resultados, guardar=True, mostrar=True):
        """Visualiza detección de llantas/neumáticos (MULTI-ALGORITMO)."""
        num_llantas = len(resultados['llantas_detectadas'])
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Panel 1: Imagen original con detecciones
        ax = fig.add_subplot(gs[0, 0:2])
        ax.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        
        titulo = f'🔘 Llantas Detectadas: {num_llantas}'
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        
        for idx, llanta in enumerate(resultados['llantas_detectadas']):
            x, y, w, h = llanta['bbox']
            conf = llanta['confidence']
            
            # Color según confianza
            if conf > 0.6:
                color = 'lime'
            elif conf > 0.4:
                color = 'yellow'
            else:
                color = 'orange'
            
            # Dibujar círculo
            cx, cy = llanta['centro']
            r = llanta['radio']
            circle = plt.Circle((cx, cy), r, color=color, fill=False, linewidth=3)
            ax.add_patch(circle)
            
            # Cruz en el centro
            ax.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
            
            # Etiqueta
            label_text = f'L{idx+1}: {conf:.2f}'
            ax.text(x, y-10, label_text,
                   bbox={'boxstyle': 'round,pad=0.5', 'facecolor': color, 'alpha': 0.8},
                   fontsize=10, fontweight='bold', color='black')
        
        ax.axis('off')
        
        # Panel 2: Máscara de colores oscuros
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(resultados['imagen_procesada']['dark_mask'], cmap='gray')
        ax.set_title('Máscara Oscura\n(Llantas negras/grises)', fontsize=11)
        ax.axis('off')
        
        # Panel 3: Gradientes Sobel
        ax = fig.add_subplot(gs[0, 3])
        ax.imshow(resultados['imagen_procesada']['sobel_magnitude'], cmap='hot')
        ax.set_title('Magnitud Gradientes\n(Sobel)', fontsize=11)
        ax.axis('off')
        
        # Panel 4: Ecualización
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(resultados['imagen_procesada']['ecualizacion'], cmap='gray')
        ax.set_title('Ecualización CLAHE', fontsize=11)
        ax.axis('off')
        
        # Panel 5: Gráfico de métricas promedio
        ax = fig.add_subplot(gs[1, 1])
        if num_llantas > 0:
            metricas_promedio = {
                'Color\nOscuro': np.mean([l['metricas']['color_score'] for l in resultados['llantas_detectadas']]),
                'Textura\nGLCM': np.mean([l['metricas']['texture_score'] for l in resultados['llantas_detectadas']]),
                'Gradientes': np.mean([l['metricas']['gradient_score'] for l in resultados['llantas_detectadas']]),
                'Keypoints': np.mean([l['metricas']['orb_density'] + l['metricas']['akaze_density'] for l in resultados['llantas_detectadas']]) / 2
            }
            
            colors_bar = ['black', 'brown', 'blue', 'green']
            bars = ax.barh(range(len(metricas_promedio)), list(metricas_promedio.values()), 
                          color=colors_bar, alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(metricas_promedio)))
            ax.set_yticklabels(list(metricas_promedio.keys()), fontsize=9)
            ax.set_xlabel('Score', fontsize=9)
            ax.set_xlim(0, 1.0)
            ax.set_title('Métricas Promedio', fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Valores en las barras
            for bar, val in zip(bars, metricas_promedio.values()):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{val:.2f}', ha='left', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Sin llantas\ndetectadas', ha='center', va='center', 
                   fontsize=12, color='red')
        ax.axis('on')
        
        # Panel 6: Distribución de radios
        ax = fig.add_subplot(gs[1, 2])
        if num_llantas > 0:
            radios = [l['radio'] for l in resultados['llantas_detectadas']]
            ax.hist(radios, bins=min(10, num_llantas), color='steelblue', 
                   alpha=0.7, edgecolor='black')
            ax.set_xlabel('Radio (px)', fontsize=9)
            ax.set_ylabel('Frecuencia', fontsize=9)
            ax.set_title('Distribución de Tamaños', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, '-', ha='center', va='center', fontsize=20, color='gray')
        
        # Panel 7: Distribución de confianza
        ax = fig.add_subplot(gs[1, 3])
        if num_llantas > 0:
            confianzas = [l['confidence'] for l in resultados['llantas_detectadas']]
            colors_scatter = ['green' if c > 0.6 else 'yellow' if c > 0.4 else 'orange' 
                            for c in confianzas]
            ax.scatter(range(num_llantas), confianzas, c=colors_scatter, 
                      s=100, alpha=0.7, edgecolors='black', linewidths=2)
            ax.set_xlabel('Índice Llanta', fontsize=9)
            ax.set_ylabel('Confianza', fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.set_title('Confianza por Llanta', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Alta')
            ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Media')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, '-', ha='center', va='center', fontsize=20, color='gray')
        
        # Paneles 8-11: ROIs de hasta 4 llantas
        for idx in range(4):
            ax = fig.add_subplot(gs[2, idx])
            
            if idx < num_llantas:
                llanta = resultados['llantas_detectadas'][idx]
                x, y, w, h = llanta['bbox']
                
                # Extraer ROI con margen
                margin = 10
                y1 = max(0, y - margin)
                y2 = min(imagen.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(imagen.shape[1], x + w + margin)
                
                roi = imagen[y1:y2, x1:x2]
                
                if roi.size > 0:
                    ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    
                    # Información detallada
                    conf = llanta['confidence']
                    r = llanta['radio']
                    dark_pct = llanta['metricas']['dark_percentage']
                    texture = llanta['metricas']['texture_score']
                    
                    title_text = f'Llanta {idx+1}\n'
                    title_text += f'Conf: {conf:.2f}\n'
                    title_text += f'R={r}px | Dark={dark_pct:.1%}\n'
                    title_text += f'Tex={texture:.2f}'
                    
                    ax.set_title(title_text, fontsize=9)
            else:
                ax.text(0.5, 0.5, '-', ha='center', va='center', fontsize=20, color='gray')
            
            ax.axis('off')
        
        # Título general
        stats = resultados['estadisticas']
        stats_text = f"Algoritmos: {stats.get('algoritmos_usados', 'Multi-método')}\n"
        stats_text += f"Círculos: {stats['circulos_hough']} → Llantas: {stats['llantas_confirmadas']} | "
        stats_text += f"Radio promedio: {stats['radio_promedio']:.1f}px"
        
        fig.suptitle(f'Detección de Llantas/Neumáticos\n{stats_text}',
                    fontsize=13, fontweight='bold', y=0.98)
        
        # Guardar imagen
        if guardar:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.results_dir, f"llantas_deteccion_{timestamp}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Imagen guardada: {output_path}")
        
        if mostrar:
            plt.show()
        else:
            plt.close(fig)
    
    def guardar_resultados_deteccion(self, formato='json'):
        """Guarda los resultados de detección en el formato especificado."""
        if not self.detection_results:
            print("❌ No hay resultados de detección para guardar.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if formato == 'json':
            filename = f"object_detection_results_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.detection_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ Resultados guardados en: {filepath}")
        
        elif formato == 'csv':
            filename = f"object_detection_summary_{timestamp}.csv"
            filepath = os.path.join(self.results_dir, filename)
            
            # Crear resumen para CSV
            summary_data = []
            for resultado in self.detection_results:
                row = {
                    'imagen': os.path.basename(resultado['imagen_path']),
                    'timestamp': resultado['timestamp']
                }
                
                for tipo_objeto, deteccion in resultado['detecciones'].items():
                    if tipo_objeto == 'vehiculos':
                        row[f'{tipo_objeto}_candidatos'] = deteccion.get('num_candidatos', 0)
                        row[f'{tipo_objeto}_confianza'] = deteccion.get('confianza_promedio', 0)
                    elif tipo_objeto == 'carriles':
                        row[f'{tipo_objeto}_lineas'] = deteccion.get('lineas_detectadas', 0)
                        row[f'{tipo_objeto}_identificados'] = deteccion.get('carriles_identificados', 0)
                    # Agregar más métricas según necesidad
                
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            df.to_csv(filepath, index=False)
            print(f"✅ Resumen guardado en: {filepath}")


def main_object_detection():
    """Función principal para prueba del módulo."""
    detector = ObjectDetectionSystem()
    
    # Ejemplo de uso
    imagen_path = "./images/cameraman.tif"  # Cambiar por ruta real
    if os.path.exists(imagen_path):
        resultados = detector.analizar_imagen_completa(
            imagen_path,
            tipos_objetos=['vehiculos', 'peatones', 'senales_circulares']
        )
        detector.guardar_resultados_deteccion('json')
    else:
        print("❌ Imagen de ejemplo no encontrada")

if __name__ == "__main__":
    main_object_detection()