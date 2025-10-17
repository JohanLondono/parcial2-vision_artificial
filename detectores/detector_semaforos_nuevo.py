#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de Semáforos en Imágenes de Tráfico Vehicular - Versión Robusta
=========================================================================

Sistema especializado para detectar semáforos utilizando configuraciones
robustas que combinan múltiples algoritmos de visión por computadora.

Configuraciones disponibles:
- CONFIG_PRECISION_ALTA: Color HSV + Estructura + Morfología + Validación
- CONFIG_ROBUSTA: Contornos + Color + GrabCut + Validación geométrica
- CONFIG_ADAPTATIVA: Color multirrango + Textura + Hough + AKAZE
- CONFIG_BALANCED: Combinación equilibrada (recomendada)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops

class DetectorSemaforos:
    """Detector especializado de semáforos con configuraciones robustas combinadas."""
    
    def __init__(self):
        """Inicializar el detector de semáforos con configuraciones combinadas."""
        
        self.CONFIG_PRECISION_ALTA = {
            'nombre': 'Precision Alta',
            'descripcion': 'Análisis color HSV + Detección estructura + Validación morfológica',
            'algoritmos': ['color_hsv', 'estructura_vertical', 'morfologia', 'validacion_geometrica'],
            'pesos': {'color': 0.40, 'estructura': 0.30, 'morfologia': 0.20, 'geometria': 0.10},
            'umbral_validacion': 0.65,
            'color_ranges': {
                'rojo_bajo1': [0, 120, 70], 'rojo_alto1': [10, 255, 255],
                'rojo_bajo2': [170, 120, 70], 'rojo_alto2': [180, 255, 255],
                'amarillo_bajo': [15, 120, 70], 'amarillo_alto': [35, 255, 255],
                'verde_bajo': [40, 120, 70], 'verde_alto': [80, 255, 255]
            },
            'hough_params': {
                'dp': 1, 'min_dist': 25, 'param1': 50, 'param2': 20,
                'min_radius': 8, 'max_radius': 45
            },
            'agrupacion': {
                'tolerancia_vertical': 80, 'tolerancia_horizontal': 25,
                'min_circulos': 2, 'max_circulos': 4
            }
        }
        
        self.CONFIG_ROBUSTA = {
            'nombre': 'Robusta',
            'descripcion': 'Contornos rectangulares + Color + GrabCut + Validación geométrica',
            'algoritmos': ['contornos_rectangulares', 'color_hsv', 'grabcut', 'validacion_geometrica'],
            'pesos': {'contornos': 0.35, 'color': 0.30, 'grabcut': 0.25, 'geometria': 0.10},
            'umbral_validacion': 0.60,
            'color_ranges': {
                'rojo_bajo1': [0, 100, 60], 'rojo_alto1': [10, 255, 255],
                'rojo_bajo2': [170, 100, 60], 'rojo_alto2': [180, 255, 255],
                'amarillo_bajo': [15, 100, 80], 'amarillo_alto': [35, 255, 255],
                'verde_bajo': [40, 80, 60], 'verde_alto': [80, 255, 255]
            },
            'contornos': {
                'min_area': 300, 'max_area': 8000,
                'aspect_ratio_min': 0.3, 'aspect_ratio_max': 3.0,
                'metodo': cv2.RETR_EXTERNAL
            },
            'grabcut': {
                'iteraciones': 3, 'modo': cv2.GC_INIT_WITH_RECT
            }
        }
        
        self.CONFIG_ADAPTATIVA = {
            'nombre': 'Adaptativa',
            'descripcion': 'Color multirrango + Análisis textural + Hough adaptativo + AKAZE',
            'algoritmos': ['color_multirrango', 'textura', 'hough_adaptativo', 'akaze'],
            'pesos': {'color': 0.30, 'textura': 0.25, 'hough': 0.25, 'keypoints': 0.20},
            'umbral_validacion': 0.55,
            'color_ranges': {
                'rangos_rojos': [
                    {'bajo': [0, 100, 50], 'alto': [10, 255, 255]},
                    {'bajo': [170, 100, 50], 'alto': [180, 255, 255]}
                ],
                'rangos_amarillos': [
                    {'bajo': [15, 100, 100], 'alto': [35, 255, 255]}
                ],
                'rangos_verdes': [
                    {'bajo': [40, 60, 60], 'alto': [80, 255, 255]}
                ]
            },
            'textura': {
                'distancias': [1, 2], 'angulos': [0, np.pi/4],
                'contraste_min': 0.2, 'energia_max': 0.4
            },
            'hough_params': {
                'dp': 1.2, 'min_dist': 20, 'param1': 40, 'param2': 25,
                'min_radius': 6, 'max_radius': 50
            },
            'akaze': {
                'threshold': 0.005, 'nOctaves': 3, 'nOctaveLayers': 3
            }
        }
        
        self.CONFIG_BALANCED = {
            'nombre': 'Equilibrada',
            'descripcion': 'Combinación equilibrada de color + estructura + geometría (recomendada)',
            'algoritmos': ['color_hsv', 'estructura_vertical', 'validacion_geometrica', 'consistencia'],
            'pesos': {'color': 0.35, 'estructura': 0.30, 'geometria': 0.20, 'consistencia': 0.15},
            'umbral_validacion': 0.58,
            'color_ranges': {
                'rojo_bajo1': [0, 110, 60], 'rojo_alto1': [10, 255, 255],
                'rojo_bajo2': [170, 110, 60], 'rojo_alto2': [180, 255, 255],
                'amarillo_bajo': [15, 110, 80], 'amarillo_alto': [35, 255, 255],
                'verde_bajo': [40, 70, 60], 'verde_alto': [80, 255, 255]
            },
            'hough_params': {
                'dp': 1, 'min_dist': 25, 'param1': 45, 'param2': 22,
                'min_radius': 8, 'max_radius': 45
            },
            'agrupacion': {
                'tolerancia_vertical': 90, 'tolerancia_horizontal': 30,
                'min_circulos': 2, 'max_circulos': 4
            },
            'geometria': {
                'ratio_min': 0.7, 'ratio_max': 1.4
            }
        }
    
    def detectar_semaforos(self, imagen, configuracion='CONFIG_BALANCED', visualizar=True, guardar=False, ruta_salida=None):
        """
        Detecta semáforos en la imagen usando la configuración especificada.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            configuracion (str): Configuración a usar ('CONFIG_PRECISION_ALTA', 'CONFIG_ROBUSTA', 
                               'CONFIG_ADAPTATIVA', 'CONFIG_BALANCED')
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imagen resultado
            ruta_salida (str): Ruta donde guardar resultado
            
        Returns:
            dict: Resultados de la detección
        """
        print(f"Detectando semáforos con configuración: {configuracion}")
        
        if not hasattr(self, configuracion):
            print(f"Configuración no reconocida: {configuracion}. Usando CONFIG_BALANCED por defecto")
            configuracion = 'CONFIG_BALANCED'
        
        config = getattr(self, configuracion)
        
        imagen_preprocesada = self._preprocesar_imagen(imagen, config)
        
        candidatos = []
        
        if 'color_hsv' in config['algoritmos']:
            candidatos.extend(self._detectar_color_hsv(imagen_preprocesada, config))
        
        if 'color_multirrango' in config['algoritmos']:
            candidatos.extend(self._detectar_color_multirrango(imagen_preprocesada, config))
        
        if 'estructura_vertical' in config['algoritmos']:
            candidatos.extend(self._detectar_estructura_vertical(imagen_preprocesada, config))
        
        if 'contornos_rectangulares' in config['algoritmos']:
            candidatos.extend(self._detectar_contornos_rectangulares(imagen_preprocesada, config))
        
        if 'hough_adaptativo' in config['algoritmos']:
            candidatos.extend(self._detectar_hough_adaptativo(imagen_preprocesada, config))
        
        semaforos_validados = self._validar_candidatos(imagen, candidatos, config)
        
        semaforos_finales = self._aplicar_nms_semaforos(semaforos_validados, umbral_iou=0.3)
        
        imagen_resultado = self._dibujar_resultados(imagen.copy(), semaforos_finales, config['nombre'])
        
        resultado = {
            'configuracion': config['nombre'],
            'num_semaforos': len(semaforos_finales),
            'semaforos_detectados': semaforos_finales,
            'candidatos_iniciales': len(candidatos),
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': np.mean([semaforo[5] for semaforo in semaforos_finales]) if semaforos_finales else 0.0
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, f"Detección de Semáforos - {config['nombre']}")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, configuracion.lower())
        
        return resultado
    
    def _preprocesar_imagen(self, imagen, config):
        """Preprocesa la imagen según la configuración."""
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        gris = cv2.GaussianBlur(gris, (5, 5), 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gris = clahe.apply(gris)
        
        return {'gris': gris, 'hsv': hsv, 'original': imagen}
    
    def _detectar_color_hsv(self, imagen_prep, config):
        """Detecta candidatos usando análisis de color HSV."""
        candidatos = []
        hsv = imagen_prep['hsv']
        
        colors = config['color_ranges']
        
        # Máscara roja (dos rangos)
        mask_rojo1 = cv2.inRange(hsv, np.array(colors['rojo_bajo1']), np.array(colors['rojo_alto1']))
        mask_rojo2 = cv2.inRange(hsv, np.array(colors['rojo_bajo2']), np.array(colors['rojo_alto2']))
        mask_rojo = cv2.bitwise_or(mask_rojo1, mask_rojo2)
        
        # Máscara amarilla
        mask_amarillo = cv2.inRange(hsv, np.array(colors['amarillo_bajo']), np.array(colors['amarillo_alto']))
        
        # Máscara verde
        mask_verde = cv2.inRange(hsv, np.array(colors['verde_bajo']), np.array(colors['verde_alto']))
        
        for mask, color_name in [(mask_rojo, 'rojo'), (mask_amarillo, 'amarillo'), (mask_verde, 'verde')]:
            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if 100 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contorno)
                    centro_x, centro_y = x + w//2, y + h//2
                    
                    candidatos.append({
                        'x': centro_x, 'y': centro_y, 'w': w, 'h': h,
                        'area': area, 'color': color_name,
                        'metodo': 'color_hsv', 'confianza_base': 0.7
                    })
        
        return candidatos
    
    def _detectar_color_multirrango(self, imagen_prep, config):
        """Detecta candidatos usando múltiples rangos de color."""
        candidatos = []
        hsv = imagen_prep['hsv']
        
        ranges = config['color_ranges']
        
        # Procesar rangos rojos
        mask_rojo_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rango in ranges['rangos_rojos']:
            mask = cv2.inRange(hsv, np.array(rango['bajo']), np.array(rango['alto']))
            mask_rojo_total = cv2.bitwise_or(mask_rojo_total, mask)
        
        # Procesar rangos amarillos
        mask_amarillo_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rango in ranges['rangos_amarillos']:
            mask = cv2.inRange(hsv, np.array(rango['bajo']), np.array(rango['alto']))
            mask_amarillo_total = cv2.bitwise_or(mask_amarillo_total, mask)
        
        # Procesar rangos verdes
        mask_verde_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rango in ranges['rangos_verdes']:
            mask = cv2.inRange(hsv, np.array(rango['bajo']), np.array(rango['alto']))
            mask_verde_total = cv2.bitwise_or(mask_verde_total, mask)
        
        for mask, color_name in [(mask_rojo_total, 'rojo'), (mask_amarillo_total, 'amarillo'), (mask_verde_total, 'verde')]:
            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if 80 < area < 6000:
                    x, y, w, h = cv2.boundingRect(contorno)
                    centro_x, centro_y = x + w//2, y + h//2
                    
                    candidatos.append({
                        'x': centro_x, 'y': centro_y, 'w': w, 'h': h,
                        'area': area, 'color': color_name,
                        'metodo': 'color_multirrango', 'confianza_base': 0.8
                    })
        
        return candidatos
    
    def _detectar_estructura_vertical(self, imagen_prep, config):
        """Detecta estructuras verticales típicas de semáforos."""
        candidatos = []
        gris = imagen_prep['gris']
        
        # Detectar bordes
        bordes = cv2.Canny(gris, 50, 150)
        
        # Operaciones morfológicas para conectar elementos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)
        
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if 200 < area < 10000:
                x, y, w, h = cv2.boundingRect(contorno)
                aspect_ratio = h / w if w > 0 else 0
                
                # Buscar estructuras alargadas verticalmente (típico de semáforos)
                if 1.5 < aspect_ratio < 5.0:
                    centro_x, centro_y = x + w//2, y + h//2
                    
                    candidatos.append({
                        'x': centro_x, 'y': centro_y, 'w': w, 'h': h,
                        'area': area, 'aspect_ratio': aspect_ratio,
                        'metodo': 'estructura_vertical', 'confianza_base': 0.6
                    })
        
        return candidatos
    
    def _detectar_contornos_rectangulares(self, imagen_prep, config):
        """Detecta contornos rectangulares característicos de semáforos."""
        candidatos = []
        gris = imagen_prep['gris']
        
        # Umbralización adaptativa
        umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        contornos, _ = cv2.findContours(umbral, config['contornos']['metodo'], cv2.CHAIN_APPROX_SIMPLE)
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if config['contornos']['min_area'] < area < config['contornos']['max_area']:
                x, y, w, h = cv2.boundingRect(contorno)
                aspect_ratio = h / w if w > 0 else 0
                
                if config['contornos']['aspect_ratio_min'] < aspect_ratio < config['contornos']['aspect_ratio_max']:
                    centro_x, centro_y = x + w//2, y + h//2
                    
                    candidatos.append({
                        'x': centro_x, 'y': centro_y, 'w': w, 'h': h,
                        'area': area, 'aspect_ratio': aspect_ratio,
                        'metodo': 'contornos_rectangulares', 'confianza_base': 0.65
                    })
        
        return candidatos
    
    def _detectar_hough_adaptativo(self, imagen_prep, config):
        """Detecta círculos usando Hough adaptativo."""
        candidatos = []
        gris = imagen_prep['gris']
        
        params = config['hough_params']
        
        circulos = cv2.HoughCircles(
            gris,
            cv2.HOUGH_GRADIENT,
            dp=params['dp'],
            minDist=params['min_dist'],
            param1=params['param1'],
            param2=params['param2'],
            minRadius=params['min_radius'],
            maxRadius=params['max_radius']
        )
        
        if circulos is not None:
            circulos = np.round(circulos[0, :]).astype("int")
            for (x, y, r) in circulos:
                candidatos.append({
                    'x': x, 'y': y, 'w': r*2, 'h': r*2,
                    'radio': r, 'metodo': 'hough_adaptativo',
                    'confianza_base': 0.75
                })
        
        return candidatos
    
    def _validar_candidatos(self, imagen, candidatos, config):
        """Valida candidatos usando múltiples criterios."""
        semaforos_validados = []
        
        for candidato in candidatos:
            scores = {}
            
            # Score de color
            if 'color' in config['pesos']:
                scores['color'] = self._evaluar_color(imagen, candidato, config)
            
            # Score de estructura
            if 'estructura' in config['pesos']:
                scores['estructura'] = self._evaluar_estructura(imagen, candidato, config)
            
            # Score de geometría
            if 'geometria' in config['pesos']:
                scores['geometria'] = self._evaluar_geometria(candidato, config)
            
            # Score de morfología
            if 'morfologia' in config['pesos']:
                scores['morfologia'] = self._evaluar_morfologia(imagen, candidato)
            
            # Score de textura
            if 'textura' in config['pesos']:
                scores['textura'] = self._evaluar_textura(imagen, candidato, config)
            
            # Score de keypoints (AKAZE)
            if 'keypoints' in config['pesos']:
                scores['keypoints'] = self._evaluar_keypoints(imagen, candidato, config)
            
            # Score de GrabCut
            if 'grabcut' in config['pesos']:
                scores['grabcut'] = self._evaluar_grabcut(imagen, candidato, config)
            
            # Score de consistencia
            if 'consistencia' in config['pesos']:
                scores['consistencia'] = self._evaluar_consistencia(candidato)
            
            # Calcular score ponderado
            score_total = 0
            for criterio, peso in config['pesos'].items():
                if criterio in scores:
                    score_total += scores[criterio] * peso
            
            # Validar umbral
            if score_total >= config['umbral_validacion']:
                semaforo = [
                    candidato['x'], candidato['y'], candidato['w'], candidato['h'],
                    candidato.get('color', 'desconocido'), score_total
                ]
                semaforos_validados.append(semaforo)
        
        return semaforos_validados
    
    def _evaluar_color(self, imagen, candidato, config):
        """Evalúa la presencia de colores típicos de semáforo."""
        try:
            x, y, w, h = candidato['x'], candidato['y'], candidato['w'], candidato['h']
            
            x1, y1 = max(0, x - w//2), max(0, y - h//2)
            x2, y2 = min(imagen.shape[1], x + w//2), min(imagen.shape[0], y + h//2)
            
            roi = imagen[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0
            
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Verificar colores de semáforo
            colors = config.get('color_ranges', {})
            score = 0.0
            
            if 'rojo_bajo1' in colors:
                mask_rojo1 = cv2.inRange(hsv_roi, np.array(colors['rojo_bajo1']), np.array(colors['rojo_alto1']))
                mask_rojo2 = cv2.inRange(hsv_roi, np.array(colors['rojo_bajo2']), np.array(colors['rojo_alto2']))
                mask_rojo = cv2.bitwise_or(mask_rojo1, mask_rojo2)
                score += cv2.countNonZero(mask_rojo) / (roi.shape[0] * roi.shape[1])
                
                mask_amarillo = cv2.inRange(hsv_roi, np.array(colors['amarillo_bajo']), np.array(colors['amarillo_alto']))
                score += cv2.countNonZero(mask_amarillo) / (roi.shape[0] * roi.shape[1])
                
                mask_verde = cv2.inRange(hsv_roi, np.array(colors['verde_bajo']), np.array(colors['verde_alto']))
                score += cv2.countNonZero(mask_verde) / (roi.shape[0] * roi.shape[1])
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _evaluar_estructura(self, imagen, candidato, config):
        """Evalúa la estructura típica de semáforo."""
        try:
            aspect_ratio = candidato.get('aspect_ratio', candidato['h'] / candidato['w'] if candidato['w'] > 0 else 0)
            
            # Preferir estructuras verticales
            if 1.5 < aspect_ratio < 4.0:
                return 0.8
            elif 1.0 < aspect_ratio < 1.5:
                return 0.6
            else:
                return 0.3
        except:
            return 0.0
    
    def _evaluar_geometria(self, candidato, config):
        """Evalúa propiedades geométricas."""
        try:
            geometria = config.get('geometria', {})
            aspect_ratio = candidato['h'] / candidato['w'] if candidato['w'] > 0 else 0
            
            ratio_min = geometria.get('ratio_min', 0.5)
            ratio_max = geometria.get('ratio_max', 2.0)
            
            if ratio_min <= aspect_ratio <= ratio_max:
                return 0.8
            else:
                return 0.3
        except:
            return 0.0
    
    def _evaluar_morfologia(self, imagen, candidato):
        """Evalúa características morfológicas."""
        try:
            x, y, w, h = candidato['x'], candidato['y'], candidato['w'], candidato['h']
            
            x1, y1 = max(0, x - w//2), max(0, y - h//2)
            x2, y2 = min(imagen.shape[1], x + w//2), min(imagen.shape[0], y + h//2)
            
            roi = imagen[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0
            
            gris_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Operaciones morfológicas
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(gris_roi, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            
            # Evaluar uniformidad después de morfología
            uniformidad = 1.0 - (np.std(closing) / 255.0)
            return max(0.0, min(1.0, uniformidad))
        except:
            return 0.0
    
    def _evaluar_textura(self, imagen, candidato, config):
        """Evalúa características texturales usando GLCM."""
        try:
            x, y, w, h = candidato['x'], candidato['y'], candidato['w'], candidato['h']
            
            x1, y1 = max(0, x - w//2), max(0, y - h//2)
            x2, y2 = min(imagen.shape[1], x + w//2), min(imagen.shape[0], y + h//2)
            
            roi = imagen[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0
            
            gris_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Redimensionar si es muy pequeño
            if gris_roi.shape[0] < 8 or gris_roi.shape[1] < 8:
                gris_roi = cv2.resize(gris_roi, (16, 16))
            
            # GLCM
            distancias = config['textura']['distancias']
            angulos = config['textura']['angulos']
            
            glcm = graycomatrix(gris_roi, distancias, angulos, 256, symmetric=True, normed=True)
            
            contraste = graycoprops(glcm, 'contrast')[0, 0]
            energia = graycoprops(glcm, 'energy')[0, 0]
            
            # Evaluar según umbrales
            score = 0.0
            if contraste >= config['textura']['contraste_min']:
                score += 0.5
            if energia <= config['textura']['energia_max']:
                score += 0.5
            
            return score
        except:
            return 0.0
    
    def _evaluar_keypoints(self, imagen, candidato, config):
        """Evalúa usando detectores de puntos clave AKAZE."""
        try:
            x, y, w, h = candidato['x'], candidato['y'], candidato['w'], candidato['h']
            
            x1, y1 = max(0, x - w//2), max(0, y - h//2)
            x2, y2 = min(imagen.shape[1], x + w//2), min(imagen.shape[0], y + h//2)
            
            roi = imagen[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0
            
            gris_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # AKAZE
            akaze_params = config['akaze']
            akaze = cv2.AKAZE_create(
                threshold=akaze_params['threshold'],
                nOctaves=akaze_params['nOctaves'],
                nOctaveLayers=akaze_params['nOctaveLayers']
            )
            
            keypoints = akaze.detect(gris_roi, None)
            
            # Score basado en número de keypoints
            num_keypoints = len(keypoints)
            if num_keypoints > 10:
                return 0.8
            elif num_keypoints > 5:
                return 0.6
            elif num_keypoints > 2:
                return 0.4
            else:
                return 0.2
        except:
            return 0.0
    
    def _evaluar_grabcut(self, imagen, candidato, config):
        """Evalúa usando segmentación GrabCut."""
        try:
            x, y, w, h = candidato['x'], candidato['y'], candidato['w'], candidato['h']
            
            x1, y1 = max(0, x - w//2), max(0, y - h//2)
            x2, y2 = min(imagen.shape[1], x + w//2), min(imagen.shape[0], y + h//2)
            
            if x2 - x1 < 10 or y2 - y1 < 10:
                return 0.0
            
            roi = imagen[y1:y2, x1:x2].copy()
            
            # Inicializar máscara
            mask = np.zeros(roi.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Rectángulo para GrabCut (interior de la ROI)
            rect = (2, 2, roi.shape[1]-4, roi.shape[0]-4)
            
            cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, 
                       config['grabcut']['iteraciones'], config['grabcut']['modo'])
            
            # Evaluar calidad de segmentación
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Porcentaje de primer plano
            fg_ratio = np.sum(mask2) / mask2.size
            
            # Score basado en ratio de primer plano
            if 0.2 <= fg_ratio <= 0.8:
                return 0.8
            elif 0.1 <= fg_ratio <= 0.9:
                return 0.6
            else:
                return 0.3
        except:
            return 0.0
    
    def _evaluar_consistencia(self, candidato):
        """Evalúa consistencia del candidato."""
        try:
            # Verificar si tiene atributos básicos
            score = 0.0
            
            if 'area' in candidato and candidato['area'] > 0:
                score += 0.3
            
            if 'metodo' in candidato:
                score += 0.2
            
            if 'confianza_base' in candidato:
                score += candidato['confianza_base'] * 0.5
            
            return min(score, 1.0)
        except:
            return 0.0
    
    def _aplicar_nms_semaforos(self, semaforos, umbral_iou=0.3):
        """Aplica Non-Maximum Suppression para eliminar detecciones duplicadas."""
        if len(semaforos) == 0:
            return []
        
        # Convertir a formato compatible
        boxes = []
        scores = []
        
        for semaforo in semaforos:
            x, y, w, h = semaforo[0], semaforo[1], semaforo[2], semaforo[3]
            boxes.append([x - w//2, y - h//2, x + w//2, y + h//2])
            scores.append(semaforo[5])  # confianza
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Aplicar NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.3, umbral_iou)
        
        semaforos_filtrados = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                semaforos_filtrados.append(semaforos[i])
        
        return semaforos_filtrados
    
    def _dibujar_resultados(self, imagen, semaforos, nombre_config):
        """Dibuja los resultados en la imagen."""
        for i, semaforo in enumerate(semaforos):
            x, y, w, h = semaforo[0], semaforo[1], semaforo[2], semaforo[3]
            color_detectado = semaforo[4] if len(semaforo) > 4 else 'desconocido'
            confianza = semaforo[5] if len(semaforo) > 5 else 0.0
            
            # Rectángulo del semáforo
            x1, y1 = x - w//2, y - h//2
            x2, y2 = x + w//2, y + h//2
            
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Etiqueta
            etiqueta = f"Semaforo {i+1}: {color_detectado} ({confianza:.2f})"
            cv2.putText(imagen, etiqueta, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Centro
            cv2.circle(imagen, (x, y), 3, (0, 0, 255), -1)
        
        return imagen
    
    def _mostrar_resultado(self, resultado, titulo):
        """Muestra resultado de la detección."""
        print(f"\nResultado - {titulo}")
        print(f"Configuración: {resultado['configuracion']}")
        print(f"Semáforos detectados: {resultado['num_semaforos']}")
        print(f"Candidatos iniciales: {resultado['candidatos_iniciales']}")
        print(f"Confianza promedio: {resultado['confianza_promedio']:.3f}")
        
        plt.figure(figsize=(12, 8))
        imagen_rgb = cv2.cvtColor(resultado['imagen_resultado'], cv2.COLOR_BGR2RGB)
        plt.imshow(imagen_rgb)
        plt.title(titulo)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _guardar_resultado(self, imagen_resultado, ruta_base, config_nombre):
        """Guarda imagen resultado."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"semaforos_{config_nombre}_{timestamp}.jpg"
        
        if ruta_base:
            directorio = os.path.dirname(ruta_base)
            ruta_completa = os.path.join(directorio, nombre_archivo)
        else:
            ruta_completa = nombre_archivo
        
        os.makedirs(os.path.dirname(ruta_completa), exist_ok=True)
        cv2.imwrite(ruta_completa, imagen_resultado)
        print(f"Resultado guardado: {ruta_completa}")


def detectar_semaforos_imagen(ruta_imagen, configuracion='CONFIG_BALANCED', visualizar=True, guardar=False, ruta_salida=None):
    """
    Función de conveniencia para detectar semáforos en una imagen.
    
    Args:
        ruta_imagen (str): Ruta de la imagen
        configuracion (str): Configuración a usar
        visualizar (bool): Si mostrar resultados
        guardar (bool): Si guardar resultados
        ruta_salida (str): Ruta donde guardar
        
    Returns:
        dict: Resultados de la detección
    """
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
        return None
    
    detector = DetectorSemaforos()
    return detector.detectar_semaforos(imagen, configuracion, visualizar, guardar, ruta_salida)