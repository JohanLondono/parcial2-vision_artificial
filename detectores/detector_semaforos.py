#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de Semáforos en Imágenes de Tráfico Vehicular
======================================================

Sistema especializado para detectar semáforos utilizando análisis de color,
detección de formas y agrupación de elementos característicos.

Características del detector:
- Detección de colores típicos: rojo, amarillo, verde
- Análisis de estructura vertical de semáforos
- Validación por forma y proporción
- GrabCut para segmentación precisa
- Optical Flow para análisis de secuencia (si aplica)
"""

import cv2
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

class DetectorSemaforos:
    """Detector especializado de semáforos en tráfico vehicular."""
    
    def __init__(self):
        """Inicializar el detector de semáforos."""
        self.config = {
            'color_ranges': {
                # Rojo del semáforo
                'rojo_bajo1': [0, 120, 70],
                'rojo_alto1': [10, 255, 255],
                'rojo_bajo2': [170, 120, 70],
                'rojo_alto2': [180, 255, 255],
                # Amarillo del semáforo
                'amarillo_bajo': [15, 120, 70],
                'amarillo_alto': [35, 255, 255],
                # Verde del semáforo
                'verde_bajo': [40, 120, 70],
                'verde_alto': [80, 255, 255]
            },
            'circulo_params': {
                'dp': 1,
                'min_dist': 30,
                'param1': 50,
                'param2': 20,
                'min_radius': 8,
                'max_radius': 50
            },
            'agrupacion': {
                'tolerancia_vertical': 100,
                'tolerancia_horizontal': 30,
                'min_circulos': 2,
                'max_circulos': 4
            }
        }
    
    def detectar_semaforos(self, imagen, metodo='combinado', visualizar=True, guardar=False, ruta_salida=None):
        """
        Detecta semáforos en la imagen usando el método especificado.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            metodo (str): Método a usar ('color', 'estructura', 'grabcut', 'combinado')
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imagen resultado
            ruta_salida (str): Ruta donde guardar resultado
            
        Returns:
            dict: Resultados de la detección
        """
        print(f"🚦 Detectando semáforos usando método: {metodo}")
        
        if metodo == 'color':
            return self._detectar_semaforos_color(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'estructura':
            return self._detectar_semaforos_estructura(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'grabcut':
            return self._detectar_semaforos_grabcut(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'combinado':
            return self._detectar_semaforos_combinado(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'multialgoritmo':
            return self._detectar_semaforos_multialgoritmo(imagen, visualizar, guardar, ruta_salida)
        else:
            print(f"❌ Método no reconocido: {metodo}")
            return None
    
    def _detectar_semaforos_color(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Detecta semáforos usando análisis de color específico."""
        # Convertir a HSV
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Crear máscaras para cada color
        mascara_rojo1 = cv2.inRange(hsv, np.array(self.config['color_ranges']['rojo_bajo1']),
                                   np.array(self.config['color_ranges']['rojo_alto1']))
        mascara_rojo2 = cv2.inRange(hsv, np.array(self.config['color_ranges']['rojo_bajo2']),
                                   np.array(self.config['color_ranges']['rojo_alto2']))
        mascara_rojo = cv2.bitwise_or(mascara_rojo1, mascara_rojo2)
        
        mascara_amarillo = cv2.inRange(hsv, np.array(self.config['color_ranges']['amarillo_bajo']),
                                      np.array(self.config['color_ranges']['amarillo_alto']))
        
        mascara_verde = cv2.inRange(hsv, np.array(self.config['color_ranges']['verde_bajo']),
                                   np.array(self.config['color_ranges']['verde_alto']))
        
        # Detectar círculos en cada máscara de color
        circulos_por_color = {}
        colores_info = [
            (mascara_rojo, 'rojo', (0, 0, 255)),
            (mascara_amarillo, 'amarillo', (0, 255, 255)),
            (mascara_verde, 'verde', (0, 255, 0))
        ]
        
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        for mascara, nombre_color, color_bgr in colores_info:
            # Aplicar máscara a imagen en gris
            imagen_color = cv2.bitwise_and(imagen_gris, imagen_gris, mask=mascara)
            
            # Detectar círculos
            circulos = cv2.HoughCircles(
                imagen_color,
                cv2.HOUGH_GRADIENT,
                dp=self.config['circulo_params']['dp'],
                minDist=self.config['circulo_params']['min_dist'],
                param1=self.config['circulo_params']['param1'],
                param2=self.config['circulo_params']['param2'],
                minRadius=self.config['circulo_params']['min_radius'],
                maxRadius=self.config['circulo_params']['max_radius']
            )
            
            if circulos is not None:
                circulos = np.round(circulos[0, :]).astype("int")
                circulos_por_color[nombre_color] = [(x, y, r, color_bgr) for x, y, r in circulos]
            else:
                circulos_por_color[nombre_color] = []
        
        # Agrupar círculos en estructuras de semáforos
        semaforos_detectados = self._agrupar_circulos_semaforos(circulos_por_color)
        
        # Crear imagen resultado
        imagen_resultado = imagen.copy()
        for i, semaforo in enumerate(semaforos_detectados):
            # Dibujar rectángulo del semáforo
            x_min, y_min, x_max, y_max = semaforo['bbox']
            cv2.rectangle(imagen_resultado, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
            cv2.putText(imagen_resultado, f"Semaforo {i+1}", (x_min, y_min-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Dibujar círculos individuales
            for circulo in semaforo['circulos']:
                x, y, r, color = circulo
                cv2.circle(imagen_resultado, (x, y), r, color, 2)
                cv2.circle(imagen_resultado, (x, y), 2, color, -1)
        
        resultado = {
            'metodo': 'Análisis de Color HSV',
            'num_semaforos': len(semaforos_detectados),
            'semaforos': semaforos_detectados,
            'circulos_por_color': circulos_por_color,
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': self._calcular_confianza_color(semaforos_detectados)
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Semáforos - Color")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "color")
        
        return resultado
    
    def _detectar_semaforos_estructura(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Detecta semáforos analizando estructura y forma general."""
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtros para resaltar estructuras
        blur = cv2.GaussianBlur(gris, (5, 5), 0)
        
        # Detectar bordes
        bordes = cv2.Canny(blur, 50, 150)
        
        # Operaciones morfológicas para conectar elementos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)
        
        # Detectar contornos
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidatos_semaforos = []
        
        for contorno in contornos:
            # Filtrar por área
            area = cv2.contourArea(contorno)
            if area < 500 or area > 20000:
                continue
            
            # Obtener rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contorno)
            
            # Verificar proporciones típicas de semáforo (más alto que ancho)
            aspect_ratio = h / w
            if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                continue
            
            # Verificar dimensiones mínimas
            if w < 20 or h < 40:
                continue
            
            # Analizar contenido de la región
            roi = imagen[y:y+h, x:x+w]
            score_semaforo = self._evaluar_region_semaforo(roi)
            
            if score_semaforo > 0.3:
                candidatos_semaforos.append({
                    'bbox': (x, y, x+w, y+h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'score': score_semaforo,
                    'centro': (x + w//2, y + h//2)
                })
        
        # Ordenar por score y filtrar los mejores
        candidatos_semaforos.sort(key=lambda x: x['score'], reverse=True)
        semaforos_detectados = candidatos_semaforos[:5]  # Máximo 5 semáforos
        
        # Crear imagen resultado
        imagen_resultado = imagen.copy()
        for i, semaforo in enumerate(semaforos_detectados):
            x_min, y_min, x_max, y_max = semaforo['bbox']
            cv2.rectangle(imagen_resultado, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.putText(imagen_resultado, f"Semaforo {i+1} ({semaforo['score']:.2f})", 
                       (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        resultado = {
            'metodo': 'Análisis de Estructura',
            'num_semaforos': len(semaforos_detectados),
            'semaforos': semaforos_detectados,
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': np.mean([s['score'] for s in semaforos_detectados]) if semaforos_detectados else 0
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Semáforos - Estructura")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "estructura")
        
        return resultado
    
    def _detectar_semaforos_grabcut(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Detecta semáforos usando segmentación GrabCut."""
        # Primero detectar regiones candidatas con método de estructura
        candidatos = self._detectar_semaforos_estructura(imagen, False, False)
        
        if not candidatos['semaforos']:
            resultado = {
                'metodo': 'GrabCut',
                'num_semaforos': 0,
                'semaforos': [],
                'imagen_resultado': imagen.copy(),
                'confianza_promedio': 0
            }
            
            if visualizar:
                self._mostrar_resultado(resultado, "Detección de Semáforos - GrabCut")
            return resultado
        
        imagen_resultado = imagen.copy()
        semaforos_refinados = []
        
        for i, candidato in enumerate(candidatos['semaforos'][:3]):  # Procesar máximo 3
            x_min, y_min, x_max, y_max = candidato['bbox']
            
            # Expandir región ligeramente
            margin = 10
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(imagen.shape[1], x_max + margin)
            y_max = min(imagen.shape[0], y_max + margin)
            
            # Extraer ROI
            roi = imagen[y_min:y_max, x_min:x_max]
            
            if roi.shape[0] < 20 or roi.shape[1] < 20:
                continue
            
            # Aplicar GrabCut
            try:
                mascara_grabcut = self._aplicar_grabcut(roi)
                
                # Analizar resultado de GrabCut
                if np.sum(mascara_grabcut) > 100:  # Suficiente área segmentada
                    # Encontrar contornos en la máscara
                    contornos_gc, _ = cv2.findContours(mascara_grabcut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contornos_gc:
                        # Obtener el contorno más grande
                        contorno_principal = max(contornos_gc, key=cv2.contourArea)
                        
                        # Calcular nuevo bounding box
                        x_gc, y_gc, w_gc, h_gc = cv2.boundingRect(contorno_principal)
                        
                        # Convertir coordenadas a imagen original
                        x_final = x_min + x_gc
                        y_final = y_min + y_gc
                        
                        semaforo_refinado = {
                            'bbox': (x_final, y_final, x_final + w_gc, y_final + h_gc),
                            'score': candidato['score'] * 1.2,  # Bonus por refinamiento
                            'metodo': 'GrabCut',
                            'area_segmentada': np.sum(mascara_grabcut)
                        }
                        
                        semaforos_refinados.append(semaforo_refinado)
                        
                        # Dibujar resultado
                        cv2.rectangle(imagen_resultado, (x_final, y_final), 
                                    (x_final + w_gc, y_final + h_gc), (255, 0, 255), 2)
                        cv2.putText(imagen_resultado, f"Semaforo GrabCut {i+1}", 
                                   (x_final, y_final-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        
                        # Superponer máscara
                        roi_resultado = imagen_resultado[y_min:y_max, x_min:x_max]
                        roi_resultado[mascara_grabcut == 1] = roi_resultado[mascara_grabcut == 1] * 0.7 + np.array([255, 0, 255]) * 0.3
            
            except Exception as e:
                print(f"Error en GrabCut para candidato {i+1}: {e}")
                continue
        
        resultado = {
            'metodo': 'GrabCut',
            'num_semaforos': len(semaforos_refinados),
            'semaforos': semaforos_refinados,
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': np.mean([s['score'] for s in semaforos_refinados]) if semaforos_refinados else 0
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Semáforos - GrabCut")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "grabcut")
        
        return resultado
    
    def _detectar_semaforos_combinado(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Combina múltiples métodos para detección robusta."""
        # Ejecutar todos los métodos
        resultado_color = self._detectar_semaforos_color(imagen, False, False)
        resultado_estructura = self._detectar_semaforos_estructura(imagen, False, False)
        resultado_grabcut = self._detectar_semaforos_grabcut(imagen, False, False)
        
        # Fusionar resultados
        todos_semaforos = []
        todos_semaforos.extend(resultado_color['semaforos'])
        todos_semaforos.extend(resultado_estructura['semaforos'])
        todos_semaforos.extend(resultado_grabcut['semaforos'])
        
        # Eliminar duplicados usando NMS
        semaforos_finales = self._aplicar_nms_semaforos(todos_semaforos)
        
        # Crear imagen resultado final
        imagen_resultado = imagen.copy()
        for i, semaforo in enumerate(semaforos_finales):
            if 'bbox' in semaforo:
                x_min, y_min, x_max, y_max = semaforo['bbox']
                cv2.rectangle(imagen_resultado, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                
                # Mostrar información del semáforo
                info_text = f"Semaforo {i+1}"
                if 'score' in semaforo:
                    info_text += f" ({semaforo['score']:.2f})"
                
                cv2.putText(imagen_resultado, info_text, (x_min, y_min-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Si tiene círculos, dibujarlos también
                if 'circulos' in semaforo:
                    for circulo in semaforo['circulos']:
                        x, y, r, color = circulo
                        cv2.circle(imagen_resultado, (x, y), r, color, 2)
        
        resultado = {
            'metodo': 'Combinado (Color + Estructura + GrabCut)',
            'num_semaforos': len(semaforos_finales),
            'semaforos': semaforos_finales,
            'detecciones_color': len(resultado_color['semaforos']),
            'detecciones_estructura': len(resultado_estructura['semaforos']),
            'detecciones_grabcut': len(resultado_grabcut['semaforos']),
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': (resultado_color['confianza_promedio'] + 
                                 resultado_estructura['confianza_promedio'] + 
                                 resultado_grabcut['confianza_promedio']) / 3
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Semáforos - Método Combinado")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "combinado")
        
        return resultado
    
    def _detectar_semaforos_multialgoritmo(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """
        Detecta semáforos usando MÚLTIPLES ALGORITMOS avanzados.
        
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
            guardar (bool): Si guardar resultados
            ruta_salida (str): Ruta donde guardar
            
        Returns:
            dict: Resultados de detección de semáforos
        """
        print("Detectando semáforos (MULTI-ALGORITMO)...")
        
        try:
            from skimage.feature import graycomatrix, graycoprops
        except ImportError:
            print("scikit-image no disponible, usando análisis simplificado")
            graycomatrix = None
            graycoprops = None
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
            
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
            print("No se detectaron círculos (luces de semáforo)")
            circles = np.array([])
        else:
            circles = np.round(circles[0, :]).astype("int")
            print(f"Círculos detectados: {len(circles)}")
        
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
            print(f"AKAZE no disponible: {e}")
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
                            if graycomatrix is not None and graycoprops is not None:
                                roi_glcm = ((roi_gris / 255.0) * 15).astype(np.uint8)
                                glcm = graycomatrix(roi_glcm, distances=[1], angles=[0],
                                                   levels=16, symmetric=True, normed=True)
                                
                                contraste = graycoprops(glcm, 'contrast')[0, 0]
                                homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
                                texture_score = min(contraste / 5.0, 1.0)
                            else:
                                # Análisis de textura simplificado
                                texture_score = np.std(roi_gris) / 128.0
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
        
        # Crear imagen resultado
        imagen_resultado = imagen.copy()
        for i, semaforo in enumerate(semaforos_detectados):
            # Dibujar bounding box
            x, y, w, h = semaforo['bbox']
            cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            # Dibujar las luces
            for j, luz in enumerate(semaforo['luces']):
                pos = luz['posicion']
                radio = luz['radio']
                color_info = luz['color']
                
                # Color para dibujar según el color detectado
                if color_info['color'] == 'rojo':
                    draw_color = (0, 0, 255)
                elif color_info['color'] == 'amarillo':
                    draw_color = (0, 255, 255)
                elif color_info['color'] == 'verde':
                    draw_color = (0, 255, 0)
                else:
                    draw_color = (128, 128, 128)
                
                cv2.circle(imagen_resultado, pos, radio, draw_color, 2)
                cv2.putText(imagen_resultado, color_info['color'][:4], 
                           (pos[0] - 15, pos[1] - radio - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)
            
            # Etiqueta de confianza
            cv2.putText(imagen_resultado, f"Semáforo {i+1}: {semaforo['confidence']:.2f}",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        resultado = {
            'metodo': 'Multi-Algoritmo Avanzado',
            'num_semaforos': len(semaforos_detectados),
            'semaforos': semaforos_detectados,
            'imagen_resultado': imagen_resultado,
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
            self._mostrar_resultado(resultado, "Detección Multi-Algoritmo de Semáforos")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "multialgoritmo")
        
        return resultado
    
    def _agrupar_circulos_semaforos(self, circulos_por_color):
        """Agrupa círculos de diferentes colores en estructuras de semáforos."""
        # Combinar todos los círculos
        todos_circulos = []
        for color, circulos in circulos_por_color.items():
            todos_circulos.extend(circulos)
        
        if len(todos_circulos) < 2:
            return []
        
        # Agrupar círculos por proximidad vertical
        grupos = []
        tolerancia_h = self.config['agrupacion']['tolerancia_horizontal']
        tolerancia_v = self.config['agrupacion']['tolerancia_vertical']
        
        circulos_procesados = set()
        
        for i, (x1, y1, r1, color1) in enumerate(todos_circulos):
            if i in circulos_procesados:
                continue
            
            grupo_actual = [(x1, y1, r1, color1)]
            circulos_procesados.add(i)
            
            # Buscar círculos cercanos verticalmente
            for j, (x2, y2, r2, color2) in enumerate(todos_circulos):
                if j in circulos_procesados:
                    continue
                
                # Verificar si están alineados verticalmente
                if (abs(x1 - x2) <= tolerancia_h and 
                    abs(y1 - y2) <= tolerancia_v and 
                    abs(y1 - y2) > 10):  # Debe haber separación mínima
                    
                    grupo_actual.append((x2, y2, r2, color2))
                    circulos_procesados.add(j)
            
            # Solo considerar grupos con al menos 2 círculos
            if len(grupo_actual) >= self.config['agrupacion']['min_circulos']:
                grupos.append(grupo_actual)
        
        # Convertir grupos a estructuras de semáforos
        semaforos = []
        for grupo in grupos:
            if len(grupo) <= self.config['agrupacion']['max_circulos']:
                # Calcular bounding box del grupo
                xs = [x for x, y, r, color in grupo]
                ys = [y for x, y, r, color in grupo]
                rs = [r for x, y, r, color in grupo]
                
                x_min = min(xs) - max(rs)
                x_max = max(xs) + max(rs)
                y_min = min(ys) - max(rs)
                y_max = max(ys) + max(rs)
                
                # Ordenar círculos por posición vertical (arriba a abajo)
                grupo_ordenado = sorted(grupo, key=lambda c: c[1])
                
                semaforo = {
                    'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                    'circulos': grupo_ordenado,
                    'num_luces': len(grupo_ordenado),
                    'colores_detectados': list(set([self._nombre_color_bgr(color) for _, _, _, color in grupo_ordenado]))
                }
                
                semaforos.append(semaforo)
        
        return semaforos
    
    def _evaluar_region_semaforo(self, roi):
        """Evalúa qué tan probable es que una región contenga un semáforo."""
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return 0.0
        
        # Convertir a HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Buscar colores típicos de semáforo
        score_total = 0.0
        
        # Verificar presencia de rojo
        mascara_rojo1 = cv2.inRange(hsv_roi, np.array(self.config['color_ranges']['rojo_bajo1']),
                                   np.array(self.config['color_ranges']['rojo_alto1']))
        mascara_rojo2 = cv2.inRange(hsv_roi, np.array(self.config['color_ranges']['rojo_bajo2']),
                                   np.array(self.config['color_ranges']['rojo_alto2']))
        mascara_rojo = cv2.bitwise_or(mascara_rojo1, mascara_rojo2)
        score_total += np.sum(mascara_rojo) / (roi.shape[0] * roi.shape[1] * 255) * 0.4
        
        # Verificar presencia de amarillo
        mascara_amarillo = cv2.inRange(hsv_roi, np.array(self.config['color_ranges']['amarillo_bajo']),
                                      np.array(self.config['color_ranges']['amarillo_alto']))
        score_total += np.sum(mascara_amarillo) / (roi.shape[0] * roi.shape[1] * 255) * 0.3
        
        # Verificar presencia de verde
        mascara_verde = cv2.inRange(hsv_roi, np.array(self.config['color_ranges']['verde_bajo']),
                                   np.array(self.config['color_ranges']['verde_alto']))
        score_total += np.sum(mascara_verde) / (roi.shape[0] * roi.shape[1] * 255) * 0.3
        
        return min(1.0, score_total)
    
    def _aplicar_grabcut(self, roi):
        """Aplica algoritmo GrabCut a una región de interés."""
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            return np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        
        # Crear máscara inicial
        mascara = np.zeros(roi.shape[:2], np.uint8)
        
        # Definir rectángulo inicial (centro de la ROI)
        altura, ancho = roi.shape[:2]
        margen_h = ancho // 4
        margen_v = altura // 4
        
        rect = (margen_h, margen_v, ancho - 2*margen_h, altura - 2*margen_v)
        
        # Modelos de fondo y primer plano (requeridos por GrabCut)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Aplicar GrabCut
            cv2.grabCut(roi, mascara, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Crear máscara binaria (solo primer plano y probablemente primer plano)
            mascara_final = np.where((mascara == 2) | (mascara == 0), 0, 1).astype('uint8')
            
            return mascara_final
            
        except Exception:
            # Si GrabCut falla, devolver máscara vacía
            return np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    
    def _aplicar_nms_semaforos(self, semaforos):
        """Aplica Non-Maximum Suppression para eliminar detecciones duplicadas."""
        if not semaforos:
            return []
        
        # Calcular IoU para cada par de semáforos
        indices_mantener = []
        
        for i in range(len(semaforos)):
            mantener = True
            
            for j in range(len(semaforos)):
                if i != j and j in indices_mantener:
                    # Calcular IoU si ambos tienen bbox
                    if 'bbox' in semaforos[i] and 'bbox' in semaforos[j]:
                        iou = self._calcular_iou(semaforos[i]['bbox'], semaforos[j]['bbox'])
                        if iou > 0.3:  # Umbral de solapamiento
                            mantener = False
                            break
            
            if mantener:
                indices_mantener.append(i)
        
        return [semaforos[i] for i in indices_mantener]
    
    def _calcular_iou(self, bbox1, bbox2):
        """Calcula Intersection over Union entre dos bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calcular intersección
        x_intersect = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_intersect = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        area_intersect = x_intersect * y_intersect
        
        # Calcular unión
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        area_union = area1 + area2 - area_intersect
        
        return area_intersect / area_union if area_union > 0 else 0
    
    def _nombre_color_bgr(self, color_bgr):
        """Convierte color BGR a nombre descriptivo."""
        b, g, r = color_bgr
        if r > 200 and g < 100 and b < 100:
            return "rojo"
        elif g > 200 and r > 150 and b < 100:
            return "amarillo"
        elif g > 200 and r < 100 and b < 100:
            return "verde"
        else:
            return "desconocido"
    
    def _calcular_confianza_color(self, semaforos):
        """Calcula confianza promedio para detecciones por color."""
        if not semaforos:
            return 0.0
        
        # Confianza basada en diversidad de colores detectados
        total_confianza = 0.0
        for semaforo in semaforos:
            if 'colores_detectados' in semaforo:
                # Más colores diferentes = mayor confianza
                confianza = len(semaforo['colores_detectados']) / 3.0  # Máximo 3 colores
                total_confianza += confianza
        
        return total_confianza / len(semaforos)
    
    def _mostrar_resultado(self, resultado, titulo):
        """Muestra resultado de la detección."""
        print(f"\nResultado - {titulo}")
        print(f"Método: {resultado['metodo']}")
        print(f"Semáforos detectados: {resultado['num_semaforos']}")
        if 'confianza_promedio' in resultado:
            print(f"Confianza promedio: {resultado['confianza_promedio']:.3f}")
        
        # Mostrar detalles de semáforos
        if resultado['semaforos']:
            print("Detalles de semáforos:")
            for i, semaforo in enumerate(resultado['semaforos']):
                info = f"  Semáforo {i+1}:"
                if 'bbox' in semaforo:
                    x_min, y_min, x_max, y_max = semaforo['bbox']
                    info += f" Posición({x_min}, {y_min}), Tamaño({x_max-x_min}x{y_max-y_min})"
                if 'num_luces' in semaforo:
                    info += f", Luces={semaforo['num_luces']}"
                if 'colores_detectados' in semaforo:
                    info += f", Colores={semaforo['colores_detectados']}"
                print(info)
        
        # Mostrar imagen resultado
        plt.figure(figsize=(12, 8))
        imagen_rgb = cv2.cvtColor(resultado['imagen_resultado'], cv2.COLOR_BGR2RGB)
        plt.imshow(imagen_rgb)
        plt.title(titulo)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _guardar_resultado(self, imagen_resultado, ruta_base, metodo):
        """Guarda imagen resultado."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"semaforos_{metodo}_{timestamp}.jpg"
        
        if ruta_base:
            directorio = os.path.dirname(ruta_base)
            ruta_completa = os.path.join(directorio, nombre_archivo)
        else:
            ruta_completa = nombre_archivo
        
        os.makedirs(os.path.dirname(ruta_completa), exist_ok=True)
        cv2.imwrite(ruta_completa, imagen_resultado)
        print(f"Resultado guardado: {ruta_completa}")

# Función de utilidad
def detectar_semaforos_imagen(ruta_imagen, metodo='combinado', visualizar=True, guardar=False, ruta_salida=None):
    """
    Función de conveniencia para detectar semáforos en una imagen.
    
    Args:
        ruta_imagen (str): Ruta de la imagen
        metodo (str): Método de detección
        visualizar (bool): Si mostrar resultados
        guardar (bool): Si guardar resultados
        ruta_salida (str): Ruta donde guardar
        
    Returns:
        dict: Resultados de la detección
    """
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
        return None
    
    # Crear detector y ejecutar
    detector = DetectorSemaforos()
    return detector.detectar_semaforos(imagen, metodo, visualizar, guardar, ruta_salida)