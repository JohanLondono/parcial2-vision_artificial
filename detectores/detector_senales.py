#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de Señales de Tráfico en Imágenes Vehiculares
======================================================

Sistema especializado para detectar señales de tráfico circulares utilizando
múltiples algoritmos de visión por computadora.

Algoritmos utilizados:
- Transformada de Hough para círculos
- FREAK (Fast Retina Keypoint) para características únicas
- Análisis de color HSV para señales típicas
- Detección de bordes avanzada (LoG)
- Combinación de métodos para mayor robustez
"""

import cv2
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.filters import gaussian, laplace
from skimage.feature import canny

class DetectorSenales:
    """Detector especializado de señales de tráfico circulares."""
    
    def __init__(self):
        """Inicializar el detector de señales."""
        self.config = {
            'hough_circles': {
                'dp': 1,
                'min_dist': 80,
                'param1': 50,
                'param2': 30,
                'min_radius': 20,
                'max_radius': 150
            },
            'color_ranges': {
                # Rojo (señales de prohibición)
                'rojo_bajo1': [0, 50, 50],
                'rojo_alto1': [10, 255, 255],
                'rojo_bajo2': [170, 50, 50],
                'rojo_alto2': [180, 255, 255],
                # Azul (señales informativas)
                'azul_bajo': [100, 50, 50],
                'azul_alto': [130, 255, 255],
                # Amarillo (señales de advertencia)
                'amarillo_bajo': [20, 50, 50],
                'amarillo_alto': [30, 255, 255]
            },
            'log_sigma': [1, 2, 3, 4],
            'freak_threshold': 60
        }
    
    def detectar_senales(self, imagen, metodo='combinado', visualizar=True, guardar=False, ruta_salida=None):
        """
        Detecta señales de tráfico usando el método especificado.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            metodo (str): Método a usar ('hough', 'freak', 'color', 'log', 'combinado')
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imagen resultado
            ruta_salida (str): Ruta donde guardar resultado
            
        Returns:
            dict: Resultados de la detección
        """
        print(f"🔍 Detectando señales de tráfico usando método: {metodo}")
        
        if metodo == 'hough':
            return self._detectar_senales_hough(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'freak':
            return self._detectar_senales_freak(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'color':
            return self._detectar_senales_color(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'log':
            return self._detectar_senales_log(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'combinado':
            return self._detectar_senales_combinado(imagen, visualizar, guardar, ruta_salida)
        else:
            print(f"❌ Método no reconocido: {metodo}")
            return None
    
    def _detectar_senales_hough(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Detecta señales usando Transformada de Hough para círculos."""
        # Preprocesar imagen
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (9, 9), 2)
        
        # Detectar círculos con Hough
        circulos = cv2.HoughCircles(
            gris,
            cv2.HOUGH_GRADIENT,
            dp=self.config['hough_circles']['dp'],
            minDist=self.config['hough_circles']['min_dist'],
            param1=self.config['hough_circles']['param1'],
            param2=self.config['hough_circles']['param2'],
            minRadius=self.config['hough_circles']['min_radius'],
            maxRadius=self.config['hough_circles']['max_radius']
        )
        
        senales_detectadas = []
        imagen_resultado = imagen.copy()
        
        if circulos is not None:
            circulos = np.round(circulos[0, :]).astype("int")
            
            for (x, y, r) in circulos:
                # Validar si es realmente una señal
                tipo_senal = self._validar_senal_hough(imagen, x, y, r)
                if tipo_senal:
                    senales_detectadas.append((x, y, r, tipo_senal))
                    
                    # Color según tipo de señal
                    color = self._obtener_color_tipo_senal(tipo_senal)
                    
                    # Dibujar círculo detectado
                    cv2.circle(imagen_resultado, (x, y), r, color, 3)
                    cv2.circle(imagen_resultado, (x, y), 2, color, -1)
                    cv2.putText(imagen_resultado, f"Senal {tipo_senal}", (x-40, y-r-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        resultado = {
            'metodo': 'Hough Circles',
            'num_senales': len(senales_detectadas),
            'senales': senales_detectadas,
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': self._calcular_confianza_hough(senales_detectadas)
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Señales - Hough")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "hough")
        
        return resultado
    
    def _detectar_senales_freak(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Detecta señales usando descriptor FREAK."""
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Crear detector FAST y descriptor FREAK
        detector_fast = cv2.FastFeatureDetector_create(threshold=self.config['freak_threshold'])
        
        try:
            # Intentar usar FREAK si está disponible
            descriptor_freak = cv2.xfeatures2d.FREAK_create()
            keypoints = detector_fast.detect(gris, None)
            keypoints, descriptors = descriptor_freak.compute(gris, keypoints)
            algoritmo_usado = 'FREAK'
        except (AttributeError, cv2.error):
            # Usar ORB como alternativa
            detector_orb = cv2.ORB_create(nfeatures=500)
            keypoints, descriptors = detector_orb.detectAndCompute(gris, None)
            algoritmo_usado = 'ORB (FREAK no disponible)'
        
        # Agrupar keypoints en regiones circulares
        senales_detectadas = self._agrupar_keypoints_senales(keypoints)
        
        # Dibujar resultados
        imagen_resultado = imagen.copy()
        
        # Dibujar keypoints
        imagen_keypoints = cv2.drawKeypoints(imagen_resultado, keypoints, None,
                                           color=(255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Dibujar señales detectadas
        for (x, y, r, tipo) in senales_detectadas:
            color = self._obtener_color_tipo_senal(tipo)
            cv2.circle(imagen_keypoints, (x, y), r, color, 3)
            cv2.putText(imagen_keypoints, f"Senal {tipo}", (x-40, y-r-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        resultado = {
            'metodo': f'{algoritmo_usado}',
            'num_senales': len(senales_detectadas),
            'senales': senales_detectadas,
            'keypoints': len(keypoints),
            'imagen_resultado': imagen_keypoints,
            'confianza_promedio': self._calcular_confianza_freak(keypoints, senales_detectadas)
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Señales - FREAK/ORB")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_keypoints, ruta_salida, "freak")
        
        return resultado
    
    def _detectar_senales_color(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Detecta señales usando análisis de color HSV avanzado con eliminación de fondo blanco."""
        print("  🌈 Método: Análisis de Color HSV Avanzado")
        
        # Convertir a HSV y escala de grises
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        altura, ancho = imagen_gris.shape
        
        # Preprocesamiento mejorado
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        imagen_eq = clahe.apply(imagen_gris)
        blur = cv2.GaussianBlur(imagen_eq, (9, 9), 2)
        
        # Detectar círculos candidatos con HoughCircles
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=30, minRadius=15, maxRadius=min(ancho, altura) // 4
        )
        
        senales_detectadas = []
        imagen_resultado = imagen.copy()
        colores_detectados = {'rojo': 0, 'azul': 0, 'amarillo': 0, 'blanco': 0}
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"    Analizando {len(circles)} círculos candidatos...")
            
            for (x, y, r) in circles:
                # Verificar límites
                if x-r < 0 or y-r < 0 or x+r >= ancho or y+r >= altura:
                    continue
                
                # Extraer ROI (Región de Interés)
                roi = imagen[max(0, y-r):min(altura, y+r), max(0, x-r):min(ancho, x+r)]
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Crear máscaras de colores mejoradas
                # ROJO - Dos rangos para cubrir la discontinuidad del matiz
                red_mask1 = cv2.inRange(roi_hsv, (0, 100, 100), (10, 255, 255))
                red_mask2 = cv2.inRange(roi_hsv, (170, 100, 100), (180, 255, 255))
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                
                # AZUL - Señales informativas
                blue_mask = cv2.inRange(roi_hsv, (100, 100, 50), (130, 255, 255))
                
                # AMARILLO - Señales de advertencia  
                yellow_mask = cv2.inRange(roi_hsv, (20, 100, 100), (30, 255, 255))
                
                # BLANCO - Fondo de señales (alta luminosidad, baja saturación)
                white_mask = cv2.inRange(roi_hsv, (0, 0, 200), (180, 30, 255))
                
                # Calcular porcentajes de cada color
                total_pixels = roi.shape[0] * roi.shape[1]
                red_pct = np.sum(red_mask > 0) / total_pixels
                blue_pct = np.sum(blue_mask > 0) / total_pixels
                yellow_pct = np.sum(yellow_mask > 0) / total_pixels
                white_pct = np.sum(white_mask > 0) / total_pixels
                
                # Determinar color dominante
                max_color_pct = max(red_pct, blue_pct, yellow_pct)
                
                # Patrón típico de señal: fondo blanco + color predominante
                has_sign_pattern = white_pct > 0.15 and max_color_pct > 0.10
                
                # Calcular confianza basada en color y patrón
                confidence = max_color_pct * 0.7 + white_pct * 0.3
                if has_sign_pattern:
                    confidence *= 1.2  # Bonus por patrón típico de señal
                
                confidence = min(confidence, 1.0)  # Limitar a 1.0
                
                # Filtrar por confianza mínima
                if confidence > 0.25:
                    # Determinar tipo de señal por color dominante
                    if red_pct == max_color_pct:
                        tipo_senal = 'Prohibicion'
                        color_draw = (0, 0, 255)  # Rojo en BGR
                        colores_detectados['rojo'] += 1
                    elif blue_pct == max_color_pct:
                        tipo_senal = 'Informativa'
                        color_draw = (255, 0, 0)  # Azul en BGR
                        colores_detectados['azul'] += 1
                    elif yellow_pct == max_color_pct:
                        tipo_senal = 'Advertencia'
                        color_draw = (0, 255, 255)  # Amarillo en BGR
                        colores_detectados['amarillo'] += 1
                    else:
                        tipo_senal = 'Detectada'
                        color_draw = (0, 255, 0)  # Verde por defecto
                    
                    # Agregar a la lista de detecciones
                    senales_detectadas.append((x, y, r, tipo_senal, confidence))
                    
                    # Dibujar en imagen resultado
                    cv2.circle(imagen_resultado, (x, y), r, color_draw, 3)
                    cv2.circle(imagen_resultado, (x, y), 2, color_draw, -1)  # Centro
                    
                    # Etiqueta con información
                    label = f"{tipo_senal} ({confidence:.2f})"
                    cv2.putText(imagen_resultado, label, (x-40, y-r-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_draw, 2)
                    
                    print(f"    ✓ Señal {tipo_senal}: centro=({x},{y}), radio={r}, confianza={confidence:.3f}")
                    print(f"      Colores: R={red_pct:.2f}, A={blue_pct:.2f}, Am={yellow_pct:.2f}, B={white_pct:.2f}")
                
                # Actualizar estadísticas globales
                colores_detectados['blanco'] += white_pct
        
        # Crear máscara combinada de todos los colores detectados (para visualización adicional)
        mascara_combinada = np.zeros((altura, ancho), dtype=np.uint8)
        
        # Crear máscaras globales para visualización
        red_mask_global1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask_global2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask_global = cv2.bitwise_or(red_mask_global1, red_mask_global2)
        
        blue_mask_global = cv2.inRange(hsv, np.array([100, 100, 50]), np.array([130, 255, 255]))
        yellow_mask_global = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        
        mascara_combinada = cv2.bitwise_or(mascara_combinada, red_mask_global)
        mascara_combinada = cv2.bitwise_or(mascara_combinada, blue_mask_global)
        mascara_combinada = cv2.bitwise_or(mascara_combinada, yellow_mask_global)
        
        # Aplicar operaciones morfológicas a la máscara combinada
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara_combinada = cv2.morphologyEx(mascara_combinada, cv2.MORPH_OPEN, kernel)
        mascara_combinada = cv2.morphologyEx(mascara_combinada, cv2.MORPH_CLOSE, kernel)
        
        resultado = {
            'metodo': 'Análisis de Color HSV Avanzado',
            'num_senales': len(senales_detectadas),
            'senales': senales_detectadas,
            'senales_detectadas': senales_detectadas,  # Para compatibilidad con extensiones
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': np.mean([s[4] for s in senales_detectadas]) if senales_detectadas else 0.0,
            'colores_detectados': colores_detectados,
            'mask': mascara_combinada,
            'imagen_procesada': {
                'hsv': hsv,
                'ecualizacion': imagen_eq,
                'red_mask': red_mask_global,
                'blue_mask': blue_mask_global,
                'yellow_mask': yellow_mask_global,
                'combined_mask': mascara_combinada
            },
            'estadisticas_color': {
                'total_rojas': colores_detectados['rojo'],
                'total_azules': colores_detectados['azul'], 
                'total_amarillas': colores_detectados['amarillo'],
                'circulos_analizados': len(circles) if circles is not None else 0,
                'patron_fondo_blanco': sum(1 for s in senales_detectadas if len(s) > 4 and s[4] > 0.4)
            }
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Señales - Color HSV Avanzado")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "color")
        
        return resultado
    
    def _detectar_senales_log(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Detecta señales usando Laplaciano de Gauss (LoG)."""
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris = gris.astype(np.float64)
        
        # Aplicar LoG con diferentes escalas
        respuestas_log = []
        for sigma in self.config['log_sigma']:
            # Aplicar filtro Gaussiano
            gauss = gaussian(gris, sigma=sigma)
            # Aplicar Laplaciano
            log_response = laplace(gauss)
            respuestas_log.append(log_response)
        
        # Combinar respuestas de diferentes escalas
        respuesta_combinada = np.max(respuestas_log, axis=0)
        
        # Normalizar y convertir a uint8
        respuesta_normalizada = np.abs(respuesta_combinada)
        respuesta_normalizada = (respuesta_normalizada / np.max(respuesta_normalizada) * 255).astype(np.uint8)
        
        # Umbralizar para encontrar regiones de interés
        _, umbral = cv2.threshold(respuesta_normalizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        umbral = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        senales_detectadas = []
        imagen_resultado = imagen.copy()
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area < 1000 or area > 30000:
                continue
            
            # Verificar circularidad
            perimetro = cv2.arcLength(contorno, True)
            if perimetro == 0:
                continue
            
            circularidad = 4 * np.pi * area / (perimetro * perimetro)
            
            if circularidad > 0.3:
                (x, y), radio = cv2.minEnclosingCircle(contorno)
                x, y, radio = int(x), int(y), int(radio)
                
                if 25 <= radio <= 120:
                    # Determinar tipo por análisis de color local
                    tipo = self._determinar_tipo_por_color(imagen, x, y, radio)
                    senales_detectadas.append((x, y, radio, tipo))
                    
                    color = self._obtener_color_tipo_senal(tipo)
                    cv2.circle(imagen_resultado, (x, y), radio, color, 3)
                    cv2.putText(imagen_resultado, f"Senal {tipo}", (x-40, y-radio-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        resultado = {
            'metodo': 'Laplaciano de Gauss (LoG)',
            'num_senales': len(senales_detectadas),
            'senales': senales_detectadas,
            'imagen_resultado': imagen_resultado,
            'respuesta_log': respuesta_normalizada,
            'confianza_promedio': self._calcular_confianza_log(senales_detectadas, circularidad if 'circularidad' in locals() else 0)
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Señales - LoG")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "log")
        
        return resultado
    
    def _detectar_senales_combinado(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Combina múltiples métodos para detección robusta."""
        # Ejecutar todos los métodos
        resultado_hough = self._detectar_senales_hough(imagen, False, False)
        resultado_freak = self._detectar_senales_freak(imagen, False, False)
        resultado_color = self._detectar_senales_color(imagen, False, False)
        resultado_log = self._detectar_senales_log(imagen, False, False)
        
        # Fusionar resultados
        todas_senales = []
        todas_senales.extend(resultado_hough['senales'])
        todas_senales.extend(resultado_freak['senales'])
        todas_senales.extend(resultado_color['senales'])
        todas_senales.extend(resultado_log['senales'])
        
        # Eliminar duplicados usando NMS
        senales_finales = self._aplicar_nms_senales(todas_senales, umbral_distancia=40)
        
        # Crear imagen resultado
        imagen_resultado = imagen.copy()
        for i, (x, y, r, tipo) in enumerate(senales_finales):
            color = self._obtener_color_tipo_senal(tipo)
            cv2.circle(imagen_resultado, (x, y), r, color, 3)
            cv2.circle(imagen_resultado, (x, y), 2, (255, 255, 255), -1)
            cv2.putText(imagen_resultado, f"Senal {i+1}: {tipo}", (x-50, y-r-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        resultado = {
            'metodo': 'Combinado (Hough + FREAK + Color + LoG)',
            'num_senales': len(senales_finales),
            'senales': senales_finales,
            'detecciones_hough': len(resultado_hough['senales']),
            'detecciones_freak': len(resultado_freak['senales']),
            'detecciones_color': len(resultado_color['senales']),
            'detecciones_log': len(resultado_log['senales']),
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': (resultado_hough['confianza_promedio'] + 
                                 resultado_freak['confianza_promedio'] + 
                                 resultado_color['confianza_promedio'] + 
                                 resultado_log['confianza_promedio']) / 4
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Señales - Método Combinado")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "combinado")
        
        return resultado
    
    def _validar_senal_hough(self, imagen, x, y, r):
        """Valida si un círculo detectado es realmente una señal."""
        # Verificar que el círculo esté dentro de la imagen
        alto, ancho = imagen.shape[:2]
        if x - r < 0 or x + r >= ancho or y - r < 0 or y + r >= alto:
            return None
        
        # Crear máscara circular
        mask = np.zeros((alto, ancho), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Analizar color en HSV
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        region_hsv = hsv[mask == 255]
        
        if len(region_hsv) == 0:
            return None
        
        # Determinar tipo de señal por color dominante
        return self._determinar_tipo_por_color_region(region_hsv)
    
    def _determinar_tipo_por_color(self, imagen, x, y, r):
        """Determina el tipo de señal por análisis de color local."""
        # Crear máscara circular
        alto, ancho = imagen.shape[:2]
        mask = np.zeros((alto, ancho), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Analizar color en HSV
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        region_hsv = hsv[mask == 255]
        
        if len(region_hsv) == 0:
            return "Desconocida"
        
        return self._determinar_tipo_por_color_region(region_hsv)
    
    def _determinar_tipo_por_color_region(self, region_hsv):
        """Determina tipo de señal por región HSV."""
        # Calcular histograma de matiz
        hist_h = cv2.calcHist([region_hsv], [0], None, [180], [0, 180])
        
        # Encontrar picos en el histograma
        max_bin = np.argmax(hist_h)
        
        if max_bin < 10 or max_bin > 170:  # Rojo
            return "Prohibicion"
        elif 100 <= max_bin <= 130:  # Azul
            return "Informativa"
        elif 20 <= max_bin <= 30:  # Amarillo
            return "Advertencia"
        else:
            return "Desconocida"
    
    def _agrupar_keypoints_senales(self, keypoints):
        """Agrupa keypoints en regiones de señales potenciales."""
        if not keypoints or len(keypoints) < 3:
            return []
        
        # Convertir keypoints a coordenadas
        puntos = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        # Usar clustering simple
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=40, min_samples=3).fit(puntos)
            labels = clustering.labels_
        except ImportError:
            # Clustering manual simple si no hay sklearn
            return self._clustering_manual(puntos)
        
        senales = []
        for label in set(labels):
            if label == -1:  # Ruido
                continue
            
            cluster_points = puntos[labels == label]
            
            # Calcular centro y radio
            centro_x = np.mean(cluster_points[:, 0])
            centro_y = np.mean(cluster_points[:, 1])
            
            distancias = np.sqrt((cluster_points[:, 0] - centro_x)**2 + 
                               (cluster_points[:, 1] - centro_y)**2)
            radio = np.mean(distancias) * 1.5  # Factor de ajuste
            
            if 20 <= radio <= 100:
                senales.append((int(centro_x), int(centro_y), int(radio), "Detectada"))
        
        return senales
    
    def _clustering_manual(self, puntos):
        """Clustering manual simple cuando sklearn no está disponible."""
        senales = []
        visitados = np.zeros(len(puntos), dtype=bool)
        
        for i, punto in enumerate(puntos):
            if visitados[i]:
                continue
            
            # Encontrar puntos cercanos
            cluster = [punto]
            visitados[i] = True
            
            for j, otro_punto in enumerate(puntos):
                if not visitados[j]:
                    distancia = np.sqrt(np.sum((punto - otro_punto)**2))
                    if distancia < 40:
                        cluster.append(otro_punto)
                        visitados[j] = True
            
            if len(cluster) >= 3:
                cluster = np.array(cluster)
                centro_x = np.mean(cluster[:, 0])
                centro_y = np.mean(cluster[:, 1])
                
                distancias = np.sqrt((cluster[:, 0] - centro_x)**2 + 
                                   (cluster[:, 1] - centro_y)**2)
                radio = np.mean(distancias) * 1.5
                
                if 20 <= radio <= 100:
                    senales.append((int(centro_x), int(centro_y), int(radio), "Detectada"))
        
        return senales
    
    def _aplicar_nms_senales(self, senales, umbral_distancia=40):
        """Aplica Non-Maximum Suppression para eliminar detecciones duplicadas."""
        if not senales:
            return []
        
        senales_array = np.array([(s[0], s[1], s[2]) for s in senales])
        tipos = [s[3] if len(s) > 3 else "Detectada" for s in senales]
        
        indices_mantener = []
        
        for i in range(len(senales_array)):
            mantener = True
            for j in range(len(senales_array)):
                if i != j and j in indices_mantener:
                    distancia = np.sqrt((senales_array[i][0] - senales_array[j][0])**2 + 
                                      (senales_array[i][1] - senales_array[j][1])**2)
                    if distancia < umbral_distancia:
                        mantener = False
                        break
            
            if mantener:
                indices_mantener.append(i)
        
        return [(int(senales_array[i][0]), int(senales_array[i][1]), 
                int(senales_array[i][2]), tipos[i]) for i in indices_mantener]
    
    def _obtener_color_tipo_senal(self, tipo):
        """Obtiene color BGR para dibujar según tipo de señal."""
        colores = {
            'Prohibicion': (0, 0, 255),      # Rojo
            'Informativa': (255, 0, 0),      # Azul
            'Advertencia': (0, 255, 255),    # Amarillo
            'Detectada': (0, 255, 0),        # Verde
            'Desconocida': (128, 128, 128)   # Gris
        }
        return colores.get(tipo, (0, 255, 0))
    
    def _calcular_confianza_hough(self, senales):
        """Calcula confianza promedio para detecciones Hough."""
        if not senales:
            return 0.0
        return min(1.0, len(senales) / 5.0)  # Normalizar por número esperado
    
    def _calcular_confianza_freak(self, keypoints, senales):
        """Calcula confianza para detecciones FREAK."""
        if not senales:
            return 0.0
        densidad_keypoints = len(keypoints) / max(1, len(senales))
        return min(1.0, densidad_keypoints / 20.0)
    
    def _calcular_confianza_color(self, senales):
        """Calcula confianza para detecciones por color."""
        if not senales:
            return 0.0
        
        # Si las señales tienen confianza individual (nuevo formato), usar esa
        if senales and len(senales[0]) > 4:
            confianzas_individuales = [s[4] for s in senales if len(s) > 4]
            if confianzas_individuales:
                return np.mean(confianzas_individuales)
        
        # Fallback al método anterior para compatibilidad
        return min(1.0, len(senales) / 3.0)
    
    def _calcular_confianza_log(self, senales, circularidad_promedio):
        """Calcula confianza para detecciones LoG."""
        if not senales:
            return 0.0
        return min(1.0, circularidad_promedio)
    
    def _mostrar_resultado(self, resultado, titulo):
        """Muestra resultado de la detección."""
        print(f"\nResultado - {titulo}")
        print(f"Método: {resultado['metodo']}")
        print(f"Señales detectadas: {resultado['num_senales']}")
        if 'confianza_promedio' in resultado:
            print(f"Confianza promedio: {resultado['confianza_promedio']:.3f}")
        
        # Mostrar detalles de señales
        if resultado['senales']:
            print("Detalles de señales:")
            for i, senal in enumerate(resultado['senales']):
                if len(senal) >= 4:
                    x, y, r, tipo = senal[:4]
                    print(f"  Señal {i+1}: Centro({x}, {y}), Radio={r}, Tipo={tipo}")
        
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
        nombre_archivo = f"senales_{metodo}_{timestamp}.jpg"
        
        if ruta_base:
            directorio = os.path.dirname(ruta_base)
            ruta_completa = os.path.join(directorio, nombre_archivo)
        else:
            ruta_completa = nombre_archivo
        
        os.makedirs(os.path.dirname(ruta_completa), exist_ok=True)
        cv2.imwrite(ruta_completa, imagen_resultado)
        print(f"Resultado guardado: {ruta_completa}")

# Función de utilidad
def detectar_senales_imagen(ruta_imagen, metodo='combinado', visualizar=True, guardar=False, ruta_salida=None):
    """
    Función de conveniencia para detectar señales en una imagen.
    
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
        print(f"❌ Error: No se pudo cargar la imagen: {ruta_imagen}")
        return None
    
    # Crear detector y ejecutar
    detector = DetectorSenales()
    return detector.detectar_senales(imagen, metodo, visualizar, guardar, ruta_salida)