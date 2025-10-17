#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de Señales de Tráfico en Imágenes Vehiculares - Versión Robusta
==========================================================================

Sistema especializado para detectar señales de tráfico circulares utilizando
configuraciones robustas que combinan múltiples algoritmos de visión por computadora.

Configuraciones disponibles:
- CONFIG_PRECISION_ALTA: Hough multiescala + Color HSV + Validación morfológica
- CONFIG_ROBUSTA: Contornos circulares + Color + Validación geométrica avanzada
- CONFIG_ADAPTATIVA: Hough adaptativo + Análisis textural + Color multirrango
- CONFIG_BALANCED: Combinación equilibrada de Hough + Contornos + Color
"""

import cv2
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

class DetectorSenales:
    """Detector especializado de señales de tráfico con múltiples formas y configuraciones."""
    
    def __init__(self):
        """Inicializar el detector de señales con configuraciones robustas."""
        
        # Configuraciones por forma de señal
        self.formas_senales = {
            'CIRCULAR': {
                'nombre': 'Señales Circulares',
                'descripcion': 'Detecta señales de forma circular (prohibición, obligación)',
                'metodos': ['hough_circles', 'contornos_circulares', 'color_hsv'],
                'colores_tipicos': ['rojo', 'azul', 'amarillo'],
                'validacion_forma': 'circular'
            },
            'RECTANGULAR': {
                'nombre': 'Señales Rectangulares',
                'descripcion': 'Detecta señales de forma rectangular (informativas)',
                'metodos': ['contornos_rectangulares', 'hough_lines', 'color_hsv'],
                'colores_tipicos': ['azul', 'verde', 'blanco'],
                'validacion_forma': 'rectangular'
            },
            'TRIANGULAR': {
                'nombre': 'Señales Triangulares',
                'descripcion': 'Detecta señales de forma triangular (advertencia)',
                'metodos': ['contornos_triangulares', 'deteccion_vertices', 'color_hsv'],
                'colores_tipicos': ['amarillo', 'rojo', 'blanco'],
                'validacion_forma': 'triangular'
            },
            'OCTAGONAL': {
                'nombre': 'Señales Octagonales',
                'descripcion': 'Detecta señales de forma octagonal (STOP)',
                'metodos': ['contornos_octagonales', 'deteccion_vertices', 'color_hsv'],
                'colores_tipicos': ['rojo', 'blanco'],
                'validacion_forma': 'octagonal'
            },
            'TODAS': {
                'nombre': 'Todas las Formas',
                'descripcion': 'Detecta señales de todas las formas disponibles',
                'metodos': ['deteccion_combinada'],
                'colores_tipicos': ['rojo', 'azul', 'amarillo', 'verde', 'blanco'],
                'validacion_forma': 'todas'
            }
        }
        
        self.configuraciones = {
            'CONFIG_PRECISION_ALTA': {
                'nombre': 'Precisión Alta',
                'descripcion': 'Hough multiescala + Color HSV + Validación morfológica estricta',
                'hough_configs': [
                    {'dp': 1.0, 'min_dist': 60, 'param1': 50, 'param2': 25, 'min_r': 15, 'max_r': 120},
                    {'dp': 1.2, 'min_dist': 50, 'param1': 40, 'param2': 30, 'min_r': 20, 'max_r': 100},
                    {'dp': 1.5, 'min_dist': 70, 'param1': 60, 'param2': 20, 'min_r': 25, 'max_r': 150}
                ],
                'color_weight': 0.5,
                'shape_weight': 0.3,
                'texture_weight': 0.2,
                'min_circularidad': 0.75,
                'nms_threshold': 0.4,
                'confianza_minima': 0.65
            },
            'CONFIG_ROBUSTA': {
                'nombre': 'Robusta',
                'descripcion': 'Contornos circulares + Color multicanal + Validación geométrica',
                'contornos_configs': [
                    {'min_area': 300, 'max_area': 50000, 'circularidad': 0.65},
                    {'min_area': 500, 'max_area': 40000, 'circularidad': 0.70},
                    {'min_area': 200, 'max_area': 60000, 'circularidad': 0.60}
                ],
                'color_weight': 0.45,
                'shape_weight': 0.35,
                'texture_weight': 0.20,
                'min_circularidad': 0.65,
                'nms_threshold': 0.35,
                'confianza_minima': 0.60
            },
            'CONFIG_ADAPTATIVA': {
                'nombre': 'Adaptativa',
                'descripcion': 'Hough adaptativo + Análisis textural + Color multirrango',
                'hough_configs': [
                    {'dp': 1.0, 'min_dist': 50, 'param1': 45, 'param2': 28, 'min_r': 18, 'max_r': 130},
                    {'dp': 1.3, 'min_dist': 60, 'param1': 55, 'param2': 22, 'min_r': 22, 'max_r': 110}
                ],
                'contornos_configs': [
                    {'min_area': 400, 'max_area': 45000, 'circularidad': 0.68}
                ],
                'color_weight': 0.4,
                'shape_weight': 0.3,
                'texture_weight': 0.3,
                'min_circularidad': 0.68,
                'nms_threshold': 0.38,
                'confianza_minima': 0.62
            },
            'CONFIG_BALANCED': {
                'nombre': 'Equilibrada',
                'descripcion': 'Combinación equilibrada Hough + Contornos + Color',
                'hough_configs': [
                    {'dp': 1.0, 'min_dist': 55, 'param1': 50, 'param2': 27, 'min_r': 20, 'max_r': 125}
                ],
                'contornos_configs': [
                    {'min_area': 350, 'max_area': 48000, 'circularidad': 0.67}
                ],
                'color_weight': 0.4,
                'shape_weight': 0.35,
                'texture_weight': 0.25,
                'min_circularidad': 0.67,
                'nms_threshold': 0.37,
                'confianza_minima': 0.63
            }
        }
        
        # Rangos de color HSV para señales de tráfico
        self.color_ranges = {
            'rojo': [
                {'bajo': np.array([0, 100, 100]), 'alto': np.array([10, 255, 255])},
                {'bajo': np.array([170, 100, 100]), 'alto': np.array([180, 255, 255])}
            ],
            'azul': [
                {'bajo': np.array([100, 100, 50]), 'alto': np.array([130, 255, 255])}
            ],
            'amarillo': [
                {'bajo': np.array([20, 100, 100]), 'alto': np.array([30, 255, 255])}
            ],
            'blanco': [
                {'bajo': np.array([0, 0, 200]), 'alto': np.array([180, 30, 255])}
            ]
        }
    
    def detectar_senales(self, imagen, configuracion='CONFIG_BALANCED', forma='CIRCULAR', visualizar=True, guardar=False, ruta_salida=None):
        """
        Detecta señales de tráfico usando una configuración robusta y forma específica.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            configuracion (str): Configuración a usar:
                - 'CONFIG_PRECISION_ALTA': Máxima precisión con multiescala
                - 'CONFIG_ROBUSTA': Robusta con validación geométrica
                - 'CONFIG_ADAPTATIVA': Adaptativa con análisis textural
                - 'CONFIG_BALANCED': Equilibrada (recomendada)
            forma (str): Forma de señal a detectar:
                - 'CIRCULAR': Señales circulares
                - 'RECTANGULAR': Señales rectangulares
                - 'TRIANGULAR': Señales triangulares
                - 'OCTAGONAL': Señales octagonales
                - 'TODAS': Todas las formas
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imagen resultado
            ruta_salida (str): Ruta donde guardar resultado
            
        Returns:
            dict: Resultados detallados de la detección
        """
        if configuracion not in self.configuraciones:
            print(f"Configuración no válida: {configuracion}. Usando CONFIG_BALANCED")
            configuracion = 'CONFIG_BALANCED'
        
        if forma not in self.formas_senales:
            print(f"Forma no válida: {forma}. Usando CIRCULAR")
            forma = 'CIRCULAR'
        
        config = self.configuraciones[configuracion]
        forma_config = self.formas_senales[forma]
        
        print(f"\nDetectando señales de tráfico")
        print(f"Forma: {forma_config['nombre']}")
        print(f"Configuración: {config['nombre']}")
        print(f"Descripción: {forma_config['descripcion']}")
        
        # Preprocesamiento común
        imagen_procesada = self._preprocesar_imagen(imagen)
        
        # Detectar candidatos según forma y configuración
        candidatos_hough = []
        candidatos_contornos = []
        candidatos_forma = []
        
        # Aplicar detección específica por forma
        if forma == 'CIRCULAR':
            candidatos_forma = self._detectar_senales_circulares(imagen_procesada, config)
        elif forma == 'RECTANGULAR':
            candidatos_forma = self._detectar_senales_rectangulares(imagen_procesada, config)
        elif forma == 'TRIANGULAR':
            candidatos_forma = self._detectar_senales_triangulares(imagen_procesada, config)
        elif forma == 'OCTAGONAL':
            candidatos_forma = self._detectar_senales_octagonales(imagen_procesada, config)
        elif forma == 'TODAS':
            candidatos_forma = self._detectar_todas_formas(imagen_procesada, config)
        
        print(f"Candidatos por forma ({forma}): {len(candidatos_forma)}")
        
        # Aplicar Hough si la configuración lo incluye (para formas circulares)
        if 'hough_configs' in config and forma in ['CIRCULAR', 'TODAS']:
            candidatos_hough = self._detectar_hough_multiescala(
                imagen_procesada['gris'],
                config['hough_configs']
            )
            print(f"Candidatos Hough adicionales: {len(candidatos_hough)}")
        
        # Aplicar detección por contornos si la configuración lo incluye
        if 'contornos_configs' in config:
            candidatos_contornos = self._detectar_contornos_por_forma(
                imagen_procesada,
                config['contornos_configs'],
                forma
            )
            print(f"Candidatos Contornos: {len(candidatos_contornos)}")
        
        # Fusionar candidatos
        todos_candidatos = candidatos_forma + candidatos_hough + candidatos_contornos
        print(f"Total candidatos: {len(todos_candidatos)}")
        
        # Validar y puntuar candidatos
        senales_validadas = self._validar_candidatos(
            imagen,
            imagen_procesada,
            todos_candidatos,
            config
        )
        
        # Aplicar NMS (Non-Maximum Suppression)
        senales_finales = self._aplicar_nms_senales(
            senales_validadas,
            config['nms_threshold']
        )
        
        # Filtrar por confianza mínima
        senales_finales = [s for s in senales_finales if s[4] >= config['confianza_minima']]
        
        print(f"Señales detectadas finales: {len(senales_finales)}")
        
        # Crear imagen resultado
        imagen_resultado = self._dibujar_resultados(imagen.copy(), senales_finales)
        
        # Preparar resultado
        resultado = {
            'metodo': f"{config['nombre']} - {config['descripcion']}",
            'configuracion': configuracion,
            'num_senales': len(senales_finales),
            'senales_detectadas': senales_finales,
            'senales': senales_finales,
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': np.mean([s[4] for s in senales_finales]) if senales_finales else 0.0,
            'candidatos_hough': len(candidatos_hough),
            'candidatos_contornos': len(candidatos_contornos),
            'estadisticas': self._calcular_estadisticas(senales_finales),
            'mask': imagen_procesada.get('mask_color', None)
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, f"Detección: {config['nombre']}")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, configuracion)
        
        return resultado
    
    def _preprocesar_imagen(self, imagen):
        """
        Preprocesamiento robusto de la imagen.
        
        Returns:
            dict: Diccionario con múltiples versiones procesadas de la imagen
        """
        # Escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Ecualización adaptativa
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gris_eq = clahe.apply(gris)
        
        # Filtrado gaussiano con múltiples escalas
        blur_suave = cv2.GaussianBlur(gris_eq, (5, 5), 1.5)
        blur_medio = cv2.GaussianBlur(gris_eq, (9, 9), 2.0)
        blur_fuerte = cv2.GaussianBlur(gris_eq, (13, 13), 2.5)
        
        # Convertir a HSV para análisis de color
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Crear máscara de colores de señales
        mask_color = self._crear_mascara_colores(hsv)
        
        # Detección de bordes multi-método
        edges_canny = cv2.Canny(blur_medio, 50, 150)
        
        # Aplicar operaciones morfológicas a la máscara de color
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_color_morph = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel_ellipse, iterations=2)
        mask_color_morph = cv2.morphologyEx(mask_color_morph, cv2.MORPH_OPEN, kernel_ellipse, iterations=1)
        
        return {
            'original': imagen,
            'gris': blur_medio,
            'gris_eq': gris_eq,
            'blur_suave': blur_suave,
            'blur_medio': blur_medio,
            'blur_fuerte': blur_fuerte,
            'hsv': hsv,
            'mask_color': mask_color,
            'mask_color_morph': mask_color_morph,
            'edges': edges_canny
        }
    
    def _crear_mascara_colores(self, hsv):
        """
        Crea una máscara combinada de todos los colores de señales de tráfico.
        """
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        try:
            for color, rangos in self.color_ranges.items():
                for rango in rangos:
                    # Asegurar que los rangos sean arrays de NumPy con el tipo correcto
                    bajo = np.array(rango['bajo'], dtype=np.uint8)
                    alto = np.array(rango['alto'], dtype=np.uint8)
                    
                    # Verificar que los arrays tengan la dimensión correcta
                    if len(bajo) != 3 or len(alto) != 3:
                        print(f"Warning: Rango de color {color} tiene dimensión incorrecta")
                        continue
                    
                    mask_temp = cv2.inRange(hsv, bajo, alto)
                    mask_total = cv2.bitwise_or(mask_total, mask_temp)
        except Exception as e:
            print(f"Error en _crear_mascara_colores: {e}")
            # Retornar máscara vacía en caso de error
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        return mask_total
    
    def _detectar_hough_multiescala(self, imagen_gris, configs):
        """
        Detecta círculos usando Transformada de Hough con múltiples configuraciones.
        
        Args:
            imagen_gris: Imagen en escala de grises
            configs: Lista de configuraciones de Hough
            
        Returns:
            Lista de candidatos (x, y, radio)
        """
        candidatos = []
        
        for config in configs:
            circulos = cv2.HoughCircles(
                imagen_gris,
                cv2.HOUGH_GRADIENT,
                dp=config['dp'],
                minDist=config['min_dist'],
                param1=config['param1'],
                param2=config['param2'],
                minRadius=config['min_r'],
                maxRadius=config['max_r']
            )
            
            if circulos is not None:
                circulos = np.uint16(np.around(circulos))
                for circulo in circulos[0, :]:
                    x, y, r = int(circulo[0]), int(circulo[1]), int(circulo[2])
                    candidatos.append((x, y, r, 'hough'))
        
        return candidatos
    
    def _detectar_contornos_circulares(self, imagen_procesada, configs):
        """
        Detecta círculos usando análisis de contornos con validación geométrica.
        Basado en el método robusto de analisis_circulos.py
        
        Args:
            imagen_procesada: Diccionario con imágenes procesadas
            configs: Lista de configuraciones para contornos
            
        Returns:
            Lista de candidatos (x, y, radio)
        """
        candidatos = []
        
        # Usar la máscara de color morfológica para encontrar contornos
        mask = imagen_procesada['mask_color_morph']
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for config in configs:
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                
                # Filtrar por área
                if area < config['min_area'] or area > config['max_area']:
                    continue
                
                # Calcular perímetro y circularidad
                perimetro = cv2.arcLength(contorno, True)
                if perimetro == 0:
                    continue
                
                circularidad = 4 * np.pi * area / (perimetro ** 2)
                
                # Validar circularidad
                if circularidad >= config['circularidad']:
                    # Calcular círculo mínimo envolvente
                    (x, y), radio = cv2.minEnclosingCircle(contorno)
                    
                    # Validación adicional: comparar área del contorno con área del círculo
                    area_circulo = np.pi * radio * radio
                    ratio_area = area / area_circulo if area_circulo > 0 else 0
                    
                    # Si el ratio está entre 0.6 y 1.0, es un buen candidato circular
                    if 0.6 <= ratio_area <= 1.0:
                        candidatos.append((int(x), int(y), int(radio), 'contorno'))
        
        return candidatos
    
    def _detectar_senales_circulares(self, imagen_procesada, config):
        """Detecta señales circulares específicamente."""
        candidatos = []
        
        # Usar Hough para círculos
        gris = imagen_procesada['gris']
        circulos = cv2.HoughCircles(
            gris,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=80,
            param2=30,
            minRadius=15,
            maxRadius=150
        )
        
        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            for circulo in circulos[0, :]:
                x, y, r = int(circulo[0]), int(circulo[1]), int(circulo[2])
                candidatos.append((x, y, r, 'circular'))
        
        return candidatos
    
    def _detectar_senales_rectangulares(self, imagen_procesada, config):
        """Detecta señales rectangulares específicamente."""
        candidatos = []
        
        # Usar detección de contornos rectangulares
        gris = imagen_procesada['gris']
        edges = cv2.Canny(gris, 50, 150)
        
        contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contorno in contornos:
            # Aproximar contorno a polígono
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)
            
            # Verificar si es rectangular (4 vértices)
            if len(approx) == 4:
                area = cv2.contourArea(contorno)
                if 500 < area < 50000:  # Filtrar por tamaño
                    x, y, w, h = cv2.boundingRect(contorno)
                    candidatos.append((x + w//2, y + h//2, max(w, h)//2, 'rectangular'))
        
        return candidatos
    
    def _detectar_senales_triangulares(self, imagen_procesada, config):
        """Detecta señales triangulares específicamente."""
        candidatos = []
        
        # Usar detección de contornos triangulares
        gris = imagen_procesada['gris']
        edges = cv2.Canny(gris, 50, 150)
        
        contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contorno in contornos:
            # Aproximar contorno a polígono
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)
            
            # Verificar si es triangular (3 vértices)
            if len(approx) == 3:
                area = cv2.contourArea(contorno)
                if 500 < area < 50000:  # Filtrar por tamaño
                    x, y, w, h = cv2.boundingRect(contorno)
                    candidatos.append((x + w//2, y + h//2, max(w, h)//2, 'triangular'))
        
        return candidatos
    
    def _detectar_senales_octagonales(self, imagen_procesada, config):
        """Detecta señales octagonales específicamente (como STOP)."""
        candidatos = []
        
        # Usar detección de contornos octagonales
        gris = imagen_procesada['gris']
        edges = cv2.Canny(gris, 50, 150)
        
        contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contorno in contornos:
            # Aproximar contorno a polígono
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)
            
            # Verificar si es octagonal (8 vértices)
            if len(approx) == 8:
                area = cv2.contourArea(contorno)
                if 1000 < area < 50000:  # Filtrar por tamaño
                    x, y, w, h = cv2.boundingRect(contorno)
                    candidatos.append((x + w//2, y + h//2, max(w, h)//2, 'octagonal'))
        
        return candidatos
    
    def _detectar_todas_formas(self, imagen_procesada, config):
        """Detecta señales de todas las formas."""
        candidatos = []
        
        # Combinar detección de todas las formas
        candidatos.extend(self._detectar_senales_circulares(imagen_procesada, config))
        candidatos.extend(self._detectar_senales_rectangulares(imagen_procesada, config))
        candidatos.extend(self._detectar_senales_triangulares(imagen_procesada, config))
        candidatos.extend(self._detectar_senales_octagonales(imagen_procesada, config))
        
        return candidatos
    
    def _detectar_contornos_por_forma(self, imagen_procesada, configs, forma):
        """Detecta contornos adaptados a la forma específica."""
        if forma == 'CIRCULAR':
            return self._detectar_contornos_circulares(imagen_procesada, configs)
        else:
            # Para otras formas, usar método general
            candidatos = []
            mask = imagen_procesada['mask_color_morph']
            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for config in configs:
                for contorno in contornos:
                    area = cv2.contourArea(contorno)
                    if area < config['min_area'] or area > config['max_area']:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contorno)
                    candidatos.append((x + w//2, y + h//2, max(w, h)//2, forma.lower()))
            
            return candidatos
    
    def _validar_candidatos(self, imagen, imagen_procesada, candidatos, config):
        """
        Valida y puntúa cada candidato usando múltiples criterios.
        
        Returns:
            Lista de señales validadas con formato: (x, y, radio, tipo, confianza)
        """
        senales_validadas = []
        hsv = imagen_procesada['hsv']
        alto, ancho = imagen.shape[:2]
        
        for candidato in candidatos:
            x, y, radio, origen = candidato
            
            # Verificar que el círculo esté dentro de la imagen
            if x - radio < 0 or x + radio >= ancho or y - radio < 0 or y + radio >= alto:
                continue
            
            # Crear máscara circular
            mask = np.zeros((alto, ancho), dtype=np.uint8)
            cv2.circle(mask, (x, y), radio, 255, -1)
            
            # Calcular puntuaciones
            score_forma = self._calcular_score_forma(imagen_procesada, x, y, radio, mask)
            score_color = self._calcular_score_color(hsv, mask)
            score_textura = self._calcular_score_textura(imagen_procesada['gris'], mask)
            
            # Calcular confianza ponderada
            confianza = (
                score_forma * config['shape_weight'] +
                score_color['score'] * config['color_weight'] +
                score_textura * config['texture_weight']
            )
            
            # Determinar tipo de señal por color dominante
            tipo_senal = score_color['tipo']
            
            senales_validadas.append((x, y, radio, tipo_senal, confianza))
        
        return senales_validadas
    
    def _calcular_score_forma(self, imagen_procesada, x, y, radio, mask):
        """
        Calcula un score basado en la forma circular del candidato.
        """
        # Extraer región de bordes
        edges = imagen_procesada['edges']
        bordes_region = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Calcular proporción de píxeles de borde
        pixels_borde = np.count_nonzero(bordes_region)
        perimetro_ideal = 2 * np.pi * radio
        
        # Score basado en cuántos píxeles de borde hay vs lo esperado
        score = min(1.0, pixels_borde / (perimetro_ideal * 0.5))
        
        return score
    
    def _calcular_score_color(self, hsv, mask):
        """
        Calcula score basado en colores típicos de señales de tráfico.
        
        Returns:
            dict con 'score' y 'tipo' de señal
        """
        try:
            # Extraer región de la imagen HSV usando la máscara
            region_hsv = hsv[mask == 255]
            
            if len(region_hsv) == 0:
                return {'score': 0.0, 'tipo': 'Desconocida'}
        except Exception as e:
            print(f"Error extrayendo región HSV: {e}")
            return {'score': 0.0, 'tipo': 'Desconocida'}
        
        max_score = 0.0
        tipo_detectado = 'Desconocida'
        tipos_senal = {
            'rojo': 'Prohibicion',
            'azul': 'Informativa',
            'amarillo': 'Advertencia'
        }
        
        for color, rangos in self.color_ranges.items():
            if color == 'blanco':  # El blanco es fondo, no el color principal
                continue
            
            pixels_color = 0
            for rango in rangos:
                try:
                    # Asegurar que los rangos sean arrays de NumPy con el tipo correcto
                    bajo = np.array(rango['bajo'], dtype=np.uint8)
                    alto = np.array(rango['alto'], dtype=np.uint8)
                    
                    # Verificar dimensiones
                    if len(bajo) != 3 or len(alto) != 3:
                        continue
                    
                    # Contar píxeles que están dentro del rango de color
                    # region_hsv es un array de forma (N, 3) donde N es el número de píxeles
                    for pixel in region_hsv:
                        if (np.all(pixel >= bajo) and np.all(pixel <= alto)):
                            pixels_color += 1
                    
                except Exception as e:
                    print(f"Error procesando color {color}: {e}")
                    continue
            
            # Calcular proporción de píxeles de este color
            score = pixels_color / len(region_hsv)
            
            if score > max_score:
                max_score = score
                tipo_detectado = tipos_senal.get(color, 'Detectada')
        
        # Validar que tenga suficiente color característico
        if max_score < 0.15:  # Al menos 15% del área debe ser color característico
            return {'score': 0.3, 'tipo': 'Detectada'}
        
        return {'score': min(1.0, max_score * 2), 'tipo': tipo_detectado}
    
    def _calcular_score_textura(self, imagen_gris, mask):
        """
        Calcula score basado en características texturales.
        Las señales suelen tener bordes definidos y patrones internos.
        """
        region = imagen_gris[mask == 255]
        
        if len(region) == 0:
            return 0.0
        
        # Calcular varianza como indicador de textura
        varianza = np.var(region)
        
        # Las señales tienen varianza moderada (no completamente uniforme ni muy ruidoso)
        # Normalizar varianza esperada entre 200 y 2000
        score = 0.0
        if 200 <= varianza <= 2000:
            score = 1.0
        elif varianza < 200:
            score = varianza / 200.0
        else:
            score = max(0.0, 1.0 - (varianza - 2000) / 3000.0)
        
        return score
    
    def _aplicar_nms_senales(self, senales, umbral_iou):
        """
        Aplica Non-Maximum Suppression basado en IoU (Intersection over Union).
        
        Args:
            senales: Lista de tuplas (x, y, radio, tipo, confianza)
            umbral_iou: Umbral de IoU para considerar duplicados
            
        Returns:
            Lista de señales filtradas
        """
        if not senales:
            return []
        
        # Ordenar por confianza (descendente)
        senales_ordenadas = sorted(senales, key=lambda s: s[4], reverse=True)
        
        senales_finales = []
        
        for senal_actual in senales_ordenadas:
            x1, y1, r1, tipo1, conf1 = senal_actual
            
            # Verificar si hay overlap con señales ya aceptadas
            mantener = True
            for senal_aceptada in senales_finales:
                x2, y2, r2, tipo2, conf2 = senal_aceptada
                
                # Calcular distancia entre centros
                distancia = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                # Si están muy cerca (menos que la suma de radios), verificar IoU
                if distancia < (r1 + r2):
                    # Calcular IoU aproximado para círculos
                    iou = self._calcular_iou_circulos(x1, y1, r1, x2, y2, r2)
                    
                    if iou > umbral_iou:
                        mantener = False
                        break
            
            if mantener:
                senales_finales.append(senal_actual)
        
        return senales_finales
    
    def _calcular_iou_circulos(self, x1, y1, r1, x2, y2, r2):
        """
        Calcula IoU aproximado entre dos círculos.
        """
        # Calcular distancia entre centros
        d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Si no hay overlap
        if d >= (r1 + r2):
            return 0.0
        
        # Si un círculo está completamente dentro del otro
        if d <= abs(r1 - r2):
            r_min = min(r1, r2)
            r_max = max(r1, r2)
            return (r_min * r_min) / (r_max * r_max)
        
        # Cálculo aproximado de IoU usando áreas
        area1 = np.pi * r1 * r1
        area2 = np.pi * r2 * r2
        
        # Aproximación: si la distancia es menor que el promedio de radios, hay overlap significativo
        if d < (r1 + r2) / 2:
            overlap_ratio = 1.0 - (d / (r1 + r2))
            area_overlap = min(area1, area2) * overlap_ratio
            area_union = area1 + area2 - area_overlap
            return area_overlap / area_union
        
        return 0.3  # Overlap pequeño
    
    def _dibujar_resultados(self, imagen, senales):
        """
        Dibuja las señales detectadas en la imagen.
        """
        for i, senal in enumerate(senales):
            x, y, radio, tipo, confianza = senal
            
            # Obtener color según tipo
            color = self._obtener_color_tipo_senal(tipo)
            
            # Dibujar círculo
            cv2.circle(imagen, (x, y), radio, color, 3)
            cv2.circle(imagen, (x, y), 2, (255, 255, 255), -1)
            
            # Dibujar etiqueta
            label = f"{tipo} {confianza:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Fondo para el texto
            cv2.rectangle(imagen, (x - w//2 - 5, y - radio - h - 15), 
                         (x + w//2 + 5, y - radio - 5), color, -1)
            
            # Texto
            cv2.putText(imagen, label, (x - w//2, y - radio - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return imagen
    
    def _obtener_color_tipo_senal(self, tipo):
        """Obtiene color BGR para dibujar según tipo de señal."""
        colores = {
            'Prohibicion': (0, 0, 255),
            'Informativa': (255, 0, 0),
            'Advertencia': (0, 255, 255),
            'Detectada': (0, 255, 0),
            'Desconocida': (128, 128, 128)
        }
        return colores.get(tipo, (0, 255, 0))
    
    def _calcular_estadisticas(self, senales):
        """
        Calcula estadísticas detalladas de las señales detectadas.
        """
        if not senales:
            return {
                'total': 0,
                'por_tipo': {},
                'radios': {'promedio': 0, 'min': 0, 'max': 0},
                'confianzas': {'promedio': 0, 'min': 0, 'max': 0}
            }
        
        # Contar por tipo
        tipos = {}
        radios = []
        confianzas = []
        
        for senal in senales:
            x, y, radio, tipo, confianza = senal
            tipos[tipo] = tipos.get(tipo, 0) + 1
            radios.append(radio)
            confianzas.append(confianza)
        
        return {
            'total': len(senales),
            'por_tipo': tipos,
            'radios': {
                'promedio': np.mean(radios),
                'min': np.min(radios),
                'max': np.max(radios)
            },
            'confianzas': {
                'promedio': np.mean(confianzas),
                'min': np.min(confianzas),
                'max': np.max(confianzas)
            }
        }
    
    def _mostrar_resultado(self, resultado, titulo):
        """Muestra resultado de la detección."""
        print(f"\nResultado - {titulo}")
        print(f"Método: {resultado['metodo']}")
        print(f"Señales detectadas: {resultado['num_senales']}")
        print(f"Confianza promedio: {resultado['confianza_promedio']:.3f}")
        
        # Mostrar estadísticas por tipo
        if 'estadisticas' in resultado and resultado['estadisticas']['total'] > 0:
            print("\nEstadísticas por tipo:")
            for tipo, cantidad in resultado['estadisticas']['por_tipo'].items():
                print(f"  {tipo}: {cantidad}")
        
        # Mostrar imagen resultado
        plt.figure(figsize=(14, 10))
        imagen_rgb = cv2.cvtColor(resultado['imagen_resultado'], cv2.COLOR_BGR2RGB)
        plt.imshow(imagen_rgb)
        plt.title(titulo)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _guardar_resultado(self, imagen_resultado, ruta_base, configuracion):
        """Guarda imagen resultado."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"senales_{configuracion}_{timestamp}.jpg"
        
        if os.path.isdir(ruta_base):
            ruta_completa = os.path.join(ruta_base, 'senales', nombre_archivo)
        else:
            directorio = os.path.dirname(ruta_base)
            ruta_completa = os.path.join(directorio, 'senales', nombre_archivo)
        
        os.makedirs(os.path.dirname(ruta_completa), exist_ok=True)
        cv2.imwrite(ruta_completa, imagen_resultado)
        print(f"Resultado guardado: {ruta_completa}")


# Función de utilidad
def detectar_senales_imagen(ruta_imagen, configuracion='CONFIG_BALANCED', forma='CIRCULAR', visualizar=True, guardar=False, ruta_salida=None):
    """
    Función de conveniencia para detectar señales en una imagen.
    
    Args:
        ruta_imagen (str): Ruta de la imagen
        configuracion (str): Configuración a usar
        forma (str): Forma de señal a detectar
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
    detector = DetectorSenales()
    return detector.detectar_senales(imagen, configuracion, forma, visualizar, guardar, ruta_salida)
