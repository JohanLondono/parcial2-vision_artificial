#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de Llantas en Imagenes de Trafico Vehicular
===================================================

Sistema especializado para detectar llantas de vehiculos utilizando
configuraciones robustas que combinan multiples algoritmos de vision
por computadora.

Configuraciones disponibles:
- CONFIG_PRECISION_ALTA: Hough multiescala + Contornos + Color + AKAZE
- CONFIG_ROBUSTA: Contornos circulares + Textura + Color + Validacion geometrica
- CONFIG_ADAPTATIVA: Hough adaptativo + Analisis textural + Color multirrango
- CONFIG_BALANCED: Combinacion equilibrada (recomendada)
"""

import cv2
import numpy as np
import os
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

class DetectorLlantas:
    """Detector especializado de llantas en vehiculos con configuraciones robustas."""
    
    def __init__(self):
        """Inicializar el detector de llantas con configuraciones robustas."""
        
        self.CONFIG_PRECISION_ALTA = {
            'nombre': 'Precision Alta',
            'descripcion': 'Hough multiescala + Contornos + Color negro/gris + AKAZE',
            'algoritmos': ['hough_multiescala', 'contornos', 'color', 'akaze'],
            'pesos': {'forma': 0.35, 'color': 0.30, 'textura': 0.20, 'keypoints': 0.15},
            'umbral_validacion': 0.55,
            'hough': {
                'escalas': [1.0, 1.2, 1.5],
                'dp': 1,
                'min_dist': 50,
                'param1': 100,
                'param2': 30,
                'min_radius': 15,
                'max_radius': 200
            },
            'contornos': {
                'min_area': 500,
                'max_area': 50000,
                'circularidad_min': 0.6,
                'metodo': cv2.RETR_EXTERNAL
            },
            'color': {
                'negro_bajo': [0, 0, 0],
                'negro_alto': [180, 255, 80],
                'gris_bajo': [0, 0, 50],
                'gris_alto': [180, 50, 200]
            },
            'akaze': {
                'threshold': 0.003,
                'nOctaves': 4,
                'nOctaveLayers': 4,
                'diffusivity': cv2.KAZE_DIFF_PM_G2
            }
        }
        
        self.CONFIG_ROBUSTA = {
            'nombre': 'Robusta',
            'descripcion': 'Contornos circulares + Textura GLCM + Color + Validacion geometrica avanzada',
            'algoritmos': ['contornos_circulares', 'textura', 'color', 'geometria'],
            'pesos': {'forma': 0.40, 'color': 0.30, 'textura': 0.25, 'geometria': 0.05},
            'umbral_validacion': 0.60,
            'contornos': {
                'min_area': 500,
                'max_area': 50000,
                'circularidad_min': 0.65,
                'metodo': cv2.RETR_LIST,
                'umbral_adaptativo': True
            },
            'textura': {
                'distancias': [1, 2, 3],
                'angulos': [0, np.pi/4, np.pi/2, 3*np.pi/4],
                'contraste_min': 0.3,
                'energia_max': 0.3
            },
            'color': {
                'negro_bajo': [0, 0, 0],
                'negro_alto': [180, 255, 90],
                'gris_bajo': [0, 0, 40],
                'gris_alto': [180, 60, 210]
            },
            'geometria': {
                'ratio_min': 0.7,
                'ratio_max': 1.3
            }
        }
        
        self.CONFIG_ADAPTATIVA = {
            'nombre': 'Adaptativa',
            'descripcion': 'Hough adaptativo + Analisis textural + Color multirrango + Morfologia',
            'algoritmos': ['hough_adaptativo', 'textura', 'color_multirrango', 'morfologia'],
            'pesos': {'forma': 0.30, 'color': 0.30, 'textura': 0.30, 'morfologia': 0.10},
            'umbral_validacion': 0.50,
            'hough': {
                'dp': 1,
                'min_dist': 40,
                'param1': 50,
                'param2': 25,
                'min_radius': 20,
                'max_radius': 180,
                'adaptativo': True
            },
            'textura': {
                'distancias': [1, 2],
                'angulos': [0, np.pi/4],
                'contraste_min': 0.25,
                'energia_max': 0.35
            },
            'color': {
                'rangos': [
                    {'bajo': [0, 0, 0], 'alto': [180, 255, 80]},
                    {'bajo': [0, 0, 50], 'alto': [180, 50, 200]},
                    {'bajo': [0, 0, 20], 'alto': [180, 100, 120]}
                ]
            },
            'morfologia': {
                'kernel_size': 5,
                'operaciones': ['close', 'open']
            }
        }
        
        self.CONFIG_BALANCED = {
            'nombre': 'Equilibrada',
            'descripcion': 'Combinacion equilibrada de todos los metodos (recomendada)',
            'algoritmos': ['hough_multiescala', 'contornos', 'color', 'textura'],
            'pesos': {'forma': 0.35, 'color': 0.30, 'textura': 0.25, 'consistencia': 0.10},
            'umbral_validacion': 0.55,
            'hough': {
                'escalas': [1.0, 1.3],
                'dp': 1,
                'min_dist': 45,
                'param1': 80,
                'param2': 28,
                'min_radius': 18,
                'max_radius': 190
            },
            'contornos': {
                'min_area': 500,
                'max_area': 50000,
                'circularidad_min': 0.6,
                'metodo': cv2.RETR_EXTERNAL
            },
            'color': {
                'negro_bajo': [0, 0, 0],
                'negro_alto': [180, 255, 85],
                'gris_bajo': [0, 0, 45],
                'gris_alto': [180, 55, 205]
            },
            'textura': {
                'distancias': [1, 2],
                'angulos': [0, np.pi/4, np.pi/2],
                'contraste_min': 0.28,
                'energia_max': 0.32
            }
        }
    
    def detectar_llantas(self, imagen, configuracion='CONFIG_BALANCED', visualizar=True, guardar=False, ruta_salida=None):
        """
        Detecta llantas en la imagen usando la configuracion especificada.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            configuracion (str): Configuracion a usar ('CONFIG_PRECISION_ALTA', 'CONFIG_ROBUSTA', 
                               'CONFIG_ADAPTATIVA', 'CONFIG_BALANCED')
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imagen resultado
            ruta_salida (str): Ruta donde guardar resultado
            
        Returns:
            dict: Resultados de la deteccion
        """
        print(f"Detectando llantas con configuracion: {configuracion}")
        
        if not hasattr(self, configuracion):
            print(f"Configuracion no reconocida: {configuracion}. Usando CONFIG_BALANCED por defecto")
            configuracion = 'CONFIG_BALANCED'
        
        config = getattr(self, configuracion)
        
        imagen_preprocesada = self._preprocesar_imagen(imagen, config)
        
        candidatos = []
        
        if 'hough_multiescala' in config['algoritmos']:
            candidatos.extend(self._detectar_hough_multiescala(imagen_preprocesada, config))
        
        if 'hough_adaptativo' in config['algoritmos']:
            candidatos.extend(self._detectar_hough_adaptativo(imagen_preprocesada, config))
        
        if 'contornos' in config['algoritmos'] or 'contornos_circulares' in config['algoritmos']:
            candidatos.extend(self._detectar_contornos_circulares(imagen_preprocesada, config))
        
        llantas_validadas = self._validar_candidatos(imagen, candidatos, config)
        
        llantas_finales = self._aplicar_nms_llantas(llantas_validadas, umbral_iou=0.3)
        
        imagen_resultado = self._dibujar_resultados(imagen.copy(), llantas_finales, config['nombre'])
        
        resultado = {
            'configuracion': config['nombre'],
            'num_llantas': len(llantas_finales),
            'llantas_detectadas': llantas_finales,
            'candidatos_iniciales': len(candidatos),
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': np.mean([llanta[3] for llanta in llantas_finales]) if llantas_finales else 0.0
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, f"Deteccion de Llantas - {config['nombre']}")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, configuracion.lower())
        
        return resultado
    
    def _preprocesar_imagen(self, imagen, config):
        """Preprocesa la imagen segun la configuracion."""
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        gris = cv2.GaussianBlur(gris, (5, 5), 0)
        
        if 'morfologia' in config.get('algoritmos', []):
            kernel_size = config.get('morfologia', {}).get('kernel_size', 5)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            operaciones = config.get('morfologia', {}).get('operaciones', ['close'])
            for op in operaciones:
                if op == 'close':
                    gris = cv2.morphologyEx(gris, cv2.MORPH_CLOSE, kernel, iterations=2)
                elif op == 'open':
                    gris = cv2.morphologyEx(gris, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return {'gris': gris, 'original': imagen}
    
    def _detectar_hough_multiescala(self, imagen_prep, config):
        """Detecta circulos usando Hough en multiples escalas."""
        candidatos = []
        gris = imagen_prep['gris']
        
        escalas = config.get('hough', {}).get('escalas', [1.0])
        params = config['hough']
        
        for escala in escalas:
            img_escalada = cv2.resize(gris, None, fx=escala, fy=escala)
            
            circulos = cv2.HoughCircles(
                img_escalada,
                cv2.HOUGH_GRADIENT,
                dp=params.get('dp', 1),
                minDist=int(params.get('min_dist', 50) * escala),
                param1=params.get('param1', 100),
                param2=params.get('param2', 30),
                minRadius=int(params.get('min_radius', 15) * escala),
                maxRadius=int(params.get('max_radius', 200) * escala)
            )
            
            if circulos is not None:
                circulos = np.round(circulos[0, :]).astype("int")
                for (x, y, r) in circulos:
                    x_orig = int(x / escala)
                    y_orig = int(y / escala)
                    r_orig = int(r / escala)
                    
                    candidatos.append({
                        'x': x_orig, 'y': y_orig, 'r': r_orig,
                        'metodo': 'hough_multiescala',
                        'escala': escala
                    })
        
        return candidatos
    
    def _detectar_hough_adaptativo(self, imagen_prep, config):
        """Detecta circulos usando Hough con parametros adaptativos."""
        candidatos = []
        gris = imagen_prep['gris']
        
        params = config.get('hough', {})
        
        media_brillo = np.mean(gris)
        
        if media_brillo < 80:
            param1 = params.get('param1', 50) - 10
            param2 = params.get('param2', 25) - 5
        elif media_brillo > 170:
            param1 = params.get('param1', 50) + 10
            param2 = params.get('param2', 25) + 5
        else:
            param1 = params.get('param1', 50)
            param2 = params.get('param2', 25)
        
        circulos = cv2.HoughCircles(
            gris,
            cv2.HOUGH_GRADIENT,
            dp=params.get('dp', 1),
            minDist=params.get('min_dist', 40),
            param1=param1,
            param2=param2,
            minRadius=params.get('min_radius', 20),
            maxRadius=params.get('max_radius', 180)
        )
        
        if circulos is not None:
            circulos = np.round(circulos[0, :]).astype("int")
            for (x, y, r) in circulos:
                candidatos.append({
                    'x': x, 'y': y, 'r': r,
                    'metodo': 'hough_adaptativo',
                    'param1': param1, 'param2': param2
                })
        
        return candidatos
    
    def _detectar_contornos_circulares(self, imagen_prep, config):
        """Detecta circulos usando analisis de contornos."""
        candidatos = []
        gris = imagen_prep['gris']
        
        params_contornos = config.get('contornos', {})
        
        if params_contornos.get('umbral_adaptativo', False):
            binaria = cv2.adaptiveThreshold(
                gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            _, binaria = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
        
        metodo = params_contornos.get('metodo', cv2.RETR_EXTERNAL)
        contornos, _ = cv2.findContours(binaria, metodo, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = params_contornos.get('min_area', 500)
        max_area = params_contornos.get('max_area', 50000)
        circularidad_min = params_contornos.get('circularidad_min', 0.6)
        
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            
            if area < min_area or area > max_area:
                continue
            
            perimetro = cv2.arcLength(contorno, True)
            if perimetro == 0:
                continue
            
            circularidad = 4 * np.pi * area / (perimetro * perimetro)
            
            if circularidad > circularidad_min:
                (x, y), radio = cv2.minEnclosingCircle(contorno)
                
                candidatos.append({
                    'x': int(x), 'y': int(y), 'r': int(radio),
                    'metodo': 'contornos',
                    'circularidad': circularidad,
                    'area': area
                })
        
        return candidatos
    
    def _validar_candidatos(self, imagen, candidatos, config):
        """Valida candidatos usando criterios multiples."""
        llantas_validadas = []
        
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        alto, ancho = imagen.shape[:2]
        
        for candidato in candidatos:
            x, y, r = candidato['x'], candidato['y'], candidato['r']
            
            if x - r < 0 or x + r >= ancho or y - r < 0 or y + r >= alto:
                continue
            
            puntuaciones = {}
            
            if 'forma' in config['pesos']:
                puntuaciones['forma'] = self._evaluar_forma(gris, x, y, r, candidato)
            
            if 'color' in config['pesos']:
                puntuaciones['color'] = self._evaluar_color(hsv, x, y, r, config)
            
            if 'textura' in config['pesos']:
                puntuaciones['textura'] = self._evaluar_textura(gris, x, y, r, config)
            
            if 'keypoints' in config['pesos']:
                puntuaciones['keypoints'] = self._evaluar_keypoints(gris, x, y, r, config)
            
            if 'geometria' in config['pesos']:
                puntuaciones['geometria'] = self._evaluar_geometria(candidato, config)
            
            if 'morfologia' in config['pesos']:
                puntuaciones['morfologia'] = self._evaluar_morfologia(gris, x, y, r)
            
            if 'consistencia' in config['pesos']:
                puntuaciones['consistencia'] = self._evaluar_consistencia(candidato)
            
            puntuacion_total = sum(
                puntuaciones.get(criterio, 0) * peso
                for criterio, peso in config['pesos'].items()
            )
            
            if puntuacion_total >= config['umbral_validacion']:
                llantas_validadas.append((x, y, r, puntuacion_total))
        
        return llantas_validadas
    
    def _evaluar_forma(self, gris, x, y, r, candidato):
        """Evalua la forma circular del candidato."""
        if 'circularidad' in candidato:
            return min(1.0, candidato['circularidad'])
        
        mask = np.zeros(gris.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        region = gris[mask == 255]
        if len(region) == 0:
            return 0.0
        
        desv_std = np.std(region)
        
        return 1.0 - min(1.0, desv_std / 128.0)
    
    def _evaluar_color(self, hsv, x, y, r, config):
        """Evalua si el color corresponde a una llanta (negro/gris)."""
        try:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            region_hsv = hsv[mask == 255]
            if len(region_hsv) == 0:
                return 0.0
            
            if 'color_multirrango' in config.get('algoritmos', []):
                rangos = config.get('color', {}).get('rangos', [])
                max_proporcion = 0.0
                
                for rango in rangos:
                    try:
                        bajo = np.array(rango['bajo'], dtype=np.uint8)
                        alto = np.array(rango['alto'], dtype=np.uint8)
                        
                        # Verificar dimensiones
                        if len(bajo) != 3 or len(alto) != 3:
                            continue
                        
                        # Contar píxeles que están dentro del rango de color
                        pixels_en_rango = 0
                        for pixel in region_hsv:
                            if (np.all(pixel >= bajo) and np.all(pixel <= alto)):
                                pixels_en_rango += 1
                        
                        proporcion = pixels_en_rango / len(region_hsv)
                        max_proporcion = max(max_proporcion, proporcion)
                    except Exception as e:
                        print(f"Error procesando rango multicolor: {e}")
                        continue
                
                return max_proporcion
            else:
                color_config = config.get('color', {})
                negro_bajo = np.array(color_config.get('negro_bajo', [0, 0, 0]), dtype=np.uint8)
                negro_alto = np.array(color_config.get('negro_alto', [180, 255, 80]), dtype=np.uint8)
                gris_bajo = np.array(color_config.get('gris_bajo', [0, 0, 50]), dtype=np.uint8)
                gris_alto = np.array(color_config.get('gris_alto', [180, 50, 200]), dtype=np.uint8)
                
                # Verificar dimensiones
                if len(negro_bajo) != 3 or len(negro_alto) != 3 or len(gris_bajo) != 3 or len(gris_alto) != 3:
                    return 0.0
                
                # Contar píxeles negros y grises
                pixels_negro = 0
                pixels_gris = 0
                
                for pixel in region_hsv:
                    if (np.all(pixel >= negro_bajo) and np.all(pixel <= negro_alto)):
                        pixels_negro += 1
                    elif (np.all(pixel >= gris_bajo) and np.all(pixel <= gris_alto)):
                        pixels_gris += 1
                
                proporcion_negro = pixels_negro / len(region_hsv)
                proporcion_gris = pixels_gris / len(region_hsv)
                
                return max(proporcion_negro, proporcion_gris)
        
        except Exception as e:
            print(f"Error evaluando color de llanta: {e}")
            return 0.0
    
    def _evaluar_textura(self, gris, x, y, r, config):
        """Evalua las caracteristicas de textura de la region."""
        y1, y2 = max(0, y-r), min(gris.shape[0], y+r)
        x1, x2 = max(0, x-r), min(gris.shape[1], x+r)
        
        region = gris[y1:y2, x1:x2]
        
        if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
            return 0.0
        
        try:
            region_uint8 = img_as_ubyte(region)
            
            params_textura = config.get('textura', {})
            distancias = params_textura.get('distancias', [1, 2])
            angulos = params_textura.get('angulos', [0, np.pi/4])
            
            glcm = graycomatrix(region_uint8, distancias, angulos, levels=256, symmetric=True, normed=True)
            
            contraste = np.mean(graycoprops(glcm, 'contrast'))
            energia = np.mean(graycoprops(glcm, 'energy'))
            
            contraste_min = params_textura.get('contraste_min', 0.3)
            energia_max = params_textura.get('energia_max', 0.3)
            
            puntuacion_contraste = 1.0 if contraste > contraste_min else contraste / contraste_min
            puntuacion_energia = 1.0 if energia < energia_max else energia_max / energia
            
            return (puntuacion_contraste + puntuacion_energia) / 2.0
        
        except Exception:
            return 0.0
    
    def _evaluar_keypoints(self, gris, x, y, r, config):
        """Evalua la densidad de keypoints en la region."""
        try:
            params_akaze = config.get('akaze', {})
            detector = cv2.AKAZE_create(
                threshold=params_akaze.get('threshold', 0.003),
                nOctaves=params_akaze.get('nOctaves', 4),
                nOctaveLayers=params_akaze.get('nOctaveLayers', 4),
                diffusivity=params_akaze.get('diffusivity', cv2.KAZE_DIFF_PM_G2)
            )
            
            mask = np.zeros(gris.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            keypoints = detector.detect(gris, mask)
            
            area = np.pi * r * r
            densidad = len(keypoints) / area if area > 0 else 0
            
            return min(1.0, densidad * 1000)
        
        except Exception:
            return 0.0
    
    def _evaluar_geometria(self, candidato, config):
        """Evalua propiedades geometricas avanzadas."""
        if 'area' not in candidato:
            return 1.0
        
        params_geom = config.get('geometria', {})
        
        r = candidato['r']
        area_teorica = np.pi * r * r
        area_real = candidato['area']
        
        ratio = area_real / area_teorica if area_teorica > 0 else 0
        
        ratio_min = params_geom.get('ratio_min', 0.7)
        ratio_max = params_geom.get('ratio_max', 1.3)
        
        if ratio_min <= ratio <= ratio_max:
            return 1.0
        elif ratio < ratio_min:
            return ratio / ratio_min
        else:
            return ratio_max / ratio
    
    def _evaluar_morfologia(self, gris, x, y, r):
        """Evalua caracteristicas morfologicas."""
        y1, y2 = max(0, y-r), min(gris.shape[0], y+r)
        x1, x2 = max(0, x-r), min(gris.shape[1], x+r)
        
        region = gris[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        bordes = cv2.Canny(region, 50, 150)
        proporcion_bordes = np.count_nonzero(bordes) / region.size
        
        return min(1.0, proporcion_bordes * 10)
    
    def _evaluar_consistencia(self, candidato):
        """Evalua la consistencia del candidato."""
        metodo = candidato.get('metodo', '')
        
        if 'hough' in metodo:
            return 0.8
        elif 'contornos' in metodo:
            return 0.9
        else:
            return 0.7
    
    def _aplicar_nms_llantas(self, llantas, umbral_iou=0.3):
        """Aplica Non-Maximum Suppression para eliminar detecciones duplicadas."""
        if not llantas:
            return []
        
        llantas = sorted(llantas, key=lambda x: x[3], reverse=True)
        
        llantas_finales = []
        
        for llanta_actual in llantas:
            agregar = True
            x1, y1, r1, _ = llanta_actual
            
            for llanta_final in llantas_finales:
                x2, y2, r2, _ = llanta_final
                
                distancia = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distancia < (r1 + r2) * 0.5:
                    iou = self._calcular_iou_circulos(x1, y1, r1, x2, y2, r2)
                    
                    if iou > umbral_iou:
                        agregar = False
                        break
            
            if agregar:
                llantas_finales.append(llanta_actual)
        
        return llantas_finales
    
    def _calcular_iou_circulos(self, x1, y1, r1, x2, y2, r2):
        """Calcula Intersection over Union para dos circulos."""
        distancia = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        if distancia >= r1 + r2:
            return 0.0
        
        if distancia <= abs(r1 - r2):
            r_min = min(r1, r2)
            r_max = max(r1, r2)
            return (r_min * r_min) / (r_max * r_max)
        
        alpha1 = 2 * np.arccos((distancia**2 + r1**2 - r2**2) / (2 * distancia * r1))
        alpha2 = 2 * np.arccos((distancia**2 + r2**2 - r1**2) / (2 * distancia * r2))
        
        area_interseccion = 0.5 * (r1**2 * (alpha1 - np.sin(alpha1)) + r2**2 * (alpha2 - np.sin(alpha2)))
        
        area_union = np.pi * (r1**2 + r2**2) - area_interseccion
        
        return area_interseccion / area_union if area_union > 0 else 0.0
    
    def _dibujar_resultados(self, imagen, llantas, nombre_config):
        """Dibuja las llantas detectadas en la imagen."""
        for i, (x, y, r, confianza) in enumerate(llantas):
            color = (0, 255, 0)
            cv2.circle(imagen, (x, y), r, color, 3)
            cv2.circle(imagen, (x, y), 2, (255, 0, 0), -1)
            
            texto = f"Llanta {i+1}"
            cv2.putText(imagen, texto, (x-30, y-r-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            texto_conf = f"{confianza:.2f}"
            cv2.putText(imagen, texto_conf, (x-20, y-r-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        texto_info = f"Config: {nombre_config} - Llantas: {len(llantas)}"
        cv2.putText(imagen, texto_info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return imagen
    
    def _mostrar_resultado(self, resultado, titulo):
        """Muestra resultado de la deteccion."""
        print(f"\nResultado - {titulo}")
        print(f"Configuracion: {resultado['configuracion']}")
        print(f"Llantas detectadas: {resultado['num_llantas']}")
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
        nombre_archivo = f"llantas_{config_nombre}_{timestamp}.jpg"
        
        if ruta_base:
            directorio = os.path.dirname(ruta_base)
            ruta_completa = os.path.join(directorio, nombre_archivo)
        else:
            ruta_completa = nombre_archivo
        
        os.makedirs(os.path.dirname(ruta_completa), exist_ok=True)
        cv2.imwrite(ruta_completa, imagen_resultado)
        print(f"Resultado guardado: {ruta_completa}")


def detectar_llantas_imagen(ruta_imagen, configuracion='CONFIG_BALANCED', visualizar=True, guardar=False, ruta_salida=None):
    """
    Funcion de conveniencia para detectar llantas en una imagen.
    
    Args:
        ruta_imagen (str): Ruta de la imagen
        configuracion (str): Configuracion a usar
        visualizar (bool): Si mostrar resultados
        guardar (bool): Si guardar resultados
        ruta_salida (str): Ruta donde guardar
        
    Returns:
        dict: Resultados de la deteccion
    """
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
        return None
    
    detector = DetectorLlantas()
    return detector.detectar_llantas(imagen, configuracion, visualizar, guardar, ruta_salida)
