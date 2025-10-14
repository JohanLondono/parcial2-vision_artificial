#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo HOG y KAZE para Análisis de Tráfico Vehicular
===================================================

Implementación de descriptores HOG (Histogram of Oriented Gradients) y 
KAZE para identificación de objetos en imágenes de tráfico vehicular.

FUNDAMENTOS TEÓRICOS
====================

HOG (Histogram of Oriented Gradients)
--------------------------------------
HOG es un descriptor de características basado en la distribución de gradientes
de intensidad locales en una imagen.

**Base Matemática:**

1. Cálculo de Gradientes:
   - Gradiente horizontal: Gx(x,y) = I(x+1,y) - I(x-1,y)
   - Gradiente vertical: Gy(x,y) = I(x,y+1) - I(x,y-1)
   - Magnitud: |G(x,y)| = √(Gx² + Gy²)
   - Orientación: θ(x,y) = arctan(Gy/Gx)

2. Agrupación en Celdas:
   - La imagen se divide en celdas de tamaño fijo (típicamente 8×8 píxeles)
   - Para cada celda se calcula un histograma de orientaciones
   - El histograma típicamente usa 9 bins cubriendo 0°-180° (sin signo)

3. Normalización por Bloques:
   - Los bloques son grupos de celdas (típicamente 2×2 celdas)
   - Normalización L2-Hys: v' = v / √(||v||₂² + ε²)
     donde v es el vector del bloque y ε es una constante pequeña
   - Esto proporciona invarianza a cambios de iluminación

4. Vector de Características Final:
   - Concatenación de todos los histogramas normalizados
   - Dimensión: n_blocks × cells_per_block² × orientations

**Ventajas:**
- Robusto ante cambios de iluminación
- Captura información de forma local
- Invariante a pequeñas deformaciones
- Alta tasa de detección para objetos rígidos

KAZE
----
KAZE es un detector y descriptor de características que utiliza difusión
no lineal en lugar de suavizado gaussiano.

**Base Matemática:**

1. Ecuación de Difusión No Lineal:
   ∂L/∂t = div(c(x,y,t) · ∇L)
   
   donde:
   - L es el espacio de escala no lineal
   - c(x,y,t) es la función de conductividad
   - div es el operador divergencia
   - ∇L es el gradiente de la imagen

2. Función de Conductividad (Perona-Malik):
   c(x,y,t) = g(|∇Lσ(x,y,t)|)
   
   Tipo PM_G2: g(x) = 1 / (1 + (|∇Lσ|/K)²)
   donde K es un parámetro de contraste

3. Construcción del Espacio de Escala:
   - Se resuelve la ecuación de difusión en múltiples octavas
   - Cada octava duplica el tiempo de difusión
   - Preserva mejor los bordes que el suavizado gaussiano

4. Detección de Puntos Clave:
   - Se buscan extremos en el determinante del Hessiano:
     Det(H) = Lxx·Lyy - Lxy²
   - Se refinan las posiciones usando interpolación cuadrática

**Ventajas:**
- Preserva bordes durante el suavizado
- Mayor precisión en localización de características
- Robusto ante cambios de escala y rotación
- Mejor para imágenes con detalles finos

APLICACIONES EN TRÁFICO VEHICULAR
==================================

HOG es especialmente efectivo para:
- Detección de vehículos y peatones (forma característica)
- Análisis de señales de tráfico (contornos geométricos)
- Clasificación de objetos basada en gradientes
- Identificación de patrones estructurados

KAZE es útil para:
- Detección de puntos clave robustos en señales
- Análisis de texturas locales en superficies vehiculares
- Matching de características para seguimiento
- Reconocimiento de matrículas y elementos texturizados
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from skimage import exposure, img_as_ubyte
import seaborn as sns

class HOGKAZEAnalyzer:
    """Analizador HOG y KAZE para tráfico vehicular."""
    
    def __init__(self, output_dir="./resultados"):
        """
        Inicializar el analizador HOG-KAZE.
        
        Args:
            output_dir (str): Directorio de salida para resultados
        """
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "hog_kaze_analysis")
        os.makedirs(self.results_dir, exist_ok=True)
        self.current_results = []
        
        # Configuración HOG para tráfico vehicular
        self.hog_config = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'transform_sqrt': True,
            'feature_vector': True
        }
        
        # Configuración KAZE
        self.kaze_config = {
            'threshold': 0.003,
            'nOctaves': 4,
            'nOctaveLayers': 4,
            'diffusivity': cv2.KAZE_DIFF_PM_G2
        }
    
    def extraer_caracteristicas_hog(self, imagen, visualizar=True, mostrar_descriptores=True, guardar_resultados=False, nombre_imagen="imagen_hog"):
        """
        Extrae características HOG (Histogram of Oriented Gradients) de la imagen.
        
        PROCESO DEL ALGORITMO:
        ======================
        
        1. **Preprocesamiento:**
           - Conversión a escala de grises para simplificar el cálculo
           - Ecualización de histograma para normalizar la iluminación
           - Fórmula de ecualización: h'(i) = ⌊(L-1) · CDF(i)⌋
             donde CDF es la función de distribución acumulativa
        
        2. **Cálculo de Gradientes:**
           - Se aplica el operador Sobel o diferencias centradas
           - Gx = [-1, 0, 1] convolución horizontal
           - Gy = [-1, 0, 1]ᵀ convolución vertical
           - Magnitud y orientación en cada píxel
        
        3. **Construcción de Histogramas:**
           - División en celdas de 8×8 píxeles (configurable)
           - Cada celda genera un histograma de 9 orientaciones
           - Votación ponderada por magnitud del gradiente
           - Interpolación bilineal entre bins adyacentes
        
        4. **Normalización por Bloques:**
           - Bloques de 2×2 celdas con solapamiento
           - Normalización L2-Hys con umbral en 0.2:
             * v_norm = v / √(||v||₂² + ε²)
             * v_clipped = min(v_norm, 0.2)
             * v_final = v_clipped / √(||v_clipped||₂² + ε²)
           - Proporciona robustez ante variaciones de iluminación
        
        5. **Análisis Estadístico:**
           - Media y desviación estándar del vector HOG
           - Energía: E = Σ(features²) - medida de intensidad total
           - Entropía: H = -Σ(p·log₂(p)) - medida de información
           - Sparsity: porcentaje de características cercanas a cero
        
        6. **Análisis Direccional:**
           - Separación de características por orientación
           - Identificación de orientación dominante
           - Índice de estructura: max(E_orient) / mean(E_orient)
           - Alto índice → objeto con dirección preferente (ej: vehículo)
        
        MÉTRICAS EXTRAÍDAS:
        ===================
        - num_features: Dimensión del vector HOG
        - hog_mean, hog_std: Estadísticas básicas
        - hog_energy: Suma de cuadrados de características
        - hog_entropy: Medida de información contenida
        - dominant_orientation: Dirección con mayor energía
        - structure_index: Medida de organización direccional
        - orientation_entropy: Uniformidad de distribución direccional
        
        Args:
            imagen (np.ndarray): Imagen de entrada (puede ser BGR o escala de grises)
            visualizar (bool): Si True, genera visualización del mapa HOG
            mostrar_descriptores (bool): Si True, imprime estadísticas en consola
            guardar_resultados (bool): Si True, guarda archivos CSV y TXT
            nombre_imagen (str): Nombre base para archivos de salida
            
        Returns:
            dict: Diccionario con características HOG y metadatos:
                - Estadísticas globales (mean, std, energy, entropy)
                - Estadísticas por orientación
                - Imágenes procesadas (gray_image, normalized_image, hog_image)
                - Vector de características crudas (hog_features_raw)
                
        Example:
            >>> analyzer = HOGKAZEAnalyzer()
            >>> imagen = cv2.imread('vehiculo.jpg')
            >>> resultados = analyzer.extraer_caracteristicas_hog(imagen)
            >>> print(f"Características extraídas: {resultados['num_features']}")
            >>> print(f"Orientación dominante: {resultados['dominant_orientation']}°")
        """
        print("🔍 Extrayendo características HOG...")
        print("="*60)
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # Normalizar imagen
        imagen_norm = exposure.equalize_hist(imagen_gris)
        
        # Extraer HOG con visualización
        hog_features, hog_image = hog(imagen_norm,
                                    orientations=self.hog_config['orientations'],
                                    pixels_per_cell=self.hog_config['pixels_per_cell'],
                                    cells_per_block=self.hog_config['cells_per_block'],
                                    block_norm=self.hog_config['block_norm'],
                                    transform_sqrt=self.hog_config['transform_sqrt'],
                                    visualize=True,
                                    feature_vector=self.hog_config['feature_vector'])
        
        # Mostrar información detallada en consola
        if mostrar_descriptores:
            print(f"📊 ANÁLISIS DE CARACTERÍSTICAS HOG - {nombre_imagen.upper()}")
            print("="*60)
            print(f"📐 Dimensiones de la imagen: {imagen_gris.shape}")
            print(f"🎯 Número de características HOG extraídas: {len(hog_features)}")
            print(f"📊 Forma del vector de características: {hog_features.shape}")
            print("\n📈 Estadísticas de las características:")
            print(f"   • Valor mínimo: {np.min(hog_features):.8f}")
            print(f"   • Valor máximo: {np.max(hog_features):.8f}")
            print(f"   • Promedio: {np.mean(hog_features):.8f}")
            print(f"   • Desviación estándar: {np.std(hog_features):.8f}")
            print(f"   • Energía total: {np.sum(hog_features**2):.6f}")
            
            print(f"\n🔍 Primeras 20 características HOG:")
            for i in range(min(20, len(hog_features))):
                print(f"   Característica {i+1:3d}: {hog_features[i]:15.8f}")
            
            if len(hog_features) > 20:
                print(f"\n🔍 Últimas 10 características HOG:")
                start_idx = max(0, len(hog_features) - 10)
                for i in range(start_idx, len(hog_features)):
                    print(f"   Característica {i+1:3d}: {hog_features[i]:15.8f}")
            
            print(f"\n💾 Para ver todas las {len(hog_features)} características, active guardar_resultados=True")
            print("="*60)
        
        # Análisis estadístico de características HOG
        hog_stats = {
            'num_features': len(hog_features),
            'hog_mean': np.mean(hog_features),
            'hog_std': np.std(hog_features),
            'hog_min': np.min(hog_features),
            'hog_max': np.max(hog_features),
            'hog_energy': np.sum(hog_features ** 2),
            'hog_entropy': self._calculate_entropy(hog_features),
            'hog_sparsity': np.sum(hog_features == 0) / len(hog_features)
        }
        
        # Análisis direccional de gradientes
        orientations = self.hog_config['orientations']
        
        # Dividir características por orientación
        features_per_cell = orientations * self.hog_config['cells_per_block'][0] * self.hog_config['cells_per_block'][1]
        orientation_histograms = []
        
        for i in range(0, len(hog_features), features_per_cell):
            block_features = hog_features[i:i+features_per_cell]
            if len(block_features) == features_per_cell:
                # Promedio por orientación en el bloque
                for j in range(orientations):
                    orient_features = block_features[j::orientations]
                    if j >= len(orientation_histograms):
                        orientation_histograms.append([])
                    orientation_histograms[j].extend(orient_features)
        
        # Estadísticas por orientación
        orientation_stats = {}
        for i, orient_hist in enumerate(orientation_histograms):
            if orient_hist:
                orientation_stats[f'orientation_{i}_mean'] = np.mean(orient_hist)
                orientation_stats[f'orientation_{i}_std'] = np.std(orient_hist)
                orientation_stats[f'orientation_{i}_energy'] = np.sum(np.array(orient_hist) ** 2)
        
        # Análisis de dominancia direccional
        orientation_energies = [orientation_stats.get(f'orientation_{i}_energy', 0) 
                               for i in range(orientations)]
        dominant_orientation = np.argmax(orientation_energies)
        orientation_entropy = self._calculate_entropy(orientation_energies)
        
        # Métricas específicas para vehículos
        # Índice de estructura (alta energía en orientaciones específicas)
        structure_index = np.max(orientation_energies) / (np.mean(orientation_energies) + 1e-10)
        
        # Índice de textura direccional
        directional_variance = np.var(orientation_energies)
        
        resultados = {
            **hog_stats,
            **orientation_stats,
            'dominant_orientation': dominant_orientation,
            'orientation_entropy': orientation_entropy,
            'structure_index': structure_index,
            'directional_variance': directional_variance,
            'hog_image': hog_image,
            'hog_features_raw': hog_features,
            'gray_image': imagen_gris,
            'normalized_image': imagen_norm,
            'nombre_imagen': nombre_imagen
        }
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            self._guardar_resultados_hog(resultados, nombre_imagen)
        
        if visualizar:
            self._visualizar_hog(resultados)
        
        return resultados
    
    def extraer_caracteristicas_kaze(self, imagen, visualizar=True, mostrar_descriptores=True, guardar_resultados=False, nombre_imagen="imagen_kaze", usar_config_default=False):
        """
        Extrae características KAZE (Accelerated KAZE) de la imagen usando difusión no lineal.
        
        PROCESO DEL ALGORITMO:
        ======================
        
        1. **Construcción del Espacio de Escala No Lineal:**
           - Se resuelve la ecuación de difusión: ∂L/∂t = div(c(x,y,t)·∇L)
           - Difusividad Perona-Malik G2: c = 1/(1 + (|∇L|/K)²)
           - Ventaja: preserva bordes mientras suaviza regiones uniformes
           - Se construyen 4 octavas con 4 capas cada una (configurable)
        
        2. **Detección de Puntos Clave:**
           - Búsqueda de extremos en el determinante del Hessiano:
             Det(H) = Lxx·Lyy - Lxy²
           - Umbralización: solo se mantienen respuestas > threshold
           - Refinamiento sub-píxel usando interpolación cuadrática
           - Eliminación de puntos con bajo contraste o en bordes
        
        3. **Asignación de Orientación:**
           - Para cada punto clave se calcula un histograma de gradientes
           - Radio de búsqueda proporcional a la escala: r = 6·σ
           - Gradientes ponderados por función gaussiana circular
           - Orientación principal = pico del histograma suavizado
        
        4. **Construcción de Descriptores:**
           - Región de 20σ × 20σ alrededor del punto clave
           - Rotación según orientación principal (invarianza rotacional)
           - Subdivisión en 4×4 = 16 subregiones
           - Para cada subregión: descriptor basado en gradientes
           - Vector final de 64 dimensiones (KAZE estándar)
        
        5. **Análisis de Distribución Espacial:**
           - Cálculo del centroide: (x̄,ȳ) = (Σxi/n, Σyi/n)
           - Dispersión espacial: σ² = Σ(xi-x̄)² + Σ(yi-ȳ)²
           - Cobertura de imagen: porcentaje de área con puntos clave
           - Clustering: agrupación de puntos cercanos
        
        6. **Análisis Multi-Escala:**
           - Distribución de escalas (tamaños) de puntos clave
           - Consistencia multi-escala: correlación entre escalas
           - Respuestas fuertes en múltiples escalas → características estables
        
        MÉTRICAS EXTRAÍDAS:
        ===================
        
        **Puntos Clave:**
        - num_keypoints: Cantidad total detectada
        - mean_response: Respuesta promedio del detector
        - scale_mean/std: Estadísticas de escalas
        - centroid_x/y: Centro de masa de los puntos
        - spatial_spread: Dispersión espacial
        
        **Descriptores:**
        - descriptor_mean/std: Estadísticas del vector
        - descriptor_diversity: Medida de variabilidad entre descriptores
        - descriptor_sparsity: Porcentaje de valores cercanos a cero
        
        **Distribución Espacial:**
        - coverage_percentage: Porcentaje de imagen cubierta
        - density_per_region: Densidad local de puntos
        - uniformity_score: Uniformidad de distribución
        
        CONFIGURACIÓN:
        ==============
        - threshold: 0.003 (sensibilidad de detección)
        - nOctaves: 4 (número de escalas principales)
        - nOctaveLayers: 4 (subdivisiones por octava)
        - diffusivity: PM_G2 (función de difusión Perona-Malik)
        
        Args:
            imagen (np.ndarray): Imagen de entrada (puede ser BGR o escala de grises)
            visualizar (bool): Si True, genera visualización de puntos clave
            mostrar_descriptores (bool): Si True, imprime estadísticas detalladas
            guardar_resultados (bool): Si True, guarda archivos CSV y TXT
            nombre_imagen (str): Nombre base para archivos de salida
            usar_config_default (bool): Si True, usa configuración básica sin parámetros avanzados
            
        Returns:
            dict: Diccionario con características KAZE:
                - Estadísticas de puntos clave (cantidad, respuestas, escalas)
                - Estadísticas de descriptores (mean, std, diversity)
                - Análisis espacial (centroide, dispersión, cobertura)
                - Datos crudos (keypoints, descriptors, gray_image)
                
        Example:
            >>> analyzer = HOGKAZEAnalyzer()
            >>> imagen = cv2.imread('señal_transito.jpg')
            >>> resultados = analyzer.extraer_caracteristicas_kaze(imagen)
            >>> print(f"Puntos clave detectados: {resultados['num_keypoints']}")
            >>> print(f"Cobertura espacial: {resultados['coverage_percentage']:.1f}%")
        
        Note:
            KAZE es especialmente efectivo para imágenes con bordes nítidos y detalles finos,
            como señales de tráfico, matrículas y texturas vehiculares.
        """
        print("🔍 Extrayendo características KAZE...")
        print("="*60)
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # Crear detector KAZE
        if usar_config_default:
            # Configuración por defecto (como la profesora)
            kaze = cv2.KAZE_create()
            print("🔧 Usando configuración KAZE por defecto (como la profesora)")
        else:
            # Configuración avanzada personalizada
            kaze = cv2.KAZE_create(
                extended=False,
                upright=False,
                threshold=self.kaze_config['threshold'],
                nOctaves=self.kaze_config['nOctaves'],
                nOctaveLayers=self.kaze_config['nOctaveLayers'],
                diffusivity=self.kaze_config['diffusivity']
            )
            print("🔧 Usando configuración KAZE avanzada")
        
        # Detectar puntos clave y calcular descriptores
        keypoints, descriptors = kaze.detectAndCompute(imagen_gris, None)
        
        # Mostrar información detallada en consola
        if mostrar_descriptores:
            print(f"🔑 ANÁLISIS DE CARACTERÍSTICAS KAZE - {nombre_imagen.upper()}")
            print("="*60)
            print(f"📐 Dimensiones de la imagen: {imagen_gris.shape}")
            print(f"🎯 Número de puntos clave detectados: {len(keypoints)}")
            
            if descriptors is not None:
                print(f"📊 Forma de la matriz de descriptores: {descriptors.shape}")
                print(f"🔢 Dimensión de cada descriptor: {descriptors.shape[1]}")
                print(f"📈 Total de descriptores: {descriptors.shape[0]}")
                
                print("\n📊 Estadísticas de los descriptores:")
                print(f"   • Valor mínimo: {np.min(descriptors):.8f}")
                print(f"   • Valor máximo: {np.max(descriptors):.8f}")
                print(f"   • Promedio: {np.mean(descriptors):.8f}")
                print(f"   • Desviación estándar: {np.std(descriptors):.8f}")
                
                print(f"\n🔍 Información detallada de los primeros 5 puntos clave:")
                for i in range(min(5, len(keypoints))):
                    kp = keypoints[i]
                    print(f"   📍 Punto clave {i+1}:")
                    print(f"      • Posición (x,y): ({kp.pt[0]:.2f}, {kp.pt[1]:.2f})")
                    print(f"      • Tamaño: {kp.size:.2f}")
                    print(f"      • Ángulo: {kp.angle:.2f}°")
                    print(f"      • Respuesta: {kp.response:.8f}")
                    print(f"      • Descriptor (primeros 10 valores): {descriptors[i][:10]}")
                
                if len(keypoints) > 0:
                    print(f"\n🎯 Descriptor completo del primer punto clave:")
                    descriptor_str = ', '.join([f"{val:.8f}" for val in descriptors[0]])
                    print(f"   Descriptor 1: [{descriptor_str}]")
                    
                    if len(keypoints) > 1:
                        print(f"\n🎯 Descriptor completo del último punto clave:")
                        descriptor_str = ', '.join([f"{val:.8f}" for val in descriptors[-1]])
                        print(f"   Descriptor {len(keypoints)}: [{descriptor_str}]")
                
                print(f"\n💾 Para ver todos los {len(keypoints)} descriptores completos, active guardar_resultados=True")
            else:
                print("❌ No se pudieron calcular descriptores")
            
            print("="*60)
        
        # Análisis de puntos clave
        num_keypoints = len(keypoints)
        
        if num_keypoints > 0:
            # Estadísticas de localización de puntos clave
            kp_x = [kp.pt[0] for kp in keypoints]
            kp_y = [kp.pt[1] for kp in keypoints]
            kp_size = [kp.size for kp in keypoints]
            kp_angle = [kp.angle for kp in keypoints]
            kp_response = [kp.response for kp in keypoints]
            
            # Estadísticas espaciales
            keypoint_stats = {
                'num_keypoints': num_keypoints,
                'kp_mean_x': np.mean(kp_x),
                'kp_std_x': np.std(kp_x),
                'kp_mean_y': np.mean(kp_y),
                'kp_std_y': np.std(kp_y),
                'kp_mean_size': np.mean(kp_size),
                'kp_std_size': np.std(kp_size),
                'kp_mean_response': np.mean(kp_response),
                'kp_std_response': np.std(kp_response),
                'kp_density': num_keypoints / (imagen_gris.shape[0] * imagen_gris.shape[1])
            }
            
            # Análisis de ángulos
            valid_angles = [angle for angle in kp_angle if angle >= 0]
            if valid_angles:
                keypoint_stats.update({
                    'kp_mean_angle': np.mean(valid_angles),
                    'kp_std_angle': np.std(valid_angles),
                    'kp_angle_entropy': self._calculate_entropy(np.histogram(valid_angles, bins=36)[0])
                })
            
            # Análisis de descriptores
            if descriptors is not None:
                descriptor_stats = {
                    'descriptor_length': descriptors.shape[1],
                    'descriptor_mean': np.mean(descriptors),
                    'descriptor_std': np.std(descriptors),
                    'descriptor_sparsity': np.sum(descriptors == 0) / descriptors.size,
                    'descriptor_energy': np.sum(descriptors ** 2),
                    'descriptor_entropy': self._calculate_entropy(descriptors.flatten())
                }
                
                # Análisis de diversidad de descriptores
                descriptor_diversity = self._calculate_descriptor_diversity(descriptors)
                descriptor_stats['descriptor_diversity'] = descriptor_diversity
            else:
                descriptor_stats = {
                    'descriptor_length': 0,
                    'descriptor_mean': 0,
                    'descriptor_std': 0,
                    'descriptor_sparsity': 0,
                    'descriptor_energy': 0,
                    'descriptor_entropy': 0,
                    'descriptor_diversity': 0
                }
            
            # Análisis espacial avanzado
            spatial_stats = self._analisis_espacial_keypoints(kp_x, kp_y, imagen_gris.shape)
            
        else:
            # No se detectaron puntos clave
            keypoint_stats = {
                'num_keypoints': 0,
                'kp_mean_x': 0, 'kp_std_x': 0,
                'kp_mean_y': 0, 'kp_std_y': 0,
                'kp_mean_size': 0, 'kp_std_size': 0,
                'kp_mean_response': 0, 'kp_std_response': 0,
                'kp_density': 0,
                'kp_mean_angle': 0, 'kp_std_angle': 0,
                'kp_angle_entropy': 0
            }
            
            descriptor_stats = {
                'descriptor_length': 0, 'descriptor_mean': 0,
                'descriptor_std': 0, 'descriptor_sparsity': 0,
                'descriptor_energy': 0, 'descriptor_entropy': 0,
                'descriptor_diversity': 0
            }
            
            spatial_stats = {
                'spatial_uniformity': 0,
                'spatial_clustering': 0,
                'edge_keypoint_ratio': 0
            }
        
        resultados = {
            **keypoint_stats,
            **descriptor_stats,
            **spatial_stats,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'gray_image': imagen_gris,
            'nombre_imagen': nombre_imagen
        }
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            self._guardar_resultados_kaze(resultados, nombre_imagen)
        
        if visualizar:
            self._visualizar_kaze(resultados)
        
        return resultados
    
    def analisis_combinado_hog_kaze(self, imagen_path, nombre_imagen=None):
        """
        Realiza análisis combinado HOG + KAZE.
        
        Args:
            imagen_path (str): Ruta a la imagen
            nombre_imagen (str): Nombre personalizado
            
        Returns:
            dict: Resultados combinados
        """
        try:
            # Cargar imagen
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
            
            if nombre_imagen is None:
                nombre_imagen = os.path.basename(imagen_path)
            
            print(f"🔄 Análisis HOG-KAZE para: {nombre_imagen}")
            
            # Análisis HOG
            resultados_hog = self.extraer_caracteristicas_hog(imagen, visualizar=False)
            
            # Análisis KAZE
            resultados_kaze = self.extraer_caracteristicas_kaze(imagen, visualizar=False)
            
            # Análisis adicional: LBP (Local Binary Patterns)
            lbp_stats = self._extraer_lbp(imagen)
            
            # Combinar resultados
            resultado_completo = {
                'Imagen': nombre_imagen,
                'Ruta': imagen_path,
                'Dimensiones': imagen.shape,
                'Fecha_Analisis': datetime.now().isoformat(),
                **{f'hog_{k}': v for k, v in resultados_hog.items() 
                   if not isinstance(v, np.ndarray)},
                **{f'kaze_{k}': v for k, v in resultados_kaze.items() 
                   if not isinstance(v, (list, np.ndarray))},
                **{f'lbp_{k}': v for k, v in lbp_stats.items()}
            }
            
            self.current_results.append(resultado_completo)
            
            print(f"✅ Análisis HOG-KAZE completado para: {nombre_imagen}")
            return resultado_completo
            
        except Exception as e:
            print(f"❌ Error al procesar {imagen_path}: {str(e)}")
            return None
    
    def _extraer_lbp(self, imagen):
        """Extrae características Local Binary Pattern."""
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # Calcular LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(imagen_gris, n_points, radius, method='uniform')
        
        # Histograma LBP
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                             range=(0, n_points + 2), density=True)
        
        return {
            'lbp_entropy': self._calculate_entropy(hist),
            'lbp_uniformity': np.sum(hist ** 2),
            'lbp_variance': np.var(hist),
            'lbp_mean': np.mean(lbp),
            'lbp_std': np.std(lbp)
        }
    
    def _calculate_entropy(self, data):
        """Calcula la entropía de los datos."""
        if isinstance(data, np.ndarray):
            data = data.flatten()
        
        # Crear histograma
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist[hist > 0]  # Remover bins vacíos
        
        if len(hist) == 0:
            return 0
        
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_descriptor_diversity(self, descriptors):
        """Calcula diversidad entre descriptores."""
        if descriptors is None or len(descriptors) < 2:
            return 0
        
        # Calcular distancias entre pares de descriptores
        from scipy.spatial.distance import pdist
        distances = pdist(descriptors, metric='euclidean')
        return np.mean(distances)
    
    def _analisis_espacial_keypoints(self, kp_x, kp_y, image_shape):
        """Analiza la distribución espacial de puntos clave."""
        if not kp_x or not kp_y:
            return {
                'spatial_uniformity': 0,
                'spatial_clustering': 0,
                'edge_keypoint_ratio': 0
            }
        
        h, w = image_shape[:2]
        
        # Uniformidad espacial (coeficiente de variación de distancias)
        points = np.column_stack((kp_x, kp_y))
        if len(points) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(points)
            spatial_uniformity = np.std(distances) / (np.mean(distances) + 1e-10)
        else:
            spatial_uniformity = 0
        
        # Clustering espacial (usando densidad local)
        edge_margin = min(w, h) * 0.1
        edge_points = sum(1 for x, y in zip(kp_x, kp_y) 
                         if x < edge_margin or x > w - edge_margin or 
                            y < edge_margin or y > h - edge_margin)
        edge_keypoint_ratio = edge_points / len(kp_x) if kp_x else 0
        
        return {
            'spatial_uniformity': spatial_uniformity,
            'spatial_clustering': 1.0 / (spatial_uniformity + 1),  # Inverso de uniformidad
            'edge_keypoint_ratio': edge_keypoint_ratio
        }
    
    def _visualizar_hog(self, resultados):
        """Visualiza los resultados HOG."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Imagen original
        axes[0].imshow(resultados['gray_image'], cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        # Imagen normalizada
        axes[1].imshow(resultados['normalized_image'], cmap='gray')
        axes[1].set_title('Imagen Normalizada')
        axes[1].axis('off')
        
        # Visualización HOG
        hog_image_rescaled = exposure.rescale_intensity(resultados['hog_image'], in_range=(0, 10))
        axes[2].imshow(hog_image_rescaled, cmap='hot')
        axes[2].set_title(f'HOG Features ({resultados["num_features"]} características)')
        axes[2].axis('off')
        
        plt.tight_layout()
        archivo_viz = os.path.join(self.results_dir, f'hog_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(archivo_viz, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Gráfico de estadísticas direccionales
        orientations = self.hog_config['orientations']
        orientation_energies = [resultados.get(f'orientation_{i}_energy', 0) for i in range(orientations)]
        
        if any(orientation_energies):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            angles = np.linspace(0, 180, orientations, endpoint=False)
            ax.bar(angles, orientation_energies, width=180/orientations*0.8)
            ax.set_xlabel('Orientación (grados)')
            ax.set_ylabel('Energía')
            ax.set_title('Distribución Direccional HOG')
            ax.grid(True, alpha=0.3)
            
            archivo_direccional = os.path.join(self.results_dir, f'hog_directions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(archivo_direccional, dpi=300, bbox_inches='tight')
            plt.show()
    
    def _visualizar_kaze(self, resultados):
        """Visualiza los resultados KAZE."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Imagen original
        axes[0].imshow(resultados['gray_image'], cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        # Puntos clave KAZE - Crear imagen en color para mostrar puntos amarillos
        if len(resultados['gray_image'].shape) == 2:
            # Convertir imagen gris a color (RGB) para poder mostrar puntos amarillos
            img_kp = cv2.cvtColor(resultados['gray_image'], cv2.COLOR_GRAY2RGB)
        else:
            img_kp = resultados['gray_image'].copy()
        
        if resultados['keypoints']:
            # Dibujar puntos clave con color amarillo (255, 255, 0) y sin flags para puntos más pequeños
            img_kp = cv2.drawKeypoints(img_kp, resultados['keypoints'], None, 
                                     color=(255, 255, 0), flags=0)
        
        axes[1].imshow(img_kp)
        axes[1].set_title(f'Puntos Clave KAZE ({resultados["num_keypoints"]})')
        axes[1].axis('off')
        
        plt.tight_layout()
        archivo_viz = os.path.join(self.results_dir, f'kaze_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(archivo_viz, dpi=300, bbox_inches='tight')
        plt.show()
    
    def guardar_resultados(self, formato='csv'):
        """Guarda los resultados del análisis."""
        if not self.current_results:
            print("❌ No hay resultados para guardar.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if formato.lower() == 'csv':
            df = pd.DataFrame(self.current_results)
            archivo_csv = os.path.join(self.results_dir, f'hog_kaze_analysis_{timestamp}.csv')
            df.to_csv(archivo_csv, index=False)
            print(f"💾 Resultados CSV guardados: {archivo_csv}")
        
        elif formato.lower() == 'json':
            import json
            archivo_json = os.path.join(self.results_dir, f'hog_kaze_analysis_{timestamp}.json')
            with open(archivo_json, 'w', encoding='utf-8') as f:
                json.dump(self.current_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Resultados JSON guardados: {archivo_json}")
    
    def generar_reporte_hog_kaze(self):
        """Genera reporte del análisis HOG-KAZE."""
        if not self.current_results:
            print("❌ No hay resultados para el reporte.")
            return
        
        print("\n📋 REPORTE ANÁLISIS HOG + KAZE")
        print("=" * 40)
        print(f"📊 Imágenes analizadas: {len(self.current_results)}")
        
        # Estadísticas HOG
        hog_features = [r.get('hog_num_features', 0) for r in self.current_results]
        hog_energies = [r.get('hog_hog_energy', 0) for r in self.current_results]
        
        print(f"\n🎯 ESTADÍSTICAS HOG:")
        print(f"   Características promedio: {np.mean(hog_features):.0f}")
        print(f"   Energía promedio: {np.mean(hog_energies):.2f}")
        
        # Estadísticas KAZE
        kaze_keypoints = [r.get('kaze_num_keypoints', 0) for r in self.current_results]
        kaze_density = [r.get('kaze_kp_density', 0) for r in self.current_results]
        
        print(f"\n🔑 ESTADÍSTICAS KAZE:")
        print(f"   Puntos clave promedio: {np.mean(kaze_keypoints):.1f}")
        print(f"   Densidad promedio: {np.mean(kaze_density):.6f}")
        
        # Top imágenes
        imagenes_hog = [(r['Imagen'], r.get('hog_hog_energy', 0)) for r in self.current_results]
        imagenes_hog.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP 3 - MAYOR ENERGÍA HOG:")
        for i, (imagen, energia) in enumerate(imagenes_hog[:3], 1):
            print(f"   {i}. {imagen}: {energia:.2f}")
        
        imagenes_kaze = [(r['Imagen'], r.get('kaze_num_keypoints', 0)) for r in self.current_results]
        imagenes_kaze.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 TOP 3 - MÁS PUNTOS CLAVE KAZE:")
        for i, (imagen, puntos) in enumerate(imagenes_kaze[:3], 1):
            print(f"   {i}. {imagen}: {puntos} puntos")
        
        print("\n" + "=" * 40)
    
    def _guardar_resultados_hog(self, resultados, nombre_imagen):
        """Guarda los resultados HOG en archivos CSV y TXT."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio si no existe
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Datos para guardar
        hog_data = {
            'imagen': nombre_imagen,
            'dimensiones': f"{resultados['gray_image'].shape[0]}x{resultados['gray_image'].shape[1]}",
            'num_caracteristicas': resultados['num_features'],
            'valor_min': resultados['hog_min'],
            'valor_max': resultados['hog_max'],
            'promedio': resultados['hog_mean'],
            'desviacion_std': resultados['hog_std'],
            'energia_total': resultados['hog_energy'],
            'entropia': resultados['hog_entropy'],
            'fecha_analisis': datetime.now().isoformat()
        }
        
        # Guardar estadísticas en CSV
        df_stats = pd.DataFrame([hog_data])
        archivo_csv_stats = os.path.join(self.results_dir, f'hog_estadisticas_{nombre_imagen}_{timestamp}.csv')
        df_stats.to_csv(archivo_csv_stats, index=False, encoding='utf-8')
        print(f"✅ Estadísticas HOG guardadas en: {archivo_csv_stats}")
        
        # Guardar características completas en CSV
        df_features = pd.DataFrame({
            'indice': range(1, len(resultados['hog_features_raw']) + 1),
            'caracteristica_hog': resultados['hog_features_raw'].flatten()
        })
        archivo_csv_features = os.path.join(self.results_dir, f'hog_caracteristicas_{nombre_imagen}_{timestamp}.csv')
        df_features.to_csv(archivo_csv_features, index=False, encoding='utf-8')
        print(f"✅ Características HOG completas guardadas en: {archivo_csv_features}")
        
        # Guardar reporte completo en TXT
        archivo_txt = os.path.join(self.results_dir, f'hog_reporte_completo_{nombre_imagen}_{timestamp}.txt')
        with open(archivo_txt, 'w', encoding='utf-8') as f:
            f.write("REPORTE COMPLETO - ANÁLISIS HOG\n")
            f.write("="*60 + "\n\n")
            f.write(f"Imagen analizada: {hog_data['imagen']}\n")
            f.write(f"Fecha de análisis: {hog_data['fecha_analisis']}\n")
            f.write(f"Dimensiones: {hog_data['dimensiones']}\n")
            f.write(f"Número de características: {hog_data['num_caracteristicas']}\n\n")
            
            f.write("ESTADÍSTICAS GENERALES:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Valor mínimo: {hog_data['valor_min']:.8f}\n")
            f.write(f"Valor máximo: {hog_data['valor_max']:.8f}\n")
            f.write(f"Promedio: {hog_data['promedio']:.8f}\n")
            f.write(f"Desviación estándar: {hog_data['desviacion_std']:.8f}\n")
            f.write(f"Energía total: {hog_data['energia_total']:.8f}\n")
            f.write(f"Entropía: {hog_data['entropia']:.8f}\n\n")
            
            f.write("CARACTERÍSTICAS HOG COMPLETAS:\n")
            f.write("-" * 35 + "\n")
            for i, feature in enumerate(resultados['hog_features_raw'].flatten(), 1):
                f.write(f"Característica {i:4d}: {feature:18.8f}\n")
        
        print(f"✅ Reporte HOG completo guardado en: {archivo_txt}")
        print(f"📊 Total de archivos generados: 3 (CSV estadísticas, CSV características, TXT reporte)")
    
    def _guardar_resultados_kaze(self, resultados, nombre_imagen):
        """Guarda los resultados KAZE en archivos CSV y TXT."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio si no existe
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Datos estadísticos generales
        kaze_stats = {
            'imagen': nombre_imagen,
            'dimensiones': f"{resultados['gray_image'].shape[0]}x{resultados['gray_image'].shape[1]}",
            'num_puntos_clave': resultados['num_keypoints'],
            'dimension_descriptor': resultados['descriptor_length'],
            'densidad_puntos': resultados['kp_density'],
            'fecha_analisis': datetime.now().isoformat()
        }
        
        if resultados['descriptors'] is not None:
            kaze_stats.update({
                'valor_min_descriptores': resultados['descriptor_mean'],
                'valor_max_descriptores': resultados['descriptor_std'],
                'promedio_descriptores': resultados['descriptor_energy'],
                'desviacion_std_descriptores': resultados['descriptor_entropy']
            })
        
        # Guardar estadísticas generales en CSV
        df_stats = pd.DataFrame([kaze_stats])
        archivo_csv_stats = os.path.join(self.results_dir, f'kaze_estadisticas_{nombre_imagen}_{timestamp}.csv')
        df_stats.to_csv(archivo_csv_stats, index=False, encoding='utf-8')
        print(f"✅ Estadísticas KAZE guardadas en: {archivo_csv_stats}")
        
        # Guardar información de puntos clave en CSV
        if resultados['keypoints']:
            keypoints_data = []
            for i, kp in enumerate(resultados['keypoints']):
                keypoints_data.append({
                    'punto_clave_id': i + 1,
                    'posicion_x': kp.pt[0],
                    'posicion_y': kp.pt[1],
                    'tamaño': kp.size,
                    'angulo': kp.angle,
                    'respuesta': kp.response
                })
            
            df_keypoints = pd.DataFrame(keypoints_data)
            archivo_csv_keypoints = os.path.join(self.results_dir, f'kaze_puntos_clave_{nombre_imagen}_{timestamp}.csv')
            df_keypoints.to_csv(archivo_csv_keypoints, index=False, encoding='utf-8')
            print(f"✅ Puntos clave KAZE guardados en: {archivo_csv_keypoints}")
        
        # Guardar descriptores completos en CSV
        if resultados['descriptors'] is not None:
            descriptor_columns = [f'descriptor_{j+1}' for j in range(resultados['descriptors'].shape[1])]
            df_descriptors = pd.DataFrame(resultados['descriptors'], columns=descriptor_columns)
            df_descriptors.insert(0, 'punto_clave_id', range(1, len(resultados['descriptors']) + 1))
            
            archivo_csv_descriptors = os.path.join(self.results_dir, f'kaze_descriptores_{nombre_imagen}_{timestamp}.csv')
            df_descriptors.to_csv(archivo_csv_descriptors, index=False, encoding='utf-8')
            print(f"✅ Descriptores KAZE completos guardados en: {archivo_csv_descriptors}")
        
        # Guardar reporte completo en TXT
        archivo_txt = os.path.join(self.results_dir, f'kaze_reporte_completo_{nombre_imagen}_{timestamp}.txt')
        with open(archivo_txt, 'w', encoding='utf-8') as f:
            f.write("REPORTE COMPLETO - ANÁLISIS KAZE\n")
            f.write("="*60 + "\n\n")
            f.write(f"Imagen analizada: {kaze_stats['imagen']}\n")
            f.write(f"Fecha de análisis: {kaze_stats['fecha_analisis']}\n")
            f.write(f"Dimensiones: {kaze_stats['dimensiones']}\n")
            f.write(f"Número de puntos clave: {kaze_stats['num_puntos_clave']}\n")
            f.write(f"Dimensión de descriptores: {kaze_stats['dimension_descriptor']}\n\n")
            
            if resultados['descriptors'] is not None:
                f.write("ESTADÍSTICAS DE DESCRIPTORES:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Promedio: {resultados['descriptor_mean']:.8f}\n")
                f.write(f"Desviación estándar: {resultados['descriptor_std']:.8f}\n")
                f.write(f"Energía: {resultados['descriptor_energy']:.8f}\n")
                f.write(f"Entropía: {resultados['descriptor_entropy']:.8f}\n\n")
            
            f.write("INFORMACIÓN DETALLADA DE PUNTOS CLAVE:\n")
            f.write("-" * 45 + "\n")
            for i, kp in enumerate(resultados['keypoints']):
                f.write(f"Punto clave {i+1}:\n")
                f.write(f"  Posición (x,y): ({kp.pt[0]:.2f}, {kp.pt[1]:.2f})\n")
                f.write(f"  Tamaño: {kp.size:.2f}\n")
                f.write(f"  Ángulo: {kp.angle:.2f}°\n")
                f.write(f"  Respuesta: {kp.response:.8f}\n")
                if resultados['descriptors'] is not None:
                    f.write("  Descriptor completo:\n")
                    descriptor_str = ', '.join([f"{val:.8f}" for val in resultados['descriptors'][i]])
                    f.write(f"    [{descriptor_str}]\n")
                f.write("\n")
        
        print(f"✅ Reporte KAZE completo guardado en: {archivo_txt}")
        archivos_generados = 2 + (1 if resultados['keypoints'] else 0) + (1 if resultados['descriptors'] is not None else 0)
        print(f"📊 Total de archivos generados: {archivos_generados}")

# Función de utilidad
def analizar_hog_kaze_imagen(imagen_path, output_dir="./resultados"):
    """
    Función de conveniencia para análisis HOG-KAZE.
    
    Args:
        imagen_path (str): Ruta a la imagen
        output_dir (str): Directorio de salida
        
    Returns:
        dict: Resultados del análisis
    """
    analyzer = HOGKAZEAnalyzer(output_dir)
    resultado = analyzer.analisis_combinado_hog_kaze(imagen_path)
    if resultado:
        analyzer.guardar_resultados('csv')
        analyzer.generar_reporte_hog_kaze()
    return resultado