#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Métricas de Comparación
=================================

Implementación de métricas para comparar el desempeño de diferentes
algoritmos de extracción de características en imágenes de tráfico vehicular.

Métricas implementadas:
- Tiempo de ejecución
- Número de características detectadas
- Calidad de características
- Robustez ante transformaciones
- Repetibilidad
- Precisión de matching
"""

import cv2
import numpy as np
import time
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from datetime import datetime


class ComparisonAnalyzer:
    """Analizador para comparar algoritmos de extracción de características."""
    
    def __init__(self):
        """Inicializar el analizador de comparación."""
        self.comparison_results = []
        
    def compare_feature_detectors(self, imagen, algorithms=None):
        """
        Compara diferentes detectores de características.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            algorithms (list): Lista de algoritmos a comparar
            
        Returns:
            dict: Resultados de comparación
        """
        if algorithms is None:
            algorithms = ['ORB', 'SIFT', 'AKAZE', 'KAZE']
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen
        
        resultados = {}
        
        for alg_name in algorithms:
            print(f"Evaluando {alg_name}...")
            
            try:
                # Medir tiempo y extraer características
                inicio = time.time()
                keypoints, descriptors = self._extract_features(imagen_gris, alg_name)
                tiempo_ejecucion = time.time() - inicio
                
                # Calcular métricas
                metricas = {
                    'tiempo_ejecucion': tiempo_ejecucion,
                    'num_keypoints': len(keypoints) if keypoints else 0,
                    'descriptor_dimension': descriptors.shape[1] if descriptors is not None else 0,
                    'descriptor_type': descriptors.dtype if descriptors is not None else 'N/A',
                    'distribucion_espacial': self._calculate_spatial_distribution(keypoints, imagen.shape),
                    'promedio_response': np.mean([kp.response for kp in keypoints]) if keypoints else 0,
                    'promedio_size': np.mean([kp.size for kp in keypoints]) if keypoints else 0
                }
                
                resultados[alg_name] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'metricas': metricas,
                    'success': True
                }
                
            except Exception as e:
                print(f"Error con {alg_name}: {str(e)}")
                resultados[alg_name] = {
                    'keypoints': None,
                    'descriptors': None,
                    'metricas': {},
                    'success': False,
                    'error': str(e)
                }
        
        return resultados
    
    def _extract_features(self, imagen, algorithm):
        """
        Extrae características usando un algoritmo específico.
        
        Args:
            imagen (np.ndarray): Imagen en escala de grises
            algorithm (str): Nombre del algoritmo
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        if algorithm == 'ORB':
            detector = cv2.ORB_create(nfeatures=500)
        elif algorithm == 'SIFT':
            detector = cv2.SIFT_create()
        elif algorithm == 'AKAZE':
            detector = cv2.AKAZE_create()
        elif algorithm == 'KAZE':
            detector = cv2.KAZE_create()
        elif algorithm == 'BRISK':
            detector = cv2.BRISK_create()
        else:
            raise ValueError(f"Algoritmo '{algorithm}' no soportado")
        
        keypoints, descriptors = detector.detectAndCompute(imagen, None)
        return keypoints, descriptors
    
    def _calculate_spatial_distribution(self, keypoints, image_shape):
        """
        Calcula la distribución espacial de los keypoints.
        
        Args:
            keypoints (list): Lista de keypoints
            image_shape (tuple): Dimensiones de la imagen
            
        Returns:
            dict: Métricas de distribución espacial
        """
        if not keypoints or len(keypoints) == 0:
            return {'uniformidad': 0, 'cobertura': 0}
        
        h, w = image_shape[:2]
        
        # Crear mapa de calor
        heatmap = np.zeros((h, w), dtype=np.float32)
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < w and 0 <= y < h:
                heatmap[y, x] += 1
        
        # Dividir imagen en cuadrantes
        n_quadrants = 4
        quad_h, quad_w = h // 2, w // 2
        quadrant_counts = []
        
        for i in range(2):
            for j in range(2):
                y1, y2 = i * quad_h, (i + 1) * quad_h
                x1, x2 = j * quad_w, (j + 1) * quad_w
                count = sum(1 for kp in keypoints 
                           if x1 <= kp.pt[0] < x2 and y1 <= kp.pt[1] < y2)
                quadrant_counts.append(count)
        
        # Calcular uniformidad (menor desviación = más uniforme)
        uniformidad = 1.0 - (np.std(quadrant_counts) / (np.mean(quadrant_counts) + 1e-6))
        uniformidad = max(0, min(1, uniformidad))
        
        # Calcular cobertura (porcentaje de la imagen con keypoints)
        cobertura = np.count_nonzero(heatmap) / (h * w)
        
        return {
            'uniformidad': uniformidad,
            'cobertura': cobertura,
            'quadrant_counts': quadrant_counts
        }
    
    def compare_matching_performance(self, img1, img2, algorithms=None):
        """
        Compara el desempeño de matching entre dos imágenes.
        
        Args:
            img1 (np.ndarray): Primera imagen
            img2 (np.ndarray): Segunda imagen
            algorithms (list): Lista de algoritmos
            
        Returns:
            dict: Resultados de matching
        """
        if algorithms is None:
            algorithms = ['ORB', 'AKAZE']
        
        resultados = {}
        
        for alg in algorithms:
            print(f"Evaluando matching con {alg}...")
            
            try:
                # Extraer características de ambas imágenes
                if len(img1.shape) == 3:
                    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                else:
                    img1_gray = img1
                    
                if len(img2.shape) == 3:
                    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                else:
                    img2_gray = img2
                
                kp1, desc1 = self._extract_features(img1_gray, alg)
                kp2, desc2 = self._extract_features(img2_gray, alg)
                
                if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
                    resultados[alg] = {'success': False, 'error': 'No se detectaron características'}
                    continue
                
                # Matching
                inicio = time.time()
                
                if alg in ['ORB', 'AKAZE', 'BRISK']:
                    # Descriptor binario - usar Hamming
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                else:
                    # Descriptor flotante - usar L2
                    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                
                matches = matcher.knnMatch(desc1, desc2, k=2)
                
                # Ratio test
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                tiempo_matching = time.time() - inicio
                
                # Calcular métricas
                matching_ratio = len(good_matches) / len(kp1) if len(kp1) > 0 else 0
                
                resultados[alg] = {
                    'num_keypoints_img1': len(kp1),
                    'num_keypoints_img2': len(kp2),
                    'num_matches_total': len(matches),
                    'num_good_matches': len(good_matches),
                    'matching_ratio': matching_ratio,
                    'tiempo_matching': tiempo_matching,
                    'success': True
                }
                
            except Exception as e:
                print(f"Error en matching con {alg}: {str(e)}")
                resultados[alg] = {'success': False, 'error': str(e)}
        
        return resultados
    
    def generate_comparison_report(self, resultados):
        """
        Genera un reporte de comparación visual.
        
        Args:
            resultados (dict): Resultados de comparación
        """
        algorithms = [alg for alg in resultados.keys() if resultados[alg]['success']]
        
        if not algorithms:
            print("No hay resultados exitosos para comparar")
            return
        
        # Preparar datos para gráficos
        nombres = []
        num_features = []
        tiempos = []
        uniformidades = []
        coberturas = []
        
        for alg in algorithms:
            metricas = resultados[alg]['metricas']
            nombres.append(alg)
            num_features.append(metricas['num_keypoints'])
            tiempos.append(metricas['tiempo_ejecucion'])
            
            dist = metricas.get('distribucion_espacial', {})
            uniformidades.append(dist.get('uniformidad', 0))
            coberturas.append(dist.get('cobertura', 0))
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparación de Algoritmos de Extracción de Características', 
                     fontsize=16, fontweight='bold')
        
        # 1. Número de características
        axes[0, 0].bar(nombres, num_features, color='steelblue')
        axes[0, 0].set_title('Número de Características Detectadas')
        axes[0, 0].set_ylabel('Cantidad')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Tiempo de ejecución
        axes[0, 1].bar(nombres, tiempos, color='coral')
        axes[0, 1].set_title('Tiempo de Ejecución')
        axes[0, 1].set_ylabel('Tiempo (segundos)')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Uniformidad de distribución
        axes[1, 0].bar(nombres, uniformidades, color='mediumseagreen')
        axes[1, 0].set_title('Uniformidad de Distribución Espacial')
        axes[1, 0].set_ylabel('Score (0-1)')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Cobertura de imagen
        axes[1, 1].bar(nombres, coberturas, color='mediumpurple')
        axes[1, 1].set_title('Cobertura de la Imagen')
        axes[1, 1].set_ylabel('Proporción (0-1)')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Reporte textual
        print("\n" + "="*70)
        print("REPORTE DE COMPARACIÓN DE ALGORITMOS")
        print("="*70)
        
        for alg in algorithms:
            metricas = resultados[alg]['metricas']
            print(f"\n{alg}:")
            print(f"  • Características: {metricas['num_keypoints']}")
            print(f"  • Tiempo: {metricas['tiempo_ejecucion']:.4f}s")
            print(f"  • Dimensión descriptor: {metricas['descriptor_dimension']}")
            print(f"  • Tipo descriptor: {metricas['descriptor_type']}")
            
            dist = metricas.get('distribucion_espacial', {})
            print(f"  • Uniformidad: {dist.get('uniformidad', 0):.3f}")
            print(f"  • Cobertura: {dist.get('cobertura', 0):.3f}")
        
        print("\n" + "="*70)
        
        # Recomendaciones
        print("\n📊 RECOMENDACIONES:")
        
        # Algoritmo más rápido
        idx_rapido = np.argmin(tiempos)
        print(f"⚡ Más rápido: {nombres[idx_rapido]} ({tiempos[idx_rapido]:.4f}s)")
        
        # Algoritmo con más características
        idx_features = np.argmax(num_features)
        print(f"🔍 Más características: {nombres[idx_features]} ({num_features[idx_features]} puntos)")
        
        # Algoritmo más uniforme
        idx_uniforme = np.argmax(uniformidades)
        print(f"📐 Distribución más uniforme: {nombres[idx_uniforme]} ({uniformidades[idx_uniforme]:.3f})")
        
        print("\n")
    
    def evaluate_algorithm_robustness(self, imagen, algorithm='ORB', transformations=None):
        """
        Evalúa la robustez de un algoritmo ante transformaciones.
        
        Args:
            imagen (np.ndarray): Imagen original
            algorithm (str): Algoritmo a evaluar
            transformations (list): Lista de transformaciones a aplicar
            
        Returns:
            dict: Resultados de robustez
        """
        if transformations is None:
            transformations = ['rotation', 'scale', 'brightness', 'noise']
        
        # Extraer características de imagen original
        if len(imagen.shape) == 3:
            img_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = imagen
        
        kp_orig, desc_orig = self._extract_features(img_gray, algorithm)
        
        resultados = {'original': {'num_keypoints': len(kp_orig)}}
        
        for transform in transformations:
            print(f"Evaluando robustez ante {transform}...")
            
            # Aplicar transformación
            img_transformed = self._apply_transformation(imagen, transform)
            
            if len(img_transformed.shape) == 3:
                img_t_gray = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2GRAY)
            else:
                img_t_gray = img_transformed
            
            # Extraer características
            kp_trans, desc_trans = self._extract_features(img_t_gray, algorithm)
            
            # Matching para ver repetibilidad
            if desc_orig is not None and desc_trans is not None:
                if algorithm in ['ORB', 'AKAZE', 'BRISK']:
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                else:
                    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                
                matches = matcher.match(desc_orig, desc_trans)
                repetibilidad = len(matches) / len(kp_orig) if len(kp_orig) > 0 else 0
            else:
                repetibilidad = 0
            
            resultados[transform] = {
                'num_keypoints': len(kp_trans),
                'num_matches': len(matches) if desc_orig is not None and desc_trans is not None else 0,
                'repetibilidad': repetibilidad
            }
        
        return resultados
    
    def _apply_transformation(self, imagen, transform_type):
        """
        Aplica una transformación a la imagen.
        
        Args:
            imagen (np.ndarray): Imagen original
            transform_type (str): Tipo de transformación
            
        Returns:
            np.ndarray: Imagen transformada
        """
        h, w = imagen.shape[:2]
        
        if transform_type == 'rotation':
            # Rotar 15 grados
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
            return cv2.warpAffine(imagen, matrix, (w, h))
        
        elif transform_type == 'scale':
            # Escalar 1.2x
            return cv2.resize(imagen, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
        
        elif transform_type == 'brightness':
            # Aumentar brillo
            return cv2.convertScaleAbs(imagen, alpha=1.0, beta=30)
        
        elif transform_type == 'noise':
            # Agregar ruido gaussiano
            noise = np.random.normal(0, 10, imagen.shape).astype(np.uint8)
            return cv2.add(imagen, noise)
        
        return imagen
