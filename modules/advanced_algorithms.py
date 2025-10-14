#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Algoritmos Avanzados para Análisis de Tráfico Vehicular
=================================================================

Implementación de algoritmos avanzados para identificación de objetos 
en imágenes de tráfico vehicular:

- FREAK (Fast Retina Keypoint): Descriptor binario inspirado en la retina humana
- AKAZE (Accelerated KAZE): Versión acelerada del detector KAZE
- GrabCut: Segmentación interactiva de objetos
- LoG (Laplaciano de Gauss): Detección de bordes avanzada
- Optical Flow: Análisis de movimiento entre frames
"""

import os
import csv
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from skimage import feature, segmentation, filters, morphology
from skimage.filters import gaussian, laplace
from skimage.measure import label, regionprops
import seaborn as sns

class AdvancedAnalyzer:
    """Analizador de algoritmos avanzados para tráfico vehicular."""
    
    def __init__(self, output_dir="./resultados"):
        """
        Inicializar el analizador avanzado.
        
        Args:
            output_dir (str): Directorio de salida para resultados
        """
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "advanced_analysis")
        os.makedirs(self.results_dir, exist_ok=True)
        self.current_results = []
        
        # Configuraciones por defecto
        self.config = {
            'akaze': {
                'threshold': 0.003,
                'nOctaves': 4,
                'nOctaveLayers': 4,
                'diffusivity': cv2.KAZE_DIFF_PM_G2,
                'descriptor_type': cv2.AKAZE_DESCRIPTOR_MLDB,
                'descriptor_size': 0,
                'descriptor_channels': 3
            },
            'log': {
                'sigma_values': [1, 2, 3, 4, 5],
                'threshold_factor': 0.1
            },
            'grabcut': {
                'iterations': 5,
                'margin': 10
            },
            'optical_flow': {
                'pyr_scale': 0.5,
                'levels': 3,
                'winsize': 15,
                'iterations': 3,
                'poly_n': 5,
                'poly_sigma': 1.2,
                'flags': 0
            }
        }
    
    def extraer_caracteristicas_freak(self, imagen, visualizar=True, mostrar_descriptores=True, guardar_resultados=False, nombre_imagen="imagen_freak"):
        """
        Extrae características FREAK (Fast Retina Keypoint).
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si generar visualización
            mostrar_descriptores (bool): Si mostrar descriptores en consola
            guardar_resultados (bool): Si guardar resultados en archivos
            nombre_imagen (str): Nombre base para archivos guardados
            
        Returns:
            dict: Características FREAK extraídas
        """
        print("🔍 Extrayendo características FREAK...")
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        try:
            # Crear detector STAR para puntos clave
            star = cv2.xfeatures2d.StarDetector_create()
            keypoints = star.detect(imagen_gris, None)
            
            # Crear descriptor FREAK
            freak = cv2.xfeatures2d.FREAK_create()
            keypoints, descriptors = freak.compute(imagen_gris, keypoints)
            
            freak_available = True
        except (AttributeError, cv2.error):
            # Fallback usando ORB para puntos clave y análisis similar
            print("⚠️  FREAK no disponible, usando análisis alternativo...")
            orb = cv2.ORB_create(nfeatures=500)
            keypoints, descriptors = orb.detectAndCompute(imagen_gris, None)
            freak_available = False
        
        # Análisis de puntos clave
        num_keypoints = len(keypoints)
        
        if num_keypoints > 0 and descriptors is not None:
            # Estadísticas de localización
            kp_x = [kp.pt[0] for kp in keypoints]
            kp_y = [kp.pt[1] for kp in keypoints]
            kp_size = [kp.size for kp in keypoints]
            kp_response = [kp.response for kp in keypoints]
            
            # Análisis de descriptores FREAK (binarios, 512 bits)
            bit_counts = np.unpackbits(descriptors, axis=1) if descriptors.dtype == np.uint8 else descriptors
            
            # Estadísticas básicas
            freak_stats = {
                'freak_num_keypoints': num_keypoints,
                'freak_kp_mean_x': np.mean(kp_x),
                'freak_kp_std_x': np.std(kp_x),
                'freak_kp_mean_y': np.mean(kp_y),
                'freak_kp_std_y': np.std(kp_y),
                'freak_kp_mean_size': np.mean(kp_size),
                'freak_kp_std_size': np.std(kp_size),
                'freak_kp_mean_response': np.mean(kp_response),
                'freak_kp_density': num_keypoints / (imagen_gris.shape[0] * imagen_gris.shape[1])
            }
            
            # Análisis específico de FREAK (inspirado en retina)
            if descriptors.dtype == np.uint8:  # Descriptores binarios
                # Análisis de patrones binarios
                freak_stats.update({
                    'freak_bit_ratio': np.mean(bit_counts),
                    'freak_bit_entropy': self._calculate_entropy(bit_counts.flatten()),
                    'freak_descriptor_diversity': self._calculate_binary_diversity(descriptors),
                    'freak_pattern_complexity': self._calculate_pattern_complexity(descriptors)
                })
            else:  # Fallback con descriptores continuos
                freak_stats.update({
                    'freak_descriptor_mean': np.mean(descriptors),
                    'freak_descriptor_std': np.std(descriptors),
                    'freak_descriptor_energy': np.sum(descriptors ** 2),
                    'freak_descriptor_sparsity': np.sum(np.abs(descriptors) < 0.01) / descriptors.size
                })
            
            # Análisis espacial tipo retina (círculos concéntricos)
            retina_stats = self._analizar_patron_retina(kp_x, kp_y, imagen_gris.shape)
            freak_stats.update(retina_stats)
            
        else:
            freak_stats = self._get_empty_freak_stats()
        
        freak_stats['freak_algorithm_available'] = freak_available
        
        # Mostrar información detallada en consola
        if mostrar_descriptores:
            descriptor_name = 'FREAK' if freak_available else 'ORB (Fallback)'
            print(f"\n🔑 ANÁLISIS DE CARACTERÍSTICAS {descriptor_name}")
            print("=" * 60)
            print(f"📐 Dimensiones de la imagen: {imagen_gris.shape}")
            print(f"🎯 Número de puntos clave detectados: {num_keypoints}")
            
            print(f"\n📊 Estadísticas:")
            print(f"• Algoritmo usado: {descriptor_name}")
            print(f"• Puntos clave encontrados: {len(keypoints)}")
            if descriptors is not None:
                print(f"• Dimensión del descriptor: {descriptors.shape[1] if len(descriptors.shape) > 1 else 'N/A'}")
                print(f"• Tipo de descriptor: {'Binario' if descriptors.dtype == np.uint8 else 'Flotante'}")
                
            if num_keypoints > 0:
                print(f"• Ubicación promedio: ({np.mean(kp_x):.1f}, {np.mean(kp_y):.1f})")
                print(f"• Tamaño promedio: {np.mean(kp_size):.2f} ± {np.std(kp_size):.2f}")
                print(f"• Respuesta promedio: {np.mean(kp_response):.6f}")
                print(f"• Densidad de puntos: {freak_stats['freak_kp_density']:.6f}")
                
                # Mostrar análisis específico de FREAK
                if freak_available and descriptors is not None and descriptors.dtype == np.uint8:
                    print(f"• Ratio de bits activos: {freak_stats.get('freak_bit_ratio', 0):.4f}")
                    print(f"• Complejidad de patrón: {freak_stats.get('freak_pattern_complexity', 0):.4f}")
                    print(f"• Densidad centro retina: {freak_stats.get('retina_center_density', 0):.4f}")
                    print(f"• Densidad periferia retina: {freak_stats.get('retina_periphery_density', 0):.4f}")
            
            print("=" * 60)
        
        resultados = {
            **freak_stats,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'gray_image': imagen_gris,
            'nombre_imagen': nombre_imagen
        }
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            self._guardar_resultados_freak(resultados, nombre_imagen)
        
        if visualizar:
            # Establecer flag para guardar visualización si se están guardando resultados
            if guardar_resultados:
                self._save_visualization = True
            self._visualizar_freak(resultados)
            if hasattr(self, '_save_visualization'):
                delattr(self, '_save_visualization')
        
        return resultados
    
    def extraer_caracteristicas_akaze(self, imagen, visualizar=True, mostrar_descriptores=True, guardar_resultados=False, nombre_imagen="imagen_akaze"):
        """
        Extrae características AKAZE (Accelerated KAZE).
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si generar visualización
            mostrar_descriptores (bool): Si mostrar descriptores en consola
            guardar_resultados (bool): Si guardar resultados en archivos
            nombre_imagen (str): Nombre base para archivos guardados
            
        Returns:
            dict: Características AKAZE extraídas
        """
        print("🔍 Extrayendo características AKAZE...")
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # Crear detector AKAZE
        akaze = cv2.AKAZE_create(
            descriptor_type=self.config['akaze']['descriptor_type'],
            descriptor_size=self.config['akaze']['descriptor_size'],
            descriptor_channels=self.config['akaze']['descriptor_channels'],
            threshold=self.config['akaze']['threshold'],
            nOctaves=self.config['akaze']['nOctaves'],
            nOctaveLayers=self.config['akaze']['nOctaveLayers'],
            diffusivity=self.config['akaze']['diffusivity']
        )
        
        # Detectar puntos clave y calcular descriptores
        keypoints, descriptors = akaze.detectAndCompute(imagen_gris, None)
        
        # Análisis de puntos clave
        num_keypoints = len(keypoints)
        
        if num_keypoints > 0:
            # Estadísticas de localización
            kp_x = [kp.pt[0] for kp in keypoints]
            kp_y = [kp.pt[1] for kp in keypoints]
            kp_size = [kp.size for kp in keypoints]
            kp_angle = [kp.angle for kp in keypoints]
            kp_response = [kp.response for kp in keypoints]
            kp_octave = [kp.octave for kp in keypoints]
            
            # Estadísticas básicas
            akaze_stats = {
                'akaze_num_keypoints': num_keypoints,
                'akaze_kp_mean_x': np.mean(kp_x),
                'akaze_kp_std_x': np.std(kp_x),
                'akaze_kp_mean_y': np.mean(kp_y),
                'akaze_kp_std_y': np.std(kp_y),
                'akaze_kp_mean_size': np.mean(kp_size),
                'akaze_kp_std_size': np.std(kp_size),
                'akaze_kp_mean_response': np.mean(kp_response),
                'akaze_kp_std_response': np.std(kp_response),
                'akaze_kp_density': num_keypoints / (imagen_gris.shape[0] * imagen_gris.shape[1])
            }
            
            # Análisis de escalas
            octave_counts = np.bincount([oct & 0xFF for oct in kp_octave])
            akaze_stats.update({
                'akaze_num_scales': len(octave_counts),
                'akaze_scale_entropy': self._calculate_entropy(octave_counts),
                'akaze_dominant_scale': np.argmax(octave_counts) if len(octave_counts) > 0 else 0
            })
            
            # Análisis de orientaciones
            valid_angles = [angle for angle in kp_angle if angle >= 0]
            if valid_angles:
                angle_hist, _ = np.histogram(valid_angles, bins=36, range=(0, 360))
                akaze_stats.update({
                    'akaze_mean_angle': np.mean(valid_angles),
                    'akaze_std_angle': np.std(valid_angles),
                    'akaze_angle_entropy': self._calculate_entropy(angle_hist)
                })
            
            # Análisis de descriptores AKAZE
            if descriptors is not None:
                descriptor_stats = self._analizar_descriptores_akaze(descriptors)
                akaze_stats.update(descriptor_stats)
            
            # Análisis de estabilidad multi-escala
            stability_stats = self._analizar_estabilidad_multiescala(keypoints, imagen_gris)
            akaze_stats.update(stability_stats)
            
        else:
            akaze_stats = self._get_empty_akaze_stats()
        
        # Mostrar información detallada en consola
        if mostrar_descriptores:
            print(f"\n🔑 ANÁLISIS DE CARACTERÍSTICAS AKAZE")
            print("=" * 60)
            print(f"📐 Dimensiones de la imagen: {imagen_gris.shape}")
            
            if num_keypoints > 0:
                print(f"\n📊 Análisis AKAZE:")
                print(f"• Puntos clave detectados: {len(keypoints)}")
                print(f"• Tamaño promedio: {np.mean(kp_size):.2f} ± {np.std(kp_size):.2f}")
                print(f"• Respuesta promedio: {np.mean(kp_response):.4f}")
                print(f"• Rango de orientaciones: {np.min(kp_angle):.1f}° a {np.max(kp_angle):.1f}°")
                if descriptors is not None:
                    print(f"• Dimensión del descriptor: {descriptors.shape[1]}")
                    print(f"• Tipo de descriptor: {descriptors.dtype}")
                print(f"• Densidad de puntos: {akaze_stats['akaze_kp_density']:.6f}")
                print(f"• Número de escalas: {akaze_stats['akaze_num_scales']}")
                print(f"• Estabilidad multi-escala: {akaze_stats.get('akaze_scale_stability', 0):.4f}")
            else:
                print("⚠️ No se detectaron puntos clave con AKAZE")
            
            print("=" * 60)
        
        resultados = {
            **akaze_stats,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'gray_image': imagen_gris,
            'nombre_imagen': nombre_imagen
        }
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            self._guardar_resultados_akaze(resultados, nombre_imagen)
        
        if visualizar:
            # Establecer flag para guardar visualización si se están guardando resultados
            if guardar_resultados:
                self._save_visualization = True
            self._visualizar_akaze(resultados)
            if hasattr(self, '_save_visualization'):
                delattr(self, '_save_visualization')
        
        return resultados
    
    def analizar_grabcut_segmentation(self, imagen, visualizar=True, mostrar_descriptores=True, guardar_resultados=False, nombre_imagen="imagen_grabcut"):
        """
        Realiza segmentación usando GrabCut.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si generar visualización
            mostrar_descriptores (bool): Si mostrar estadísticas en consola
            guardar_resultados (bool): Si guardar resultados en archivos
            nombre_imagen (str): Nombre base para archivos guardados
            
        Returns:
            dict: Resultados de segmentación GrabCut
        """
        print("🔍 Analizando segmentación GrabCut...")
        
        if len(imagen.shape) != 3:
            print("⚠️  GrabCut requiere imagen en color, convirtiendo...")
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        
        h, w = imagen.shape[:2]
        margin = self.config['grabcut']['margin']
        
        # Crear rectángulo inicial (región central)
        rect = (margin, margin, w - 2*margin, h - 2*margin)
        
        # Inicializar máscaras
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Aplicar GrabCut
        cv2.grabCut(imagen, mask, rect, bgd_model, fgd_model, 
                   self.config['grabcut']['iterations'], cv2.GC_INIT_WITH_RECT)
        
        # Crear máscara binaria
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Análisis de la segmentación
        foreground_pixels = np.sum(mask2)
        background_pixels = np.sum(1 - mask2)
        total_pixels = h * w
        
        # Análisis de componentes conectados
        labeled_mask = label(mask2)
        regions = regionprops(labeled_mask)
        
        # Estadísticas de regiones
        if regions:
            areas = [region.area for region in regions]
            centroids = [region.centroid for region in regions]
            
            grabcut_stats = {
                'grabcut_foreground_ratio': foreground_pixels / total_pixels,
                'grabcut_background_ratio': background_pixels / total_pixels,
                'grabcut_num_regions': len(regions),
                'grabcut_largest_region_area': max(areas) if areas else 0,
                'grabcut_mean_region_area': np.mean(areas) if areas else 0,
                'grabcut_std_region_area': np.std(areas) if areas else 0,
                'grabcut_region_compactness': np.mean([region.extent for region in regions]) if regions else 0,
                'grabcut_region_solidity': np.mean([region.solidity for region in regions]) if regions else 0
            }
            
            # Análisis de distribución espacial de regiones
            if centroids:
                centroid_x = [c[1] for c in centroids]  # columna
                centroid_y = [c[0] for c in centroids]  # fila
                
                grabcut_stats.update({
                    'grabcut_centroid_mean_x': np.mean(centroid_x),
                    'grabcut_centroid_std_x': np.std(centroid_x),
                    'grabcut_centroid_mean_y': np.mean(centroid_y),
                    'grabcut_centroid_std_y': np.std(centroid_y)
                })
        else:
            grabcut_stats = self._get_empty_grabcut_stats()
        
        # Análisis de calidad de segmentación
        edge_coherence = self._calcular_coherencia_bordes(imagen, mask2)
        grabcut_stats['grabcut_edge_coherence'] = edge_coherence
        
        # Mostrar información detallada en consola
        if mostrar_descriptores:
            print(f"\n🔑 ANÁLISIS DE SEGMENTACIÓN GRABCUT")
            print("=" * 60)
            print(f"📐 Dimensiones de la imagen: {imagen.shape}")
            
            print(f"\n📊 Estadísticas GrabCut:")
            print(f"• Píxeles totales: {total_pixels:,}")
            print(f"• Píxeles de primer plano: {foreground_pixels:,} ({100*foreground_pixels/total_pixels:.1f}%)")
            print(f"• Píxeles de fondo: {background_pixels:,} ({100*background_pixels/total_pixels:.1f}%)")
            print(f"• Iteraciones: {self.config['grabcut']['iterations']}")
            
            if regions:
                print(f"• Número de regiones: {len(regions)}")
                print(f"• Área de región más grande: {max(areas):,} píxeles")
                print(f"• Área promedio de regiones: {np.mean(areas):.1f} píxeles")
                print(f"• Compacidad promedio: {grabcut_stats['grabcut_region_compactness']:.4f}")
                print(f"• Solidez promedio: {grabcut_stats['grabcut_region_solidity']:.4f}")
            
            print(f"• Coherencia de bordes: {edge_coherence:.4f}")
            print("=" * 60)
        
        resultados = {
            **grabcut_stats,
            'original_image': imagen,
            'segmentation_mask': mask2,
            'labeled_regions': labeled_mask,
            'regions': regions,
            'nombre_imagen': nombre_imagen
        }
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            self._guardar_resultados_grabcut(resultados, nombre_imagen)
        
        if visualizar:
            # Establecer flag para guardar visualización si se están guardando resultados
            if guardar_resultados:
                self._save_visualization = True
            self._visualizar_grabcut(resultados)
            if hasattr(self, '_save_visualization'):
                delattr(self, '_save_visualization')
        
        return resultados
    
    def analizar_log_detector(self, imagen, visualizar=True, mostrar_descriptores=True, guardar_resultados=False, nombre_imagen="imagen_log"):
        """
        Analiza usando Laplaciano de Gauss (LoG).
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si generar visualización
            mostrar_descriptores (bool): Si mostrar estadísticas en consola
            guardar_resultados (bool): Si guardar resultados en archivos
            nombre_imagen (str): Nombre base para archivos guardados
            
        Returns:
            dict: Resultados del análisis LoG
        """
        print("🔍 Analizando con Laplaciano de Gauss (LoG)...")
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # Normalizar imagen
        imagen_norm = imagen_gris.astype(np.float64) / 255.0
        
        # Aplicar LoG con diferentes sigmas
        log_responses = []
        log_stats = {}
        
        for i, sigma in enumerate(self.config['log']['sigma_values']):
            # Aplicar filtro Gaussiano seguido de Laplaciano
            gaussian_img = gaussian(imagen_norm, sigma=sigma)
            log_response = laplace(gaussian_img)
            log_responses.append(log_response)
            
            # Estadísticas por escala
            log_stats[f'log_sigma_{sigma}_mean'] = np.mean(np.abs(log_response))
            log_stats[f'log_sigma_{sigma}_std'] = np.std(log_response)
            log_stats[f'log_sigma_{sigma}_energy'] = np.sum(log_response ** 2)
            log_stats[f'log_sigma_{sigma}_entropy'] = self._calculate_entropy(log_response.flatten())
        
        # Análisis multi-escala
        log_stack = np.stack(log_responses, axis=2)
        
        # Detección de blobs usando máximos locales
        threshold = self.config['log']['threshold_factor'] * np.max(np.abs(log_stack))
        
        # Encontrar máximos locales en el espacio de escalas
        local_maxima = []
        for i in range(1, len(self.config['log']['sigma_values']) - 1):
            current_scale = np.abs(log_responses[i])
            prev_scale = np.abs(log_responses[i-1])
            next_scale = np.abs(log_responses[i+1])
            
            # Máximos locales en escala y espacio
            mask = (current_scale > prev_scale) & (current_scale > next_scale) & (current_scale > threshold)
            coords = np.where(mask)
            
            for y, x in zip(coords[0], coords[1]):
                local_maxima.append({
                    'x': x, 'y': y, 'sigma': self.config['log']['sigma_values'][i],
                    'response': current_scale[y, x]
                })
        
        # Estadísticas de detección
        num_blobs = len(local_maxima)
        
        if local_maxima:
            blob_x = [blob['x'] for blob in local_maxima]
            blob_y = [blob['y'] for blob in local_maxima]
            blob_sigma = [blob['sigma'] for blob in local_maxima]
            blob_response = [blob['response'] for blob in local_maxima]
            
            detection_stats = {
                'log_num_blobs': num_blobs,
                'log_blob_density': num_blobs / (imagen_gris.shape[0] * imagen_gris.shape[1]),
                'log_mean_x': np.mean(blob_x),
                'log_std_x': np.std(blob_x),
                'log_mean_y': np.mean(blob_y),
                'log_std_y': np.std(blob_y),
                'log_mean_sigma': np.mean(blob_sigma),
                'log_std_sigma': np.std(blob_sigma),
                'log_mean_response': np.mean(blob_response),
                'log_std_response': np.std(blob_response)
            }
        else:
            detection_stats = {
                'log_num_blobs': 0, 'log_blob_density': 0,
                'log_mean_x': 0, 'log_std_x': 0,
                'log_mean_y': 0, 'log_std_y': 0,
                'log_mean_sigma': 0, 'log_std_sigma': 0,
                'log_mean_response': 0, 'log_std_response': 0
            }
        
        # Análisis de estructura multi-escala
        scale_consistency = self._analizar_consistencia_escalas(log_responses)
        
        all_stats = {**log_stats, **detection_stats, **scale_consistency}
        
        # Mostrar información detallada en consola
        if mostrar_descriptores:
            print(f"\n🔑 ANÁLISIS LAPLACIANO DE GAUSS (LoG)")
            print("=" * 60)
            print(f"📐 Dimensiones de la imagen: {imagen_gris.shape}")
            
            print(f"\n📊 Estadísticas LoG:")
            print(f"• Valores de sigma analizados: {self.config['log']['sigma_values']}")
            print(f"• Número de blobs detectados: {num_blobs}")
            print(f"• Densidad de blobs: {detection_stats['log_blob_density']:.6f}")
            
            if num_blobs > 0:
                print(f"• Ubicación promedio: ({detection_stats['log_mean_x']:.1f}, {detection_stats['log_mean_y']:.1f})")
                print(f"• Sigma promedio: {detection_stats['log_mean_sigma']:.2f} ± {detection_stats['log_std_sigma']:.2f}")
                print(f"• Respuesta promedio: {detection_stats['log_mean_response']:.6f}")
                print(f"• Consistencia entre escalas: {scale_consistency.get('log_scale_consistency', 0):.4f}")
            
            # Estadísticas por escala
            print(f"\n📈 Análisis multi-escala:")
            for i, sigma in enumerate(self.config['log']['sigma_values']):
                if i < len(log_stats):
                    response_mean = log_stats.get(f'log_sigma_{sigma}_mean', 0)
                    response_std = log_stats.get(f'log_sigma_{sigma}_std', 0)
                    print(f"• Sigma {sigma}: μ={response_mean:.4f}, σ={response_std:.4f}")
            
            print("=" * 60)
        
        resultados = {
            **all_stats,
            'log_responses': log_responses,
            'local_maxima': local_maxima,
            'gray_image': imagen_gris,
            'normalized_image': imagen_norm,
            'nombre_imagen': nombre_imagen
        }
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            self._guardar_resultados_log(resultados, nombre_imagen)
        
        if visualizar:
            self._visualizar_log(resultados)
        
        return resultados
    
    def analizar_optical_flow(self, imagen1, imagen2=None, visualizar=True, mostrar_descriptores=True, guardar_resultados=False, nombre_imagen="imagen_optical_flow"):
        """
        Analiza flujo óptico entre dos imágenes.
        
        Args:
            imagen1 (np.ndarray): Primera imagen o imagen única
            imagen2 (np.ndarray o str, optional): Segunda imagen (array) o ruta a la segunda imagen
            visualizar (bool): Si generar visualización
            mostrar_descriptores (bool): Si mostrar estadísticas en consola
            guardar_resultados (bool): Si guardar resultados en archivos
            nombre_imagen (str): Nombre base para archivos guardados
            
        Returns:
            dict: Resultados del análisis de flujo óptico
        """
        print("🔍 Analizando flujo óptico...")
        
        # Si imagen2 es una ruta (string), cargarla
        if isinstance(imagen2, str):
            try:
                imagen2_ruta = imagen2  # Guardar la ruta original
                imagen2_cargada = cv2.imread(imagen2)
                if imagen2_cargada is None:
                    print(f"❌ Error: No se pudo cargar la imagen: {imagen2}")
                    return None
                imagen2 = imagen2_cargada
                print(f"✅ Segunda imagen cargada: {os.path.basename(imagen2_ruta)}")
            except Exception as e:
                print(f"❌ Error cargando segunda imagen: {e}")
                return None
        
        # Si solo se proporciona una imagen, crear una versión desplazada
        if imagen2 is None:
            print("⚠️  Solo una imagen proporcionada, creando desplazamiento artificial...")
            # Crear desplazamiento artificial para demostrar
            if len(imagen1.shape) == 3:
                imagen1_gris = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
            else:
                imagen1_gris = imagen1.copy()
            
            # Desplazamiento y rotación ligera
            h, w = imagen1_gris.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), 1, 1.02)  # Rotación de 1° y escala 1.02
            M[0, 2] += 2  # Desplazamiento en X
            M[1, 2] += 1  # Desplazamiento en Y
            
            imagen2_gris = cv2.warpAffine(imagen1_gris, M, (w, h))
        else:
            # Convertir ambas imágenes a escala de grises
            if len(imagen1.shape) == 3:
                imagen1_gris = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
            else:
                imagen1_gris = imagen1.copy()
            
            if len(imagen2.shape) == 3:
                imagen2_gris = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
            else:
                imagen2_gris = imagen2.copy()
        
        # Calcular flujo óptico denso (Farneback)
        flow = cv2.calcOpticalFlowFarneback(imagen1_gris, imagen2_gris,
                                           None,
                                           pyr_scale=self.config['optical_flow']['pyr_scale'],
                                           levels=self.config['optical_flow']['levels'],
                                           winsize=self.config['optical_flow']['winsize'],
                                           iterations=self.config['optical_flow']['iterations'],
                                           poly_n=self.config['optical_flow']['poly_n'],
                                           poly_sigma=self.config['optical_flow']['poly_sigma'],
                                           flags=self.config['optical_flow']['flags'])
        
        # Calcular magnitud y ángulo del flujo
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Estadísticas del flujo óptico
        flow_stats = {
            'optical_flow_mean_magnitude': np.mean(magnitude),
            'optical_flow_std_magnitude': np.std(magnitude),
            'optical_flow_max_magnitude': np.max(magnitude),
            'optical_flow_mean_angle': np.mean(angle),
            'optical_flow_std_angle': np.std(angle)
        }
        
        # Análisis direccional del movimiento
        angle_deg = np.degrees(angle)
        angle_hist, _ = np.histogram(angle_deg.flatten(), bins=36, range=(0, 360))
        flow_stats['optical_flow_direction_entropy'] = self._calculate_entropy(angle_hist)
        flow_stats['optical_flow_dominant_direction'] = np.argmax(angle_hist) * 10  # grados
        
        # Análisis de regiones de movimiento
        movement_threshold = np.mean(magnitude) + np.std(magnitude)
        moving_pixels = magnitude > movement_threshold
        flow_stats['optical_flow_movement_ratio'] = np.sum(moving_pixels) / moving_pixels.size
        
        # Análisis de coherencia espacial
        coherence = self._calcular_coherencia_flujo(flow)
        flow_stats['optical_flow_spatial_coherence'] = coherence
        
        # Mostrar información detallada en consola
        if mostrar_descriptores:
            print(f"\n🔑 ANÁLISIS DE FLUJO ÓPTICO")
            print("=" * 60)
            print(f"📐 Dimensiones de la imagen: {imagen1_gris.shape}")
            
            print(f"\n📊 Estadísticas Optical Flow:")
            print(f"• Algoritmo: Farneback (Flujo óptico denso)")
            print(f"• Magnitud promedio: {flow_stats['optical_flow_mean_magnitude']:.4f}")
            print(f"• Magnitud máxima: {flow_stats['optical_flow_max_magnitude']:.4f}")
            print(f"• Desviación estándar magnitud: {flow_stats['optical_flow_std_magnitude']:.4f}")
            print(f"• Ángulo promedio: {np.degrees(flow_stats['optical_flow_mean_angle']):.1f}°")
            print(f"• Dirección dominante: {flow_stats['optical_flow_dominant_direction']:.0f}°")
            print(f"• Entropía direccional: {flow_stats['optical_flow_direction_entropy']:.4f}")
            print(f"• Ratio de píxeles en movimiento: {flow_stats['optical_flow_movement_ratio']:.4f}")
            print(f"• Coherencia espacial: {flow_stats['optical_flow_spatial_coherence']:.4f}")
            
            # Información sobre el método
            print(f"\n⚙️ Configuración:")
            print(f"• Escala piramidal: {self.config['optical_flow']['pyr_scale']}")
            print(f"• Niveles: {self.config['optical_flow']['levels']}")
            print(f"• Tamaño ventana: {self.config['optical_flow']['winsize']}")
            print(f"• Iteraciones: {self.config['optical_flow']['iterations']}")
            
            print("=" * 60)
        
        resultados = {
            **flow_stats,
            'flow_field': flow,
            'magnitude': magnitude,
            'angle': angle,
            'image1': imagen1_gris,
            'image2': imagen2_gris if imagen2 is not None else imagen2_gris,
            'nombre_imagen': nombre_imagen
        }
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            self._guardar_resultados_optical_flow(resultados, nombre_imagen)
        
        if visualizar:
            self._visualizar_optical_flow(resultados)
        
        return resultados
    
    def analizar_optical_flow_profesora(self, imagen1, imagen2=None, visualizar=True, mostrar_descriptores=True, 
                                      guardar_resultados=False, nombre_imagen="imagen_optical_flow_prof"):
        """
        Analiza flujo óptico usando exactamente el método de la profesora.
        Implementación directa del código proporcionado por la profesora.
        
        Args:
            imagen1 (np.ndarray): Primera imagen o imagen única
            imagen2 (np.ndarray o str, optional): Segunda imagen (array) o ruta a la segunda imagen
            visualizar (bool): Si generar visualización
            mostrar_descriptores (bool): Si mostrar estadísticas en consola
            guardar_resultados (bool): Si guardar resultados en archivos
            nombre_imagen (str): Nombre base para archivos guardados
            
        Returns:
            dict: Resultados del análisis de flujo óptico (método profesora)
        """
        print("🔍 Analizando flujo óptico (Método de la Profesora)...")
        
        # Si imagen2 es una ruta (string), cargarla
        if isinstance(imagen2, str):
            try:
                frame2 = cv2.imread(imagen2, cv2.IMREAD_GRAYSCALE)
                if frame2 is None:
                    print(f"❌ Error: No se pudo cargar la imagen: {imagen2}")
                    frame2 = None
                else:
                    print(f"✅ Segunda imagen cargada: {os.path.basename(imagen2)}")
            except Exception as e:
                print(f"❌ Error cargando segunda imagen: {e}")
                frame2 = None
        elif imagen2 is not None:
            # Convertir a escala de grises si es necesario
            if len(imagen2.shape) == 3:
                frame2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
            else:
                frame2 = imagen2.copy()
        else:
            # Como en el código de la profesora: usar la misma imagen para frame2
            frame2 = None
        
        # Cargar dos imágenes consecutivas de un video (como en código de la profesora)
        if isinstance(imagen1, str):
            frame1 = cv2.imread(imagen1, cv2.IMREAD_GRAYSCALE)
        else:
            if len(imagen1.shape) == 3:
                frame1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
            else:
                frame1 = imagen1.copy()
        
        # Si no hay segunda imagen, usar la misma (como en el ejemplo de la profesora)
        if frame2 is None:
            frame2 = frame1.copy()
        
        # Verificar si las imágenes se cargaron correctamente
        if frame1 is None or frame2 is None:
            print("⚠️ Problema con las imágenes, creando frames de ejemplo...")
            # Crear frames de ejemplo (código exacto de la profesora)
            frame1 = np.zeros((300, 400), dtype=np.uint8)
            cv2.circle(frame1, (100, 150), 30, 255, -1)
            cv2.circle(frame1, (200, 100), 25, 255, -1)
            
            # Frame2 con ligero desplazamiento
            frame2 = np.zeros((300, 400), dtype=np.uint8)
            cv2.circle(frame2, (105, 150), 30, 255, -1)  # Desplazado 5 píxeles
            cv2.circle(frame2, (205, 100), 25, 255, -1)  # Desplazado 5 píxeles
        
        # Calcular el flujo óptico usando el método de Farneback (código exacto de la profesora)
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Convertir el flujo óptico a un formato visualizable (código exacto de la profesora)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame1, dtype=np.float32)
        # Apilar imágenes en escala de grises para tener 3 canales
        hsv = np.stack((hsv, hsv, hsv), axis=-1)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Estadísticas básicas
        mag_promedio = np.mean(mag)
        mag_maxima = np.max(mag)
        mag_std = np.std(mag)
        ang_promedio = np.mean(ang)
        
        # Estadísticas del flujo óptico (método profesora)
        flow_stats = {
            'optical_flow_mean_magnitude': mag_promedio,
            'optical_flow_max_magnitude': mag_maxima,
            'optical_flow_std_magnitude': mag_std,
            'optical_flow_mean_angle': ang_promedio,
            'optical_flow_method': 'profesora_farneback',
            'optical_flow_parameters': {
                'pyr_scale': 0.5,
                'levels': 3,
                'winsize': 15,
                'iterations': 3,
                'poly_n': 5,
                'poly_sigma': 1.2,
                'flags': 0
            }
        }
        
        # Mostrar información detallada en consola
        if mostrar_descriptores:
            print(f"\n🔑 ANÁLISIS DE FLUJO ÓPTICO (MÉTODO PROFESORA)")
            print("=" * 60)
            print(f"📐 Dimensiones de la imagen: {frame1.shape}")
            
            print(f"\n📊 Estadísticas Optical Flow:")
            print(f"• Método: Farneback (Implementación Profesora)")
            print(f"• Magnitud promedio: {mag_promedio:.4f}")
            print(f"• Magnitud máxima: {mag_maxima:.4f}")
            print(f"• Desviación estándar magnitud: {mag_std:.4f}")
            print(f"• Ángulo promedio: {np.degrees(ang_promedio):.1f}°")
            
            # Información sobre el método
            print(f"\n⚙️ Parámetros utilizados:")
            print(f"• Escala piramidal: 0.5")
            print(f"• Niveles: 3")
            print(f"• Tamaño ventana: 15")
            print(f"• Iteraciones: 3")
            print(f"• Polinomio N: 5")
            print(f"• Polinomio Sigma: 1.2")
            
            print("=" * 60)
        
        resultados = {
            **flow_stats,
            'flow_field': flow,
            'magnitude': mag,
            'angle': ang,
            'flow_rgb': flow_rgb,
            'hsv_representation': hsv,
            'frame1': frame1,
            'frame2': frame2,
            'nombre_imagen': nombre_imagen
        }
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            self._guardar_resultados_optical_flow_profesora(resultados, nombre_imagen)
        
        if visualizar:
            self._visualizar_optical_flow_profesora(resultados)
        
        return resultados
    
    def analizar_secuencia_imagenes_carpeta(self, carpeta_path, patron_archivos="*.jpg", 
                                          visualizar=True, guardar_resultados=True,
                                          nombre_secuencia="secuencia_movimiento"):
        """
        Analiza una secuencia de imágenes en una carpeta para detectar cambios de movimiento.
        Útil para analizar series temporales de imágenes y detectar patrones.
        
        Args:
            carpeta_path (str): Ruta a la carpeta con las imágenes
            patron_archivos (str): Patrón para filtrar archivos (ej: "*.jpg", "*.png")
            visualizar (bool): Si generar visualizaciones
            guardar_resultados (bool): Si guardar resultados
            nombre_secuencia (str): Nombre para los archivos de resultados
            
        Returns:
            dict: Análisis completo de la secuencia de imágenes
        """
        print(f"📁 Analizando secuencia de imágenes en carpeta: {carpeta_path}")
        
        import glob
        
        # Verificar que la carpeta existe
        if not os.path.exists(carpeta_path):
            print(f"❌ Error: La carpeta no existe: {carpeta_path}")
            return None
        
        # Buscar archivos de imagen
        patron_completo = os.path.join(carpeta_path, patron_archivos)
        archivos_imagen = sorted(glob.glob(patron_completo))
        
        if len(archivos_imagen) < 2:
            print(f"❌ Error: Se necesitan al menos 2 imágenes. Encontradas: {len(archivos_imagen)}")
            return None
        
        print(f"📊 Encontradas {len(archivos_imagen)} imágenes")
        print(f"🔍 Analizando secuencia de movimientos...")
        
        # Variables para análisis de secuencia
        secuencia_flujos = []
        estadisticas_secuencia = []
        cambios_movimiento = []
        magnitudes_globales = []
        
        # Cargar primera imagen
        try:
            imagen_anterior = cv2.imread(archivos_imagen[0], cv2.IMREAD_GRAYSCALE)
            if imagen_anterior is None:
                print(f"❌ Error cargando primera imagen: {archivos_imagen[0]}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
        
        print("🔄 Procesando secuencia...")
        
        for i in range(1, len(archivos_imagen)):
            try:
                # Cargar imagen actual
                imagen_actual = cv2.imread(archivos_imagen[i], cv2.IMREAD_GRAYSCALE)
                if imagen_actual is None:
                    print(f"⚠️ Saltando imagen corrupta: {archivos_imagen[i]}")
                    continue
                
                # Redimensionar si es necesario para que coincidan
                if imagen_actual.shape != imagen_anterior.shape:
                    imagen_actual = cv2.resize(imagen_actual, 
                                             (imagen_anterior.shape[1], imagen_anterior.shape[0]))
                
                # Calcular optical flow entre imágenes consecutivas (método profesora)
                flow = cv2.calcOpticalFlowFarneback(imagen_anterior, imagen_actual, None, 
                                                  0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Calcular magnitud y ángulo
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Estadísticas de esta transición
                stats_transicion = {
                    'imagen_anterior': os.path.basename(archivos_imagen[i-1]),
                    'imagen_actual': os.path.basename(archivos_imagen[i]),
                    'indice': i,
                    'magnitude_mean': np.mean(magnitude),
                    'magnitude_max': np.max(magnitude),
                    'magnitude_std': np.std(magnitude),
                    'angle_mean': np.mean(angle),
                    'movement_intensity': np.sum(magnitude > np.mean(magnitude)),
                    'movement_area': np.sum(magnitude > (np.mean(magnitude) + np.std(magnitude)))
                }
                
                estadisticas_secuencia.append(stats_transicion)
                secuencia_flujos.append(flow)
                magnitudes_globales.append(np.mean(magnitude))
                
                # Detectar cambios significativos de movimiento
                if len(magnitudes_globales) > 1:
                    cambio_relativo = (magnitudes_globales[-1] - magnitudes_globales[-2]) / (magnitudes_globales[-2] + 1e-10)
                    
                    if abs(cambio_relativo) > 0.5:  # Cambio mayor al 50%
                        tipo_cambio = "incremento" if cambio_relativo > 0 else "reduccion"
                        cambios_movimiento.append({
                            'posicion': i,
                            'tipo': tipo_cambio,
                            'magnitude_anterior': magnitudes_globales[-2],
                            'magnitude_actual': magnitudes_globales[-1],
                            'cambio_relativo': cambio_relativo,
                            'imagen_anterior': archivos_imagen[i-1],
                            'imagen_actual': archivos_imagen[i]
                        })
                
                # Actualizar imagen anterior
                imagen_anterior = imagen_actual.copy()
                
                # Mostrar progreso
                if i % 5 == 0:
                    print(f"📈 Procesadas {i}/{len(archivos_imagen)-1} transiciones...")
                    
            except Exception as e:
                print(f"⚠️ Error procesando {archivos_imagen[i]}: {e}")
                continue
        
        # Análisis global de la secuencia
        if estadisticas_secuencia:
            analisis_global = self._analizar_secuencia_global(estadisticas_secuencia, cambios_movimiento)
        else:
            print("❌ No se pudieron procesar las imágenes")
            return None
        
        # Resultados finales
        resultados = {
            'carpeta_path': carpeta_path,
            'archivos_procesados': archivos_imagen,
            'num_transiciones': len(estadisticas_secuencia),
            'estadisticas_secuencia': estadisticas_secuencia,
            'cambios_movimiento': cambios_movimiento,
            'magnitudes_globales': magnitudes_globales,
            'analisis_global': analisis_global,
            'secuencia_flujos': secuencia_flujos[:10] if len(secuencia_flujos) > 10 else secuencia_flujos,  # Limitar memoria
            'nombre_secuencia': nombre_secuencia
        }
        
        print(f"✅ Análisis de secuencia completado: {len(estadisticas_secuencia)} transiciones procesadas")
        print(f"🔄 Cambios de movimiento detectados: {len(cambios_movimiento)}")
        
        # Guardar resultados
        if guardar_resultados:
            self._guardar_resultados_secuencia(resultados, nombre_secuencia)
        
        # Visualizar resultados
        if visualizar:
            self._visualizar_secuencia_imagenes(resultados)
        
        return resultados
    
    def analizar_video_optical_flow(self, video_path, max_frames=None, skip_frames=1, extraer_caracteristicas=True, 
                                  guardar_resultados=True, nombre_video="video_analisis"):
        """
        Analiza secuencias de video usando optical flow para detectar patrones de movimiento.
        Útil para entrenar modelos de visión artificial y detectar eventos en videos.
        
        Args:
            video_path (str): Ruta al archivo de video (.avi, .mp4, etc.)
            max_frames (int, optional): Número máximo de frames a procesar
            skip_frames (int): Procesar cada N frames (para acelerar análisis)
            extraer_caracteristicas (bool): Si extraer características temporales
            guardar_resultados (bool): Si guardar resultados de análisis
            nombre_video (str): Nombre base para archivos guardados
            
        Returns:
            dict: Análisis temporal del video con optical flow
        """
        print(f"🎬 Analizando video: {video_path}")
        
        # Verificar que el archivo existe
        if not os.path.exists(video_path):
            print(f"❌ Error: No se encontró el video: {video_path}")
            return None
        
        try:
            # Abrir video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Error: No se pudo abrir el video: {video_path}")
                return None
            
            # Propiedades del video
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"📊 Propiedades del video:")
            print(f"• Resolución: {width}x{height}")
            print(f"• FPS: {fps:.2f}")
            print(f"• Total de frames: {total_frames}")
            print(f"• Duración: {duration:.2f} segundos")
            
            # Limitar frames si se especifica
            frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
            print(f"• Frames a procesar: {frames_to_process} (cada {skip_frames} frames)")
            
            # Leer primer frame
            ret, frame1 = cap.read()
            if not ret:
                print("❌ Error: No se pudo leer el primer frame")
                cap.release()
                return None
            
            # Convertir a escala de grises
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            
            # Variables para análisis temporal
            flow_sequences = []
            movement_patterns = []
            frame_statistics = []
            motion_history = []
            
            frame_count = 0
            processed_count = 0
            
            print("🔄 Procesando frames...")
            
            while True:
                ret, frame2 = cap.read()
                if not ret or processed_count >= frames_to_process:
                    break
                
                frame_count += 1
                
                # Saltar frames si es necesario
                if frame_count % skip_frames != 0:
                    continue
                
                # Convertir a escala de grises
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                
                # Calcular optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None,
                    pyr_scale=self.config['optical_flow']['pyr_scale'],
                    levels=self.config['optical_flow']['levels'],
                    winsize=self.config['optical_flow']['winsize'],
                    iterations=self.config['optical_flow']['iterations'],
                    poly_n=self.config['optical_flow']['poly_n'],
                    poly_sigma=self.config['optical_flow']['poly_sigma'],
                    flags=self.config['optical_flow']['flags']
                )
                
                # Calcular magnitud y ángulo
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Estadísticas del frame actual
                frame_stats = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'mean_magnitude': np.mean(magnitude),
                    'max_magnitude': np.max(magnitude),
                    'std_magnitude': np.std(magnitude),
                    'motion_density': np.sum(magnitude > np.mean(magnitude)) / magnitude.size,
                    'dominant_direction': np.degrees(np.mean(angle)),
                    'movement_area': np.sum(magnitude > (np.mean(magnitude) + np.std(magnitude)))
                }
                
                frame_statistics.append(frame_stats)
                flow_sequences.append(flow)
                
                # Detectar patrones de movimiento
                if extraer_caracteristicas:
                    pattern = self._extraer_patron_movimiento(flow, magnitude, angle)
                    movement_patterns.append(pattern)
                
                # Actualizar historial de movimiento
                motion_intensity = np.mean(magnitude)
                motion_history.append(motion_intensity)
                
                # Actualizar frame anterior
                gray1 = gray2.copy()
                processed_count += 1
                
                # Mostrar progreso
                if processed_count % 10 == 0:
                    print(f"📈 Procesados {processed_count}/{frames_to_process} frames...")
            
            cap.release()
            
            # Análisis temporal completo
            video_analysis = self._analizar_secuencia_temporal(
                frame_statistics, movement_patterns, motion_history, fps
            )
            
            # Resultados finales
            resultados = {
                'video_path': video_path,
                'video_properties': {
                    'width': width, 'height': height, 'fps': fps,
                    'total_frames': total_frames, 'duration': duration,
                    'processed_frames': processed_count
                },
                'frame_statistics': frame_statistics,
                'movement_patterns': movement_patterns,
                'motion_history': motion_history,
                'temporal_analysis': video_analysis,
                'flow_sequences': flow_sequences[:10] if len(flow_sequences) > 10 else flow_sequences,  # Guardar solo los primeros 10 para memoria
                'nombre_video': nombre_video
            }
            
            print(f"✅ Análisis de video completado: {processed_count} frames procesados")
            
            # Guardar resultados
            if guardar_resultados:
                self._guardar_resultados_video(resultados, nombre_video)
            
            # Generar visualización
            self._visualizar_analisis_video(resultados)
            
            return resultados
            
        except Exception as e:
            print(f"❌ Error durante el análisis del video: {e}")
            return None
    
    def _extraer_patron_movimiento(self, flow, magnitude, angle):
        """Extrae patrones específicos de movimiento del optical flow."""
        h, w = flow.shape[:2]
        
        # Dividir imagen en regiones para análisis local
        regions = {
            'centro': magnitude[h//4:3*h//4, w//4:3*w//4],
            'esquina_superior_izq': magnitude[0:h//2, 0:w//2],
            'esquina_superior_der': magnitude[0:h//2, w//2:w],
            'esquina_inferior_izq': magnitude[h//2:h, 0:w//2],
            'esquina_inferior_der': magnitude[h//2:h, w//2:w]
        }
        
        # Análisis direccional
        angle_deg = np.degrees(angle)
        direcciones = {
            'arriba': np.sum((angle_deg >= 45) & (angle_deg < 135)),
            'abajo': np.sum((angle_deg >= 225) & (angle_deg < 315)),
            'izquierda': np.sum((angle_deg >= 135) & (angle_deg < 225)),
            'derecha': np.sum(((angle_deg >= 315) & (angle_deg <= 360)) | ((angle_deg >= 0) & (angle_deg < 45)))
        }
        
        # Detectar tipos de movimiento
        total_motion = np.sum(magnitude > np.mean(magnitude))
        movement_type = "estatico"
        
        if total_motion > 0.1 * magnitude.size:
            if direcciones['derecha'] > direcciones['izquierda'] * 1.5:
                movement_type = "movimiento_derecha"
            elif direcciones['izquierda'] > direcciones['derecha'] * 1.5:
                movement_type = "movimiento_izquierda"
            elif direcciones['arriba'] > direcciones['abajo'] * 1.5:
                movement_type = "movimiento_arriba"
            elif direcciones['abajo'] > direcciones['arriba'] * 1.5:
                movement_type = "movimiento_abajo"
            else:
                movement_type = "movimiento_multiple"
        
        return {
            'type': movement_type,
            'total_motion': total_motion,
            'region_activity': {region: np.mean(mag) for region, mag in regions.items()},
            'directional_distribution': direcciones,
            'motion_intensity': np.mean(magnitude),
            'motion_complexity': np.std(angle_deg)
        }
    
    def _analizar_secuencia_temporal(self, frame_stats, patterns, motion_history, fps):
        """Analiza la secuencia temporal completa del video."""
        if not frame_stats:
            return {}
        
        # Análisis de tendencias temporales
        magnitudes = [stat['mean_magnitude'] for stat in frame_stats]
        motion_densities = [stat['motion_density'] for stat in frame_stats]
        timestamps = [stat['timestamp'] for stat in frame_stats]
        
        # Detectar eventos significativos (picos de movimiento)
        threshold = np.mean(magnitudes) + 2 * np.std(magnitudes)
        eventos = []
        
        for i, (mag, timestamp) in enumerate(zip(magnitudes, timestamps)):
            if mag > threshold:
                eventos.append({
                    'frame': frame_stats[i]['frame_number'],
                    'timestamp': timestamp,
                    'magnitude': mag,
                    'type': 'high_motion_event'
                })
        
        # Análisis de periodicidad
        periodicidad = self._detectar_periodicidad(motion_history)
        
        # Clasificación de actividad general
        actividad_promedio = np.mean(magnitudes)
        if actividad_promedio < 0.5:
            nivel_actividad = "baja"
        elif actividad_promedio < 2.0:
            nivel_actividad = "media"
        else:
            nivel_actividad = "alta"
        
        # Análisis de patrones de movimiento dominantes
        if patterns:
            tipos_movimiento = [p['type'] for p in patterns]
            tipo_dominante = max(set(tipos_movimiento), key=tipos_movimiento.count)
        else:
            tipo_dominante = "desconocido"
        
        return {
            'duracion_total': timestamps[-1] if timestamps else 0,
            'actividad_promedio': actividad_promedio,
            'nivel_actividad': nivel_actividad,
            'eventos_significativos': eventos,
            'num_eventos': len(eventos),
            'patron_dominante': tipo_dominante,
            'periodicidad': periodicidad,
            'variabilidad_temporal': np.std(magnitudes),
            'tendencia_movimiento': 'creciente' if magnitudes[-1] > magnitudes[0] else 'decreciente',
            'picos_actividad': len([m for m in magnitudes if m > threshold]),
            'frames_activos': len([d for d in motion_densities if d > 0.1]),
            'estabilidad': 'estable' if np.std(magnitudes) < np.mean(magnitudes) * 0.5 else 'variable'
        }
    
    def _detectar_periodicidad(self, motion_history, min_period=3):
        """Detecta patrones periódicos en el historial de movimiento."""
        if len(motion_history) < min_period * 2:
            return {'detected': False, 'period': 0, 'confidence': 0}
        
        # Usar autocorrelación para detectar periodicidad
        motion_array = np.array(motion_history)
        motion_normalized = (motion_array - np.mean(motion_array)) / np.std(motion_array)
        
        # Calcular autocorrelación
        max_lag = min(len(motion_history) // 3, 50)
        autocorr = []
        
        for lag in range(1, max_lag):
            if lag < len(motion_normalized):
                corr = np.corrcoef(motion_normalized[:-lag], motion_normalized[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorr.append(corr)
                else:
                    autocorr.append(0)
        
        if autocorr:
            max_corr_idx = np.argmax(autocorr)
            max_corr = autocorr[max_corr_idx]
            
            if max_corr > 0.3:  # Umbral para periodicidad significativa
                return {
                    'detected': True,
                    'period': max_corr_idx + 1,
                    'confidence': max_corr,
                    'autocorrelation': autocorr[:20]  # Primeros 20 valores
                }
        
        return {'detected': False, 'period': 0, 'confidence': 0}
    
    def _visualizar_analisis_video(self, resultados):
        """Visualiza el análisis temporal del video."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        frame_stats = resultados['frame_statistics']
        if not frame_stats:
            print("⚠️ No hay estadísticas de frames para visualizar")
            return
        
        timestamps = [stat['timestamp'] for stat in frame_stats]
        magnitudes = [stat['mean_magnitude'] for stat in frame_stats]
        motion_densities = [stat['motion_density'] for stat in frame_stats]
        movement_areas = [stat['movement_area'] for stat in frame_stats]
        
        # 1. Magnitud del movimiento a lo largo del tiempo
        axes[0].plot(timestamps, magnitudes, 'b-', linewidth=2)
        axes[0].set_title('Magnitud de Movimiento vs Tiempo')
        axes[0].set_xlabel('Tiempo (segundos)')
        axes[0].set_ylabel('Magnitud Promedio')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Densidad de movimiento
        axes[1].plot(timestamps, motion_densities, 'g-', linewidth=2)
        axes[1].set_title('Densidad de Movimiento')
        axes[1].set_xlabel('Tiempo (segundos)')
        axes[1].set_ylabel('Densidad de Píxeles en Movimiento')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Área de movimiento
        axes[2].fill_between(timestamps, movement_areas, alpha=0.6, color='orange')
        axes[2].set_title('Área de Movimiento Significativo')
        axes[2].set_xlabel('Tiempo (segundos)')
        axes[2].set_ylabel('Píxeles')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Histograma de magnitudes
        axes[3].hist(magnitudes, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[3].set_title('Distribución de Magnitudes')
        axes[3].set_xlabel('Magnitud')
        axes[3].set_ylabel('Frecuencia')
        axes[3].grid(True, alpha=0.3)
        
        # 5. Eventos significativos
        eventos = resultados['temporal_analysis'].get('eventos_significativos', [])
        if eventos:
            event_times = [e['timestamp'] for e in eventos]
            event_mags = [e['magnitude'] for e in eventos]
            axes[4].scatter(event_times, event_mags, color='red', s=100, alpha=0.7)
            axes[4].plot(timestamps, magnitudes, 'b-', alpha=0.5)
            axes[4].set_title(f'Eventos Significativos ({len(eventos)})')
            axes[4].set_xlabel('Tiempo (segundos)')
            axes[4].set_ylabel('Magnitud')
        else:
            axes[4].text(0.5, 0.5, 'No se detectaron\neventos significativos', 
                        ha='center', va='center', transform=axes[4].transAxes, fontsize=12)
            axes[4].set_title('Eventos Significativos')
        axes[4].grid(True, alpha=0.3)
        
        # 6. Análisis de periodicidad
        periodicidad = resultados['temporal_analysis'].get('periodicidad', {})
        if periodicidad.get('detected', False):
            autocorr = periodicidad.get('autocorrelation', [])
            if autocorr:
                axes[5].plot(range(1, len(autocorr) + 1), autocorr, 'r-', linewidth=2)
                axes[5].axhline(y=0.3, color='g', linestyle='--', alpha=0.7, label='Umbral significativo')
                axes[5].set_title(f'Periodicidad Detectada (Período: {periodicidad["period"]})')
                axes[5].set_xlabel('Lag (frames)')
                axes[5].set_ylabel('Autocorrelación')
                axes[5].legend()
        else:
            axes[5].text(0.5, 0.5, 'No se detectó\nperiodicidad', 
                        ha='center', va='center', transform=axes[5].transAxes, fontsize=12)
            axes[5].set_title('Análisis de Periodicidad')
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar visualización si se especifica
        if hasattr(self, '_save_visualization') and self._save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            imagen_filename = os.path.join(self.results_dir, f"video_analysis_{resultados['nombre_video']}_{timestamp}.png")
            self.asegurar_directorio_existe(imagen_filename)
            plt.savefig(imagen_filename, dpi=300, bbox_inches='tight')
            print(f"🖼️ Análisis de video guardado: {imagen_filename}")
        
        plt.show()
        
        # Mostrar resumen textual
        temporal_analysis = resultados['temporal_analysis']
        print(f"\n📊 RESUMEN DEL ANÁLISIS DE VIDEO")
        print("=" * 60)
        print(f"🎬 Video: {os.path.basename(resultados['video_path'])}")
        print(f"⏱️ Duración: {temporal_analysis['duracion_total']:.2f} segundos")
        print(f"📈 Nivel de actividad: {temporal_analysis['nivel_actividad'].upper()}")
        print(f"🎯 Patrón dominante: {temporal_analysis['patron_dominante']}")
        print(f"🔄 Periodicidad: {'SÍ' if temporal_analysis['periodicidad']['detected'] else 'NO'}")
        if temporal_analysis['periodicidad']['detected']:
            print(f"   → Período: {temporal_analysis['periodicidad']['period']} frames")
            print(f"   → Confianza: {temporal_analysis['periodicidad']['confidence']:.3f}")
        print(f"⚡ Eventos significativos: {temporal_analysis['num_eventos']}")
        print(f"📊 Estabilidad: {temporal_analysis['estabilidad']}")
        print(f"📈 Tendencia: {temporal_analysis['tendencia_movimiento']}")
        print("=" * 60)

    def analisis_completo_avanzado(self, imagen_path, imagen2_path=None, nombre_imagen=None):
        """
        Realiza análisis completo con algoritmos avanzados.
        
        Args:
            imagen_path (str): Ruta a la primera imagen
            imagen2_path (str, optional): Ruta a la segunda imagen para optical flow
            nombre_imagen (str): Nombre personalizado
            
        Returns:
            dict: Resultados combinados
        """
        try:
            # Cargar primera imagen
            imagen1 = cv2.imread(imagen_path)
            if imagen1 is None:
                raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
            
            # Cargar segunda imagen si se proporciona
            imagen2 = None
            if imagen2_path:
                imagen2 = cv2.imread(imagen2_path)
                if imagen2 is None:
                    print(f"⚠️  No se pudo cargar la segunda imagen: {imagen2_path}")
            
            if nombre_imagen is None:
                nombre_imagen = os.path.basename(imagen_path)
            
            print(f"🔄 Análisis avanzado para: {nombre_imagen}")
            
            # Análisis FREAK
            resultados_freak = self.extraer_caracteristicas_freak(imagen1, visualizar=False)
            
            # Análisis AKAZE
            resultados_akaze = self.extraer_caracteristicas_akaze(imagen1, visualizar=False)
            
            # Análisis GrabCut
            resultados_grabcut = self.analizar_grabcut_segmentation(imagen1, visualizar=False)
            
            # Análisis LoG
            resultados_log = self.analizar_log_detector(imagen1, visualizar=False)
            
            # Análisis Optical Flow
            resultados_flow = self.analizar_optical_flow(imagen1, imagen2, visualizar=False)
            
            # Combinar resultados
            resultado_completo = {
                'Imagen': nombre_imagen,
                'Ruta': imagen_path,
                'Ruta_Imagen2': imagen2_path if imagen2_path else 'N/A',
                'Dimensiones': imagen1.shape,
                'Fecha_Analisis': datetime.now().isoformat(),
                **{k: v for k, v in resultados_freak.items() 
                   if not isinstance(v, (list, np.ndarray))},
                **{k: v for k, v in resultados_akaze.items() 
                   if not isinstance(v, (list, np.ndarray))},
                **{k: v for k, v in resultados_grabcut.items() 
                   if not isinstance(v, (list, np.ndarray))},
                **{k: v for k, v in resultados_log.items() 
                   if not isinstance(v, (list, np.ndarray))},
                **{k: v for k, v in resultados_flow.items() 
                   if not isinstance(v, (list, np.ndarray))}
            }
            
            self.current_results.append(resultado_completo)
            
            print(f"✅ Análisis avanzado completado para: {nombre_imagen}")
            return resultado_completo
            
        except Exception as e:
            print(f"❌ Error al procesar {imagen_path}: {str(e)}")
            return None
    
    def _analizar_secuencia_global(self, estadisticas_secuencia, cambios_movimiento):
        """Analiza globalmente una secuencia de imágenes."""
        if not estadisticas_secuencia:
            return {}
        
        # Extraer magnitudes
        magnitudes = [stat['magnitude_mean'] for stat in estadisticas_secuencia]
        intensidades = [stat['movement_intensity'] for stat in estadisticas_secuencia]
        
        # Análisis de tendencias
        tendencia = "estable"
        if len(magnitudes) > 1:
            diferencia_total = magnitudes[-1] - magnitudes[0]
            if diferencia_total > 0.1:
                tendencia = "creciente"
            elif diferencia_total < -0.1:
                tendencia = "decreciente"
        
        # Detectar picos y valles
        picos = []
        valles = []
        threshold_pico = np.mean(magnitudes) + np.std(magnitudes)
        threshold_valle = np.mean(magnitudes) - np.std(magnitudes)
        
        for i, mag in enumerate(magnitudes):
            if mag > threshold_pico:
                picos.append(i)
            elif mag < threshold_valle:
                valles.append(i)
        
        # Estabilidad de la secuencia
        variabilidad = np.std(magnitudes) / (np.mean(magnitudes) + 1e-10)
        if variabilidad < 0.3:
            estabilidad = "muy_estable"
        elif variabilidad < 0.6:
            estabilidad = "estable"
        elif variabilidad < 1.0:
            estabilidad = "variable"
        else:
            estabilidad = "muy_variable"
        
        # Análisis de periodicidad simple
        periodicidad_detectada = False
        if len(magnitudes) > 6:
            # Buscar patrones simples
            autocorr = np.correlate(magnitudes, magnitudes, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            if len(autocorr) > 3 and np.max(autocorr[2:]) > 0.7 * autocorr[0]:
                periodicidad_detectada = True
        
        return {
            'tendencia_general': tendencia,
            'estabilidad': estabilidad,
            'variabilidad': variabilidad,
            'magnitude_promedio': np.mean(magnitudes),
            'magnitude_max': np.max(magnitudes),
            'magnitude_min': np.min(magnitudes),
            'num_picos': len(picos),
            'num_valles': len(valles),
            'posiciones_picos': picos,
            'posiciones_valles': valles,
            'num_cambios_significativos': len(cambios_movimiento),
            'periodicidad_detectada': periodicidad_detectada,
            'intensidad_promedio': np.mean(intensidades),
            'transiciones_procesadas': len(estadisticas_secuencia)
        }
    
    # Métodos auxiliares
    def _calculate_entropy(self, data):
        """Calcula la entropía de los datos."""
        if isinstance(data, np.ndarray):
            data = data.flatten()
        
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0
        
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_binary_diversity(self, descriptors):
        """Calcula diversidad entre descriptores binarios."""
        if descriptors is None or len(descriptors) < 2:
            return 0
        
        # Calcular distancias de Hamming
        hamming_distances = []
        for i in range(min(len(descriptors), 50)):
            for j in range(i+1, min(len(descriptors), 50)):
                hamming_dist = np.sum(descriptors[i] != descriptors[j])
                hamming_distances.append(hamming_dist)
        
        return np.mean(hamming_distances) if hamming_distances else 0
    
    def _calculate_pattern_complexity(self, descriptors):
        """Calcula complejidad de patrones en descriptores binarios."""
        if descriptors is None or len(descriptors) == 0:
            return 0
        
        # Analizar transiciones de bits (cambios 0->1 o 1->0)
        transitions = []
        for descriptor in descriptors[:50]:  # Limitar para eficiencia
            bit_array = np.unpackbits(descriptor) if descriptor.dtype == np.uint8 else descriptor
            transition_count = np.sum(bit_array[:-1] != bit_array[1:])
            transitions.append(transition_count)
        
        return np.mean(transitions) if transitions else 0
    
    def _analizar_patron_retina(self, kp_x, kp_y, image_shape):
        """Analiza distribución de puntos clave en patrón tipo retina."""
        if not kp_x or not kp_y:
            return {
                'retina_center_density': 0,
                'retina_periphery_density': 0,
                'retina_radial_gradient': 0
            }
        
        h, w = image_shape[:2]
        center_x, center_y = w/2, h/2
        max_radius = min(w, h) / 2
        
        # Dividir en anillos concéntricos (como la retina)
        distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) 
                    for x, y in zip(kp_x, kp_y)]
        
        # Contar puntos en diferentes anillos
        center_points = sum(1 for d in distances if d < max_radius * 0.3)
        mid_points = sum(1 for d in distances if max_radius * 0.3 <= d < max_radius * 0.7)
        periphery_points = sum(1 for d in distances if d >= max_radius * 0.7)
        
        total_points = len(kp_x)
        
        return {
            'retina_center_density': center_points / total_points if total_points > 0 else 0,
            'retina_mid_density': mid_points / total_points if total_points > 0 else 0,
            'retina_periphery_density': periphery_points / total_points if total_points > 0 else 0,
            'retina_radial_gradient': (center_points - periphery_points) / total_points if total_points > 0 else 0
        }
    
    def _analizar_descriptores_akaze(self, descriptors):
        """Analiza descriptores AKAZE."""
        if descriptors is None or len(descriptors) == 0:
            return {
                'akaze_descriptor_length': 0,
                'akaze_descriptor_mean': 0,
                'akaze_descriptor_std': 0,
                'akaze_descriptor_sparsity': 0,
                'akaze_descriptor_energy': 0
            }
        
        # Los descriptores AKAZE pueden ser binarios o de punto flotante
        if descriptors.dtype == np.uint8:
            # Descriptores binarios
            bit_counts = np.unpackbits(descriptors, axis=1)
            return {
                'akaze_descriptor_length': descriptors.shape[1] * 8,
                'akaze_descriptor_bit_ratio': np.mean(bit_counts),
                'akaze_descriptor_entropy': self._calculate_entropy(bit_counts.flatten()),
                'akaze_descriptor_diversity': self._calculate_binary_diversity(descriptors)
            }
        else:
            # Descriptores de punto flotante
            return {
                'akaze_descriptor_length': descriptors.shape[1],
                'akaze_descriptor_mean': np.mean(descriptors),
                'akaze_descriptor_std': np.std(descriptors),
                'akaze_descriptor_sparsity': np.sum(np.abs(descriptors) < 0.01) / descriptors.size,
                'akaze_descriptor_energy': np.sum(descriptors ** 2)
            }
    
    def _analizar_estabilidad_multiescala(self, keypoints, imagen):
        """Analiza estabilidad de puntos clave a través de escalas."""
        if not keypoints:
            return {
                'akaze_scale_stability': 0,
                'akaze_response_consistency': 0
            }
        
        # Agrupar puntos por octava
        octave_groups = {}
        for kp in keypoints:
            octave = kp.octave & 0xFF
            if octave not in octave_groups:
                octave_groups[octave] = []
            octave_groups[octave].append(kp)
        
        # Calcular estabilidad como consistencia entre escalas
        if len(octave_groups) > 1:
            responses_by_octave = []
            for octave, kps in octave_groups.items():
                avg_response = np.mean([kp.response for kp in kps])
                responses_by_octave.append(avg_response)
            
            stability = 1.0 / (np.std(responses_by_octave) + 1e-10)
            consistency = np.corrcoef(range(len(responses_by_octave)), responses_by_octave)[0, 1]
            consistency = consistency if not np.isnan(consistency) else 0
        else:
            stability = 1.0
            consistency = 1.0
        
        return {
            'akaze_scale_stability': stability,
            'akaze_response_consistency': consistency
        }
    
    def _calcular_coherencia_bordes(self, imagen, mask):
        """Calcula coherencia entre bordes y segmentación."""
        # Detectar bordes
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        edges = cv2.Canny(imagen_gris, 50, 150)
        
        # Calcular bordes de la máscara
        mask_edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        
        # Coherencia como superposición de bordes
        overlap = np.sum((edges > 0) & (mask_edges > 0))
        total_mask_edges = np.sum(mask_edges > 0)
        
        return overlap / (total_mask_edges + 1e-10)
    
    def _analizar_consistencia_escalas(self, log_responses):
        """Analiza consistencia entre escalas en LoG."""
        if len(log_responses) < 2:
            return {'log_scale_consistency': 0}
        
        # Calcular correlación entre escalas adyacentes
        correlations = []
        for i in range(len(log_responses) - 1):
            corr = np.corrcoef(log_responses[i].flatten(), log_responses[i+1].flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        consistency = np.mean(correlations) if correlations else 0
        return {'log_scale_consistency': consistency}
    
    def _calcular_coherencia_flujo(self, flow):
        """Calcula coherencia espacial del flujo óptico."""
        # Calcular divergencia del campo de flujo
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        
        # Gradientes
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        
        # Divergencia
        divergence = du_dx + dv_dy
        
        # Coherencia como inverso de la varianza de la divergencia
        coherence = 1.0 / (np.std(divergence) + 1e-10)
        
        return coherence
    
    def _get_empty_freak_stats(self):
        """Estadísticas vacías para FREAK."""
        return {
            'freak_num_keypoints': 0, 'freak_kp_mean_x': 0, 'freak_kp_std_x': 0,
            'freak_kp_mean_y': 0, 'freak_kp_std_y': 0, 'freak_kp_mean_size': 0,
            'freak_kp_std_size': 0, 'freak_kp_mean_response': 0, 'freak_kp_density': 0,
            'freak_bit_ratio': 0, 'freak_bit_entropy': 0, 'freak_descriptor_diversity': 0,
            'freak_pattern_complexity': 0, 'retina_center_density': 0,
            'retina_mid_density': 0, 'retina_periphery_density': 0, 'retina_radial_gradient': 0
        }
    
    def _get_empty_akaze_stats(self):
        """Estadísticas vacías para AKAZE."""
        return {
            'akaze_num_keypoints': 0, 'akaze_kp_mean_x': 0, 'akaze_kp_std_x': 0,
            'akaze_kp_mean_y': 0, 'akaze_kp_std_y': 0, 'akaze_kp_mean_size': 0,
            'akaze_kp_std_size': 0, 'akaze_kp_mean_response': 0, 'akaze_kp_std_response': 0,
            'akaze_kp_density': 0, 'akaze_num_scales': 0, 'akaze_scale_entropy': 0,
            'akaze_dominant_scale': 0, 'akaze_mean_angle': 0, 'akaze_std_angle': 0,
            'akaze_angle_entropy': 0, 'akaze_descriptor_length': 0,
            'akaze_scale_stability': 0, 'akaze_response_consistency': 0
        }
    
    def _get_empty_grabcut_stats(self):
        """Estadísticas vacías para GrabCut."""
        return {
            'grabcut_foreground_ratio': 0, 'grabcut_background_ratio': 1,
            'grabcut_num_regions': 0, 'grabcut_largest_region_area': 0,
            'grabcut_mean_region_area': 0, 'grabcut_std_region_area': 0,
            'grabcut_region_compactness': 0, 'grabcut_region_solidity': 0,
            'grabcut_centroid_mean_x': 0, 'grabcut_centroid_std_x': 0,
            'grabcut_centroid_mean_y': 0, 'grabcut_centroid_std_y': 0,
            'grabcut_edge_coherence': 0
        }
    
    # Métodos de visualización
    def _visualizar_freak(self, resultados):
        """Visualiza resultados FREAK."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        axes[0].imshow(resultados['gray_image'], cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        img_kp = resultados['gray_image'].copy()
        if resultados['keypoints']:
            img_kp = cv2.drawKeypoints(img_kp, resultados['keypoints'], None,
                                     color=(0, 0, 255), 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        axes[1].imshow(img_kp, cmap='gray')
        axes[1].set_title(f'Puntos Clave FREAK ({resultados.get("freak_num_keypoints", 0)})')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización si se especifica
        if hasattr(self, '_save_visualization') and self._save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            imagen_filename = os.path.join(self.results_dir, f"freak_visualization_{resultados['nombre_imagen']}_{timestamp}.png")
            self.asegurar_directorio_existe(imagen_filename)
            plt.savefig(imagen_filename, dpi=300, bbox_inches='tight')
            print(f"🖼️ Visualización FREAK guardada: {imagen_filename}")
        
        plt.show()
    
    def _visualizar_akaze(self, resultados):
        """Visualiza resultados AKAZE."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        axes[0].imshow(resultados['gray_image'], cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        img_kp = resultados['gray_image'].copy()
        if resultados['keypoints']:
            img_kp = cv2.drawKeypoints(img_kp, resultados['keypoints'], None,
                                     color=(0, 0, 255), 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        axes[1].imshow(img_kp, cmap='gray')
        axes[1].set_title(f'Puntos Clave AKAZE ({resultados.get("akaze_num_keypoints", 0)})')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización si se especifica
        if hasattr(self, '_save_visualization') and self._save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            imagen_filename = os.path.join(self.results_dir, f"akaze_visualization_{resultados['nombre_imagen']}_{timestamp}.png")
            self.asegurar_directorio_existe(imagen_filename)
            plt.savefig(imagen_filename, dpi=300, bbox_inches='tight')
            print(f"🖼️ Visualización AKAZE guardada: {imagen_filename}")
        
        plt.show()
    
    def _visualizar_grabcut(self, resultados):
        """Visualiza resultados GrabCut."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Imagen original
        if len(resultados['original_image'].shape) == 3:
            img_rgb = cv2.cvtColor(resultados['original_image'], cv2.COLOR_BGR2RGB)
            axes[0].imshow(img_rgb)
        else:
            axes[0].imshow(resultados['original_image'], cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        # Máscara de segmentación
        axes[1].imshow(resultados['segmentation_mask'], cmap='gray')
        axes[1].set_title('Segmentación GrabCut')
        axes[1].axis('off')
        
        # Imagen segmentada
        if len(resultados['original_image'].shape) == 3:
            segmented = resultados['original_image'].copy()
            segmented[resultados['segmentation_mask'] == 0] = 0
            segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
            axes[2].imshow(segmented_rgb)
        else:
            segmented = resultados['original_image'] * resultados['segmentation_mask']
            axes[2].imshow(segmented, cmap='gray')
        axes[2].set_title('Resultado Segmentado')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización si se especifica
        if hasattr(self, '_save_visualization') and self._save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            imagen_filename = os.path.join(self.results_dir, f"grabcut_visualization_{resultados['nombre_imagen']}_{timestamp}.png")
            self.asegurar_directorio_existe(imagen_filename)
            plt.savefig(imagen_filename, dpi=300, bbox_inches='tight')
            print(f"🖼️ Visualización GrabCut guardada: {imagen_filename}")
        
        plt.show()
    
    def _visualizar_log(self, resultados):
        """Visualiza resultados LoG."""
        num_scales = len(resultados['log_responses'])
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Imagen original
        axes[0].imshow(resultados['gray_image'], cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        # Respuestas LoG para diferentes sigmas
        for i, (sigma, response) in enumerate(zip(self.config['log']['sigma_values'], resultados['log_responses'])):
            if i + 1 < len(axes):
                axes[i + 1].imshow(np.abs(response), cmap='hot')
                axes[i + 1].set_title(f'LoG σ={sigma}')
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización si se especifica
        if hasattr(self, '_save_visualization') and self._save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            imagen_filename = os.path.join(self.results_dir, f"log_visualization_{resultados['nombre_imagen']}_{timestamp}.png")
            self.asegurar_directorio_existe(imagen_filename)
            plt.savefig(imagen_filename, dpi=300, bbox_inches='tight')
            print(f"🖼️ Visualización LoG guardada: {imagen_filename}")
        
        plt.show()
    
    def _visualizar_optical_flow(self, resultados):
        """Visualiza resultados de flujo óptico."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Imagen 1
        axes[0, 0].imshow(resultados['image1'], cmap='gray')
        axes[0, 0].set_title('Imagen 1')
        axes[0, 0].axis('off')
        
        # Imagen 2
        axes[0, 1].imshow(resultados['image2'], cmap='gray')
        axes[0, 1].set_title('Imagen 2')
        axes[0, 1].axis('off')
        
        # Magnitud del flujo
        axes[1, 0].imshow(resultados['magnitude'], cmap='hot')
        axes[1, 0].set_title('Magnitud del Flujo')
        axes[1, 0].axis('off')
        
        # Campo de flujo visualizado
        step = 20
        h, w = resultados['image1'].shape[:2]
        
        # Crear grilla de coordenadas alineada con el submuestreo del flujo
        y_coords = np.arange(step//2, h, step)
        x_coords = np.arange(step//2, w, step)
        
        # Asegurar que no excedamos las dimensiones del flujo
        y_coords = y_coords[y_coords < h]
        x_coords = x_coords[x_coords < w]
        
        # Crear meshgrid
        x, y = np.meshgrid(x_coords, y_coords)
        
        # Extraer componentes del flujo con las mismas dimensiones
        u = resultados['flow_field'][y_coords[:, np.newaxis], x_coords, 0]
        v = resultados['flow_field'][y_coords[:, np.newaxis], x_coords, 1]
        
        # Asegurar que las dimensiones coincidan
        if u.shape != x.shape:
            # Redimensionar para que coincidan
            min_rows = min(u.shape[0], x.shape[0])
            min_cols = min(u.shape[1], x.shape[1])
            u = u[:min_rows, :min_cols]
            v = v[:min_rows, :min_cols]
            x = x[:min_rows, :min_cols]
            y = y[:min_rows, :min_cols]
        
        axes[1, 1].imshow(resultados['image1'], cmap='gray', alpha=0.8)
        
        # Solo dibujar si tenemos datos válidos
        if u.size > 0 and v.size > 0:
            axes[1, 1].quiver(x, y, u, v, color='red', alpha=0.8, scale_units='xy', scale=1)
        
        axes[1, 1].set_title('Campo de Flujo Óptico')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización si se especifica
        if hasattr(self, '_save_visualization') and self._save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            imagen_filename = os.path.join(self.results_dir, f"optical_flow_visualization_{resultados['nombre_imagen']}_{timestamp}.png")
            self.asegurar_directorio_existe(imagen_filename)
            plt.savefig(imagen_filename, dpi=300, bbox_inches='tight')
            print(f"🖼️ Visualización Optical Flow guardada: {imagen_filename}")
        
        plt.show()
    
    def _visualizar_optical_flow_profesora(self, resultados):
        """Visualiza resultados de optical flow usando exactamente el método de la profesora."""
        # Implementación exacta del código de visualización de la profesora
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(resultados['frame1'], cmap='gray')
        plt.title('Frame 1')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(resultados['frame2'], cmap='gray')
        plt.title('Frame 2')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('Optical Flow (HSV)')
        plt.imshow(resultados['flow_rgb'])
        plt.axis('off')
        
        plt.tight_layout()
        
        # Guardar visualización si se especifica
        if hasattr(self, '_save_visualization') and self._save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            imagen_filename = os.path.join(self.results_dir, f"optical_flow_profesora_{resultados['nombre_imagen']}_{timestamp}.png")
            self.asegurar_directorio_existe(imagen_filename)
            plt.savefig(imagen_filename, dpi=300, bbox_inches='tight')
            print(f"🖼️ Visualización Optical Flow (Profesora) guardada: {imagen_filename}")
        
        plt.show()
        
        # Mostrar estadísticas como en el código original
        print("✅ Optical Flow básico completado")
        print(f"📊 Magnitud promedio del flujo: {resultados['optical_flow_mean_magnitude']:.4f}")
        print(f"📊 Magnitud máxima: {resultados['optical_flow_max_magnitude']:.4f}")
    
    def _visualizar_secuencia_imagenes(self, resultados):
        """Visualiza el análisis de secuencia de imágenes."""
        estadisticas = resultados['estadisticas_secuencia']
        cambios = resultados['cambios_movimiento']
        analisis_global = resultados['analisis_global']
        
        if not estadisticas:
            print("⚠️ No hay estadísticas para visualizar")
            return
        
        # Crear visualización completa
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Extraer datos para gráficos
        indices = [stat['indice'] for stat in estadisticas]
        magnitudes = [stat['magnitude_mean'] for stat in estadisticas]
        intensidades = [stat['movement_intensity'] for stat in estadisticas]
        areas_movimiento = [stat['movement_area'] for stat in estadisticas]
        magnitudes_max = [stat['magnitude_max'] for stat in estadisticas]
        
        # 1. Magnitud promedio a lo largo de la secuencia
        axes[0].plot(indices, magnitudes, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0].set_title('Magnitud de Movimiento en la Secuencia')
        axes[0].set_xlabel('Índice de Transición')
        axes[0].set_ylabel('Magnitud Promedio')
        axes[0].grid(True, alpha=0.3)
        
        # Marcar cambios significativos
        if cambios:
            cambio_indices = [c['posicion'] for c in cambios]
            cambio_magnitudes = [magnitudes[c['posicion']-1] for c in cambios if c['posicion']-1 < len(magnitudes)]
            axes[0].scatter(cambio_indices, cambio_magnitudes, color='red', s=100, alpha=0.8, 
                          label=f'Cambios Significativos ({len(cambios)})')
            axes[0].legend()
        
        # 2. Intensidad de movimiento
        axes[1].bar(indices, intensidades, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_title('Intensidad de Movimiento')
        axes[1].set_xlabel('Índice de Transición')
        axes[1].set_ylabel('Píxeles en Movimiento')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Área de movimiento significativo
        axes[2].fill_between(indices, areas_movimiento, alpha=0.6, color='orange')
        axes[2].set_title('Área de Movimiento Significativo')
        axes[2].set_xlabel('Índice de Transición')
        axes[2].set_ylabel('Píxeles')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Comparación magnitud promedio vs máxima
        axes[3].plot(indices, magnitudes, 'b-', label='Magnitud Promedio', linewidth=2)
        axes[3].plot(indices, magnitudes_max, 'r--', label='Magnitud Máxima', linewidth=2)
        axes[3].set_title('Comparación de Magnitudes')
        axes[3].set_xlabel('Índice de Transición')
        axes[3].set_ylabel('Magnitud')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 5. Histograma de magnitudes
        axes[4].hist(magnitudes, bins=15, alpha=0.7, color='purple', edgecolor='black')
        axes[4].axvline(np.mean(magnitudes), color='red', linestyle='--', 
                       label=f'Promedio: {np.mean(magnitudes):.3f}')
        axes[4].set_title('Distribución de Magnitudes')
        axes[4].set_xlabel('Magnitud')
        axes[4].set_ylabel('Frecuencia')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # 6. Resumen de cambios detectados
        if cambios:
            tipos_cambio = [c['tipo'] for c in cambios]
            incrementos = tipos_cambio.count('incremento')
            reducciones = tipos_cambio.count('reduccion')
            
            axes[5].pie([incrementos, reducciones], 
                       labels=[f'Incrementos ({incrementos})', f'Reducciones ({reducciones})'],
                       autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
            axes[5].set_title('Tipos de Cambios Detectados')
        else:
            axes[5].text(0.5, 0.5, 'No se detectaron\ncambios significativos', 
                        ha='center', va='center', transform=axes[5].transAxes, fontsize=12)
            axes[5].set_title('Cambios Detectados')
        
        plt.tight_layout()
        
        # Guardar visualización si se especifica
        if hasattr(self, '_save_visualization') and self._save_visualization:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            imagen_filename = os.path.join(self.results_dir, f"secuencia_analysis_{resultados['nombre_secuencia']}_{timestamp}.png")
            self.asegurar_directorio_existe(imagen_filename)
            plt.savefig(imagen_filename, dpi=300, bbox_inches='tight')
            print(f"🖼️ Análisis de secuencia guardado: {imagen_filename}")
        
        plt.show()
        
        # Mostrar resumen textual
        print(f"\n📊 RESUMEN DEL ANÁLISIS DE SECUENCIA")
        print("=" * 60)
        print(f"📁 Carpeta: {os.path.basename(resultados['carpeta_path'])}")
        print(f"🖼️ Imágenes procesadas: {len(resultados['archivos_procesados'])}")
        print(f"🔄 Transiciones analizadas: {analisis_global['transiciones_procesadas']}")
        print(f"📈 Tendencia general: {analisis_global['tendencia_general'].upper()}")
        print(f"📊 Estabilidad: {analisis_global['estabilidad']}")
        print(f"📉 Variabilidad: {analisis_global['variabilidad']:.3f}")
        print(f"⚡ Cambios significativos: {analisis_global['num_cambios_significativos']}")
        print(f"🔝 Picos detectados: {analisis_global['num_picos']}")
        print(f"🔽 Valles detectados: {analisis_global['num_valles']}")
        print(f"🔄 Periodicidad: {'SÍ' if analisis_global['periodicidad_detectada'] else 'NO'}")
        print(f"📏 Magnitud promedio: {analisis_global['magnitude_promedio']:.4f}")
        print(f"📏 Rango: {analisis_global['magnitude_min']:.4f} - {analisis_global['magnitude_max']:.4f}")
        print("=" * 60)

    def guardar_resultados(self, formato='csv'):
        """Guarda los resultados del análisis."""
        if not self.current_results:
            print("❌ No hay resultados para guardar.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if formato.lower() == 'csv':
            df = pd.DataFrame(self.current_results)
            archivo_csv = os.path.join(self.results_dir, f'advanced_analysis_{timestamp}.csv')
            os.makedirs(os.path.dirname(archivo_csv), exist_ok=True)
            df.to_csv(archivo_csv, index=False)
            print(f"💾 Resultados CSV guardados: {archivo_csv}")
        
        elif formato.lower() == 'json':
            import json
            archivo_json = os.path.join(self.results_dir, f'advanced_analysis_{timestamp}.json')
            os.makedirs(os.path.dirname(archivo_json), exist_ok=True)
            with open(archivo_json, 'w', encoding='utf-8') as f:
                json.dump(self.current_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Resultados JSON guardados: {archivo_json}")
    
    def generar_reporte_avanzado(self):
        """Genera reporte del análisis avanzado."""
        if not self.current_results:
            print("❌ No hay resultados para el reporte.")
            return
        
        print("\n📋 REPORTE ANÁLISIS ALGORITMOS AVANZADOS")
        print("=" * 50)
        print(f"📊 Imágenes analizadas: {len(self.current_results)}")
        
        # Estadísticas por algoritmo
        algoritmos = ['freak', 'akaze', 'grabcut', 'log', 'optical_flow']
        
        for algoritmo in algoritmos:
            keypoints_key = f'{algoritmo}_num_keypoints'
            if algoritmo == 'grabcut':
                keypoints_key = 'grabcut_num_regions'
            elif algoritmo == 'log':
                keypoints_key = 'log_num_blobs'
            elif algoritmo == 'optical_flow':
                keypoints_key = 'optical_flow_mean_magnitude'
            
            valores = [r.get(keypoints_key, 0) for r in self.current_results]
            if any(valores):
                print(f"\n🔍 {algoritmo.upper()}:")
                print(f"   Promedio: {np.mean(valores):.2f}")
                print(f"   Desviación: {np.std(valores):.2f}")
                print(f"   Máximo: {np.max(valores):.2f}")
        
        print("\n" + "=" * 50)
    
    def _guardar_resultados_freak(self, resultados, nombre_imagen):
        """Guarda los resultados FREAK en archivos CSV y TXT."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Archivo CSV con estadísticas principales
        csv_filename = os.path.join(self.results_dir, f"freak_analysis_{nombre_imagen}_{timestamp}.csv")
        self.asegurar_directorio_existe(csv_filename)
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Métrica', 'Valor'])
            
            # Estadísticas básicas
            writer.writerow(['Algoritmo', 'FREAK' if resultados.get('freak_algorithm_available', False) else 'ORB (Fallback)'])
            writer.writerow(['Puntos clave detectados', resultados.get('freak_num_keypoints', 0)])
            writer.writerow(['Densidad de puntos', f"{resultados.get('freak_kp_density', 0):.8f}"])
            writer.writerow(['Ubicación promedio X', f"{resultados.get('freak_kp_mean_x', 0):.2f}"])
            writer.writerow(['Ubicación promedio Y', f"{resultados.get('freak_kp_mean_y', 0):.2f}"])
            writer.writerow(['Tamaño promedio', f"{resultados.get('freak_kp_mean_size', 0):.2f}"])
            writer.writerow(['Respuesta promedio', f"{resultados.get('freak_kp_mean_response', 0):.6f}"])
            writer.writerow(['Ratio de bits activos', f"{resultados.get('freak_bit_ratio', 0):.4f}"])
            writer.writerow(['Entropía de bits', f"{resultados.get('freak_bit_entropy', 0):.4f}"])
            writer.writerow(['Diversidad de descriptores', f"{resultados.get('freak_descriptor_diversity', 0):.4f}"])
            writer.writerow(['Complejidad de patrón', f"{resultados.get('freak_pattern_complexity', 0):.4f}"])
            writer.writerow(['Densidad centro retina', f"{resultados.get('retina_center_density', 0):.4f}"])
            writer.writerow(['Densidad periferia retina', f"{resultados.get('retina_periphery_density', 0):.4f}"])
        
        # Archivo TXT con información detallada
        txt_filename = os.path.join(self.results_dir, f"freak_detailed_{nombre_imagen}_{timestamp}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as txtfile:
            txtfile.write("ANÁLISIS DETALLADO FREAK\n")
            txtfile.write("=" * 50 + "\n\n")
            txtfile.write(f"Imagen analizada: {nombre_imagen}\n")
            txtfile.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            txtfile.write(f"Algoritmo disponible: {'Sí' if resultados.get('freak_algorithm_available', False) else 'No (usando ORB)'}\n\n")
            
            # Estadísticas de puntos clave
            keypoints = resultados.get('keypoints', [])
            txtfile.write(f"PUNTOS CLAVE DETECTADOS: {len(keypoints)}\n")
            txtfile.write("-" * 30 + "\n")
            
            for i, kp in enumerate(keypoints[:10]):  # Primeros 10 puntos
                txtfile.write(f"Punto {i+1}:\n")
                txtfile.write(f"  Posición: ({kp.pt[0]:.2f}, {kp.pt[1]:.2f})\n")
                txtfile.write(f"  Tamaño: {kp.size:.2f}\n")
                txtfile.write(f"  Ángulo: {kp.angle:.2f}°\n")
                txtfile.write(f"  Respuesta: {kp.response:.6f}\n\n")
            
            if len(keypoints) > 10:
                txtfile.write(f"... y {len(keypoints) - 10} puntos más\n\n")
        
        print(f"💾 Resultados FREAK guardados:")
        print(f"   📊 CSV: {csv_filename}")
        print(f"   📝 TXT: {txt_filename}")
    
    def _guardar_resultados_akaze(self, resultados, nombre_imagen):
        """Guarda los resultados AKAZE en archivos CSV y TXT."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Similar estructura para AKAZE
        csv_filename = os.path.join(self.results_dir, f"akaze_analysis_{nombre_imagen}_{timestamp}.csv")
        self.asegurar_directorio_existe(csv_filename)
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Métrica', 'Valor'])
            
            writer.writerow(['Algoritmo', 'AKAZE'])
            writer.writerow(['Puntos clave detectados', resultados.get('akaze_num_keypoints', 0)])
            writer.writerow(['Densidad de puntos', f"{resultados.get('akaze_kp_density', 0):.8f}"])
            writer.writerow(['Ubicación promedio X', f"{resultados.get('akaze_kp_mean_x', 0):.2f}"])
            writer.writerow(['Ubicación promedio Y', f"{resultados.get('akaze_kp_mean_y', 0):.2f}"])
            writer.writerow(['Tamaño promedio', f"{resultados.get('akaze_kp_mean_size', 0):.2f}"])
            writer.writerow(['Respuesta promedio', f"{resultados.get('akaze_kp_mean_response', 0):.6f}"])
            writer.writerow(['Número de escalas', resultados.get('akaze_num_scales', 0)])
            writer.writerow(['Entropía de escalas', f"{resultados.get('akaze_scale_entropy', 0):.4f}"])
            writer.writerow(['Estabilidad multi-escala', f"{resultados.get('akaze_scale_stability', 0):.4f}"])
        
        txt_filename = os.path.join(self.results_dir, f"akaze_detailed_{nombre_imagen}_{timestamp}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as txtfile:
            txtfile.write("ANÁLISIS DETALLADO AKAZE\n")
            txtfile.write("=" * 50 + "\n\n")
            txtfile.write(f"Imagen analizada: {nombre_imagen}\n")
            txtfile.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            keypoints = resultados.get('keypoints', [])
            txtfile.write(f"PUNTOS CLAVE DETECTADOS: {len(keypoints)}\n")
            txtfile.write("-" * 30 + "\n")
            
            for i, kp in enumerate(keypoints[:10]):
                txtfile.write(f"Punto {i+1}:\n")
                txtfile.write(f"  Posición: ({kp.pt[0]:.2f}, {kp.pt[1]:.2f})\n")
                txtfile.write(f"  Tamaño: {kp.size:.2f}\n")
                txtfile.write(f"  Ángulo: {kp.angle:.2f}°\n")
                txtfile.write(f"  Respuesta: {kp.response:.6f}\n")
                txtfile.write(f"  Octava: {kp.octave}\n\n")
        
        print(f"💾 Resultados AKAZE guardados:")
        print(f"   📊 CSV: {csv_filename}")
        print(f"   📝 TXT: {txt_filename}")
    
    def _guardar_resultados_grabcut(self, resultados, nombre_imagen):
        """Guarda los resultados GrabCut en archivos CSV y TXT."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_filename = os.path.join(self.results_dir, f"grabcut_analysis_{nombre_imagen}_{timestamp}.csv")
        self.asegurar_directorio_existe(csv_filename)
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Métrica', 'Valor'])
            
            writer.writerow(['Algoritmo', 'GrabCut'])
            writer.writerow(['Ratio primer plano', f"{resultados.get('grabcut_foreground_ratio', 0):.6f}"])
            writer.writerow(['Ratio fondo', f"{resultados.get('grabcut_background_ratio', 0):.6f}"])
            writer.writerow(['Número de regiones', resultados.get('grabcut_num_regions', 0)])
            writer.writerow(['Área región más grande', resultados.get('grabcut_largest_region_area', 0)])
            writer.writerow(['Área promedio regiones', f"{resultados.get('grabcut_mean_region_area', 0):.2f}"])
            writer.writerow(['Compacidad promedio', f"{resultados.get('grabcut_region_compactness', 0):.4f}"])
            writer.writerow(['Solidez promedio', f"{resultados.get('grabcut_region_solidity', 0):.4f}"])
            writer.writerow(['Coherencia de bordes', f"{resultados.get('grabcut_edge_coherence', 0):.4f}"])
        
        txt_filename = os.path.join(self.results_dir, f"grabcut_detailed_{nombre_imagen}_{timestamp}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as txtfile:
            txtfile.write("ANÁLISIS DETALLADO GRABCUT\n")
            txtfile.write("=" * 50 + "\n\n")
            txtfile.write(f"Imagen analizada: {nombre_imagen}\n")
            txtfile.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            txtfile.write("ESTADÍSTICAS DE SEGMENTACIÓN:\n")
            txtfile.write("-" * 30 + "\n")
            txtfile.write(f"Iteraciones GrabCut: {self.config['grabcut']['iterations']}\n")
            txtfile.write(f"Píxeles de primer plano: {int(resultados.get('grabcut_foreground_ratio', 0) * resultados['original_image'].size)}\n")
            txtfile.write(f"Píxeles de fondo: {int(resultados.get('grabcut_background_ratio', 0) * resultados['original_image'].size)}\n")
            txtfile.write(f"Número de regiones conectadas: {resultados.get('grabcut_num_regions', 0)}\n")
        
        print(f"💾 Resultados GrabCut guardados:")
        print(f"   📊 CSV: {csv_filename}")
        print(f"   📝 TXT: {txt_filename}")
    
    def _guardar_resultados_log(self, resultados, nombre_imagen):
        """Guarda los resultados LoG en archivos CSV y TXT."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_filename = os.path.join(self.results_dir, f"log_analysis_{nombre_imagen}_{timestamp}.csv")
        self.asegurar_directorio_existe(csv_filename)
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Métrica', 'Valor'])
            
            writer.writerow(['Algoritmo', 'Laplaciano de Gauss (LoG)'])
            writer.writerow(['Número de blobs', resultados.get('log_num_blobs', 0)])
            writer.writerow(['Densidad de blobs', f"{resultados.get('log_blob_density', 0):.8f}"])
            writer.writerow(['Ubicación promedio X', f"{resultados.get('log_mean_x', 0):.2f}"])
            writer.writerow(['Ubicación promedio Y', f"{resultados.get('log_mean_y', 0):.2f}"])
            writer.writerow(['Sigma promedio', f"{resultados.get('log_mean_sigma', 0):.2f}"])
            writer.writerow(['Respuesta promedio', f"{resultados.get('log_mean_response', 0):.6f}"])
            writer.writerow(['Consistencia escalas', f"{resultados.get('log_scale_consistency', 0):.4f}"])
        
        txt_filename = os.path.join(self.results_dir, f"log_detailed_{nombre_imagen}_{timestamp}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as txtfile:
            txtfile.write("ANÁLISIS DETALLADO LAPLACIANO DE GAUSS (LoG)\n")
            txtfile.write("=" * 60 + "\n\n")
            txtfile.write(f"Imagen analizada: {nombre_imagen}\n")
            txtfile.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            txtfile.write("CONFIGURACIÓN DEL ALGORITMO:\n")
            txtfile.write("-" * 35 + "\n")
            txtfile.write(f"Valores sigma analizados: {self.config['log']['sigma_values']}\n")
            txtfile.write(f"Factor de umbral: {self.config['log']['threshold_factor']}\n")
            txtfile.write(f"Número total de blobs detectados: {resultados.get('log_num_blobs', 0)}\n")
            txtfile.write(f"Densidad de blobs: {resultados.get('log_blob_density', 0):.8f}\n")
            txtfile.write(f"Consistencia entre escalas: {resultados.get('log_scale_consistency', 0):.4f}\n\n")
            
            # Estadísticas por escala
            txtfile.write("ESTADÍSTICAS POR ESCALA:\n")
            txtfile.write("-" * 30 + "\n")
            for sigma in self.config['log']['sigma_values']:
                txtfile.write(f"Sigma {sigma}:\n")
                media_key = f'log_sigma_{sigma}_mean'
                std_key = f'log_sigma_{sigma}_std'
                energia_key = f'log_sigma_{sigma}_energy'
                entropia_key = f'log_sigma_{sigma}_entropy'
                
                txtfile.write(f"  Media: {resultados.get(media_key, 0):.6f}\n")
                txtfile.write(f"  Desviación estándar: {resultados.get(std_key, 0):.6f}\n")
                txtfile.write(f"  Energía: {resultados.get(energia_key, 0):.6f}\n")
                txtfile.write(f"  Entropía: {resultados.get(entropia_key, 0):.6f}\n\n")
            
            # Estadísticas de distribución de blobs
            if resultados.get('log_num_blobs', 0) > 0:
                txtfile.write("ESTADÍSTICAS DE DISTRIBUCIÓN DE BLOBS:\n")
                txtfile.write("-" * 40 + "\n")
                txtfile.write(f"Posición promedio X: {resultados.get('log_mean_x', 0):.2f} ± {resultados.get('log_std_x', 0):.2f}\n")
                txtfile.write(f"Posición promedio Y: {resultados.get('log_mean_y', 0):.2f} ± {resultados.get('log_std_y', 0):.2f}\n")
                txtfile.write(f"Sigma promedio: {resultados.get('log_mean_sigma', 0):.2f} ± {resultados.get('log_std_sigma', 0):.2f}\n")
                txtfile.write(f"Respuesta promedio: {resultados.get('log_mean_response', 0):.6f} ± {resultados.get('log_std_response', 0):.6f}\n\n")
            
            # Lista detallada de blobs detectados
            txtfile.write("BLOBS DETECTADOS (Primeros 20):\n")
            txtfile.write("-" * 35 + "\n")
            local_maxima = resultados.get('local_maxima', [])
            if local_maxima:
                for i, blob in enumerate(local_maxima[:20]):
                    try:
                        # Los blobs son diccionarios con claves 'x', 'y', 'sigma', 'response'
                        if isinstance(blob, dict):
                            x = float(blob.get('x', 0))
                            y = float(blob.get('y', 0))
                            sigma = float(blob.get('sigma', 0))
                            response = float(blob.get('response', 0))
                            
                            txtfile.write(f"Blob {i+1:2d}: Pos=({x:6.2f}, {y:6.2f}) Sigma={sigma:4.2f} Resp={response:8.6f}\n")
                        else:
                            txtfile.write(f"Blob {i+1:2d}: Formato inesperado - {blob}\n")
                    except (ValueError, TypeError) as e:
                        txtfile.write(f"Blob {i+1:2d}: Error de conversión - {blob}\n")
            else:
                txtfile.write("No se detectaron blobs.\n")
            
            if len(local_maxima) > 20:
                txtfile.write(f"\n... y {len(local_maxima) - 20} blobs adicionales.\n")
        
        print(f"💾 Resultados LoG guardados:")
        print(f"   📊 CSV: {csv_filename}")
        print(f"   📝 TXT: {txt_filename}")
    
    def _guardar_resultados_optical_flow(self, resultados, nombre_imagen):
        """Guarda los resultados Optical Flow en archivos CSV y TXT."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_filename = os.path.join(self.results_dir, f"optical_flow_analysis_{nombre_imagen}_{timestamp}.csv")
        self.asegurar_directorio_existe(csv_filename)
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Métrica', 'Valor'])
            
            writer.writerow(['Algoritmo', 'Farneback Optical Flow'])
            writer.writerow(['Magnitud promedio', f"{resultados.get('optical_flow_mean_magnitude', 0):.6f}"])
            writer.writerow(['Magnitud máxima', f"{resultados.get('optical_flow_max_magnitude', 0):.6f}"])
            writer.writerow(['Desviación magnitud', f"{resultados.get('optical_flow_std_magnitude', 0):.6f}"])
            writer.writerow(['Ángulo promedio (rad)', f"{resultados.get('optical_flow_mean_angle', 0):.6f}"])
            writer.writerow(['Ángulo promedio (grados)', f"{np.degrees(resultados.get('optical_flow_mean_angle', 0)):.2f}"])
            writer.writerow(['Dirección dominante', f"{resultados.get('optical_flow_dominant_direction', 0):.0f}°"])
            writer.writerow(['Entropía direccional', f"{resultados.get('optical_flow_direction_entropy', 0):.4f}"])
            writer.writerow(['Ratio píxeles en movimiento', f"{resultados.get('optical_flow_movement_ratio', 0):.4f}"])
            writer.writerow(['Coherencia espacial', f"{resultados.get('optical_flow_spatial_coherence', 0):.4f}"])
        
        txt_filename = os.path.join(self.results_dir, f"optical_flow_detailed_{nombre_imagen}_{timestamp}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as txtfile:
            txtfile.write("ANÁLISIS DETALLADO FLUJO ÓPTICO\n")
            txtfile.write("=" * 50 + "\n\n")
            txtfile.write(f"Imagen analizada: {nombre_imagen}\n")
            txtfile.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            txtfile.write("CONFIGURACIÓN DEL ALGORITMO:\n")
            txtfile.write("-" * 30 + "\n")
            txtfile.write(f"Algoritmo: Farneback (Flujo óptico denso)\n")
            txtfile.write(f"Escala piramidal: {self.config['optical_flow']['pyr_scale']}\n")
            txtfile.write(f"Niveles piramidales: {self.config['optical_flow']['levels']}\n")
            txtfile.write(f"Tamaño de ventana: {self.config['optical_flow']['winsize']}\n")
            txtfile.write(f"Iteraciones: {self.config['optical_flow']['iterations']}\n")
            txtfile.write(f"Orden polinomial: {self.config['optical_flow']['poly_n']}\n")
            txtfile.write(f"Sigma polinomial: {self.config['optical_flow']['poly_sigma']}\n\n")
            
            txtfile.write("ESTADÍSTICAS DE MOVIMIENTO:\n")
            txtfile.write("-" * 30 + "\n")
            txtfile.write(f"Campo de flujo calculado para imagen de dimensiones: {resultados['image1'].shape}\n")
            txtfile.write(f"Magnitud promedio del movimiento: {resultados.get('optical_flow_mean_magnitude', 0):.6f}\n")
            txtfile.write(f"Porcentaje de píxeles en movimiento: {resultados.get('optical_flow_movement_ratio', 0)*100:.2f}%\n")
        
        print(f"💾 Resultados Optical Flow guardados:")
        print(f"   📊 CSV: {csv_filename}")
        print(f"   📝 TXT: {txt_filename}")
    
    def _guardar_resultados_optical_flow_profesora(self, resultados, nombre_imagen):
        """Guarda resultados del optical flow usando el método de la profesora."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Archivo CSV
        csv_filename = os.path.join(self.results_dir, f"optical_flow_profesora_{nombre_imagen}_{timestamp}.csv")
        self.asegurar_directorio_existe(csv_filename)
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Métrica', 'Valor'])
            writer.writerow(['Magnitud promedio', f"{resultados['optical_flow_mean_magnitude']:.6f}"])
            writer.writerow(['Magnitud máxima', f"{resultados['optical_flow_max_magnitude']:.6f}"])
            writer.writerow(['Desviación estándar magnitud', f"{resultados['optical_flow_std_magnitude']:.6f}"])
            writer.writerow(['Ángulo promedio (radianes)', f"{resultados['optical_flow_mean_angle']:.6f}"])
            writer.writerow(['Ángulo promedio (grados)', f"{np.degrees(resultados['optical_flow_mean_angle']):.2f}"])
            writer.writerow(['Método', resultados['optical_flow_method']])
            
            # Parámetros utilizados
            params = resultados['optical_flow_parameters']
            writer.writerow(['Parámetro pyr_scale', params['pyr_scale']])
            writer.writerow(['Parámetro levels', params['levels']])
            writer.writerow(['Parámetro winsize', params['winsize']])
            writer.writerow(['Parámetro iterations', params['iterations']])
            writer.writerow(['Parámetro poly_n', params['poly_n']])
            writer.writerow(['Parámetro poly_sigma', params['poly_sigma']])
        
        # Archivo TXT
        txt_filename = os.path.join(self.results_dir, f"optical_flow_profesora_{nombre_imagen}_{timestamp}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as txtfile:
            txtfile.write("ANÁLISIS OPTICAL FLOW - MÉTODO PROFESORA\n")
            txtfile.write("=" * 50 + "\n")
            txtfile.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            txtfile.write(f"Imagen analizada: {nombre_imagen}\n\n")
            
            txtfile.write("CONFIGURACIÓN UTILIZADA:\n")
            txtfile.write("-" * 30 + "\n")
            txtfile.write("Algoritmo: Farneback Optical Flow\n")
            params = resultados['optical_flow_parameters']
            txtfile.write(f"• Escala piramidal: {params['pyr_scale']}\n")
            txtfile.write(f"• Niveles: {params['levels']}\n")
            txtfile.write(f"• Tamaño ventana: {params['winsize']}\n")
            txtfile.write(f"• Iteraciones: {params['iterations']}\n")
            txtfile.write(f"• Polinomio N: {params['poly_n']}\n")
            txtfile.write(f"• Polinomio Sigma: {params['poly_sigma']}\n\n")
            
            txtfile.write("RESULTADOS:\n")
            txtfile.write("-" * 30 + "\n")
            txtfile.write(f"Magnitud promedio del flujo: {resultados['optical_flow_mean_magnitude']:.6f}\n")
            txtfile.write(f"Magnitud máxima: {resultados['optical_flow_max_magnitude']:.6f}\n")
            txtfile.write(f"Desviación estándar: {resultados['optical_flow_std_magnitude']:.6f}\n")
            txtfile.write(f"Ángulo promedio: {np.degrees(resultados['optical_flow_mean_angle']):.2f}°\n")
            txtfile.write(f"Dimensiones de imagen: {resultados['frame1'].shape}\n\n")
            
            txtfile.write("NOTA:\n")
            txtfile.write("Este análisis fue realizado usando exactamente el mismo código\n")
            txtfile.write("y parámetros proporcionados por la profesora.\n")
        
        # Guardar imagen HSV del flujo
        imagen_filename = os.path.join(self.results_dir, f"optical_flow_hsv_{nombre_imagen}_{timestamp}.png")
        plt.figure(figsize=(10, 6))
        plt.imshow(resultados['flow_rgb'])
        plt.title(f'Optical Flow HSV - {nombre_imagen}')
        plt.axis('off')
        plt.savefig(imagen_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("💾 Resultados Optical Flow (Método Profesora) guardados:")
        print(f"   📊 CSV: {csv_filename}")
        print(f"   📝 TXT: {txt_filename}")
        print(f"   🖼️ Imagen HSV: {imagen_filename}")
    
    def _guardar_resultados_secuencia(self, resultados, nombre_secuencia):
        """Guarda resultados del análisis de secuencia de imágenes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio específico para secuencias
        secuencia_dir = os.path.join(self.results_dir, "sequence_analysis")
        os.makedirs(secuencia_dir, exist_ok=True)
        
        # Archivo CSV con estadísticas de transiciones
        csv_filename = os.path.join(secuencia_dir, f"sequence_analysis_{nombre_secuencia}_{timestamp}.csv")
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Sección de análisis global
            writer.writerow(['Tipo', 'Métrica', 'Valor'])
            analisis = resultados['analisis_global']
            writer.writerow(['Global', 'Tendencia general', analisis['tendencia_general']])
            writer.writerow(['Global', 'Estabilidad', analisis['estabilidad']])
            writer.writerow(['Global', 'Variabilidad', f"{analisis['variabilidad']:.4f}"])
            writer.writerow(['Global', 'Magnitud promedio', f"{analisis['magnitude_promedio']:.6f}"])
            writer.writerow(['Global', 'Magnitud máxima', f"{analisis['magnitude_max']:.6f}"])
            writer.writerow(['Global', 'Magnitud mínima', f"{analisis['magnitude_min']:.6f}"])
            writer.writerow(['Global', 'Número de picos', analisis['num_picos']])
            writer.writerow(['Global', 'Número de valles', analisis['num_valles']])
            writer.writerow(['Global', 'Cambios significativos', analisis['num_cambios_significativos']])
            writer.writerow(['Global', 'Periodicidad detectada', 'Sí' if analisis['periodicidad_detectada'] else 'No'])
            
            # Estadísticas por transición
            writer.writerow([])  # Línea vacía
            writer.writerow(['Transición', 'Imagen Anterior', 'Imagen Actual', 'Magnitud Media', 
                           'Magnitud Máxima', 'Intensidad Movimiento', 'Área Movimiento'])
            
            for stat in resultados['estadisticas_secuencia']:
                writer.writerow([
                    stat['indice'],
                    stat['imagen_anterior'],
                    stat['imagen_actual'],
                    f"{stat['magnitude_mean']:.6f}",
                    f"{stat['magnitude_max']:.6f}",
                    stat['movement_intensity'],
                    stat['movement_area']
                ])
            
            # Cambios detectados
            if resultados['cambios_movimiento']:
                writer.writerow([])  # Línea vacía
                writer.writerow(['Cambio', 'Posición', 'Tipo', 'Magnitud Anterior', 'Magnitud Actual', 'Cambio Relativo'])
                
                for i, cambio in enumerate(resultados['cambios_movimiento']):
                    writer.writerow([
                        i + 1,
                        cambio['posicion'],
                        cambio['tipo'],
                        f"{cambio['magnitude_anterior']:.6f}",
                        f"{cambio['magnitude_actual']:.6f}",
                        f"{cambio['cambio_relativo']:.4f}"
                    ])
        
        # Archivo TXT con análisis detallado
        txt_filename = os.path.join(secuencia_dir, f"sequence_analysis_{nombre_secuencia}_{timestamp}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as txtfile:
            txtfile.write("ANÁLISIS DE SECUENCIA DE IMÁGENES\n")
            txtfile.write("=" * 60 + "\n")
            txtfile.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            txtfile.write(f"Carpeta analizada: {resultados['carpeta_path']}\n")
            txtfile.write(f"Secuencia: {nombre_secuencia}\n\n")
            
            # Información general
            txtfile.write("INFORMACIÓN GENERAL:\n")
            txtfile.write("-" * 30 + "\n")
            txtfile.write(f"Número de imágenes: {len(resultados['archivos_procesados'])}\n")
            txtfile.write(f"Transiciones analizadas: {resultados['num_transiciones']}\n")
            txtfile.write(f"Algoritmo: Farneback Optical Flow\n\n")
            
            # Análisis global
            txtfile.write("ANÁLISIS GLOBAL:\n")
            txtfile.write("-" * 30 + "\n")
            analisis = resultados['analisis_global']
            txtfile.write(f"Tendencia general: {analisis['tendencia_general'].upper()}\n")
            txtfile.write(f"Estabilidad de la secuencia: {analisis['estabilidad']}\n")
            txtfile.write(f"Variabilidad: {analisis['variabilidad']:.4f}\n")
            txtfile.write(f"Magnitud promedio: {analisis['magnitude_promedio']:.6f}\n")
            txtfile.write(f"Rango de magnitudes: {analisis['magnitude_min']:.6f} - {analisis['magnitude_max']:.6f}\n")
            txtfile.write(f"Picos detectados: {analisis['num_picos']} en posiciones {analisis['posiciones_picos']}\n")
            txtfile.write(f"Valles detectados: {analisis['num_valles']} en posiciones {analisis['posiciones_valles']}\n")
            txtfile.write(f"Periodicidad: {'SÍ' if analisis['periodicidad_detectada'] else 'NO'}\n\n")
            
            # Cambios significativos
            txtfile.write("CAMBIOS SIGNIFICATIVOS DETECTADOS:\n")
            txtfile.write("-" * 30 + "\n")
            if resultados['cambios_movimiento']:
                for i, cambio in enumerate(resultados['cambios_movimiento']):
                    txtfile.write(f"{i+1}. Posición {cambio['posicion']}: {cambio['tipo'].upper()}\n")
                    txtfile.write(f"   De {os.path.basename(cambio['imagen_anterior'])}\n")
                    txtfile.write(f"   A {os.path.basename(cambio['imagen_actual'])}\n")
                    txtfile.write(f"   Cambio: {cambio['magnitude_anterior']:.4f} → {cambio['magnitude_actual']:.4f} ")
                    txtfile.write(f"({cambio['cambio_relativo']:+.1%})\n\n")
            else:
                txtfile.write("No se detectaron cambios significativos en la secuencia.\n\n")
            
            # Lista de archivos procesados
            txtfile.write("ARCHIVOS PROCESADOS:\n")
            txtfile.write("-" * 30 + "\n")
            for i, archivo in enumerate(resultados['archivos_procesados']):
                txtfile.write(f"{i+1:3d}. {os.path.basename(archivo)}\n")
        
        # Archivo JSON para uso programático
        json_filename = os.path.join(secuencia_dir, f"sequence_analysis_{nombre_secuencia}_{timestamp}.json")
        
        # Función auxiliar para convertir tipos NumPy a tipos nativos de Python
        def convert_numpy_types(obj):
            """Convierte tipos de NumPy a tipos nativos de Python para serialización JSON."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        # Preparar datos JSON serializables
        json_data = {
            'carpeta_path': resultados['carpeta_path'],
            'nombre_secuencia': nombre_secuencia,
            'num_transiciones': resultados['num_transiciones'],
            'analisis_global': convert_numpy_types(resultados['analisis_global']),
            'estadisticas_secuencia': convert_numpy_types(resultados['estadisticas_secuencia'][:50]),  # Limitar para tamaño
            'cambios_movimiento': convert_numpy_types(resultados['cambios_movimiento']),
            'magnitudes_globales': convert_numpy_types(resultados['magnitudes_globales'][:100]),  # Limitar tamaño
            'archivos_procesados': [os.path.basename(f) for f in resultados['archivos_procesados']],
            'timestamp': timestamp
        }
        
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        print("💾 Resultados de análisis de secuencia guardados:")
        print(f"   📊 CSV: {csv_filename}")
        print(f"   📝 TXT: {txt_filename}")
        print(f"   📋 JSON: {json_filename}")

    def _guardar_resultados_video(self, resultados, nombre_video):
        """Guarda resultados del análisis de video."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio específico para análisis de video
        video_dir = os.path.join(self.results_dir, "video_analysis")
        os.makedirs(video_dir, exist_ok=True)
        
        # Archivo CSV con estadísticas temporales
        csv_filename = os.path.join(video_dir, f"video_analysis_{nombre_video}_{timestamp}.csv")
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Encabezados
            writer.writerow(['Tipo', 'Métrica', 'Valor'])
            
            # Propiedades del video
            props = resultados['video_properties']
            writer.writerow(['Video', 'Resolución', f"{props['width']}x{props['height']}"])
            writer.writerow(['Video', 'FPS', f"{props['fps']:.2f}"])
            writer.writerow(['Video', 'Duración (s)', f"{props['duration']:.2f}"])
            writer.writerow(['Video', 'Frames Totales', props['total_frames']])
            writer.writerow(['Video', 'Frames Procesados', props['processed_frames']])
            
            # Análisis temporal
            temporal = resultados['temporal_analysis']
            writer.writerow(['Temporal', 'Nivel de Actividad', temporal['nivel_actividad']])
            writer.writerow(['Temporal', 'Actividad Promedio', f"{temporal['actividad_promedio']:.4f}"])
            writer.writerow(['Temporal', 'Patrón Dominante', temporal['patron_dominante']])
            writer.writerow(['Temporal', 'Eventos Significativos', temporal['num_eventos']])
            writer.writerow(['Temporal', 'Estabilidad', temporal['estabilidad']])
            writer.writerow(['Temporal', 'Tendencia', temporal['tendencia_movimiento']])
            writer.writerow(['Temporal', 'Variabilidad', f"{temporal['variabilidad_temporal']:.4f}"])
            
            # Periodicidad
            periodicidad = temporal['periodicidad']
            writer.writerow(['Periodicidad', 'Detectada', 'Sí' if periodicidad['detected'] else 'No'])
            if periodicidad['detected']:
                writer.writerow(['Periodicidad', 'Período (frames)', periodicidad['period']])
                writer.writerow(['Periodicidad', 'Confianza', f"{periodicidad['confidence']:.4f}"])
            
            # Estadísticas por frame
            writer.writerow([])  # Línea vacía
            writer.writerow(['Frame', 'Timestamp', 'Magnitud Media', 'Densidad Movimiento', 'Área Movimiento'])
            
            for stat in resultados['frame_statistics'][:50]:  # Primeros 50 frames
                writer.writerow([
                    stat['frame_number'],
                    f"{stat['timestamp']:.3f}",
                    f"{stat['mean_magnitude']:.4f}",
                    f"{stat['motion_density']:.4f}",
                    stat['movement_area']
                ])
        
        # Archivo TXT con análisis detallado
        txt_filename = os.path.join(video_dir, f"video_analysis_{nombre_video}_{timestamp}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as txtfile:
            txtfile.write(f"ANÁLISIS DE VIDEO CON OPTICAL FLOW\n")
            txtfile.write(f"{'='*60}\n")
            txtfile.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            txtfile.write(f"Video analizado: {resultados['video_path']}\n\n")
            
            # Propiedades del video
            txtfile.write(f"PROPIEDADES DEL VIDEO:\n")
            txtfile.write(f"{'-'*30}\n")
            props = resultados['video_properties']
            txtfile.write(f"Resolución: {props['width']}x{props['height']}\n")
            txtfile.write(f"Frames por segundo: {props['fps']:.2f}\n")
            txtfile.write(f"Duración total: {props['duration']:.2f} segundos\n")
            txtfile.write(f"Frames totales: {props['total_frames']}\n")
            txtfile.write(f"Frames procesados: {props['processed_frames']}\n\n")
            
            # Análisis temporal
            txtfile.write(f"ANÁLISIS TEMPORAL:\n")
            txtfile.write(f"{'-'*30}\n")
            temporal = resultados['temporal_analysis']
            txtfile.write(f"Nivel de actividad: {temporal['nivel_actividad'].upper()}\n")
            txtfile.write(f"Actividad promedio: {temporal['actividad_promedio']:.4f}\n")
            txtfile.write(f"Patrón de movimiento dominante: {temporal['patron_dominante']}\n")
            txtfile.write(f"Número de eventos significativos: {temporal['num_eventos']}\n")
            txtfile.write(f"Estabilidad del video: {temporal['estabilidad']}\n")
            txtfile.write(f"Tendencia del movimiento: {temporal['tendencia_movimiento']}\n")
            txtfile.write(f"Variabilidad temporal: {temporal['variabilidad_temporal']:.4f}\n")
            txtfile.write(f"Frames con actividad: {temporal['frames_activos']}\n\n")
            
            # Periodicidad
            txtfile.write(f"ANÁLISIS DE PERIODICIDAD:\n")
            txtfile.write(f"{'-'*30}\n")
            periodicidad = temporal['periodicidad']
            if periodicidad['detected']:
                txtfile.write(f"Periodicidad detectada: SÍ\n")
                txtfile.write(f"Período: {periodicidad['period']} frames\n")
                txtfile.write(f"Confianza: {periodicidad['confidence']:.4f}\n")
            else:
                txtfile.write(f"Periodicidad detectada: NO\n")
            txtfile.write(f"\n")
            
            # Eventos significativos
            eventos = temporal.get('eventos_significativos', [])
            txtfile.write(f"EVENTOS SIGNIFICATIVOS ({len(eventos)}):\n")
            txtfile.write(f"{'-'*30}\n")
            if eventos:
                for i, evento in enumerate(eventos[:10]):  # Mostrar máximo 10 eventos
                    txtfile.write(f"{i+1}. Frame {evento['frame']} "
                                f"(t={evento['timestamp']:.2f}s) - "
                                f"Magnitud: {evento['magnitude']:.4f}\n")
            else:
                txtfile.write("No se detectaron eventos significativos.\n")
            txtfile.write(f"\n")
            
            # Patrones de movimiento
            if resultados.get('movement_patterns'):
                txtfile.write(f"PATRONES DE MOVIMIENTO:\n")
                txtfile.write(f"{'-'*30}\n")
                tipos_movimiento = [p['type'] for p in resultados['movement_patterns']]
                conteo_tipos = {tipo: tipos_movimiento.count(tipo) for tipo in set(tipos_movimiento)}
                
                for tipo, conteo in sorted(conteo_tipos.items(), key=lambda x: x[1], reverse=True):
                    porcentaje = (conteo / len(tipos_movimiento)) * 100
                    txtfile.write(f"{tipo}: {conteo} frames ({porcentaje:.1f}%)\n")
                txtfile.write(f"\n")
            
            # Configuración utilizada
            txtfile.write(f"CONFIGURACIÓN DEL ANÁLISIS:\n")
            txtfile.write(f"{'-'*30}\n")
            txtfile.write(f"Algoritmo: Farneback Optical Flow\n")
            config = self.config['optical_flow']
            txtfile.write(f"Escala piramidal: {config['pyr_scale']}\n")
            txtfile.write(f"Niveles: {config['levels']}\n")
            txtfile.write(f"Tamaño de ventana: {config['winsize']}\n")
            txtfile.write(f"Iteraciones: {config['iterations']}\n")
        
        # Guardar JSON con datos completos para análisis posteriores
        json_filename = os.path.join(video_dir, f"video_analysis_{nombre_video}_{timestamp}.json")
        
        # Preparar datos serializables para JSON
        json_data = {
            'video_path': resultados['video_path'],
            'video_properties': resultados['video_properties'],
            'temporal_analysis': resultados['temporal_analysis'],
            'frame_statistics': resultados['frame_statistics'],
            'movement_patterns': resultados['movement_patterns'][:100] if resultados.get('movement_patterns') else [],  # Limitar para tamaño
            'motion_history': resultados['motion_history'],
            'analysis_timestamp': timestamp,
            'config_used': self.config['optical_flow']
        }
        
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"💾 Resultados de análisis de video guardados:")
        print(f"   📊 CSV: {csv_filename}")
        print(f"   📝 TXT: {txt_filename}")
        print(f"   📋 JSON: {json_filename}")

    def asegurar_directorio_existe(self, ruta_archivo):
        """Asegura que el directorio para un archivo existe antes de guardarlo."""
        directorio = os.path.dirname(ruta_archivo)
        os.makedirs(directorio, exist_ok=True)

# Función de utilidad
def analizar_avanzado_imagen(imagen_path, imagen2_path=None, output_dir="./resultados"):
    """
    Función de conveniencia para análisis avanzado.
    
    Args:
        imagen_path (str): Ruta a la primera imagen
        imagen2_path (str, optional): Ruta a la segunda imagen
        output_dir (str): Directorio de salida
        
    Returns:
        dict: Resultados del análisis
    """
    analyzer = AdvancedAnalyzer(output_dir)
    resultado = analyzer.analisis_completo_avanzado(imagen_path, imagen2_path)
    if resultado:
        analyzer.guardar_resultados('csv')
        analyzer.generar_reporte_avanzado()
    return resultado