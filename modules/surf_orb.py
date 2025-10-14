#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo SURF y ORB para Análisis de Tráfico Vehicular
===================================================

Implementación de algoritmos SURF (Speeded Up Robust Features) y 
ORB (Oriented FAST and Rotated BRIEF) para identificación de objetos 
en imágenes de tráfico vehicular.

SURF es efectivo para:
- Detección de características robustas ante cambios de escala
- Matching entre imágenes para seguimiento de objetos
- Análisis de texturas distintivas

ORB es útil para:
- Detección rápida de características
- Matching en tiempo real
- Análisis de patrones locales con rotación invariante
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import seaborn as sns

class SURFORBAnalyzer:
    """Analizador SURF y ORB para tráfico vehicular."""
    
    def __init__(self, output_dir="./resultados"):
        """
        Inicializar el analizador SURF-ORB.
        
        Args:
            output_dir (str): Directorio de salida para resultados
        """
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "surf_orb_analysis")
        os.makedirs(self.results_dir, exist_ok=True)
        self.current_results = []
        
        # Configuración SURF
        self.surf_config = {
            'hessianThreshold': 400,
            'nOctaves': 4,
            'nOctaveLayers': 3,
            'extended': True,
            'upright': False
        }
        
        # Configuración ORB
        self.orb_config = {
            'nfeatures': 500,
            'scaleFactor': 1.2,
            'nlevels': 8,
            'edgeThreshold': 31,
            'firstLevel': 0,
            'WTA_K': 2,
            'scoreType': cv2.ORB_HARRIS_SCORE,
            'patchSize': 31,
            'fastThreshold': 20
        }
    
    def extraer_caracteristicas_surf(self, imagen, visualizar=True, mostrar_descriptores=True, guardar_resultados=False, nombre_imagen="imagen_surf"):
        """
        Extrae características SURF de la imagen.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si generar visualización
            mostrar_descriptores (bool): Si mostrar descriptores en consola
            guardar_resultados (bool): Si guardar resultados en archivos
            nombre_imagen (str): Nombre base para archivos guardados
            
        Returns:
            dict: Características SURF extraídas
        """
        print("🔍 Extrayendo características SURF...")
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        try:
            # Crear detector SURF (requiere opencv-contrib-python)
            surf = cv2.xfeatures2d.SURF_create(
                hessianThreshold=self.surf_config['hessianThreshold'],
                nOctaves=self.surf_config['nOctaves'],
                nOctaveLayers=self.surf_config['nOctaveLayers'],
                extended=self.surf_config['extended'],
                upright=self.surf_config['upright']
            )
            
            # Detectar puntos clave y calcular descriptores
            keypoints, descriptors = surf.detectAndCompute(imagen_gris, None)
            surf_available = True
            
        except (AttributeError, cv2.error):
            # Fallback si SURF no está disponible
            print("⚠️  SURF no disponible, usando SIFT como alternativa...")
            surf = cv2.SIFT_create(nfeatures=500)
            keypoints, descriptors = surf.detectAndCompute(imagen_gris, None)
            surf_available = False
        
        # Mostrar información detallada en consola
        if mostrar_descriptores:
            algorithm_name = "SURF" if surf_available else "SIFT"
            print(f"🌊 ANÁLISIS DE CARACTERÍSTICAS {algorithm_name} - {nombre_imagen.upper()}")
            print("="*60)
            print(f"📐 Dimensiones de la imagen: {imagen_gris.shape}")
            print(f"🎯 Número de puntos clave detectados: {len(keypoints)}")
            print(f"🔧 Algoritmo usado: {algorithm_name}")
            
            if descriptors is not None:
                print(f"📊 Forma de la matriz de descriptores: {descriptors.shape}")
                print(f"🔢 Dimensión de cada descriptor: {descriptors.shape[1]} valores")
                print(f"📈 Total de descriptores: {descriptors.shape[0]}")
                
                print("\n📊 Estadísticas de los descriptores:")
                print(f"   • Valor mínimo: {np.min(descriptors):.8f}")
                print(f"   • Valor máximo: {np.max(descriptors):.8f}")
                print(f"   • Promedio: {np.mean(descriptors):.8f}")
                print(f"   • Desviación estándar: {np.std(descriptors):.8f}")
                print(f"   • Energía total: {np.sum(descriptors**2):.6f}")
                
                print(f"\n🔍 Información detallada de los primeros 5 puntos clave:")
                for i in range(min(5, len(keypoints))):
                    kp = keypoints[i]
                    print(f"   📍 Punto clave {i+1}:")
                    print(f"      • Posición (x,y): ({kp.pt[0]:.2f}, {kp.pt[1]:.2f})")
                    print(f"      • Tamaño: {kp.size:.2f}")
                    print(f"      • Ángulo: {kp.angle:.2f}°")
                    print(f"      • Respuesta: {kp.response:.8f}")
                    print(f"      • Octava: {kp.octave}")
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
            # Estadísticas de localización
            kp_x = [kp.pt[0] for kp in keypoints]
            kp_y = [kp.pt[1] for kp in keypoints]
            kp_size = [kp.size for kp in keypoints]
            kp_angle = [kp.angle for kp in keypoints]
            kp_response = [kp.response for kp in keypoints]
            kp_octave = [kp.octave for kp in keypoints]
            
            # Estadísticas básicas
            keypoint_stats = {
                'surf_num_keypoints': num_keypoints,
                'surf_kp_mean_x': np.mean(kp_x),
                'surf_kp_std_x': np.std(kp_x),
                'surf_kp_mean_y': np.mean(kp_y),
                'surf_kp_std_y': np.std(kp_y),
                'surf_kp_mean_size': np.mean(kp_size),
                'surf_kp_std_size': np.std(kp_size),
                'surf_kp_mean_response': np.mean(kp_response),
                'surf_kp_std_response': np.std(kp_response),
                'surf_kp_density': num_keypoints / (imagen_gris.shape[0] * imagen_gris.shape[1])
            }
            
            # Análisis de escalas (octavas)
            octave_counts = np.bincount([oct & 0xFF for oct in kp_octave])
            keypoint_stats.update({
                'surf_num_scales': len(octave_counts),
                'surf_scale_entropy': self._calculate_entropy(octave_counts),
                'surf_dominant_scale': np.argmax(octave_counts) if len(octave_counts) > 0 else 0
            })
            
            # Análisis de orientaciones
            valid_angles = [angle for angle in kp_angle if angle >= 0]
            if valid_angles:
                angle_hist, _ = np.histogram(valid_angles, bins=36, range=(0, 360))
                keypoint_stats.update({
                    'surf_mean_angle': np.mean(valid_angles),
                    'surf_std_angle': np.std(valid_angles),
                    'surf_angle_entropy': self._calculate_entropy(angle_hist),
                    'surf_dominant_orientation': np.argmax(angle_hist) * 10  # grados
                })
            
            # Análisis de descriptores
            if descriptors is not None:
                descriptor_stats = self._analizar_descriptores_surf(descriptors)
                keypoint_stats.update(descriptor_stats)
            
            # Análisis espacial
            spatial_stats = self._analizar_distribucion_espacial(kp_x, kp_y, kp_size, imagen_gris.shape)
            keypoint_stats.update({f'surf_{k}': v for k, v in spatial_stats.items()})
            
        else:
            # No se detectaron puntos clave
            keypoint_stats = self._get_empty_surf_stats()
        
        keypoint_stats['surf_algorithm_used'] = 'SURF' if surf_available else 'SIFT'
        
        resultados = {
            **keypoint_stats,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'gray_image': imagen_gris,
            'nombre_imagen': nombre_imagen
        }
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            self._guardar_resultados_surf(resultados, nombre_imagen)
        
        if visualizar:
            self._visualizar_surf(resultados)
        
        return resultados
    
    def extraer_caracteristicas_orb(self, imagen, visualizar=True, mostrar_descriptores=True, guardar_resultados=False, nombre_imagen="imagen_orb", usar_metodo_profesora=False):
        """
        Extrae características ORB de la imagen.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si generar visualización
            mostrar_descriptores (bool): Si mostrar descriptores en consola
            guardar_resultados (bool): Si guardar resultados en archivos
            nombre_imagen (str): Nombre base para archivos guardados
            usar_metodo_profesora (bool): Si usar configuración exacta de la profesora
            
        Returns:
            dict: Características ORB extraídas
        """
        print("🔍 Extrayendo características ORB...")
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # Crear detector ORB
        if usar_metodo_profesora:
            # Configuración exacta de la profesora (por defecto)
            orb = cv2.ORB_create()
            print("🔧 Usando configuración ORB por defecto (como la profesora)")
        else:
            # Configuración avanzada personalizada
            orb = cv2.ORB_create(
                nfeatures=self.orb_config['nfeatures'],
                scaleFactor=self.orb_config['scaleFactor'],
                nlevels=self.orb_config['nlevels'],
                edgeThreshold=self.orb_config['edgeThreshold'],
                firstLevel=self.orb_config['firstLevel'],
                WTA_K=self.orb_config['WTA_K'],
                scoreType=self.orb_config['scoreType'],
                patchSize=self.orb_config['patchSize'],
                fastThreshold=self.orb_config['fastThreshold']
            )
            print("🔧 Usando configuración ORB avanzada personalizada")
        
        # Detectar puntos clave y calcular descriptores
        keypoints, descriptors = orb.detectAndCompute(imagen_gris, None)
        
        # Mostrar información detallada en consola
        if mostrar_descriptores:
            print(f"🔑 ANÁLISIS DE CARACTERÍSTICAS ORB - {nombre_imagen.upper()}")
            print("="*60)
            print(f"📐 Dimensiones de la imagen: {imagen_gris.shape}")
            print(f"🎯 Número de puntos clave detectados: {len(keypoints)}")
            
            if descriptors is not None:
                print(f"📊 Forma de la matriz de descriptores: {descriptors.shape}")
                print(f"🔢 Dimensión de cada descriptor: {descriptors.shape[1]} bytes ({descriptors.shape[1] * 8} bits)")
                print(f"📈 Total de descriptores: {descriptors.shape[0]}")
                
                print("\n📊 Estadísticas de los descriptores (binarios):")
                # Convertir a bits para análisis
                bit_descriptors = np.unpackbits(descriptors, axis=1)
                print(f"   • Ratio de bits activos promedio: {np.mean(bit_descriptors):.4f}")
                print(f"   • Total de bits: {bit_descriptors.size}")
                print(f"   • Bits activos totales: {np.sum(bit_descriptors)}")
                
                print(f"\n🔍 Información detallada de los primeros 5 puntos clave:")
                for i in range(min(5, len(keypoints))):
                    kp = keypoints[i]
                    print(f"   📍 Punto clave {i+1}:")
                    print(f"      • Posición (x,y): ({kp.pt[0]:.2f}, {kp.pt[1]:.2f})")
                    print(f"      • Tamaño: {kp.size:.2f}")
                    print(f"      • Ángulo: {kp.angle:.2f}°")
                    print(f"      • Respuesta: {kp.response:.8f}")
                    print(f"      • Descriptor (bytes): {descriptors[i]}")
                    print(f"      • Descriptor (bits): {bit_descriptors[i]}")
                
                if len(keypoints) > 0:
                    print(f"\n🎯 Descriptor completo del primer punto clave:")
                    descriptor_bytes = ', '.join([f"{val}" for val in descriptors[0]])
                    descriptor_bits = ', '.join([f"{val}" for val in bit_descriptors[0]])
                    print(f"   Descriptor 1 (bytes): [{descriptor_bytes}]")
                    print(f"   Descriptor 1 (bits):  [{descriptor_bits}]")
                    
                    if len(keypoints) > 1:
                        print(f"\n🎯 Descriptor completo del último punto clave:")
                        descriptor_bytes = ', '.join([f"{val}" for val in descriptors[-1]])
                        descriptor_bits = ', '.join([f"{val}" for val in bit_descriptors[-1]])
                        print(f"   Descriptor {len(keypoints)} (bytes): [{descriptor_bytes}]")
                        print(f"   Descriptor {len(keypoints)} (bits):  [{descriptor_bits}]")
                
                print(f"\n💾 Para ver todos los {len(keypoints)} descriptores completos, active guardar_resultados=True")
            else:
                print("❌ No se pudieron calcular descriptores")
            
            print("="*60)
        
        # Análisis de puntos clave
        num_keypoints = len(keypoints)
        
        if num_keypoints > 0:
            # Estadísticas de localización
            kp_x = [kp.pt[0] for kp in keypoints]
            kp_y = [kp.pt[1] for kp in keypoints]
            kp_size = [kp.size for kp in keypoints]
            kp_angle = [kp.angle for kp in keypoints]
            kp_response = [kp.response for kp in keypoints]
            
            # Estadísticas básicas
            keypoint_stats = {
                'orb_num_keypoints': num_keypoints,
                'orb_kp_mean_x': np.mean(kp_x),
                'orb_kp_std_x': np.std(kp_x),
                'orb_kp_mean_y': np.mean(kp_y),
                'orb_kp_std_y': np.std(kp_y),
                'orb_kp_mean_size': np.mean(kp_size),
                'orb_kp_std_size': np.std(kp_size),
                'orb_kp_mean_response': np.mean(kp_response),
                'orb_kp_std_response': np.std(kp_response),
                'orb_kp_density': num_keypoints / (imagen_gris.shape[0] * imagen_gris.shape[1])
            }
            
            # Análisis de orientaciones (ORB es rotation-invariant)
            valid_angles = [angle for angle in kp_angle if angle >= 0]
            if valid_angles:
                angle_hist, _ = np.histogram(valid_angles, bins=36, range=(0, 360))
                keypoint_stats.update({
                    'orb_mean_angle': np.mean(valid_angles),
                    'orb_std_angle': np.std(valid_angles),
                    'orb_angle_entropy': self._calculate_entropy(angle_hist),
                    'orb_angle_uniformity': np.std(angle_hist) / (np.mean(angle_hist) + 1e-10)
                })
            
            # Análisis de descriptores ORB (binarios)
            if descriptors is not None:
                descriptor_stats = self._analizar_descriptores_orb(descriptors)
                keypoint_stats.update(descriptor_stats)
            
            # Análisis espacial específico para ORB
            spatial_stats = self._analizar_distribucion_espacial(kp_x, kp_y, kp_size, imagen_gris.shape)
            keypoint_stats.update({f'orb_{k}': v for k, v in spatial_stats.items()})
            
            # Análisis de pirámide de escalas
            scale_analysis = self._analizar_escalas_orb(keypoints)
            keypoint_stats.update(scale_analysis)
            
        else:
            # No se detectaron puntos clave
            keypoint_stats = self._get_empty_orb_stats()
        
        resultados = {
            **keypoint_stats,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'gray_image': imagen_gris,
            'nombre_imagen': nombre_imagen,
            'usar_metodo_profesora': usar_metodo_profesora
        }
        
        # Guardar resultados si se solicita
        if guardar_resultados:
            self._guardar_resultados_orb(resultados, nombre_imagen)
        
        if visualizar:
            self._visualizar_orb(resultados)
        
        return resultados
    
    def analisis_combinado_surf_orb(self, imagen_path, nombre_imagen=None):
        """
        Realiza análisis combinado SURF + ORB.
        
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
            
            print(f"🔄 Análisis SURF-ORB para: {nombre_imagen}")
            
            # Análisis SURF
            resultados_surf = self.extraer_caracteristicas_surf(imagen, visualizar=False)
            
            # Análisis ORB
            resultados_orb = self.extraer_caracteristicas_orb(imagen, visualizar=False)
            
            # Análisis comparativo
            comparacion = self._comparar_surf_orb(resultados_surf, resultados_orb)
            
            # Análisis adicional: FAST corners
            fast_stats = self._analizar_fast_corners(imagen)
            
            # Combinar resultados
            resultado_completo = {
                'Imagen': nombre_imagen,
                'Ruta': imagen_path,
                'Dimensiones': imagen.shape,
                'Fecha_Analisis': datetime.now().isoformat(),
                **{k: v for k, v in resultados_surf.items() 
                   if not isinstance(v, (list, np.ndarray))},
                **{k: v for k, v in resultados_orb.items() 
                   if not isinstance(v, (list, np.ndarray))},
                **comparacion,
                **fast_stats
            }
            
            self.current_results.append(resultado_completo)
            
            print(f"✅ Análisis SURF-ORB completado para: {nombre_imagen}")
            return resultado_completo
            
        except Exception as e:
            print(f"❌ Error al procesar {imagen_path}: {str(e)}")
            return None
    
    def _analizar_descriptores_surf(self, descriptors):
        """Analiza descriptores SURF (64 o 128 dimensiones)."""
        if descriptors is None or len(descriptors) == 0:
            return {
                'surf_descriptor_length': 0,
                'surf_descriptor_mean': 0,
                'surf_descriptor_std': 0,
                'surf_descriptor_sparsity': 0,
                'surf_descriptor_energy': 0,
                'surf_descriptor_entropy': 0
            }
        
        return {
            'surf_descriptor_length': descriptors.shape[1],
            'surf_descriptor_mean': np.mean(descriptors),
            'surf_descriptor_std': np.std(descriptors),
            'surf_descriptor_sparsity': np.sum(np.abs(descriptors) < 0.01) / descriptors.size,
            'surf_descriptor_energy': np.sum(descriptors ** 2),
            'surf_descriptor_entropy': self._calculate_entropy(descriptors.flatten()),
            'surf_descriptor_diversity': self._calculate_descriptor_diversity(descriptors)
        }
    
    def _analizar_descriptores_orb(self, descriptors):
        """Analiza descriptores ORB (binarios, 256 bits)."""
        if descriptors is None or len(descriptors) == 0:
            return {
                'orb_descriptor_length': 0,
                'orb_descriptor_mean_hamming': 0,
                'orb_descriptor_std_hamming': 0,
                'orb_descriptor_sparsity': 0,
                'orb_descriptor_entropy': 0
            }
        
        # Convertir a bits para análisis
        bit_counts = np.unpackbits(descriptors, axis=1)
        
        # Distancias de Hamming entre descriptores
        hamming_distances = []
        for i in range(min(len(descriptors), 100)):  # Limitar para eficiencia
            for j in range(i+1, min(len(descriptors), 100)):
                hamming_dist = np.sum(descriptors[i] != descriptors[j])
                hamming_distances.append(hamming_dist)
        
        return {
            'orb_descriptor_length': descriptors.shape[1] * 8,  # bits
            'orb_descriptor_mean_hamming': np.mean(hamming_distances) if hamming_distances else 0,
            'orb_descriptor_std_hamming': np.std(hamming_distances) if hamming_distances else 0,
            'orb_descriptor_bit_ratio': np.mean(bit_counts),  # Ratio de bits activos
            'orb_descriptor_entropy': self._calculate_entropy(bit_counts.flatten()),
            'orb_descriptor_uniqueness': len(set(map(tuple, descriptors))) / len(descriptors) if len(descriptors) > 0 else 0
        }
    
    def _analizar_distribucion_espacial(self, kp_x, kp_y, kp_size, image_shape):
        """Analiza la distribución espacial de puntos clave."""
        if not kp_x or not kp_y:
            return {
                'spatial_uniformity': 0,
                'spatial_clustering': 0,
                'edge_concentration': 0,
                'center_concentration': 0,
                'size_gradient': 0
            }
        
        h, w = image_shape[:2]
        
        # Dividir imagen en regiones
        h_third = h // 3
        w_third = w // 3
        
        regions = {
            'top_left': sum(1 for x, y in zip(kp_x, kp_y) if x < w_third and y < h_third),
            'top_center': sum(1 for x, y in zip(kp_x, kp_y) if w_third <= x < 2*w_third and y < h_third),
            'top_right': sum(1 for x, y in zip(kp_x, kp_y) if x >= 2*w_third and y < h_third),
            'center_left': sum(1 for x, y in zip(kp_x, kp_y) if x < w_third and h_third <= y < 2*h_third),
            'center': sum(1 for x, y in zip(kp_x, kp_y) if w_third <= x < 2*w_third and h_third <= y < 2*h_third),
            'center_right': sum(1 for x, y in zip(kp_x, kp_y) if x >= 2*w_third and h_third <= y < 2*h_third),
            'bottom_left': sum(1 for x, y in zip(kp_x, kp_y) if x < w_third and y >= 2*h_third),
            'bottom_center': sum(1 for x, y in zip(kp_x, kp_y) if w_third <= x < 2*w_third and y >= 2*h_third),
            'bottom_right': sum(1 for x, y in zip(kp_x, kp_y) if x >= 2*w_third and y >= 2*h_third)
        }
        
        # Uniformidad espacial
        region_counts = list(regions.values())
        spatial_uniformity = np.std(region_counts) / (np.mean(region_counts) + 1e-10)
        
        # Concentración en el centro
        center_concentration = regions['center'] / len(kp_x)
        
        # Concentración en bordes
        edge_points = sum(1 for x, y in zip(kp_x, kp_y) 
                         if x < w*0.1 or x > w*0.9 or y < h*0.1 or y > h*0.9)
        edge_concentration = edge_points / len(kp_x)
        
        # Gradiente de tamaño
        center_x, center_y = w/2, h/2
        distances_to_center = [np.sqrt((x - center_x)**2 + (y - center_y)**2) 
                              for x, y in zip(kp_x, kp_y)]
        
        if len(kp_size) == len(distances_to_center):
            correlation_size_distance = np.corrcoef(kp_size, distances_to_center)[0, 1]
            size_gradient = correlation_size_distance if not np.isnan(correlation_size_distance) else 0
        else:
            size_gradient = 0
        
        return {
            'spatial_uniformity': spatial_uniformity,
            'spatial_clustering': 1.0 / (spatial_uniformity + 1),
            'edge_concentration': edge_concentration,
            'center_concentration': center_concentration,
            'size_gradient': size_gradient
        }
    
    def _analizar_escalas_orb(self, keypoints):
        """Analiza la distribución de escalas en ORB."""
        if not keypoints:
            return {
                'orb_scale_diversity': 0,
                'orb_scale_entropy': 0,
                'orb_dominant_scale_level': 0
            }
        
        # Extraer niveles de escala
        scale_levels = [kp.octave & 0xFF for kp in keypoints]
        scale_counts = np.bincount(scale_levels)
        
        return {
            'orb_scale_diversity': len(np.unique(scale_levels)),
            'orb_scale_entropy': self._calculate_entropy(scale_counts),
            'orb_dominant_scale_level': np.argmax(scale_counts) if len(scale_counts) > 0 else 0
        }
    
    def _analizar_fast_corners(self, imagen):
        """Analiza corners usando FAST detector."""
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # FAST corner detection
        fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        fast_kp = fast.detect(imagen_gris, None)
        
        return {
            'fast_num_corners': len(fast_kp),
            'fast_corner_density': len(fast_kp) / (imagen_gris.shape[0] * imagen_gris.shape[1]),
            'fast_mean_response': np.mean([kp.response for kp in fast_kp]) if fast_kp else 0
        }
    
    def _comparar_surf_orb(self, resultados_surf, resultados_orb):
        """Compara resultados entre SURF y ORB."""
        surf_kp = resultados_surf.get('surf_num_keypoints', 0)
        orb_kp = resultados_orb.get('orb_num_keypoints', 0)
        
        surf_density = resultados_surf.get('surf_kp_density', 0)
        orb_density = resultados_orb.get('orb_kp_density', 0)
        
        return {
            'keypoint_ratio_surf_orb': surf_kp / (orb_kp + 1e-10),
            'density_ratio_surf_orb': surf_density / (orb_density + 1e-10),
            'total_keypoints_surf_orb': surf_kp + orb_kp,
            'algorithm_preference': 'SURF' if surf_kp > orb_kp else 'ORB' if orb_kp > 0 else 'None'
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
        
        # Calcular distancias entre pares de descriptores (muestra limitada)
        from scipy.spatial.distance import pdist
        sample_size = min(100, len(descriptors))
        sample_descriptors = descriptors[:sample_size]
        distances = pdist(sample_descriptors, metric='euclidean')
        return np.mean(distances)
    
    def _get_empty_surf_stats(self):
        """Retorna estadísticas vacías para SURF."""
        return {
            'surf_num_keypoints': 0, 'surf_kp_mean_x': 0, 'surf_kp_std_x': 0,
            'surf_kp_mean_y': 0, 'surf_kp_std_y': 0, 'surf_kp_mean_size': 0,
            'surf_kp_std_size': 0, 'surf_kp_mean_response': 0, 'surf_kp_std_response': 0,
            'surf_kp_density': 0, 'surf_num_scales': 0, 'surf_scale_entropy': 0,
            'surf_dominant_scale': 0, 'surf_mean_angle': 0, 'surf_std_angle': 0,
            'surf_angle_entropy': 0, 'surf_dominant_orientation': 0,
            'surf_descriptor_length': 0, 'surf_descriptor_mean': 0, 'surf_descriptor_std': 0,
            'surf_descriptor_sparsity': 0, 'surf_descriptor_energy': 0, 'surf_descriptor_entropy': 0,
            'surf_spatial_uniformity': 0, 'surf_spatial_clustering': 0, 'surf_edge_concentration': 0,
            'surf_center_concentration': 0, 'surf_size_gradient': 0
        }
    
    def _get_empty_orb_stats(self):
        """Retorna estadísticas vacías para ORB."""
        return {
            'orb_num_keypoints': 0, 'orb_kp_mean_x': 0, 'orb_kp_std_x': 0,
            'orb_kp_mean_y': 0, 'orb_kp_std_y': 0, 'orb_kp_mean_size': 0,
            'orb_kp_std_size': 0, 'orb_kp_mean_response': 0, 'orb_kp_std_response': 0,
            'orb_kp_density': 0, 'orb_mean_angle': 0, 'orb_std_angle': 0,
            'orb_angle_entropy': 0, 'orb_angle_uniformity': 0,
            'orb_descriptor_length': 0, 'orb_descriptor_mean_hamming': 0,
            'orb_descriptor_std_hamming': 0, 'orb_descriptor_bit_ratio': 0,
            'orb_descriptor_entropy': 0, 'orb_descriptor_uniqueness': 0,
            'orb_spatial_uniformity': 0, 'orb_spatial_clustering': 0, 'orb_edge_concentration': 0,
            'orb_center_concentration': 0, 'orb_size_gradient': 0,
            'orb_scale_diversity': 0, 'orb_scale_entropy': 0, 'orb_dominant_scale_level': 0
        }
    
    def _visualizar_surf(self, resultados):
        """Visualiza los resultados SURF."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Imagen original
        axes[0].imshow(resultados['gray_image'], cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        # Puntos clave SURF con puntos amarillos
        if resultados['keypoints']:
            # Convertir imagen gris a RGB para mostrar puntos amarillos
            img_kp = cv2.cvtColor(resultados['gray_image'], cv2.COLOR_GRAY2RGB)
            img_kp = cv2.drawKeypoints(img_kp, resultados['keypoints'], None, 
                                     color=(255, 255, 0),  # Amarillo en BGR
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            axes[1].imshow(img_kp)
        else:
            axes[1].imshow(resultados['gray_image'], cmap='gray')
        axes[1].set_title(f'Puntos Clave SURF ({resultados.get("surf_num_keypoints", 0)})')
        axes[1].axis('off')
        
        plt.tight_layout()
        archivo_viz = os.path.join(self.results_dir, f'surf_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        os.makedirs(os.path.dirname(archivo_viz), exist_ok=True)
        plt.savefig(archivo_viz, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualizar_orb(self, resultados):
        """Visualiza los resultados ORB."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Imagen original
        axes[0].imshow(resultados['gray_image'], cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        # Determinar método de visualización
        usar_metodo_profesora = resultados.get('usar_metodo_profesora', False)
        
        if usar_metodo_profesora:
            # Método exacto de la profesora
            img_kp = resultados['gray_image'].copy()
            if resultados['keypoints']:
                # Usar la misma imagen como salida y flags de la profesora
                img_kp = cv2.drawKeypoints(img_kp, resultados['keypoints'], img_kp, 
                                         color=(255, 0, 0), 
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Mostrar en escala de grises como la profesora
            axes[1].imshow(img_kp, cmap='gray')
            axes[1].set_title(f'Puntos Clave ORB - Método Profesora ({resultados.get("orb_num_keypoints", 0)})')
        else:
            # Mi método personalizado - Crear imagen en color para mostrar puntos rojos
            if len(resultados['gray_image'].shape) == 2:
                # Convertir imagen gris a color (RGB) para poder mostrar puntos rojos
                img_kp = cv2.cvtColor(resultados['gray_image'], cv2.COLOR_GRAY2RGB)
            else:
                img_kp = resultados['gray_image'].copy()
            
            if resultados['keypoints']:
                # Dibujar puntos clave con color rojo (255, 0, 0) y sin flags para puntos más pequeños
                img_kp = cv2.drawKeypoints(img_kp, resultados['keypoints'], None, 
                                         color=(255, 0, 0), flags=0)
            
            axes[1].imshow(img_kp)
            axes[1].set_title(f'Puntos Clave ORB - Método Personalizado ({resultados.get("orb_num_keypoints", 0)})')
        
        axes[1].axis('off')
        
        plt.tight_layout()
        archivo_viz = os.path.join(self.results_dir, f'orb_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        os.makedirs(os.path.dirname(archivo_viz), exist_ok=True)
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
            archivo_csv = os.path.join(self.results_dir, f'surf_orb_analysis_{timestamp}.csv')
            os.makedirs(os.path.dirname(archivo_csv), exist_ok=True)
            df.to_csv(archivo_csv, index=False)
            print(f"💾 Resultados CSV guardados: {archivo_csv}")
        
        elif formato.lower() == 'json':
            import json
            archivo_json = os.path.join(self.results_dir, f'surf_orb_analysis_{timestamp}.json')
            os.makedirs(os.path.dirname(archivo_json), exist_ok=True)
            with open(archivo_json, 'w', encoding='utf-8') as f:
                json.dump(self.current_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Resultados JSON guardados: {archivo_json}")
    
    def generar_reporte_surf_orb(self):
        """Genera reporte del análisis SURF-ORB."""
        if not self.current_results:
            print("❌ No hay resultados para el reporte.")
            return
        
        print("\n📋 REPORTE ANÁLISIS SURF + ORB")
        print("=" * 40)
        print(f"📊 Imágenes analizadas: {len(self.current_results)}")
        
        # Estadísticas SURF
        surf_keypoints = [r.get('surf_num_keypoints', 0) for r in self.current_results]
        surf_density = [r.get('surf_kp_density', 0) for r in self.current_results]
        
        print(f"\n🌊 ESTADÍSTICAS SURF:")
        print(f"   Puntos clave promedio: {np.mean(surf_keypoints):.1f}")
        print(f"   Densidad promedio: {np.mean(surf_density):.6f}")
        
        # Estadísticas ORB
        orb_keypoints = [r.get('orb_num_keypoints', 0) for r in self.current_results]
        orb_density = [r.get('orb_kp_density', 0) for r in self.current_results]
        
        print(f"\n⭕ ESTADÍSTICAS ORB:")
        print(f"   Puntos clave promedio: {np.mean(orb_keypoints):.1f}")
        print(f"   Densidad promedio: {np.mean(orb_density):.6f}")
        
        # Comparación
        ratios = [r.get('keypoint_ratio_surf_orb', 1) for r in self.current_results]
        print(f"\n⚖️  COMPARACIÓN:")
        print(f"   Ratio SURF/ORB promedio: {np.mean(ratios):.2f}")
        
        # Algoritmo preferido
        preferences = [r.get('algorithm_preference', 'None') for r in self.current_results]
        pref_counts = {pref: preferences.count(pref) for pref in set(preferences)}
        print(f"   Preferencias: {pref_counts}")
        
        print("\n" + "=" * 40)
    
    def _guardar_resultados_orb(self, resultados, nombre_imagen):
        """Guarda los resultados ORB en archivos CSV y TXT."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio si no existe
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Datos estadísticos generales
        orb_stats = {
            'imagen': nombre_imagen,
            'dimensiones': f"{resultados['gray_image'].shape[0]}x{resultados['gray_image'].shape[1]}",
            'num_puntos_clave': resultados.get('orb_num_keypoints', 0),
            'dimension_descriptor_bytes': resultados.get('orb_descriptor_length', 0) // 8,
            'dimension_descriptor_bits': resultados.get('orb_descriptor_length', 0),
            'densidad_puntos': resultados.get('orb_kp_density', 0),
            'fecha_analisis': datetime.now().isoformat()
        }
        
        if resultados['descriptors'] is not None:
            orb_stats.update({
                'hamming_distance_promedio': resultados.get('orb_descriptor_mean_hamming', 0),
                'hamming_distance_std': resultados.get('orb_descriptor_std_hamming', 0),
                'ratio_bits_activos': resultados.get('orb_descriptor_bit_ratio', 0),
                'entropia_descriptores': resultados.get('orb_descriptor_entropy', 0),
                'unicidad_descriptores': resultados.get('orb_descriptor_uniqueness', 0)
            })
        
        # Guardar estadísticas generales en CSV
        df_stats = pd.DataFrame([orb_stats])
        archivo_csv_stats = os.path.join(self.results_dir, f'orb_estadisticas_{nombre_imagen}_{timestamp}.csv')
        df_stats.to_csv(archivo_csv_stats, index=False, encoding='utf-8')
        print(f"✅ Estadísticas ORB guardadas en: {archivo_csv_stats}")
        
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
                    'respuesta': kp.response,
                    'octava': kp.octave
                })
            
            df_keypoints = pd.DataFrame(keypoints_data)
            archivo_csv_keypoints = os.path.join(self.results_dir, f'orb_puntos_clave_{nombre_imagen}_{timestamp}.csv')
            df_keypoints.to_csv(archivo_csv_keypoints, index=False, encoding='utf-8')
            print(f"✅ Puntos clave ORB guardados en: {archivo_csv_keypoints}")
        
        # Guardar descriptores completos en CSV (bytes y bits)
        if resultados['descriptors'] is not None:
            # Descriptores en bytes
            descriptor_columns_bytes = [f'descriptor_byte_{j+1}' for j in range(resultados['descriptors'].shape[1])]
            df_descriptors_bytes = pd.DataFrame(resultados['descriptors'], columns=descriptor_columns_bytes)
            df_descriptors_bytes.insert(0, 'punto_clave_id', range(1, len(resultados['descriptors']) + 1))
            
            archivo_csv_descriptors_bytes = os.path.join(self.results_dir, f'orb_descriptores_bytes_{nombre_imagen}_{timestamp}.csv')
            df_descriptors_bytes.to_csv(archivo_csv_descriptors_bytes, index=False, encoding='utf-8')
            print(f"✅ Descriptores ORB (bytes) guardados en: {archivo_csv_descriptors_bytes}")
            
            # Descriptores en bits
            bit_descriptors = np.unpackbits(resultados['descriptors'], axis=1)
            descriptor_columns_bits = [f'descriptor_bit_{j+1}' for j in range(bit_descriptors.shape[1])]
            df_descriptors_bits = pd.DataFrame(bit_descriptors, columns=descriptor_columns_bits)
            df_descriptors_bits.insert(0, 'punto_clave_id', range(1, len(bit_descriptors) + 1))
            
            archivo_csv_descriptors_bits = os.path.join(self.results_dir, f'orb_descriptores_bits_{nombre_imagen}_{timestamp}.csv')
            df_descriptors_bits.to_csv(archivo_csv_descriptors_bits, index=False, encoding='utf-8')
            print(f"✅ Descriptores ORB (bits) guardados en: {archivo_csv_descriptors_bits}")
        
        # Guardar reporte completo en TXT
        archivo_txt = os.path.join(self.results_dir, f'orb_reporte_completo_{nombre_imagen}_{timestamp}.txt')
        with open(archivo_txt, 'w', encoding='utf-8') as f:
            f.write("REPORTE COMPLETO - ANÁLISIS ORB\n")
            f.write("="*60 + "\n\n")
            f.write(f"Imagen analizada: {orb_stats['imagen']}\n")
            f.write(f"Fecha de análisis: {orb_stats['fecha_analisis']}\n")
            f.write(f"Dimensiones: {orb_stats['dimensiones']}\n")
            f.write(f"Número de puntos clave: {orb_stats['num_puntos_clave']}\n")
            f.write(f"Dimensión de descriptores: {orb_stats['dimension_descriptor_bytes']} bytes ({orb_stats['dimension_descriptor_bits']} bits)\n\n")
            
            if resultados['descriptors'] is not None:
                f.write("ESTADÍSTICAS DE DESCRIPTORES (BINARIOS):\n")
                f.write("-" * 40 + "\n")
                f.write(f"Distancia Hamming promedio: {resultados.get('orb_descriptor_mean_hamming', 0):.4f}\n")
                f.write(f"Desviación estándar Hamming: {resultados.get('orb_descriptor_std_hamming', 0):.4f}\n")
                f.write(f"Ratio de bits activos: {resultados.get('orb_descriptor_bit_ratio', 0):.4f}\n")
                f.write(f"Entropía: {resultados.get('orb_descriptor_entropy', 0):.8f}\n")
                f.write(f"Unicidad: {resultados.get('orb_descriptor_uniqueness', 0):.4f}\n\n")
            
            f.write("INFORMACIÓN DETALLADA DE PUNTOS CLAVE:\n")
            f.write("-" * 45 + "\n")
            for i, kp in enumerate(resultados['keypoints']):
                f.write(f"Punto clave {i+1}:\n")
                f.write(f"  Posición (x,y): ({kp.pt[0]:.2f}, {kp.pt[1]:.2f})\n")
                f.write(f"  Tamaño: {kp.size:.2f}\n")
                f.write(f"  Ángulo: {kp.angle:.2f}°\n")
                f.write(f"  Respuesta: {kp.response:.8f}\n")
                f.write(f"  Octava: {kp.octave}\n")
                if resultados['descriptors'] is not None:
                    f.write("  Descriptor completo (bytes):\n")
                    descriptor_bytes_str = ', '.join([f"{val}" for val in resultados['descriptors'][i]])
                    f.write(f"    [{descriptor_bytes_str}]\n")
                    f.write("  Descriptor completo (bits):\n")
                    bit_descriptors = np.unpackbits(resultados['descriptors'], axis=1)
                    descriptor_bits_str = ', '.join([f"{val}" for val in bit_descriptors[i]])
                    f.write(f"    [{descriptor_bits_str}]\n")
                f.write("\n")
        
        print(f"✅ Reporte ORB completo guardado en: {archivo_txt}")
        archivos_generados = 2 + (1 if resultados['keypoints'] else 0) + (2 if resultados['descriptors'] is not None else 0)
        print(f"📊 Total de archivos generados: {archivos_generados}")
    
    def _guardar_resultados_surf(self, resultados, nombre_imagen):
        """Guarda los resultados SURF en archivos CSV y TXT."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio si no existe
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Datos estadísticos generales
        surf_stats = {
            'imagen': nombre_imagen,
            'dimensiones': f"{resultados['gray_image'].shape[0]}x{resultados['gray_image'].shape[1]}",
            'num_puntos_clave': resultados.get('surf_num_keypoints', 0),
            'dimension_descriptor': resultados.get('surf_descriptor_length', 0),
            'densidad_puntos': resultados.get('surf_kp_density', 0),
            'algoritmo_usado': resultados.get('surf_algorithm_used', 'SURF'),
            'fecha_analisis': datetime.now().isoformat()
        }
        
        if resultados['descriptors'] is not None:
            surf_stats.update({
                'valor_min_descriptores': resultados.get('surf_descriptor_mean', 0),
                'valor_max_descriptores': resultados.get('surf_descriptor_std', 0),
                'promedio_descriptores': resultados.get('surf_descriptor_energy', 0),
                'entropia_descriptores': resultados.get('surf_descriptor_entropy', 0),
                'diversidad_descriptores': resultados.get('surf_descriptor_diversity', 0)
            })
        
        # Guardar estadísticas generales en CSV
        df_stats = pd.DataFrame([surf_stats])
        archivo_csv_stats = os.path.join(self.results_dir, f'surf_estadisticas_{nombre_imagen}_{timestamp}.csv')
        df_stats.to_csv(archivo_csv_stats, index=False, encoding='utf-8')
        print(f"✅ Estadísticas SURF guardadas en: {archivo_csv_stats}")
        
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
                    'respuesta': kp.response,
                    'octava': kp.octave
                })
            
            df_keypoints = pd.DataFrame(keypoints_data)
            archivo_csv_keypoints = os.path.join(self.results_dir, f'surf_puntos_clave_{nombre_imagen}_{timestamp}.csv')
            df_keypoints.to_csv(archivo_csv_keypoints, index=False, encoding='utf-8')
            print(f"✅ Puntos clave SURF guardados en: {archivo_csv_keypoints}")
        
        # Guardar descriptores completos en CSV
        if resultados['descriptors'] is not None:
            descriptor_columns = [f'descriptor_{j+1}' for j in range(resultados['descriptors'].shape[1])]
            df_descriptors = pd.DataFrame(resultados['descriptors'], columns=descriptor_columns)
            df_descriptors.insert(0, 'punto_clave_id', range(1, len(resultados['descriptors']) + 1))
            
            archivo_csv_descriptors = os.path.join(self.results_dir, f'surf_descriptores_{nombre_imagen}_{timestamp}.csv')
            df_descriptors.to_csv(archivo_csv_descriptors, index=False, encoding='utf-8')
            print(f"✅ Descriptores SURF completos guardados en: {archivo_csv_descriptors}")
        
        # Guardar reporte completo en TXT
        archivo_txt = os.path.join(self.results_dir, f'surf_reporte_completo_{nombre_imagen}_{timestamp}.txt')
        with open(archivo_txt, 'w', encoding='utf-8') as f:
            f.write("REPORTE COMPLETO - ANÁLISIS SURF\n")
            f.write("="*60 + "\n\n")
            f.write(f"Imagen analizada: {surf_stats['imagen']}\n")
            f.write(f"Fecha de análisis: {surf_stats['fecha_analisis']}\n")
            f.write(f"Dimensiones: {surf_stats['dimensiones']}\n")
            f.write(f"Número de puntos clave: {surf_stats['num_puntos_clave']}\n")
            f.write(f"Dimensión de descriptores: {surf_stats['dimension_descriptor']} valores\n")
            f.write(f"Algoritmo usado: {surf_stats['algoritmo_usado']}\n\n")
            
            if resultados['descriptors'] is not None:
                f.write("ESTADÍSTICAS DE DESCRIPTORES:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Promedio: {resultados.get('surf_descriptor_mean', 0):.8f}\n")
                f.write(f"Desviación estándar: {resultados.get('surf_descriptor_std', 0):.8f}\n")
                f.write(f"Energía: {resultados.get('surf_descriptor_energy', 0):.8f}\n")
                f.write(f"Entropía: {resultados.get('surf_descriptor_entropy', 0):.8f}\n")
                f.write(f"Diversidad: {resultados.get('surf_descriptor_diversity', 0):.8f}\n\n")
            
            f.write("INFORMACIÓN DETALLADA DE PUNTOS CLAVE:\n")
            f.write("-" * 45 + "\n")
            for i, kp in enumerate(resultados['keypoints']):
                f.write(f"Punto clave {i+1}:\n")
                f.write(f"  Posición (x,y): ({kp.pt[0]:.2f}, {kp.pt[1]:.2f})\n")
                f.write(f"  Tamaño: {kp.size:.2f}\n")
                f.write(f"  Ángulo: {kp.angle:.2f}°\n")
                f.write(f"  Respuesta: {kp.response:.8f}\n")
                f.write(f"  Octava: {kp.octave}\n")
                if resultados['descriptors'] is not None:
                    f.write("  Descriptor completo:\n")
                    descriptor_str = ', '.join([f"{val:.8f}" for val in resultados['descriptors'][i]])
                    f.write(f"    [{descriptor_str}]\n")
                f.write("\n")
        
        print(f"✅ Reporte SURF completo guardado en: {archivo_txt}")
        archivos_generados = 2 + (1 if resultados['keypoints'] else 0) + (1 if resultados['descriptors'] is not None else 0)
        print(f"📊 Total de archivos generados: {archivos_generados}")

# Función de utilidad
def analizar_surf_orb_imagen(imagen_path, output_dir="./resultados"):
    """
    Función de conveniencia para análisis SURF-ORB.
    
    Args:
        imagen_path (str): Ruta a la imagen
        output_dir (str): Directorio de salida
        
    Returns:
        dict: Resultados del análisis
    """
    analyzer = SURFORBAnalyzer(output_dir)
    resultado = analyzer.analisis_combinado_surf_orb(imagen_path)
    if resultado:
        analyzer.guardar_resultados('csv')
        analyzer.generar_reporte_surf_orb()
    return resultado