#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Análisis con Transformada de Hough para Tráfico Vehicular
==================================================================

Implementación de la Transformada de Hough para detección de líneas y círculos
en imágenes de tráfico vehicular. Especialmente útil para:
- Detección de carriles y líneas de la carretera
- Identificación de señales circulares (señales de tráfico)
- Detección de ruedas y elementos circulares de vehículos
- Análisis de la geometría de la infraestructura vial

Basado en el código de análisis de texturas con Hough existente.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage import measure, morphology
import seaborn as sns

class HoughAnalyzer:
    """Analizador de Transformada de Hough para tráfico vehicular."""
    
    def __init__(self, output_dir="./resultados"):
        """
        Inicializar el analizador de Hough.
        
        Args:
            output_dir (str): Directorio de salida para resultados
        """
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "hough_analysis")
        os.makedirs(self.results_dir, exist_ok=True)
        self.current_results = []
        
        # Parámetros por defecto para tráfico vehicular
        self.config = {
            'canny_low': 50,
            'canny_high': 150,
            'gaussian_sigma': 2,
            'hough_line_threshold': 100,
            'min_line_length': 50,
            'max_line_gap': 10,
            'circle_radii_range': range(10, 100, 2),
            'circle_threshold': 0.6
        }
    
    def preprocesar_imagen(self, imagen):
        """
        Preprocesa la imagen para optimizar la detección de Hough.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            
        Returns:
            tuple: (imagen_gris, bordes_canny)
        """
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # Filtro Gaussiano para reducir ruido
        imagen_suavizada = cv2.GaussianBlur(imagen_gris, 
                                          (5, 5), 
                                          self.config['gaussian_sigma'])
        
        # Detección de bordes con Canny
        bordes = canny(imagen_suavizada, 
                      low_threshold=self.config['canny_low'],
                      high_threshold=self.config['canny_high'])
        
        return imagen_gris, bordes
    
    def detectar_lineas_hough(self, imagen, visualizar=True):
        """
        Detecta líneas usando la Transformada de Hough.
        Especialmente útil para carriles y elementos lineales del tráfico.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si mostrar visualización
            
        Returns:
            dict: Resultados de la detección de líneas
        """
        print("Detectando líneas con Transformada de Hough...")
        
        # Preprocesamiento
        imagen_gris, bordes = self.preprocesar_imagen(imagen)
        
        # Método 1: Hough Líneas Estándar (scikit-image)
        tested_angles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False)
        h, theta, d = hough_line(bordes, theta=tested_angles)
        
        # Encontrar picos en el espacio de Hough
        hough_peaks = hough_line_peaks(h, theta, d, 
                                     threshold=self.config['hough_line_threshold']//2,
                                     min_angle=10, min_distance=25)
        
        # Método 2: Hough Líneas Probabilístico (OpenCV)
        lineas_opencv = cv2.HoughLinesP(bordes.astype(np.uint8) * 255,
                                       rho=1,
                                       theta=np.pi/180,
                                       threshold=self.config['hough_line_threshold'],
                                       minLineLength=self.config['min_line_length'],
                                       maxLineGap=self.config['max_line_gap'])
        
        # Analizar líneas detectadas
        num_lineas_skimage = len(hough_peaks[0]) if hough_peaks[0] is not None else 0
        num_lineas_opencv = len(lineas_opencv) if lineas_opencv is not None else 0
        
        # Calcular estadísticas de las líneas
        angulos_detectados = []
        distancias_detectadas = []
        longitudes_lineas = []
        
        # Análisis de líneas de scikit-image
        if num_lineas_skimage > 0:
            for _, angle, dist in zip(*hough_peaks):
                angulos_detectados.append(np.degrees(angle))
                distancias_detectadas.append(dist)
        
        # Análisis de líneas de OpenCV
        if lineas_opencv is not None:
            for linea in lineas_opencv:
                x1, y1, x2, y2 = linea[0]
                longitud = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                longitudes_lineas.append(longitud)
                
                # Calcular ángulo
                angulo = np.degrees(np.arctan2(y2-y1, x2-x1))
                angulos_detectados.append(angulo)
        
        # Estadísticas de ángulos y distancias
        stats_angulos = self._calcular_estadisticas(angulos_detectados)
        stats_distancias = self._calcular_estadisticas(distancias_detectadas)
        stats_longitudes = self._calcular_estadisticas(longitudes_lineas)
        
        # Análisis específico para tráfico vehicular
        lineas_horizontales = [ang for ang in angulos_detectados if abs(ang) < 30 or abs(ang) > 150]
        lineas_verticales = [ang for ang in angulos_detectados if 60 < abs(ang) < 120]
        lineas_diagonales = [ang for ang in angulos_detectados if 30 <= abs(ang) <= 60 or 120 <= abs(ang) <= 150]
        
        # Momentos del espacio de Hough
        momento_segundo_orden = np.sum(h * h)
        momento_energia = np.sum(h ** 2)
        momento_inercia = np.sum(h * (np.arange(len(h))[:, np.newaxis] ** 2))
        
        # Resultados
        resultados = {
            'num_lineas_skimage': num_lineas_skimage,
            'num_lineas_opencv': num_lineas_opencv,
            'num_lineas_total': num_lineas_skimage + num_lineas_opencv,
            'bordes_detectados': np.sum(bordes),
            'intensidad_promedio': np.mean(imagen_gris),
            'desviacion_intensidad': np.std(imagen_gris),
            
            # Estadísticas de ángulos
            'media_angulos': stats_angulos['media'],
            'std_angulos': stats_angulos['std'],
            'min_angulos': stats_angulos['min'],
            'max_angulos': stats_angulos['max'],
            
            # Estadísticas de distancias
            'media_distancias': stats_distancias['media'],
            'std_distancias': stats_distancias['std'],
            
            # Estadísticas de longitudes
            'media_longitudes': stats_longitudes['media'],
            'std_longitudes': stats_longitudes['std'],
            
            # Análisis direccional
            'num_lineas_horizontales': len(lineas_horizontales),
            'num_lineas_verticales': len(lineas_verticales),
            'num_lineas_diagonales': len(lineas_diagonales),
            'ratio_horizontal_vertical': len(lineas_horizontales) / (len(lineas_verticales) + 1),
            
            # Momentos del espacio de Hough
            'momento_segundo_orden_hough': momento_segundo_orden,
            'momento_energia_hough': momento_energia,
            'momento_inercia_hough': momento_inercia,
            
            # Datos para visualización
            'hough_space': h,
            'theta': theta,
            'distances': d,
            'peaks': hough_peaks,
            'opencv_lines': lineas_opencv,
            'edges': bordes,
            'gray_image': imagen_gris
        }
        
        if visualizar:
            self._visualizar_deteccion_lineas(resultados)
        
        return resultados
    
    def detectar_circulos_hough(self, imagen, visualizar=True):
        """
        Detecta círculos usando la Transformada de Hough.
        Útil para detectar señales de tráfico, ruedas, etc.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si mostrar visualización
            
        Returns:
            dict: Resultados de la detección de círculos
        """
        print("Detectando círculos con Transformada de Hough...")
        
        # Preprocesamiento
        imagen_gris, bordes = self.preprocesar_imagen(imagen)
        
        # Método 1: Hough Círculos (scikit-image)
        hough_radii = self.config['circle_radii_range']
        hough_res = hough_circle(bordes, hough_radii)
        
        # Encontrar picos en el espacio de Hough
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                  threshold=self.config['circle_threshold'])
        
        # Método 2: Hough Círculos (OpenCV)
        circles_opencv = cv2.HoughCircles(imagen_gris,
                                        cv2.HOUGH_GRADIENT, 1, 20,
                                        param1=50, param2=30,
                                        minRadius=10, maxRadius=100)
        
        # Analizar círculos detectados
        num_circulos_skimage = len(accums) if accums is not None else 0
        num_circulos_opencv = len(circles_opencv[0]) if circles_opencv is not None else 0
        
        # Estadísticas de círculos
        radios_detectados = list(radii) if radii is not None else []
        centros_x = list(cx) if cx is not None else []
        centros_y = list(cy) if cy is not None else []
        
        # Agregar círculos de OpenCV
        if circles_opencv is not None:
            for circle in circles_opencv[0]:
                x, y, r = circle
                centros_x.append(x)
                centros_y.append(y)
                radios_detectados.append(r)
        
        # Estadísticas
        stats_radios = self._calcular_estadisticas(radios_detectados)
        stats_centros_x = self._calcular_estadisticas(centros_x)
        stats_centros_y = self._calcular_estadisticas(centros_y)
        
        # Análisis espacial de círculos
        circulo_densidad = len(radios_detectados) / (imagen_gris.shape[0] * imagen_gris.shape[1]) if radios_detectados else 0
        
        # Clasificación por tamaño (útil para distinguir tipos de objetos)
        circulos_pequenos = [r for r in radios_detectados if r < 25]  # Elementos pequeños
        circulos_medianos = [r for r in radios_detectados if 25 <= r <= 50]  # Señales medianas
        circulos_grandes = [r for r in radios_detectados if r > 50]  # Ruedas, elementos grandes
        
        # Momentos del espacio de Hough para círculos
        momento_segundo_orden_circ = np.sum(hough_res ** 2) if hough_res.size > 0 else 0
        momento_energia_circ = np.sum(hough_res ** 2) if hough_res.size > 0 else 0
        
        # Resultados
        resultados = {
            'num_circulos_skimage': num_circulos_skimage,
            'num_circulos_opencv': num_circulos_opencv,
            'num_circulos_total': num_circulos_skimage + num_circulos_opencv,
            
            # Estadísticas de radios
            'media_radios': stats_radios['media'],
            'std_radios': stats_radios['std'],
            'min_radios': stats_radios['min'],
            'max_radios': stats_radios['max'],
            
            # Estadísticas de centros
            'media_centros_x': stats_centros_x['media'],
            'std_centros_x': stats_centros_x['std'],
            'media_centros_y': stats_centros_y['media'],
            'std_centros_y': stats_centros_y['std'],
            
            # Análisis por tamaño
            'num_circulos_pequenos': len(circulos_pequenos),
            'num_circulos_medianos': len(circulos_medianos),
            'num_circulos_grandes': len(circulos_grandes),
            'densidad_circulos': circulo_densidad,
            
            # Momentos del espacio de Hough
            'momento_segundo_orden_hough_circ': momento_segundo_orden_circ,
            'momento_energia_hough_circ': momento_energia_circ,
            
            # Datos para visualización
            'hough_space_circles': hough_res,
            'circles_skimage': (accums, cx, cy, radii),
            'circles_opencv': circles_opencv,
            'edges': bordes,
            'gray_image': imagen_gris
        }
        
        if visualizar:
            self._visualizar_deteccion_circulos(resultados)
        
        return resultados
    
    def calcular_momentos_geometricos(self, imagen):
        """
        Calcula momentos geométricos de la imagen.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            
        Returns:
            dict: Momentos geométricos calculados
        """
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()
        
        # Calcular momentos usando OpenCV
        momentos = cv2.moments(imagen_gris)
        
        # Momentos geométricos básicos
        m00 = momentos['m00']
        m10 = momentos['m10']
        m01 = momentos['m01']
        m20 = momentos['m20']
        m11 = momentos['m11']
        m02 = momentos['m02']
        
        # Centroides
        if m00 != 0:
            centroide_x = m10 / m00
            centroide_y = m01 / m00
        else:
            centroide_x = 0
            centroide_y = 0
        
        # Momentos centrales
        mu20 = m20 - centroide_x * m10
        mu11 = m11 - centroide_x * m01
        mu02 = m02 - centroide_y * m01
        
        # Momentos invariantes de Hu
        hu_moments = cv2.HuMoments(momentos).flatten()
        
        return {
            'momento_m00': m00,
            'momento_m10': m10,
            'momento_m01': m01,
            'momento_m20': m20,
            'momento_m11': m11,
            'momento_m02': m02,
            'centroide_x': centroide_x,
            'centroide_y': centroide_y,
            'momento_central_mu20': mu20,
            'momento_central_mu11': mu11,
            'momento_central_mu02': mu02,
            'momento_hu1': hu_moments[0],
            'momento_hu2': hu_moments[1],
            'momento_hu3': hu_moments[2],
            'momento_hu4': hu_moments[3],
            'momento_hu5': hu_moments[4],
            'momento_hu6': hu_moments[5],
            'momento_hu7': hu_moments[6]
        }
    
    def analisis_completo_hough(self, imagen_path, nombre_imagen=None):
        """
        Realiza análisis completo con Transformada de Hough.
        
        Args:
            imagen_path (str): Ruta a la imagen
            nombre_imagen (str): Nombre personalizado
            
        Returns:
            dict: Resultados completos del análisis
        """
        try:
            # Cargar imagen
            imagen = cv2.imread(imagen_path)
            if imagen is None:
                raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
            
            if nombre_imagen is None:
                nombre_imagen = os.path.basename(imagen_path)
            
            print(f"Análisis Hough completo para: {nombre_imagen}")
            
            # Análisis de líneas
            resultados_lineas = self.detectar_lineas_hough(imagen, visualizar=False)
            
            # Análisis de círculos
            resultados_circulos = self.detectar_circulos_hough(imagen, visualizar=False)
            
            # Momentos geométricos
            momentos = self.calcular_momentos_geometricos(imagen)
            
            # Combinar resultados
            resultado_completo = {
                'Imagen': nombre_imagen,
                'Ruta': imagen_path,
                'Dimensiones': imagen.shape,
                'Fecha_Analisis': datetime.now().isoformat(),
                **momentos,
                **resultados_lineas,
                **resultados_circulos
            }
            
            # Limpiar datos de visualización para almacenamiento
            keys_to_remove = ['hough_space', 'theta', 'distances', 'peaks', 'opencv_lines',
                             'hough_space_circles', 'circles_skimage', 'circles_opencv',
                             'edges', 'gray_image']
            for key in keys_to_remove:
                resultado_completo.pop(key, None)
            
            self.current_results.append(resultado_completo)
            
            print(f"Análisis Hough completado para: {nombre_imagen}")
            return resultado_completo
            
        except Exception as e:
            print(f"Error al procesar {imagen_path}: {str(e)}")
            return None
    
    def _calcular_estadisticas(self, datos):
        """Calcula estadísticas básicas de una lista de datos."""
        if not datos:
            return {'media': 0, 'std': 0, 'min': 0, 'max': 0}
        
        datos_array = np.array(datos)
        return {
            'media': np.mean(datos_array),
            'std': np.std(datos_array),
            'min': np.min(datos_array),
            'max': np.max(datos_array)
        }
    
    def _visualizar_deteccion_lineas(self, resultados):
        """Visualiza los resultados de detección de líneas."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Imagen original
        axes[0, 0].imshow(resultados['gray_image'], cmap='gray')
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        # Bordes detectados
        axes[0, 1].imshow(resultados['edges'], cmap='gray')
        axes[0, 1].set_title(f"Bordes Canny ({resultados['bordes_detectados']} píxeles)")
        axes[0, 1].axis('off')
        
        # Espacio de Hough
        axes[1, 0].imshow(resultados['hough_space'], cmap='hot', 
                         extent=[np.rad2deg(resultados['theta'][-1]), 
                                np.rad2deg(resultados['theta'][0]),
                                resultados['distances'][-1], 
                                resultados['distances'][0]])
        axes[1, 0].set_title('Espacio de Hough (Líneas)')
        axes[1, 0].set_xlabel('Ángulo (grados)')
        axes[1, 0].set_ylabel('Distancia (píxeles)')
        
        # Líneas detectadas
        img_lines = cv2.cvtColor(resultados['gray_image'], cv2.COLOR_GRAY2RGB)
        if resultados['opencv_lines'] is not None:
            for linea in resultados['opencv_lines']:
                x1, y1, x2, y2 = linea[0]
                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Amarillo (BGR)
        
        axes[1, 1].imshow(img_lines, cmap='gray')
        axes[1, 1].set_title(f"Líneas Detectadas ({resultados['num_lineas_total']})")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        archivo_viz = os.path.join(self.results_dir, f'deteccion_lineas_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(archivo_viz, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _visualizar_deteccion_circulos(self, resultados):
        """Visualiza los resultados de detección de círculos."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Imagen original
        axes[0, 0].imshow(resultados['gray_image'], cmap='gray')
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        # Bordes detectados
        axes[0, 1].imshow(resultados['edges'], cmap='gray')
        axes[0, 1].set_title('Bordes para Detección de Círculos')
        axes[0, 1].axis('off')
        
        # Círculos detectados (scikit-image)
        img_circles_ski = cv2.cvtColor(resultados['gray_image'], cv2.COLOR_GRAY2RGB)
        accums, cx, cy, radii = resultados['circles_skimage']
        if cx is not None:
            for center_y, center_x, radius in zip(cy, cx, radii):
                circy, circx = circle_perimeter(int(center_y), int(center_x), int(radius),
                                               shape=img_circles_ski.shape[:2])
                # Aplicar color amarillo (0, 255, 255) en formato BGR
                img_circles_ski[circy, circx] = [0, 255, 255]
        
        axes[1, 0].imshow(img_circles_ski, cmap='gray')
        axes[1, 0].set_title(f"Círculos Scikit-image ({resultados['num_circulos_skimage']})")
        axes[1, 0].axis('off')
        
        # Círculos detectados (OpenCV)
        img_circles_cv = cv2.cvtColor(resultados['gray_image'], cv2.COLOR_GRAY2RGB)
        if resultados['circles_opencv'] is not None:
            circles = np.round(resultados['circles_opencv'][0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(img_circles_cv, (x, y), r, (0, 255, 255), 2)  # Amarillo (BGR)
                cv2.circle(img_circles_cv, (x, y), 2, (0, 255, 255), 3)  # Centro amarillo
        
        axes[1, 1].imshow(img_circles_cv)
        axes[1, 1].set_title(f"Círculos OpenCV ({resultados['num_circulos_opencv']})")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        archivo_viz = os.path.join(self.results_dir, f'deteccion_circulos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(archivo_viz, dpi=300, bbox_inches='tight')
        plt.show()
    
    def guardar_resultados(self, formato='csv'):
        """Guarda los resultados del análisis."""
        if not self.current_results:
            print("No hay resultados para guardar.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if formato.lower() == 'csv':
            df = pd.DataFrame(self.current_results)
            archivo_csv = os.path.join(self.results_dir, f'hough_analysis_{timestamp}.csv')
            df.to_csv(archivo_csv, index=False)
            print(f"Resultados CSV guardados: {archivo_csv}")
            
        elif formato.lower() == 'json':
            import json
            archivo_json = os.path.join(self.results_dir, f'hough_analysis_{timestamp}.json')
            with open(archivo_json, 'w', encoding='utf-8') as f:
                json.dump(self.current_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"Resultados JSON guardados: {archivo_json}")
    
    def generar_reporte_hough(self):
        """Genera un reporte del análisis de Hough."""
        if not self.current_results:
            print("No hay resultados para el reporte.")
            return
        
        print("\nREPORTE ANÁLISIS TRANSFORMADA DE HOUGH")
        print("=" * 50)
        print(f"Imágenes analizadas: {len(self.current_results)}")
        
        # Estadísticas de líneas
        total_lineas = sum(r.get('num_lineas_total', 0) for r in self.current_results)
        promedio_lineas = total_lineas / len(self.current_results)
        
        print(f"Total líneas detectadas: {total_lineas}")
        print(f"Promedio líneas por imagen: {promedio_lineas:.2f}")
        
        # Estadísticas de círculos
        total_circulos = sum(r.get('num_circulos_total', 0) for r in self.current_results)
        promedio_circulos = total_circulos / len(self.current_results)
        
        print(f"Total círculos detectados: {total_circulos}")
        print(f"Promedio círculos por imagen: {promedio_circulos:.2f}")
        
        # Top imágenes por detecciones
        imagenes_lineas = [(r['Imagen'], r.get('num_lineas_total', 0)) for r in self.current_results]
        imagenes_lineas.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTOP 3 - MAYOR DETECCIÓN DE LÍNEAS:")
        for i, (imagen, num_lineas) in enumerate(imagenes_lineas[:3], 1):
            print(f"   {i}. {imagen}: {num_lineas} líneas")
        
        imagenes_circulos = [(r['Imagen'], r.get('num_circulos_total', 0)) for r in self.current_results]
        imagenes_circulos.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTOP 3 - MAYOR DETECCIÓN DE CÍRCULOS:")
        for i, (imagen, num_circulos) in enumerate(imagenes_circulos[:3], 1):
            print(f"   {i}. {imagen}: {num_circulos} círculos")
        
        print("\n" + "=" * 50)

# Función auxiliar para perímetros de círculo
def circle_perimeter(r, c, radius, shape=None):
    """Genera coordenadas de perímetro de círculo."""
    from skimage.draw import circle_perimeter as skimage_circle_perimeter
    return skimage_circle_perimeter(r, c, radius, shape=shape)

# Función de utilidad
def analizar_hough_imagen(imagen_path, output_dir="./resultados"):
    """
    Función de conveniencia para analizar una imagen con Hough.
    
    Args:
        imagen_path (str): Ruta a la imagen
        output_dir (str): Directorio de salida
        
    Returns:
        dict: Resultados del análisis
    """
    analyzer = HoughAnalyzer(output_dir)
    resultado = analyzer.analisis_completo_hough(imagen_path)
    if resultado:
        analyzer.guardar_resultados('csv')
        analyzer.generar_reporte_hough()
    return resultado