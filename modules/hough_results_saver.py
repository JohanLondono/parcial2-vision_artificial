#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para Guardar Resultados de Análisis de Hough
=================================================

Este módulo proporciona funcionalidades para guardar los resultados del análisis
de transformada de Hough y momentos, tanto para imágenes individuales como para
procesamiento por lotes.

Funcionalidades:
- Guardado en CSV, TXT y Excel
- Visualizaciones comparativas
- Procesamiento individual y por lotes
- Estadísticas detalladas

Basado en: analisis_texturas_hough_python.py
Autor: Sistema de Detección Vehicular
Fecha: Octubre 2025
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage import img_as_ubyte
from scipy.spatial.distance import euclidean
from scipy import ndimage

class HoughResultsSaver:
    """Clase para guardar resultados de análisis de Hough."""
    
    def __init__(self, directorio_resultados):
        """
        Inicializar el guardador de resultados.
        
        Args:
            directorio_resultados (str): Directorio donde guardar los resultados
        """
        self.directorio_resultados = directorio_resultados
        self.directorio_hough = os.path.join(directorio_resultados, 'hough_analysis')
        self._crear_directorios()
    
    def _crear_directorios(self):
        """Crear directorios necesarios."""
        os.makedirs(self.directorio_hough, exist_ok=True)
        os.makedirs(os.path.join(self.directorio_hough, 'individual'), exist_ok=True)
        os.makedirs(os.path.join(self.directorio_hough, 'batch'), exist_ok=True)
        os.makedirs(os.path.join(self.directorio_hough, 'visualizations'), exist_ok=True)
    
    def convertir_a_gris(self, imagen):
        """Convierte una imagen a escala de grises si es necesario."""
        if len(imagen.shape) == 3:
            return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return imagen
    
    def preprocesar_imagen(self, imagen):
        """Realiza el preprocesamiento de la imagen para la Transformada de Hough."""
        # 1. Conversión a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen
        
        # 2. Filtro Gaussiano para reducir ruido
        imagen_filtrada = cv2.GaussianBlur(imagen_gris, (5, 5), 1.0)
        
        # 3. Detección de bordes con Canny
        bordes = cv2.Canny(imagen_filtrada, 50, 150, apertureSize=7)
        
        return imagen_gris, imagen_filtrada, bordes
    
    def calcular_momentos_imagen(self, imagen):
        """Calcula los momentos de la imagen original."""
        # Normalizar imagen para cálculos
        imagen_norm = imagen.astype(np.float64) / 255.0
        
        # Momentos geométricos hasta orden 2
        m00 = np.sum(imagen_norm)  # Momento de orden 0,0 (masa total)
        
        # Momentos de primer orden
        y_coords, x_coords = np.mgrid[:imagen.shape[0], :imagen.shape[1]]
        m10 = np.sum(x_coords * imagen_norm)
        m01 = np.sum(y_coords * imagen_norm)
        
        # Momentos de segundo orden
        m20 = np.sum(x_coords**2 * imagen_norm)
        m11 = np.sum(x_coords * y_coords * imagen_norm)
        m02 = np.sum(y_coords**2 * imagen_norm)
        
        # Centro de masa
        if m00 != 0:
            centroide_x = m10 / m00
            centroide_y = m01 / m00
        else:
            centroide_x = centroide_y = 0
        
        # Momentos centrales
        mu20 = m20 - centroide_x * m10
        mu11 = m11 - centroide_x * m01
        mu02 = m02 - centroide_y * m01
        
        # Momentos invariantes de Hu
        eta20 = mu20 / (m00 ** 1.2) if m00 != 0 else 0
        eta11 = mu11 / (m00 ** 1.2) if m00 != 0 else 0
        eta02 = mu02 / (m00 ** 1.2) if m00 != 0 else 0
        
        hu1 = eta20 + eta02
        hu2 = (eta20 - eta02)**2 + 4 * eta11**2
        
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
            'momento_hu1': hu1,
            'momento_hu2': hu2
        }
    
    def detectar_lineas_hough(self, bordes):
        """Detecta líneas usando la Transformada de Hough con OpenCV."""
        # Convertir bordes a formato OpenCV (uint8)
        if bordes.dtype != np.uint8:
            bordes_cv = img_as_ubyte(bordes)
        else:
            bordes_cv = bordes
        
        # Aplicar transformada de Hough para líneas usando OpenCV
        lines = cv2.HoughLines(bordes_cv, 1, np.pi / 180, threshold=50)
        
        # Si no se detectan líneas, intentar con umbral más bajo
        if lines is None:
            lines = cv2.HoughLines(bordes_cv, 1, np.pi / 180, threshold=30)
        
        # Si aún no se detectan líneas, intentar con umbral aún más bajo
        if lines is None:
            lines = cv2.HoughLines(bordes_cv, 1, np.pi / 180, threshold=20)
        
        # Calcular estadísticas de las líneas detectadas
        if lines is not None:
            num_lineas = len(lines)
            
            # Extraer ángulos (theta) y distancias (rho) de las líneas detectadas
            angulos = []
            distancias = []
            
            for line in lines:
                rho, theta = line[0]
                angulos.append(theta)
                distancias.append(abs(rho))
            
            angulos = np.array(angulos)
            distancias = np.array(distancias)
            
            # Convertir ángulos a grados
            angulos_grados = np.degrees(angulos)
            
            # Momentos de la distribución de ángulos
            media_angulos = np.mean(angulos_grados)
            std_angulos = np.std(angulos_grados)
            
            # Momentos de la distribución de distancias
            media_distancias = np.mean(distancias)
            std_distancias = np.std(distancias)
            
            # Para los momentos de la matriz de Hough, usar scikit-image
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
            h, theta_sk, d_sk = hough_line(bordes, theta=tested_angles)
            
            # Momento de segundo orden de la matriz de Hough
            momento_segundo_orden_h = np.sum(h**2)
            momento_energia_h = np.sum(h**2) / np.sum(h)**2 if np.sum(h) != 0 else 0
            
        else:
            num_lineas = 0
            media_angulos = std_angulos = 0
            media_distancias = std_distancias = 0
            
            # Calcular momentos de Hough aunque no se detecten líneas
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
            h, theta_sk, d_sk = hough_line(bordes, theta=tested_angles)
            momento_segundo_orden_h = np.sum(h**2)
            momento_energia_h = np.sum(h**2) / np.sum(h)**2 if np.sum(h) != 0 else 0
        
        return {
            'num_lineas_detectadas': num_lineas,
            'momento_media_angulos': media_angulos,
            'momento_std_angulos': std_angulos,
            'momento_media_distancias': media_distancias,
            'momento_std_distancias': std_distancias,
            'momento_segundo_orden_hough': momento_segundo_orden_h,
            'momento_energia_hough': momento_energia_h,
            'matriz_hough_max': np.max(h),
            'matriz_hough_sum': np.sum(h)
        }, h, lines
    
    def detectar_circulos_hough(self, bordes, imagen_gris=None):
        """Detecta círculos usando la Transformada de Hough con OpenCV."""
        # Convertir bordes a formato OpenCV si es necesario
        if bordes.dtype != np.uint8:
            bordes_cv = img_as_ubyte(bordes)
        else:
            bordes_cv = bordes
        
        # Si no tenemos la imagen en escala de grises, usar los bordes
        if imagen_gris is None:
            gray_for_circles = bordes_cv
        else:
            gray_for_circles = imagen_gris
        
        # Aplicar suavizado gaussiano
        gray_blur = cv2.GaussianBlur(gray_for_circles, (9, 9), 2)
        
        # Estimar rango de radios adaptado
        min_radio = max(6, min(gray_for_circles.shape) // 30)
        max_radio = min(100, min(gray_for_circles.shape) // 3)
        
        # Aplicar HoughCircles con parámetros optimizados
        circles = cv2.HoughCircles(
            gray_blur, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=max(20, min_radio), 
            param1=100, 
            param2=30, 
            minRadius=min_radio, 
            maxRadius=max_radio
        )
        
        # Si no se detectan círculos, intentar con parámetros más relajados
        if circles is None:
            circles = cv2.HoughCircles(
                gray_blur, 
                cv2.HOUGH_GRADIENT, 
                dp=1.5, 
                minDist=max(15, min_radio), 
                param1=50, 
                param2=25, 
                minRadius=max(3, min_radio // 2),
                maxRadius=max_radio
            )
        
        # Calcular estadísticas de los círculos detectados
        if circles is not None:
            circles = np.uint16(np.around(circles))
            num_circulos = circles.shape[1]
            
            # Extraer coordenadas y radios
            cx = circles[0, :, 0]  # coordenadas x de los centros
            cy = circles[0, :, 1]  # coordenadas y de los centros  
            radii = circles[0, :, 2]  # radios
            
            # Momentos de la distribución de radios
            media_radios = np.mean(radii)
            std_radios = np.std(radii)
            
            # Momentos de la distribución de centros
            media_centros_x = np.mean(cx)
            media_centros_y = np.mean(cy)
            std_centros_x = np.std(cx)
            std_centros_y = np.std(cy)
            
            # Para los momentos de Hough, usar scikit-image
            radii_range = np.arange(max(5, min_radio), min(150, max_radio), 2)
            hough_res = hough_circle(bordes, radii_range)
            
            # Momento de segundo orden de la matriz de Hough
            momento_segundo_orden_hc = np.sum(hough_res**2)
            momento_energia_hc = np.sum(hough_res**2) / np.sum(hough_res)**2 if np.sum(hough_res) != 0 else 0
            
        else:
            num_circulos = 0
            media_radios = std_radios = 0
            media_centros_x = media_centros_y = 0
            std_centros_x = std_centros_y = 0
            
            # Calcular momentos de Hough aunque no se detecten círculos
            radii_range = np.arange(max(5, min_radio), min(150, max_radio), 2)
            hough_res = hough_circle(bordes, radii_range)
            momento_segundo_orden_hc = np.sum(hough_res**2)
            momento_energia_hc = np.sum(hough_res**2) / np.sum(hough_res)**2 if np.sum(hough_res) != 0 else 0
        
        return {
            'num_circulos_detectados': num_circulos,
            'momento_media_radios': media_radios,
            'momento_std_radios': std_radios,
            'momento_media_centros_x': media_centros_x,
            'momento_media_centros_y': media_centros_y,
            'momento_std_centros_x': std_centros_x,
            'momento_std_centros_y': std_centros_y,
            'momento_segundo_orden_hough_circ': momento_segundo_orden_hc,
            'momento_energia_hough_circ': momento_energia_hc,
            'matriz_hough_circ_max': np.max(hough_res),
            'matriz_hough_circ_sum': np.sum(hough_res)
        }, hough_res, circles
    
    def procesar_imagen_individual(self, imagen, nombre_imagen):
        """
        Procesa una imagen individual y calcula todos los momentos de Hough.
        
        Args:
            imagen: Imagen a procesar
            nombre_imagen: Nombre de la imagen
            
        Returns:
            dict: Diccionario con todos los resultados
            dict: Datos de visualización
        """
        # Preprocesamiento
        imagen_gris, imagen_filtrada, bordes = self.preprocesar_imagen(imagen)
        
        # Calcular momentos de la imagen original
        momentos_imagen = self.calcular_momentos_imagen(imagen_gris)
        
        # Detectar líneas usando Transformada de Hough
        stats_lineas, h_lineas, lines_detected = self.detectar_lineas_hough(bordes)
        
        # Detectar círculos usando Transformada de Hough
        stats_circulos, h_circulos, circles_detected = self.detectar_circulos_hough(bordes, imagen_gris)
        
        # Combinar resultados
        resultados = {'Imagen': nombre_imagen}
        resultados.update(momentos_imagen)
        resultados.update(stats_lineas)
        resultados.update(stats_circulos)
        
        # Agregar información del preprocesamiento
        resultados['bordes_detectados'] = np.sum(bordes)
        resultados['intensidad_promedio'] = np.mean(imagen_gris)
        resultados['desviacion_intensidad'] = np.std(imagen_gris)
        
        # Datos de visualización
        datos_viz = {
            'imagen_original': imagen_gris,
            'imagen_filtrada': imagen_filtrada,
            'bordes': bordes,
            'hough_lineas': (h_lineas, lines_detected),
            'hough_circulos': (h_circulos, circles_detected)
        }
        
        return resultados, datos_viz
    
    def guardar_procesamiento_imagen_individual(self, ruta_imagen, datos_viz):
        """Guarda las etapas del procesamiento de una imagen individual."""
        nombre_imagen = os.path.basename(ruta_imagen)
        nombre_sin_ext = os.path.splitext(nombre_imagen)[0]
        
        # Crear figura con el procesamiento de la imagen
        plt.figure(figsize=(20, 12))
        
        # Imagen original
        plt.subplot(2, 4, 1)
        plt.imshow(datos_viz['imagen_original'], cmap='gray')
        plt.title(f'Original\n{nombre_imagen}')
        plt.axis('off')
        
        # Imagen filtrada
        plt.subplot(2, 4, 2)
        plt.imshow(datos_viz['imagen_filtrada'], cmap='gray')
        plt.title('Filtro Gaussiano\n(σ=1.0, kernel 5x5)')
        plt.axis('off')
        
        # Bordes detectados
        plt.subplot(2, 4, 3)
        plt.imshow(datos_viz['bordes'], cmap='gray')
        plt.title('Bordes Detectados\n(Canny)')
        plt.axis('off')
        
        # Matriz de Hough para líneas
        h_lineas, lines_detected = datos_viz['hough_lineas']
        plt.subplot(2, 4, 4)
        plt.imshow(h_lineas, cmap='hot')
        plt.title('Espacio de Hough\n(Líneas)')
        plt.xlabel('Theta (índice)')
        plt.ylabel('Rho (índice)')
        plt.colorbar()
        
        # Líneas detectadas superpuestas en la imagen original
        plt.subplot(2, 4, 5)
        img_lines = cv2.cvtColor(datos_viz['imagen_original'], cv2.COLOR_GRAY2RGB)
        if lines_detected is not None:
            for line in lines_detected:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                # Asegurar que las coordenadas estén dentro de los límites
                h, w = img_lines.shape[:2]
                x1 = max(0, min(w-1, x1))
                y1 = max(0, min(h-1, y1))
                x2 = max(0, min(w-1, x2))
                y2 = max(0, min(h-1, y2))
                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Amarillo BGR
        
        plt.imshow(img_lines)
        num_lineas = len(lines_detected) if lines_detected is not None else 0
        plt.title(f'Líneas Detectadas\n({num_lineas} líneas)')
        plt.axis('off')
        
        # Círculos detectados
        h_circulos, circles_detected = datos_viz['hough_circulos']
        plt.subplot(2, 4, 6)
        img_circles = cv2.cvtColor(datos_viz['imagen_original'], cv2.COLOR_GRAY2RGB)
        if circles_detected is not None:
            circles = np.uint16(np.around(circles_detected))
            for i in circles[0, :]:
                # Dibujar el círculo en amarillo
                cv2.circle(img_circles, (i[0], i[1]), i[2], (0, 255, 255), 2)  # Amarillo BGR
                # Dibujar el centro en amarillo
                cv2.circle(img_circles, (i[0], i[1]), 2, (0, 255, 255), 3)
        
        plt.imshow(img_circles)
        num_circulos = circles_detected.shape[1] if circles_detected is not None else 0
        plt.title(f'Círculos Detectados\n({num_circulos} círculos)')
        plt.axis('off')
        
        # Histograma de intensidades
        plt.subplot(2, 4, 7)
        plt.hist(datos_viz['imagen_original'].flatten(), bins=50, alpha=0.7, color='blue', density=True)
        plt.title('Histograma de\nIntensidades')
        plt.xlabel('Intensidad')
        plt.ylabel('Densidad')
        plt.grid(True, alpha=0.3)
        
        # Información estadística
        plt.subplot(2, 4, 8)
        plt.axis('off')
        info_text = f"""Estadísticas de Procesamiento:

Líneas detectadas: {num_lineas}
Círculos detectados: {num_circulos}
Bordes detectados: {np.sum(datos_viz['bordes'])}
Intensidad promedio: {np.mean(datos_viz['imagen_original']):.2f}
Desviación estándar: {np.std(datos_viz['imagen_original']):.2f}

Dimensiones: {datos_viz['imagen_original'].shape}
Máximo Hough (líneas): {np.max(h_lineas):.0f}
Suma Hough (líneas): {np.sum(h_lineas):.0f}
"""
        plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, fontsize=10, 
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.suptitle(f'Procesamiento de Transformada de Hough: {nombre_imagen}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Guardar la figura
        ruta_salida = os.path.join(self.directorio_hough, 'individual', f'{nombre_sin_ext}_procesamiento_hough.png')
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
        plt.close()  # Cerrar la figura para liberar memoria
        
        print(f"✅ Procesamiento de {nombre_imagen} guardado en {ruta_salida}")
        return ruta_salida
    
    def guardar_resultados_csv_txt_excel(self, resultados_list, prefijo="hough_analysis"):
        """
        Guarda los resultados en formato CSV, TXT y Excel.
        
        Args:
            resultados_list: Lista de diccionarios con resultados
            prefijo: Prefijo para los nombres de archivo
        """
        if not resultados_list:
            print("❌ No hay resultados para guardar")
            return
        
        # Crear DataFrame
        df = pd.DataFrame(resultados_list)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar CSV
        ruta_csv = os.path.join(self.directorio_hough, f'{prefijo}_{timestamp}.csv')
        os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
        df.to_csv(ruta_csv, index=False)
        print(f"Resultados CSV guardados: {ruta_csv}")
        
        # Guardar Excel
        ruta_excel = os.path.join(self.directorio_hough, f'{prefijo}_{timestamp}.xlsx')
        os.makedirs(os.path.dirname(ruta_excel), exist_ok=True)
        df.to_excel(ruta_excel, index=False)
        print(f"Resultados Excel guardados: {ruta_excel}")
        
        # Guardar TXT con formato detallado
        ruta_txt = os.path.join(self.directorio_hough, f'{prefijo}_{timestamp}.txt')
        os.makedirs(os.path.dirname(ruta_txt), exist_ok=True)
        
        txt_content = "ANÁLISIS DE TRANSFORMADA DE HOUGH - RESULTADOS\n"
        txt_content += "=" * 60 + "\n\n"
        txt_content += "PREPROCESAMIENTO APLICADO:\n"
        txt_content += "1. Conversión a escala de grises\n"
        txt_content += "2. Filtro Gaussiano (kernel 5x5, sigma=1.0)\n"
        txt_content += "3. Detección de bordes con Canny (low=50, high=150)\n"
        txt_content += "4. Transformada de Hough para líneas y círculos\n\n"
        txt_content += "JUSTIFICACIÓN DEL PREPROCESAMIENTO:\n"
        txt_content += "- Filtro Gaussiano: Reduce el ruido que puede interferir con la detección de bordes\n"
        txt_content += "- Canny: Proporciona bordes precisos y continuos necesarios para Hough\n"
        txt_content += "- Parámetros ajustados para imágenes de textura con detalles finos\n\n"
        txt_content += "=" * 60 + "\n\n"
        
        for resultado in resultados_list:
            txt_content += f"IMAGEN: {resultado['Imagen']}\n"
            txt_content += "-" * 40 + "\n"
            
            txt_content += "MOMENTOS DE LA IMAGEN ORIGINAL:\n"
            txt_content += f"  Momento m00: {resultado['momento_m00']:.6f}\n"
            txt_content += f"  Momento m10: {resultado['momento_m10']:.6f}\n"
            txt_content += f"  Momento m01: {resultado['momento_m01']:.6f}\n"
            txt_content += f"  Centroide X: {resultado['centroide_x']:.2f}\n"
            txt_content += f"  Centroide Y: {resultado['centroide_y']:.2f}\n"
            txt_content += f"  Momento Hu1: {resultado['momento_hu1']:.6f}\n"
            txt_content += f"  Momento Hu2: {resultado['momento_hu2']:.6f}\n\n"
            
            txt_content += "ANÁLISIS DE LÍNEAS (HOUGH):\n"
            txt_content += f"  Líneas detectadas: {resultado['num_lineas_detectadas']}\n"
            txt_content += f"  Media ángulos: {resultado['momento_media_angulos']:.2f}°\n"
            txt_content += f"  Std ángulos: {resultado['momento_std_angulos']:.2f}°\n"
            txt_content += f"  Media distancias: {resultado['momento_media_distancias']:.2f}\n"
            txt_content += f"  Std distancias: {resultado['momento_std_distancias']:.2f}\n"
            txt_content += f"  Momento segundo orden: {resultado['momento_segundo_orden_hough']:.0f}\n"
            txt_content += f"  Momento energía: {resultado['momento_energia_hough']:.6f}\n\n"
            
            txt_content += "ANÁLISIS DE CÍRCULOS (HOUGH):\n"
            txt_content += f"  Círculos detectados: {resultado['num_circulos_detectados']}\n"
            txt_content += f"  Media radios: {resultado['momento_media_radios']:.2f}\n"
            txt_content += f"  Std radios: {resultado['momento_std_radios']:.2f}\n"
            txt_content += f"  Media centros X: {resultado['momento_media_centros_x']:.2f}\n"
            txt_content += f"  Media centros Y: {resultado['momento_media_centros_y']:.2f}\n"
            txt_content += f"  Momento segundo orden círc: {resultado['momento_segundo_orden_hough_circ']:.0f}\n"
            txt_content += f"  Momento energía círc: {resultado['momento_energia_hough_circ']:.6f}\n\n"
            
            txt_content += "ESTADÍSTICAS ADICIONALES:\n"
            txt_content += f"  Bordes detectados: {resultado['bordes_detectados']}\n"
            txt_content += f"  Intensidad promedio: {resultado['intensidad_promedio']:.2f}\n"
            txt_content += f"  Desviación intensidad: {resultado['desviacion_intensidad']:.2f}\n"
            txt_content += "\n" + "="*60 + "\n\n"
        
        with open(ruta_txt, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        
        print(f"Resultados TXT guardados: {ruta_txt}")
        
        return {
            'csv': ruta_csv,
            'excel': ruta_excel, 
            'txt': ruta_txt
        }
    
    def crear_visualizacion_comparativa_lotes(self, df, prefijo="comparacion_hough"):
        """
        Crea la visualización comparativa para procesamiento por lotes.
        Similar a comparacion_hough_python.png del archivo original.
        
        Args:
            df: DataFrame con los resultados
            prefijo: Prefijo para el nombre del archivo
        """
        if df is None or len(df) == 0:
            print("❌ No hay datos para visualizar")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Gráfico de características de Hough para líneas y círculos
        plt.figure(figsize=(16, 12))
        
        # Subplot 1: Detección de líneas
        plt.subplot(3, 2, 1)
        plt.bar(range(len(df)), df['num_lineas_detectadas'], color='skyblue', alpha=0.7)
        plt.title('Número de Líneas Detectadas por Imagen')
        plt.xlabel('Imagen')
        plt.ylabel('Número de Líneas')
        plt.xticks(range(len(df)), [os.path.splitext(os.path.basename(img))[0] for img in df['Imagen']], rotation=45, ha='right')
        
        # Subplot 2: Detección de círculos
        plt.subplot(3, 2, 2)
        plt.bar(range(len(df)), df['num_circulos_detectados'], color='lightcoral', alpha=0.7)
        plt.title('Número de Círculos Detectados por Imagen')
        plt.xlabel('Imagen')
        plt.ylabel('Número de Círculos')
        plt.xticks(range(len(df)), [os.path.splitext(os.path.basename(img))[0] for img in df['Imagen']], rotation=45, ha='right')
        
        # Subplot 3: Momentos de ángulos de líneas
        plt.subplot(3, 2, 3)
        plt.plot(df['momento_media_angulos'], 'o-', label='Media Ángulos', color='green', markersize=6)
        plt.plot(df['momento_std_angulos'], 's-', label='Std Ángulos', color='orange', markersize=6)
        plt.title('Momentos de Distribución de Ángulos')
        plt.xlabel('Imagen')
        plt.ylabel('Ángulo (grados)')
        plt.legend()
        plt.xticks(range(len(df)), [os.path.splitext(os.path.basename(img))[0] for img in df['Imagen']], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Momentos de radios de círculos
        plt.subplot(3, 2, 4)
        plt.plot(df['momento_media_radios'], 'o-', label='Media Radios', color='purple', markersize=6)
        plt.plot(df['momento_std_radios'], 's-', label='Std Radios', color='brown', markersize=6)
        plt.title('Momentos de Distribución de Radios')
        plt.xlabel('Imagen')
        plt.ylabel('Radio (píxeles)')
        plt.legend()
        plt.xticks(range(len(df)), [os.path.splitext(os.path.basename(img))[0] for img in df['Imagen']], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Subplot 5: Energía de matrices Hough
        plt.subplot(3, 2, 5)
        plt.semilogy(df['momento_energia_hough'], 'o-', label='Energía Hough Líneas', color='red', markersize=6)
        plt.semilogy(df['momento_energia_hough_circ'], 's-', label='Energía Hough Círculos', color='blue', markersize=6)
        plt.title('Energía de Matrices de Hough')
        plt.xlabel('Imagen')
        plt.ylabel('Energía (log scale)')
        plt.legend()
        plt.xticks(range(len(df)), [os.path.splitext(os.path.basename(img))[0] for img in df['Imagen']], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Subplot 6: Momentos geométricos de la imagen
        plt.subplot(3, 2, 6)
        plt.plot(df['momento_hu1'], 'o-', label='Momento Hu1', color='cyan', markersize=6)
        plt.plot(df['momento_hu2'], 's-', label='Momento Hu2', color='magenta', markersize=6)
        plt.title('Momentos Invariantes de Hu')
        plt.xlabel('Imagen')
        plt.ylabel('Valor del Momento')
        plt.legend()
        plt.xticks(range(len(df)), [os.path.splitext(os.path.basename(img))[0] for img in df['Imagen']], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        ruta_grafico = os.path.join(self.directorio_hough, 'visualizations', f'{prefijo}_{timestamp}.png')
        os.makedirs(os.path.dirname(ruta_grafico), exist_ok=True)
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico comparativo guardado: {ruta_grafico}")
        
        # 2. Matriz de correlación entre características
        plt.figure(figsize=(14, 12))
        
        # Seleccionar características numéricas para la correlación
        caracteristicas_numericas = df.select_dtypes(include=[np.number]).columns
        # Excluir la columna 'Imagen' si existe
        caracteristicas_numericas = [col for col in caracteristicas_numericas if col != 'Imagen']
        
        if len(caracteristicas_numericas) > 1:
            corr = df[caracteristicas_numericas].corr()
            
            plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(label='Correlación')
            plt.xticks(range(len(caracteristicas_numericas)), caracteristicas_numericas, rotation=90)
            plt.yticks(range(len(caracteristicas_numericas)), caracteristicas_numericas)
            plt.title('Matriz de Correlación entre Características de Hough y Momentos')
            
            # Añadir valores de correlación en las celdas
            for i in range(len(caracteristicas_numericas)):
                for j in range(len(caracteristicas_numericas)):
                    text = plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.5 else "white",
                                  fontsize=8)
            
            plt.tight_layout()
            ruta_matriz = os.path.join(self.directorio_hough, 'visualizations', f'matriz_correlacion_{prefijo}_{timestamp}.png')
            os.makedirs(os.path.dirname(ruta_matriz), exist_ok=True)
            plt.savefig(ruta_matriz, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Matriz de correlación guardada: {ruta_matriz}")
        
        return ruta_grafico
    
    def procesar_carpeta_imagenes(self, carpeta_imagenes, patron="*.png,*.jpg,*.jpeg,*.tif,*.tiff,*.bmp", 
                                guardar_procesamiento_individual=False):
        """
        Procesa todas las imágenes en una carpeta y genera todos los formatos de salida.
        
        Args:
            carpeta_imagenes: Ruta a la carpeta con imágenes
            patron: Patrones de archivos a procesar
            guardar_procesamiento_individual: Si guardar el procesamiento de cada imagen
            
        Returns:
            dict: Rutas de archivos generados
        """
        # Expandir los patrones separados por comas
        patrones = patron.split(',')
        archivos = []
        
        # Obtener todos los archivos que coinciden con los patrones
        for p in patrones:
            patron_ruta = os.path.join(carpeta_imagenes, p.strip())
            archivos.extend(glob.glob(patron_ruta))
        
        if not archivos:
            print(f"❌ No se encontraron imágenes en {carpeta_imagenes} con el patrón {patron}")
            return None
        
        print(f"Encontradas {len(archivos)} imágenes para procesar")
        
        # Procesar cada imagen
        resultados = []
        datos_visualizacion = []
        archivos_procesamiento = []
        
        for archivo in archivos:
            print(f"Procesando {os.path.basename(archivo)}...")
            
            # Cargar imagen
            imagen = cv2.imread(archivo)
            if imagen is None:
                print(f"Error al cargar {archivo}")
                continue
            
            # Procesar imagen
            resultado, datos_viz = self.procesar_imagen_individual(imagen, os.path.basename(archivo))
            resultados.append(resultado)
            datos_visualizacion.append((archivo, datos_viz))
            
            # Guardar el procesamiento individual si se solicita
            if guardar_procesamiento_individual:
                ruta_proc = self.guardar_procesamiento_imagen_individual(archivo, datos_viz)
                archivos_procesamiento.append(ruta_proc)
        
        if not resultados:
            print("No se pudieron procesar imágenes")
            return None
        
        # Crear DataFrame con los resultados
        df = pd.DataFrame(resultados)
        
        # Guardar resultados en CSV, TXT, Excel
        rutas_archivos = self.guardar_resultados_csv_txt_excel(resultados, "hough_batch_results")
        
        # Crear visualización comparativa
        ruta_comparacion = self.crear_visualizacion_comparativa_lotes(df, "comparacion_hough")
        
        # Mostrar resumen estadístico
        print("\nRESUMEN ESTADÍSTICO DE LOS RESULTADOS:")
        print("=" * 60)
        print(f"Número de imágenes procesadas: {len(resultados)}")
        print(f"Promedio de líneas detectadas: {df['num_lineas_detectadas'].mean():.2f}")
        print(f"Promedio de círculos detectados: {df['num_circulos_detectados'].mean():.2f}")
        print(f"Promedio de bordes detectados: {df['bordes_detectados'].mean():.0f}")
        
        # Preparar diccionario de retorno
        resultado_final = {
            'csv': rutas_archivos['csv'],
            'excel': rutas_archivos['excel'],
            'txt': rutas_archivos['txt'],
            'comparacion': ruta_comparacion,
            'dataframe': df,
            'num_imagenes': len(resultados)
        }
        
        if guardar_procesamiento_individual:
            resultado_final['procesamiento_individual'] = archivos_procesamiento
        
        print(f"\nProcesamiento por lotes completado exitosamente")
        print(f"Resultados guardados en: {self.directorio_hough}")
        
        return resultado_final