#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handlers para Menús de Análisis de Características
=================================================

Implementaciones de las funciones de análisis para el sistema principal.
Estas funciones manejan la extracción de características, análisis de texturas,
detección de bordes, formas y métodos avanzados.

Autor: Estudiante
Fecha: Octubre 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class AnalysisHandlers:
    """Clase que maneja las implementaciones de análisis."""
    
    def __init__(self, sistema_principal):
        """
        Inicializa el handler con referencia al sistema principal.
        
        Args:
            sistema_principal: Instancia del SistemaDeteccionVehicular
        """
        self.sistema = sistema_principal

    # =========================================================================
    # IMPLEMENTACIONES DE ANÁLISIS DE CARACTERÍSTICAS
    # =========================================================================
    
    def estadisticas_primer_orden(self):
        """Ejecuta análisis de estadísticas de primer orden."""
        try:
            print("Analizando estadísticas de primer orden...")
            
            # Usar el analizador de texturas
            resultados = self.sistema.texture_analyzer.estadisticas_primer_orden(
                self.sistema.imagen_actual
            )
            
            # Mostrar resultados
            print("\nESTADÍSTICAS DE PRIMER ORDEN")
            print("-" * 40)
            for clave, valor in resultados.items():
                print(f"{clave}: {valor:.4f}")
            
            # Opción de guardar
            self._guardar_resultados_analisis("estadisticas_primer_orden", resultados)
            
        except Exception as e:
            print(f"Error en análisis de primer orden: {e}")

    def estadisticas_segundo_orden(self):
        """Ejecuta análisis de estadísticas de segundo orden (GLCM)."""
        try:
            print("Analizando estadísticas de segundo orden (GLCM)...")
            
            # Usar el analizador de texturas
            resultados = self.sistema.texture_analyzer.estadisticas_segundo_orden(
                self.sistema.imagen_actual
            )
            
            # Mostrar resultados
            print("\nESTADÍSTICAS DE SEGUNDO ORDEN (GLCM)")
            print("-" * 50)
            for clave, valor in resultados.items():
                print(f"{clave}: {valor:.4f}")
            
            # Opción de guardar
            self._guardar_resultados_analisis("estadisticas_segundo_orden", resultados)
            
        except Exception as e:
            print(f"Error en análisis de segundo orden: {e}")

    def analisis_texturas_completo(self):
        """Ejecuta análisis completo de texturas con opción de procesamiento por lotes."""
        try:
            print("ANÁLISIS COMPLETO DE TEXTURAS")
            print("=" * 50)
            print("1. Imagen actual")
            print("2. Procesamiento por lotes")
            print("0. Cancelar")
            
            opcion = input("\nSeleccione opción: ").strip()
            
            if opcion == '1':
                self._analizar_imagen_individual()
            elif opcion == '2':
                self._analizar_lote_texturas()
            elif opcion == '0':
                print("Operación cancelada")
            else:
                print("Opción no válida")
                
        except Exception as e:
            print(f"Error en análisis completo de texturas: {e}")
    
    def _analizar_imagen_individual(self):
        """Analiza texturas en imagen individual."""
        try:
            print("Analizando imagen actual...")
            
            # Ejecutar análisis completo
            resultados = self.sistema.texture_analyzer.procesar_imagen_completa(
                self.sistema.ruta_imagen_actual
            )
            
            if resultados:
                print("Análisis de texturas completado")
                print(f"Estadísticas disponibles en: {self.sistema.texture_analyzer.results_dir}")
                
                # Mostrar visualización si está disponible
                respuesta = input("¿Mostrar visualización? (s/n): ").strip().lower()
                if respuesta.startswith('s'):
                    self.sistema.texture_analyzer.visualizar_resultados()
            else:
                print("Error en análisis de texturas")
                
        except Exception as e:
            print(f"Error analizando imagen individual: {e}")
    
    def _analizar_lote_texturas(self):
        """Analiza texturas en lote de múltiples imágenes."""
        try:
            print("PROCESAMIENTO POR LOTES - ANÁLISIS DE TEXTURAS")
            print("=" * 60)
            
            # Carpeta de imágenes
            carpeta_predeterminada = self.sistema.directorio_imagenes
            print(f"Carpeta actual: {carpeta_predeterminada}")
            carpeta = input("Nueva carpeta (Enter para actual): ").strip()
            if not carpeta:
                carpeta = carpeta_predeterminada
            
            if not os.path.exists(carpeta):
                print(f"La carpeta {carpeta} no existe")
                return
            
            # Patrones de archivos
            print("\nPatrones de archivos:")
            print("1. *.jpg,*.jpeg,*.png")
            print("2. *.jpg,*.jpeg,*.png,*.tif,*.tiff")
            print("3. Personalizado")
            
            patron_opcion = input("\nSeleccione patrón (1): ").strip() or "1"
            
            if patron_opcion == '1':
                patron = "*.jpg,*.jpeg,*.png"
            elif patron_opcion == '2':
                patron = "*.jpg,*.jpeg,*.png,*.tif,*.tiff"
            elif patron_opcion == '3':
                patron = input("Ingrese patrón personalizado: ").strip()
            else:
                patron = "*.jpg,*.jpeg,*.png"
            
            print(f"\nProcesando imágenes en: {carpeta}")
            print(f"Patrón: {patron}")
            print("-" * 60)
            
            # Procesar carpeta
            resultados = self.sistema.texture_analyzer.procesar_carpeta(carpeta, patron)
            
            if resultados is not None and len(resultados) > 0:
                print(f"\nProcesamiento completado: {len(resultados)} imágenes")
                print(f"Resultados guardados en: {self.sistema.texture_analyzer.results_dir}")
                
                # Opción de visualización
                respuesta = input("\n¿Mostrar visualizaciones? (s/n): ").strip().lower()
                if respuesta.startswith('s'):
                    self.sistema.texture_analyzer.visualizar_resultados_batch(resultados, carpeta)
                
                # Mostrar resumen de estadísticas
                print("\nRESUMEN ESTADÍSTICO:")
                print("=" * 50)
                print(f"Total de imágenes procesadas: {len(resultados)}")
                
                # Mostrar estadísticas promedio
                caracteristicas_numericas = ['Media', 'Varianza', 'Desviación_Estándar', 'Entropía',
                                           'Contraste', 'Homogeneidad', 'Energía', 'Correlación']
                
                print("\nESTADÍSTICAS PROMEDIO:")
                for caract in caracteristicas_numericas:
                    if caract in resultados.columns:
                        promedio = resultados[caract].mean()
                        print(f"  {caract}: {promedio:.4f}")
                
            else:
                print("No se pudieron procesar las imágenes")
                
        except Exception as e:
            print(f"Error en procesamiento por lotes: {e}")

    def comparar_regiones_textura(self):
        """Compara texturas en diferentes regiones de la imagen."""
        try:
            print("Analizando regiones de textura...")
            
            # Analizar regiones vehiculares
            resultados = self.sistema.texture_analyzer.analizar_regiones_vehiculares(
                self.sistema.imagen_actual, mostrar_regiones=True
            )
            
            if resultados:
                print("Análisis de regiones completado")
            else:
                print("Error en análisis de regiones")
                
        except Exception as e:
            print(f"Error comparando regiones: {e}")

    def detectar_bordes_canny(self):
        """Detecta bordes usando Canny."""
        try:
            print("Detectando bordes con Canny...")
            
            # Parámetros del usuario
            low_thresh = int(input("Umbral bajo (ej: 50): ") or "50")
            high_thresh = int(input("Umbral alto (ej: 150): ") or "150")
            
            # Convertir a escala de grises si es necesario
            imagen_gris = cv2.cvtColor(self.sistema.imagen_actual, cv2.COLOR_BGR2GRAY) \
                if len(self.sistema.imagen_actual.shape) == 3 else self.sistema.imagen_actual
            
            # Aplicar Canny
            bordes = cv2.Canny(imagen_gris, low_thresh, high_thresh)
            
            # Mostrar resultado
            self._mostrar_resultado_bordes("Canny", bordes)
            
        except Exception as e:
            print(f"Error detectando bordes Canny: {e}")

    def detectar_bordes_sobel(self):
        """Detecta bordes usando Sobel."""
        try:
            print("Detectando bordes con Sobel...")
            
            # Convertir a escala de grises
            imagen_gris = cv2.cvtColor(self.sistema.imagen_actual, cv2.COLOR_BGR2GRAY) \
                if len(self.sistema.imagen_actual.shape) == 3 else self.sistema.imagen_actual
            
            # Aplicar Sobel
            sobel_x = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalizar
            sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
            
            # Mostrar resultado
            self._mostrar_resultado_bordes("Sobel", sobel_combined)
            
        except Exception as e:
            print(f"Error detectando bordes Sobel: {e}")

    def detectar_bordes_log(self):
        """Detecta bordes usando Laplaciano de Gauss."""
        try:
            print("🔍 Detectando bordes con Laplaciano de Gauss (LoG)...")
            
            # Obtener parámetros del usuario
            mostrar_descriptores = input("¿Mostrar análisis detallado en consola? (s/n): ").lower() == 's'
            guardar_resultados = input("¿Guardar estadísticas en archivos CSV y TXT? (s/n): ").lower() == 's'
            guardar_imagen = input("¿Guardar imagen de visualización? (s/n): ").lower() == 's'
            
            if guardar_resultados or guardar_imagen:
                nombre_imagen = input("Nombre para los archivos (sin extensión): ").strip()
                if not nombre_imagen:
                    nombre_imagen = "log_bordes_analysis"
            else:
                nombre_imagen = "log_bordes_analysis"
            
            # Configurar guardado de imagen si se solicita
            if guardar_imagen:
                self.sistema.advanced_analyzer._save_visualization = True
            
            # Usar el analizador avanzado con parámetros completos
            resultados = self.sistema.advanced_analyzer.analizar_log_detector(
                self.sistema.imagen_actual,
                visualizar=True,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen
            )
            
            # Limpiar flag de guardado de imagen
            if hasattr(self.sistema.advanced_analyzer, '_save_visualization'):
                delattr(self.sistema.advanced_analyzer, '_save_visualization')
            
            if resultados:
                print("Detección LoG completada")
                print(f"Estadísticas principales:")
                print(f"  • Blobs detectados: {resultados.get('log_num_blobs', 0)}")
                print(f"  • Densidad de blobs: {resultados.get('log_blob_density', 0):.8f}")
                print(f"  • Consistencia escalas: {resultados.get('log_scale_consistency', 0):.4f}")
                
                if resultados.get('log_num_blobs', 0) > 0:
                    print(f"  • Posición promedio: ({resultados.get('log_mean_x', 0):.1f}, {resultados.get('log_mean_y', 0):.1f})")
                    print(f"  • Sigma promedio: {resultados.get('log_mean_sigma', 0):.2f}")
                    print(f"  • Respuesta promedio: {resultados.get('log_mean_response', 0):.6f}")
                
                if guardar_resultados:
                    print(f"Estadísticas guardadas con nombre: {nombre_imagen}")
                if guardar_imagen:
                    print(f"Imagen de visualización guardada en: {self.sistema.advanced_analyzer.results_dir}")
            else:
                print("Error en detección LoG")
                
        except Exception as e:
            print(f"Error detectando bordes LoG: {e}")

    def analizar_gradientes(self):
        """Analiza gradientes de la imagen."""
        try:
            print("Analizando gradientes...")
            
            # Convertir a escala de grises
            imagen_gris = cv2.cvtColor(self.sistema.imagen_actual, cv2.COLOR_BGR2GRAY) \
                if len(self.sistema.imagen_actual.shape) == 3 else self.sistema.imagen_actual
            
            # Calcular gradientes
            grad_x = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=3)
            
            # Magnitud y dirección
            magnitud = np.sqrt(grad_x**2 + grad_y**2)
            direccion = np.arctan2(grad_y, grad_x)
            
            # Estadísticas
            print(f"Magnitud promedio: {np.mean(magnitud):.2f}")
            print(f"Magnitud máxima: {np.max(magnitud):.2f}")
            print(f"Direcciones dominantes: {np.std(direccion):.2f}")
            
            # Visualizar
            self._mostrar_analisis_gradientes(magnitud, direccion)
            
        except Exception as e:
            print(f"Error analizando gradientes: {e}")

    def comparar_metodos_bordes(self):
        """Compara diferentes métodos de detección de bordes."""
        try:
            print("Comparando métodos de detección de bordes...")
            
            imagen_gris = cv2.cvtColor(self.sistema.imagen_actual, cv2.COLOR_BGR2GRAY) \
                if len(self.sistema.imagen_actual.shape) == 3 else self.sistema.imagen_actual
            
            # Aplicar diferentes métodos
            canny = cv2.Canny(imagen_gris, 50, 150)
            sobel_x = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            laplacian = cv2.Laplacian(imagen_gris, cv2.CV_64F)
            
            # Normalizar
            sobel = np.uint8(sobel / sobel.max() * 255)
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Mostrar comparación
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes[0, 0].imshow(imagen_gris, cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(canny, cmap='gray')
            axes[0, 1].set_title('Canny')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(sobel, cmap='gray')
            axes[0, 2].set_title('Sobel')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(laplacian, cmap='gray')
            axes[1, 0].set_title('Laplaciano')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(np.abs(sobel_x), cmap='gray')
            axes[1, 1].set_title('Sobel X')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(np.abs(sobel_y), cmap='gray')
            axes[1, 2].set_title('Sobel Y')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print("Comparación de métodos completada")
            
        except Exception as e:
            print(f"Error comparando métodos: {e}")

    def detectar_lineas_hough(self):
        """Detecta líneas usando transformada de Hough."""
        try:
            print("Detectando líneas con Hough...")
            
            resultados = self.sistema.hough_analyzer.detectar_lineas_hough(
                self.sistema.imagen_actual, visualizar=True
            )
            
            if resultados:
                print("Detección de líneas completada")
                print(f"Líneas detectadas: {resultados.get('num_lineas_opencv', 0)}")
            else:
                print("Error en detección de líneas")
                
        except Exception as e:
            print(f"Error detectando líneas: {e}")

    def detectar_circulos_hough(self):
        """Detecta círculos usando transformada de Hough."""
        try:
            print("Detectando círculos con Hough...")
            
            resultados = self.sistema.hough_analyzer.detectar_circulos_hough(
                self.sistema.imagen_actual, visualizar=True
            )
            
            if resultados:
                print("Detección de círculos completada")
                print(f"Círculos detectados: {resultados.get('num_circulos', 0)}")
            else:
                print("Error en detección de círculos")
                
        except Exception as e:
            print(f"Error detectando círculos: {e}")

    def calcular_momentos_geometricos(self):
        """Calcula momentos geométricos."""
        try:
            print("Calculando momentos geométricos...")
            
            resultados = self.sistema.hough_analyzer.calcular_momentos_geometricos(
                self.sistema.imagen_actual
            )
            
            if resultados:
                print("Cálculo de momentos completado")
                for clave, valor in resultados.items():
                    if isinstance(valor, (int, float)):
                        print(f"{clave}: {valor:.4f}")
            else:
                print("Error calculando momentos")
                
        except Exception as e:
            print(f"Error calculando momentos: {e}")

    def analisis_formas_completo(self):
        """Ejecuta análisis completo de formas."""
        try:
            print("Ejecutando análisis completo de formas...")
            
            resultados = self.sistema.hough_analyzer.analisis_completo_hough(
                self.sistema.ruta_imagen_actual
            )
            
            if resultados:
                print("Análisis de formas completado")
                print(f"Resultados guardados en: {self.sistema.hough_analyzer.results_dir}")
            else:
                print("Error en análisis de formas")
                
        except Exception as e:
            print(f"Error en análisis de formas: {e}")

    # =========================================================================
    # MÉTODOS AVANZADOS DE CARACTERÍSTICAS
    # =========================================================================
    
    def extraer_surf(self):
        """Extrae características SURF."""
        try:
            print("\nANÁLISIS SURF (Speeded Up Robust Features)")
            print("="*60)
            print("Opciones de análisis:")
            print("1. ¿Mostrar descriptores detallados en consola? (s/N)")
            print("2. ¿Guardar resultados en archivos CSV y TXT? (s/N)")
            print()
            
            # Opciones del usuario
            mostrar_desc = input("¿Mostrar descriptores detallados? (s/N): ").strip().lower()
            mostrar_descriptores = mostrar_desc in ['s', 'sí', 'si', 'y', 'yes']
            
            guardar_res = input("¿Guardar resultados en archivos? (s/N): ").strip().lower()
            guardar_resultados = guardar_res in ['s', 'sí', 'si', 'y', 'yes']
            
            # Nombre de imagen
            nombre_imagen = input("Nombre para archivos (Enter para auto): ").strip()
            if not nombre_imagen:
                import os
                nombre_imagen = os.path.splitext(os.path.basename(self.sistema.ruta_imagen_actual))[0] if self.sistema.ruta_imagen_actual else "imagen_surf"
            
            print(f"\nExtrayendo características SURF...")
            print(f"Descriptores en consola: {'Sí' if mostrar_descriptores else 'No'}")
            print(f"Guardar archivos: {'Sí' if guardar_resultados else 'No'}")
            print(f"Nombre: {nombre_imagen}")
            
            resultados = self.sistema.surf_orb_analyzer.extraer_caracteristicas_surf(
                self.sistema.imagen_actual, 
                visualizar=True,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen
            )
            
            if resultados:
                print(f"\nExtracción SURF completada")
                print(f"Keypoints detectados: {len(resultados.get('keypoints', []))}")
                if resultados.get('descriptors') is not None:
                    print(f"Dimensión descriptores: {resultados['descriptors'].shape[1]} valores")
                if guardar_resultados:
                    print(f"Archivos guardados con nombre base: {nombre_imagen}")
            else:
                print("Error en extracción SURF")
                
        except Exception as e:
            print(f"Error extrayendo SURF: {e}")

    def extraer_orb(self):
        """Extrae características ORB."""
        try:
            print("\nANÁLISIS ORB (Oriented FAST and Rotated BRIEF)")
            print("="*60)
            print("Opciones de análisis:")
            print("1. ¿Mostrar descriptores detallados en consola? (s/N)")
            print("2. ¿Guardar resultados en archivos CSV y TXT? (s/N)")
            print("3. ¿Método de visualización?")
            print("   a) Método de la predeterminado (escala de grises, puntos detallados)")
            print("   b) Método personalizado (color, puntos pequeños)")
            print()
            
            # Opciones del usuario
            mostrar_desc = input("¿Mostrar descriptores detallados? (s/N): ").strip().lower()
            mostrar_descriptores = mostrar_desc in ['s', 'sí', 'si', 'y', 'yes']
            
            guardar_res = input("¿Guardar resultados en archivos? (s/N): ").strip().lower()
            guardar_resultados = guardar_res in ['s', 'sí', 'si', 'y', 'yes']
            
            # Método de visualización
            metodo = input("¿Método de visualización? (a=predeterminado, b=personalizado): ").strip().lower()
            usar_metodo_profesora = metodo in ['a', 'predeterminado', 'prof']
            
            # Nombre de imagen
            nombre_imagen = input("Nombre para archivos (Enter para auto): ").strip()
            if not nombre_imagen:
                import os
                nombre_imagen = os.path.splitext(os.path.basename(self.sistema.ruta_imagen_actual))[0] if self.sistema.ruta_imagen_actual else "imagen_orb"
            
            metodo_texto = "Predeterminado (config. por defecto)" if usar_metodo_profesora else "Personalizado (config. avanzada)"
            print(f"\nExtrayendo características ORB...")
            print(f"Descriptores en consola: {'Sí' if mostrar_descriptores else 'No'}")
            print(f"Guardar archivos: {'Sí' if guardar_resultados else 'No'}")
            print(f"Método: {metodo_texto}")
            print(f"Nombre: {nombre_imagen}")
            
            resultados = self.sistema.surf_orb_analyzer.extraer_caracteristicas_orb(
                self.sistema.imagen_actual, 
                visualizar=True,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen,
                usar_metodo_profesora=usar_metodo_profesora
            )
            
            if resultados:
                print(f"\nExtracción ORB completada")
                print(f"Keypoints detectados: {len(resultados.get('keypoints', []))}")
                if resultados.get('descriptors') is not None:
                    print(f"Dimensión descriptores: {resultados['descriptors'].shape[1]} bytes ({resultados['descriptors'].shape[1] * 8} bits)")
                if guardar_resultados:
                    print(f"Archivos guardados con nombre base: {nombre_imagen}")
            else:
                print("Error en extracción ORB")
                
        except Exception as e:
            print(f"Error extrayendo ORB: {e}")

    def extraer_hog(self):
        """Extrae características HOG."""
        try:
            if not self.sistema.verificar_imagen_cargada():
                return
                
            print("\nExtracción de Características HOG")
            print("="*50)
            
            # Obtener nombre de imagen
            nombre_imagen = self.sistema.ruta_imagen_actual
            if nombre_imagen:
                nombre_imagen = os.path.splitext(os.path.basename(nombre_imagen))[0]
            else:
                nombre_imagen = "imagen_actual"
            
            # Preguntar opciones al usuario
            print("Opciones de análisis HOG:")
            print("1. Solo mostrar resultados en pantalla")
            print("2. Mostrar resultados + guardar archivos (CSV y TXT)")
            print("3. Solo guardar archivos (sin mostrar en pantalla)")
            
            opcion = input("\nSeleccione una opción (1-3): ").strip()
            
            mostrar_descriptores = opcion in ['1', '2']
            guardar_resultados = opcion in ['2', '3']
            visualizar = opcion in ['1', '2']
            
            resultados = self.sistema.hog_kaze_analyzer.extraer_caracteristicas_hog(
                self.sistema.imagen_actual, 
                visualizar=visualizar,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen
            )
            
            if resultados:
                print(f"\nExtracción HOG completada para: {nombre_imagen}")
                print(f"Características extraídas: {resultados.get('num_features', 0)}")
                if guardar_resultados:
                    print(f"Archivos guardados en: {self.sistema.hog_kaze_analyzer.results_dir}")
            else:
                print("Error en extracción HOG")
                
        except Exception as e:
            print(f"Error extrayendo HOG: {e}")

    def extraer_kaze(self):
        """Extrae características KAZE."""
        try:
            if not self.sistema.verificar_imagen_cargada():
                return
                
            print("\nExtracción de Características KAZE")
            print("="*50)
            
            # Obtener nombre de imagen
            nombre_imagen = self.sistema.ruta_imagen_actual
            if nombre_imagen:
                nombre_imagen = os.path.splitext(os.path.basename(nombre_imagen))[0]
            else:
                nombre_imagen = "imagen_actual"
            
            # Preguntar opciones al usuario
            print("Opciones de análisis KAZE:")
            print("1. Solo mostrar resultados en pantalla")
            print("2. Mostrar resultados + guardar archivos (CSV y TXT)")
            print("3. Solo guardar archivos (sin mostrar en pantalla)")
            
            opcion = input("\nSeleccione una opción (1-3): ").strip()
            
            print("\nConfiguración KAZE:")
            print("1. Configuración por defecto")
            print("2. Configuración avanzada (más sensible)")
            
            config_opcion = input("\nSeleccione configuración (1-2): ").strip()
            usar_config_default = config_opcion == '1'
            
            mostrar_descriptores = opcion in ['1', '2']
            guardar_resultados = opcion in ['2', '3']
            visualizar = opcion in ['1', '2']
            
            resultados = self.sistema.hog_kaze_analyzer.extraer_caracteristicas_kaze(
                self.sistema.imagen_actual, 
                visualizar=visualizar,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen,
                usar_config_default=usar_config_default
            )
            
            if resultados:
                print(f"\nExtracción KAZE completada para: {nombre_imagen}")
                print(f"Puntos clave detectados: {len(resultados.get('keypoints', []))}")
                if resultados.get('descriptors') is not None:
                    print(f"Descriptores generados: {resultados['descriptors'].shape[0]}")
                if guardar_resultados:
                    print(f"Archivos guardados en: {self.sistema.hog_kaze_analyzer.results_dir}")
            else:
                print("Error en extracción KAZE")
                
        except Exception as e:
            print(f"Error extrayendo KAZE: {e}")
    
    def analisis_comparativo_hog_kaze(self):
        """Realiza análisis comparativo HOG + KAZE."""
        try:
            if not self.sistema.verificar_imagen_cargada():
                return
                
            print("\nAnálisis Comparativo HOG + KAZE")
            print("="*60)
            
            # Obtener nombre de imagen
            nombre_imagen = self.sistema.ruta_imagen_actual
            if nombre_imagen:
                nombre_base = os.path.splitext(os.path.basename(nombre_imagen))[0]
            else:
                nombre_base = "imagen_actual"
            
            print("Realizando análisis HOG...")
            resultados_hog = self.sistema.hog_kaze_analyzer.extraer_caracteristicas_hog(
                self.sistema.imagen_actual, 
                visualizar=False,
                mostrar_descriptores=True,
                guardar_resultados=True,
                nombre_imagen=f"{nombre_base}_hog"
            )
            
            print("\nRealizando análisis KAZE...")
            resultados_kaze = self.sistema.hog_kaze_analyzer.extraer_caracteristicas_kaze(
                self.sistema.imagen_actual, 
                visualizar=False,
                mostrar_descriptores=True,
                guardar_resultados=True,
                nombre_imagen=f"{nombre_base}_kaze"
            )
            
            # Mostrar comparación
            print(f"\nCOMPARACIÓN HOG vs KAZE - {nombre_base.upper()}")
            print("="*60)
            print(f"HOG:")
            print(f"   • Características extraídas: {resultados_hog.get('num_features', 0)}")
            print(f"   • Energía total: {resultados_hog.get('hog_energy', 0):.6f}")
            print(f"   • Entropía: {resultados_hog.get('hog_entropy', 0):.6f}")
            
            print(f"\nKAZE:")
            print(f"   • Puntos clave detectados: {len(resultados_kaze.get('keypoints', []))}")
            print(f"   • Densidad de puntos: {resultados_kaze.get('kp_density', 0):.8f}")
            if resultados_kaze.get('descriptors') is not None:
                print(f"   • Dimensión descriptores: {resultados_kaze['descriptors'].shape[1]}")
                print(f"   • Entropía descriptores: {resultados_kaze.get('descriptor_entropy', 0):.6f}")
            
            # Mostrar visualizaciones
            print(f"\nGenerando visualizaciones comparativas...")
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Imagen original
            if len(self.sistema.imagen_actual.shape) == 3:
                axes[0,0].imshow(cv2.cvtColor(self.sistema.imagen_actual, cv2.COLOR_BGR2RGB))
            else:
                axes[0,0].imshow(self.sistema.imagen_actual, cmap='gray')
            axes[0,0].set_title('Imagen Original')
            axes[0,0].axis('off')
            
            # HOG visualization
            if 'hog_image' in resultados_hog:
                from skimage import exposure
                hog_image_rescaled = exposure.rescale_intensity(resultados_hog['hog_image'], in_range=(0, 10))
                axes[0,1].imshow(hog_image_rescaled, cmap='hot')
                axes[0,1].set_title(f'HOG Features ({resultados_hog.get("num_features", 0)})')
                axes[0,1].axis('off')
            
            # KAZE keypoints
            if resultados_kaze.get('keypoints'):
                img_kp = resultados_kaze['gray_image'].copy()
                img_kp = cv2.drawKeypoints(img_kp, resultados_kaze['keypoints'], None, 
                                         color=(0, 255, 255), 
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                axes[1,0].imshow(img_kp)
                axes[1,0].set_title(f'KAZE Keypoints ({len(resultados_kaze["keypoints"])})')
                axes[1,0].axis('off')
            
            # Histograma comparativo de energías
            if 'hog_features_raw' in resultados_hog:
                axes[1,1].hist(resultados_hog['hog_features_raw'].flatten(), bins=50, 
                             alpha=0.7, color='red', label='HOG Features')
            if resultados_kaze.get('descriptors') is not None:
                axes[1,1].hist(resultados_kaze['descriptors'].flatten(), bins=50, 
                             alpha=0.7, color='blue', label='KAZE Descriptors')
            axes[1,1].set_title('Distribución de Características')
            axes[1,1].set_xlabel('Valor')
            axes[1,1].set_ylabel('Frecuencia')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nAnálisis comparativo completado")
            print(f"Archivos guardados en: {self.sistema.hog_kaze_analyzer.results_dir}")
            
        except Exception as e:
            print(f"Error en análisis comparativo: {e}")

    def extraer_akaze(self):
        """Extrae características AKAZE."""
        try:
            print("Extrayendo características AKAZE...")
            
            # Preguntar si guardar resultados
            guardar = input("¿Guardar resultados en CSV/TXT/imagen? (s/N): ").strip().lower() == 's'
            
            # Configurar guardado si se solicita
            if guardar:
                self.sistema.advanced_analyzer._save_visualization = True
                
            resultados = self.sistema.advanced_analyzer.extraer_caracteristicas_akaze(
                self.sistema.imagen_actual, 
                visualizar=True,
                guardar_resultados=guardar,
                nombre_imagen=os.path.splitext(os.path.basename(self.sistema.ruta_imagen_actual or "imagen"))[0]
            )
            
            # Limpiar flag
            if hasattr(self.sistema.advanced_analyzer, '_save_visualization'):
                delattr(self.sistema.advanced_analyzer, '_save_visualization')
            
            if resultados:
                print("Extracción AKAZE completada")
                print(f"Keypoints detectados: {len(resultados.get('keypoints', []))}")
                if guardar:
                    print(f"Resultados guardados en: {self.sistema.advanced_analyzer.results_dir}")
            else:
                print("Error en extracción AKAZE")
                
        except Exception as e:
            print(f"Error extrayendo AKAZE: {e}")

    def extraer_freak(self):
        """Extrae características FREAK."""
        try:
            print("Extrayendo características FREAK...")
            
            # Preguntar si guardar resultados
            guardar = input("¿Guardar resultados en CSV/TXT/imagen? (s/N): ").strip().lower() == 's'
            
            # Configurar guardado si se solicita
            if guardar:
                self.sistema.advanced_analyzer._save_visualization = True
                
            resultados = self.sistema.advanced_analyzer.extraer_caracteristicas_freak(
                self.sistema.imagen_actual, 
                visualizar=True,
                guardar_resultados=guardar,
                nombre_imagen=os.path.splitext(os.path.basename(self.sistema.ruta_imagen_actual or "imagen"))[0]
            )
            
            # Limpiar flag
            if hasattr(self.sistema.advanced_analyzer, '_save_visualization'):
                delattr(self.sistema.advanced_analyzer, '_save_visualization')
            
            if resultados:
                print("Extracción FREAK completada")
                print(f"Keypoints detectados: {len(resultados.get('keypoints', []))}")
                if guardar:
                    print(f"Resultados guardados en: {self.sistema.advanced_analyzer.results_dir}")
            else:
                print("Error en extracción FREAK")
                
        except Exception as e:
            print(f"Error extrayendo FREAK: {e}")

    def segmentacion_grabcut(self):
        """Ejecuta segmentación GrabCut."""
        try:
            print("Ejecutando segmentación GrabCut...")
            
            # Preguntar si guardar resultados
            guardar = input("¿Guardar resultados en CSV/TXT/imagen? (s/N): ").strip().lower() == 's'
            
            # Configurar guardado si se solicita
            if guardar:
                self.sistema.advanced_analyzer._save_visualization = True
                
            resultados = self.sistema.advanced_analyzer.analizar_grabcut_segmentation(
                self.sistema.imagen_actual, 
                visualizar=True,
                guardar_resultados=guardar,
                nombre_imagen=os.path.splitext(os.path.basename(self.sistema.ruta_imagen_actual or "imagen"))[0]
            )
            
            # Limpiar flag
            if hasattr(self.sistema.advanced_analyzer, '_save_visualization'):
                delattr(self.sistema.advanced_analyzer, '_save_visualization')
            
            if resultados:
                print("Segmentación GrabCut completada")
                print(f"Calidad de segmentación: {resultados.get('grabcut_edge_coherence', 0):.3f}")
                if guardar:
                    print(f"Resultados guardados en: {self.sistema.advanced_analyzer.results_dir}")
            else:
                print("Error en segmentación GrabCut")
                
        except Exception as e:
            print(f"Error en GrabCut: {e}")

    def analisis_optical_flow(self):
        """Analiza optical flow (requiere segunda imagen)."""
        try:
            print("Análisis de Optical Flow...")
            print("Se requiere una segunda imagen para comparación")
            
            # Opción de selección de carpeta
            print("\nSelección de carpeta de imágenes:")
            print("1. Usar carpeta por defecto (./images)")
            print("2. Especificar ruta personalizada")
            print("3. Cancelar")
            
            opcion_carpeta = input("\nSeleccione opción (1-3): ").strip()
            
            if opcion_carpeta == '1':
                carpeta_imagenes = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")
            elif opcion_carpeta == '2':
                carpeta_personalizada = input("Ingrese la ruta de la carpeta: ").strip()
                if not carpeta_personalizada or not os.path.exists(carpeta_personalizada):
                    print("Ruta inválida o carpeta no existe")
                    return
                carpeta_imagenes = carpeta_personalizada
            elif opcion_carpeta == '3':
                print("Análisis cancelado")
                return
            else:
                print("Opción inválida")
                return
            
            # Buscar imágenes en la carpeta
            imagenes_disponibles = self._buscar_imagenes_en_carpeta(carpeta_imagenes)
            
            if not imagenes_disponibles:
                print(f"No se encontraron imágenes en: {carpeta_imagenes}")
                print("   Formatos soportados: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
                return
            
            # Mostrar lista de imágenes
            print(f"\nImágenes disponibles en: {carpeta_imagenes}")
            print("-" * 60)
            for i, (nombre, ruta) in enumerate(imagenes_disponibles, 1):
                tamaño = self._obtener_info_imagen(ruta)
                print(f"{i:2d}. {nombre} {tamaño}")
            
            # Selección de imagen
            try:
                seleccion = input(f"\nSeleccione imagen (1-{len(imagenes_disponibles)}) o 'c' para cancelar: ").strip()
                if seleccion.lower() == 'c':
                    print("Análisis cancelado")
                    return
                
                indice = int(seleccion) - 1
                if 0 <= indice < len(imagenes_disponibles):
                    nombre_imagen, ruta_segunda = imagenes_disponibles[indice]
                    print(f"Imagen seleccionada: {nombre_imagen}")
                else:
                    print("Selección inválida")
                    return
                    
            except ValueError:
                print("Entrada inválida")
                return
            
            # Preguntar si guardar resultados
            guardar = input("¿Guardar resultados en CSV/TXT/imagen? (s/N): ").strip().lower() == 's'
            
            # Configurar guardado si se solicita
            if guardar:
                self.sistema.advanced_analyzer._save_visualization = True
            
            # Ejecutar análisis
            print(f"\nAnalizando flujo óptico entre imágenes...")
            print(f"Imagen 1: {os.path.basename(self.sistema.ruta_imagen_actual or 'imagen_actual')}")
            print(f"Imagen 2: {nombre_imagen}")
            
            resultados = self.sistema.advanced_analyzer.analizar_optical_flow(
                self.sistema.imagen_actual, 
                ruta_segunda, 
                visualizar=True,
                guardar_resultados=guardar,
                nombre_imagen=f"optical_flow_{os.path.splitext(os.path.basename(self.sistema.ruta_imagen_actual or 'img1'))[0]}_{os.path.splitext(nombre_imagen)[0]}"
            )
            
            # Limpiar flag
            if hasattr(self.sistema.advanced_analyzer, '_save_visualization'):
                delattr(self.sistema.advanced_analyzer, '_save_visualization')
            
            if resultados:
                print("Análisis de Optical Flow completado")
                print(f"Estadísticas principales:")
                print(f"  • Magnitud promedio: {resultados.get('optical_flow_mean_magnitude', 0):.4f}")
                print(f"  • Magnitud máxima: {resultados.get('optical_flow_max_magnitude', 0):.4f}")
                print(f"  • Dirección dominante: {resultados.get('optical_flow_dominant_direction', 0):.0f}°")
                print(f"  • Coherencia espacial: {resultados.get('optical_flow_spatial_coherence', 0):.4f}")
                
                if guardar:
                    print(f"Resultados guardados en: {self.sistema.advanced_analyzer.results_dir}")
            else:
                print("Error en análisis de Optical Flow")
                
        except Exception as e:
            print(f"Error en Optical Flow: {e}")

    def _buscar_imagenes_en_carpeta(self, carpeta):
        """Busca imágenes en una carpeta."""
        extensiones_imagen = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF')
        imagenes = []
        
        try:
            if not os.path.exists(carpeta):
                return imagenes
                
            for archivo in os.listdir(carpeta):
                if archivo.endswith(extensiones_imagen):
                    ruta_completa = os.path.join(carpeta, archivo)
                    imagenes.append((archivo, ruta_completa))
                    
            # Ordenar por nombre
            imagenes.sort(key=lambda x: x[0].lower())
            
        except Exception as e:
            print(f"Error leyendo carpeta {carpeta}: {e}")
            
        return imagenes
    
    def _obtener_info_imagen(self, ruta_imagen):
        """Obtiene información básica de una imagen."""
        try:
            import cv2
            img = cv2.imread(ruta_imagen)
            if img is not None:
                h, w = img.shape[:2]
                tamaño_kb = os.path.getsize(ruta_imagen) // 1024
                return f"({w}x{h}, {tamaño_kb}KB)"
            else:
                return "(no se pudo leer)"
        except:
            return "(error)"

    def analisis_avanzado_combinado(self):
        """Ejecuta análisis avanzado combinado."""
        try:
            print("Ejecutando análisis avanzado combinado...")
            
            resultados = self.sistema.advanced_analyzer.analisis_completo_avanzado(
                self.sistema.ruta_imagen_actual
            )
            
            if resultados:
                print("Análisis avanzado completado")
                print(f"Resultados guardados en: {self.sistema.advanced_analyzer.results_dir}")
            else:
                print("Error en análisis avanzado")
                
        except Exception as e:
            print(f"Error en análisis avanzado: {e}")

    def comparar_algoritmos(self):
        """Compara diferentes algoritmos de extracción."""
        try:
            print("Comparando algoritmos de extracción...")
            
            resultados = self.sistema.comparison_analyzer.compare_feature_detectors(
                self.sistema.imagen_actual
            )
            
            if resultados:
                print("Comparación completada")
                self.sistema.comparison_analyzer.generate_comparison_report(resultados)
            else:
                print("Error en comparación")
                
        except Exception as e:
            print(f"Error comparando algoritmos: {e}")

    def analisis_completo_caracteristicas(self):
        """Ejecuta análisis completo de todas las características."""
        try:
            print("Ejecutando análisis completo de características...")
            
            # Análisis de texturas
            print("Ejecutando análisis de texturas...")
            self.sistema.texture_analyzer.procesar_imagen_completa(self.sistema.ruta_imagen_actual)
            
            # Análisis de Hough
            print("Ejecutando análisis de formas...")
            self.sistema.hough_analyzer.analisis_completo_hough(self.sistema.ruta_imagen_actual)
            
            # Análisis HOG-KAZE
            print("Ejecutando análisis HOG-KAZE...")
            self.sistema.hog_kaze_analyzer.analisis_combinado_hog_kaze(self.sistema.ruta_imagen_actual)
            
            # Análisis SURF-ORB
            print("Ejecutando análisis SURF-ORB...")
            self.sistema.surf_orb_analyzer.analisis_combinado_surf_orb(self.sistema.ruta_imagen_actual)
            
            # Análisis avanzado
            print("Ejecutando análisis avanzado...")
            self.sistema.advanced_analyzer.analisis_completo_avanzado(self.sistema.ruta_imagen_actual)
            
            print("Análisis completo terminado")
            print(f"Todos los resultados guardados en: {self.sistema.directorio_resultados}")
            
        except Exception as e:
            print(f"Error en análisis completo: {e}")

    # =========================================================================
    # FUNCIONES AUXILIARES
    # =========================================================================
    
    def _guardar_resultados_analisis(self, nombre_analisis, resultados):
        """Guarda los resultados del análisis."""
        try:
            guardar = input("¿Guardar resultados? (s/n): ").strip().lower()
            if guardar.startswith('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"{nombre_analisis}_{timestamp}.txt"
                ruta_archivo = os.path.join(self.sistema.directorio_resultados, nombre_archivo)
                
                with open(ruta_archivo, 'w', encoding='utf-8') as f:
                    f.write(f"RESULTADOS DE {nombre_analisis.upper()}\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Imagen: {self.sistema.ruta_imagen_actual}\n")
                    f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    for clave, valor in resultados.items():
                        f.write(f"{clave}: {valor}\n")
                
                print(f"Resultados guardados: {ruta_archivo}")
        except Exception as e:
            print(f"Error guardando resultados: {e}")

    def _mostrar_resultado_bordes(self, nombre_metodo, imagen_bordes):
        """Muestra resultado de detección de bordes."""
        try:
            mostrar = input("¿Mostrar resultado? (s/n): ").strip().lower()
            if mostrar.startswith('s'):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Original
                imagen_original = self.sistema.imagen_actual
                if len(imagen_original.shape) == 3:
                    axes[0].imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))
                else:
                    axes[0].imshow(imagen_original, cmap='gray')
                axes[0].set_title('Imagen Original')
                axes[0].axis('off')
                
                # Bordes
                axes[1].imshow(imagen_bordes, cmap='gray')
                axes[1].set_title(f'Bordes - {nombre_metodo}')
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.show()
                
            # Guardar resultado
            guardar = input("¿Guardar imagen de bordes? (s/n): ").strip().lower()
            if guardar.startswith('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"bordes_{nombre_metodo.lower()}_{timestamp}.jpg"
                ruta_guardar = os.path.join(self.sistema.directorio_resultados, nombre_archivo)
                os.makedirs(os.path.dirname(ruta_guardar), exist_ok=True)
                cv2.imwrite(ruta_guardar, imagen_bordes)
                print(f"Imagen guardada: {ruta_guardar}")
                
        except Exception as e:
            print(f"Error mostrando resultado: {e}")

    def _mostrar_analisis_gradientes(self, magnitud, direccion):
        """Muestra análisis de gradientes."""
        try:
            mostrar = input("¿Mostrar análisis visual? (s/n): ").strip().lower()
            if mostrar.startswith('s'):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Imagen original
                if len(self.sistema.imagen_actual.shape) == 3:
                    axes[0].imshow(cv2.cvtColor(self.sistema.imagen_actual, cv2.COLOR_BGR2RGB))
                else:
                    axes[0].imshow(self.sistema.imagen_actual, cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                # Magnitud
                axes[1].imshow(magnitud, cmap='hot')
                axes[1].set_title('Magnitud de Gradientes')
                axes[1].axis('off')
                
                # Dirección
                axes[2].imshow(direccion, cmap='hsv')
                axes[2].set_title('Dirección de Gradientes')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error mostrando gradientes: {e}")
    
    # =========================================================================
    # IMPLEMENTACIONES DE MÉTODOS AVANZADOS
    # =========================================================================
    
    def extraer_freak(self):
        """Ejecuta extracción de características FREAK."""
        try:
            print("Extrayendo características FREAK...")
            
            # Obtener parámetros del usuario
            mostrar_descriptores = input("¿Mostrar descriptores en consola? (s/n): ").lower() == 's'
            guardar_resultados = input("¿Guardar resultados en archivos? (s/n): ").lower() == 's'
            
            if guardar_resultados:
                nombre_imagen = input("Nombre para los archivos (sin extensión): ").strip()
                if not nombre_imagen:
                    nombre_imagen = "freak_analysis"
            else:
                nombre_imagen = "freak_analysis"
            
            # Realizar análisis FREAK
            resultados = self.sistema.advanced_analyzer.extraer_caracteristicas_freak(
                self.sistema.imagen_actual,
                visualizar=True,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen
            )
            
            print(f"Análisis FREAK completado: {resultados.get('freak_num_keypoints', 0)} puntos detectados")
            
        except Exception as e:
            print(f"Error en análisis FREAK: {e}")
    
    def extraer_akaze_avanzado(self):
        """Ejecuta extracción de características AKAZE avanzado."""
        try:
            print("Extrayendo características AKAZE avanzado...")
            
            # Obtener parámetros del usuario
            mostrar_descriptores = input("¿Mostrar análisis detallado en consola? (s/n): ").lower() == 's'
            guardar_resultados = input("¿Guardar resultados en archivos? (s/n): ").lower() == 's'
            
            if guardar_resultados:
                nombre_imagen = input("Nombre para los archivos (sin extensión): ").strip()
                if not nombre_imagen:
                    nombre_imagen = "akaze_analysis"
            else:
                nombre_imagen = "akaze_analysis"
            
            # Realizar análisis AKAZE
            resultados = self.sistema.advanced_analyzer.extraer_caracteristicas_akaze(
                self.sistema.imagen_actual,
                visualizar=True,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen
            )
            
            print(f"Análisis AKAZE completado: {resultados.get('akaze_num_keypoints', 0)} puntos detectados")
            
        except Exception as e:
            print(f"Error en análisis AKAZE: {e}")
    
    def analizar_grabcut(self):
        """Ejecuta análisis de segmentación GrabCut."""
        try:
            print("Analizando segmentación GrabCut...")
            
            # Obtener parámetros del usuario
            mostrar_descriptores = input("¿Mostrar estadísticas detalladas en consola? (s/n): ").lower() == 's'
            guardar_resultados = input("¿Guardar resultados en archivos? (s/n): ").lower() == 's'
            
            if guardar_resultados:
                nombre_imagen = input("Nombre para los archivos (sin extensión): ").strip()
                if not nombre_imagen:
                    nombre_imagen = "grabcut_analysis"
            else:
                nombre_imagen = "grabcut_analysis"
            
            # Realizar análisis GrabCut
            resultados = self.sistema.advanced_analyzer.analizar_grabcut_segmentation(
                self.sistema.imagen_actual,
                visualizar=True,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen
            )
            
            print(f"Análisis GrabCut completado: {resultados.get('grabcut_num_regions', 0)} regiones detectadas")
            
        except Exception as e:
            print(f"Error en análisis GrabCut: {e}")
    
    def analizar_log(self):
        """Ejecuta análisis Laplaciano de Gauss (LoG)."""
        try:
            print("Analizando con Laplaciano de Gauss (LoG)...")
            
            # Obtener parámetros del usuario
            mostrar_descriptores = input("¿Mostrar análisis detallado en consola? (s/n): ").lower() == 's'
            guardar_resultados = input("¿Guardar resultados en archivos? (s/n): ").lower() == 's'
            
            if guardar_resultados:
                nombre_imagen = input("Nombre para los archivos (sin extensión): ").strip()
                if not nombre_imagen:
                    nombre_imagen = "log_analysis"
            else:
                nombre_imagen = "log_analysis"
            
            # Realizar análisis LoG
            resultados = self.sistema.advanced_analyzer.analizar_log_detector(
                self.sistema.imagen_actual,
                visualizar=True,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen
            )
            
            print(f"Análisis LoG completado: {resultados.get('log_num_blobs', 0)} blobs detectados")
            
        except Exception as e:
            print(f"Error en análisis LoG: {e}")
    
    def analizar_optical_flow(self):
        """Ejecuta análisis de flujo óptico."""
        try:
            print("Analizando flujo óptico...")
            
            # Preguntar si usar segunda imagen
            usar_segunda = input("¿Usar segunda imagen? (s/n - si no, se creará automáticamente): ").lower() == 's'
            
            imagen2 = None
            if usar_segunda:
                print("Selecciona la segunda imagen:")
                self.sistema.seleccionar_imagen_secundaria()
                if self.sistema.imagen_secundaria is not None:
                    imagen2 = self.sistema.imagen_secundaria
                else:
                    print("No se seleccionó segunda imagen, usando automática")
            
            # Obtener parámetros del usuario
            mostrar_descriptores = input("¿Mostrar análisis detallado en consola? (s/n): ").lower() == 's'
            guardar_resultados = input("¿Guardar resultados en archivos? (s/n): ").lower() == 's'
            
            if guardar_resultados:
                nombre_imagen = input("Nombre para los archivos (sin extensión): ").strip()
                if not nombre_imagen:
                    nombre_imagen = "optical_flow_analysis"
            else:
                nombre_imagen = "optical_flow_analysis"
            
            # Realizar análisis de flujo óptico
            resultados = self.sistema.advanced_analyzer.analizar_optical_flow(
                self.sistema.imagen_actual,
                imagen2=imagen2,
                visualizar=True,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen
            )
            
            magnitud_promedio = resultados.get('optical_flow_mean_magnitude', 0)
            print(f"Análisis Optical Flow completado: magnitud promedio {magnitud_promedio:.6f}")
            
        except Exception as e:
            print(f"Error en análisis Optical Flow: {e}")
    
    def analisis_avanzado_completo(self):
        """Ejecuta análisis completo con todos los métodos avanzados."""
        try:
            print("Ejecutando análisis avanzado completo...")
            
            # Obtener parámetros del usuario
            guardar_resultados = input("¿Guardar resultados de todos los métodos? (s/n): ").lower() == 's'
            
            if guardar_resultados:
                nombre_base = input("Nombre base para los archivos (sin extensión): ").strip()
                if not nombre_base:
                    nombre_base = "analisis_avanzado_completo"
            else:
                nombre_base = "analisis_avanzado_completo"
            
            print("\nEjecutando FREAK...")
            resultados_freak = self.sistema.advanced_analyzer.extraer_caracteristicas_freak(
                self.sistema.imagen_actual,
                visualizar=True,
                mostrar_descriptores=True,
                guardar_resultados=guardar_resultados,
                nombre_imagen=f"{nombre_base}_freak"
            )
            
            print("\nEjecutando AKAZE...")
            resultados_akaze = self.sistema.advanced_analyzer.extraer_caracteristicas_akaze(
                self.sistema.imagen_actual,
                visualizar=True,
                mostrar_descriptores=True,
                guardar_resultados=guardar_resultados,
                nombre_imagen=f"{nombre_base}_akaze"
            )
            
            print("\nEjecutando GrabCut...")
            resultados_grabcut = self.sistema.advanced_analyzer.analizar_grabcut_segmentation(
                self.sistema.imagen_actual,
                visualizar=True,
                mostrar_descriptores=True,
                guardar_resultados=guardar_resultados,
                nombre_imagen=f"{nombre_base}_grabcut"
            )
            
            print("\nEjecutando LoG...")
            resultados_log = self.sistema.advanced_analyzer.analizar_log_detector(
                self.sistema.imagen_actual,
                visualizar=True,
                mostrar_descriptores=True,
                guardar_resultados=guardar_resultados,
                nombre_imagen=f"{nombre_base}_log"
            )
            
            print("\nEjecutando Optical Flow...")
            resultados_flow = self.sistema.advanced_analyzer.analizar_optical_flow(
                self.sistema.imagen_actual,
                imagen2=None,
                visualizar=True,
                mostrar_descriptores=True,
                guardar_resultados=guardar_resultados,
                nombre_imagen=f"{nombre_base}_optical_flow"
            )
            
            # Mostrar resumen comparativo
            print("\nRESUMEN COMPARATIVO MÉTODOS AVANZADOS")
            print("=" * 60)
            print(f"FREAK:       {resultados_freak.get('freak_num_keypoints', 0)} puntos clave")
            print(f"AKAZE:       {resultados_akaze.get('akaze_num_keypoints', 0)} puntos clave")
            print(f"GrabCut:     {resultados_grabcut.get('grabcut_num_regions', 0)} regiones")
            print(f"LoG:         {resultados_log.get('log_num_blobs', 0)} blobs")
            print(f"Opt. Flow:   {resultados_flow.get('optical_flow_mean_magnitude', 0):.6f} magnitud promedio")
            print("=" * 60)
            
            print("Análisis avanzado completo finalizado")
            
        except Exception as e:
            print(f"Error en análisis avanzado completo: {e}")
    
    def extraer_surf(self):
        """Ejecuta extracción de características SURF con puntos amarillos."""
        try:
            print("Extrayendo características SURF...")
            
            # Obtener parámetros del usuario
            mostrar_descriptores = input("¿Mostrar descriptores en consola? (s/n): ").lower() == 's'
            guardar_resultados = input("¿Guardar resultados en archivos? (s/n): ").lower() == 's'
            
            if guardar_resultados:
                nombre_imagen = input("Nombre para los archivos (sin extensión): ").strip()
                if not nombre_imagen:
                    nombre_imagen = "surf_analysis"
            else:
                nombre_imagen = "surf_analysis"
            
            # Realizar análisis SURF
            resultados = self.sistema.surf_orb_analyzer.extraer_caracteristicas_surf(
                self.sistema.imagen_actual,
                visualizar=True,
                mostrar_descriptores=mostrar_descriptores,
                guardar_resultados=guardar_resultados,
                nombre_imagen=nombre_imagen
            )
            
            algoritmo_usado = 'SURF' if resultados.get('surf_algorithm_used') == 'SURF' else 'SIFT'
            print(f"Análisis {algoritmo_usado} completado: {resultados.get('surf_num_keypoints', 0)} puntos detectados")
            
        except Exception as e:
            print(f"Error en análisis SURF: {e}")