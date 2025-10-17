#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Detección de Objetos Vehiculares
===========================================

Proyecto: Identificación de objetos en imágenes de tráfico vehicular
Universidad del Quindío - Visión Artificial

Sistema completo para detectar:
- Llantas de vehículos
- Señales de tráfico circulares
- Semáforos

Algoritmos implementados:
- Transformada de Hough
- AKAZE/FREAK para puntos característicos
- Análisis de color HSV
- Laplaciano de Gauss (LoG)
- GrabCut para segmentación
- Métodos combinados para mayor robustez

Autor: Estudiante
Fecha: Octubre 2025
"""

import os
import sys
import cv2
import numpy as np
import csv
from datetime import datetime
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI para evitar ventanas
import warnings
warnings.filterwarnings('ignore')

# Agregar rutas al sistema
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar detectores
from detectores.detector_llantas import DetectorLlantas, detectar_llantas_imagen
from detectores.detector_senales import DetectorSenales, detectar_senales_imagen
from detectores.detector_semaforos_nuevo import DetectorSemaforos, detectar_semaforos_imagen
from detectores.procesador_lotes import ProcesadorLotes, procesar_carpeta_imagenes

# Importar extensiones de métodos múltiples corregidas
from scripts.extensiones_multiples_llantas_corregido import agregar_metodos_multiples_llantas
from scripts.extensiones_multiples_senales_corregido import agregar_metodos_multiples_senales
from scripts.extensiones_multiples_semaforos_corregido import agregar_metodos_multiples_semaforos
from scripts.extensiones_procesamiento_lotes import agregar_procesamiento_multiples_metodos

# Importar módulos de análisis
from modules.preprocessing import ImagePreprocessor
from modules.texture_analysis import TextureAnalyzer
from modules.hough_analysis import HoughAnalyzer
from modules.hog_kaze import HOGKAZEAnalyzer
from modules.surf_orb import SURFORBAnalyzer
from modules.advanced_algorithms import AdvancedAnalyzer
from modules.comparison_metrics import ComparisonAnalyzer
from modules.hough_results_saver import HoughResultsSaver

# Importar módulos de preprocesamiento
from modules.filtros import Filtros
from modules.operaciones_aritmeticas import OperacionesAritmeticas
from modules.operaciones_geometricas import OperacionesGeometricas
from modules.operaciones_logicas import OperacionesLogicas
from modules.operaciones_morfologicas import OperacionesMorfologicas
from modules.segmentacion import Segmentacion

# Importar utilidades
from utils.image_utils import ImageHandler

# Importar handlers de menús
from scripts.menu_handlers import AnalysisHandlers

class SistemaDeteccionVehicular:
    """Sistema principal de detección de objetos vehiculares."""
    
    def __init__(self):
        """Inicializar el sistema."""
        self.imagen_actual = None
        self.ruta_imagen_actual = None
        self.directorio_imagenes = "./images"
        self.directorio_resultados = "./resultados_deteccion"
        
        # Variables para el sistema de preprocesamiento avanzado
        self.imagen_original = None  # Imagen original sin modificaciones
        self.imagen_preprocesada = None  # Imagen con preprocesamiento acumulativo
        self.historial_preprocesamiento = []  # Historial de operaciones aplicadas
        self.imagen_secundaria = None  # Para operaciones que requieren dos imágenes
        self.ruta_imagen_secundaria = None
        
        # Verificar que existen los directorios necesarios
        self.verificar_directorios()
        
        # Inicializar detectores
        self.detector_llantas = DetectorLlantas()
        self.detector_senales = DetectorSenales()
        self.detector_semaforos = DetectorSemaforos()
        self.procesador_lotes = ProcesadorLotes(self.directorio_resultados)
        
        # Agregar extensiones de métodos múltiples
        print("🔧 Cargando extensiones de métodos múltiples...")
        agregar_metodos_multiples_llantas()
        agregar_metodos_multiples_senales()
        agregar_metodos_multiples_semaforos()
        agregar_procesamiento_multiples_metodos()
        print("Extensiones de métodos múltiples cargadas")
        
        # Inicializar analizadores de características
        self.preprocessor = ImagePreprocessor()
        self.texture_analyzer = TextureAnalyzer(self.directorio_resultados)
        self.hough_analyzer = HoughAnalyzer(self.directorio_resultados)
        self.hog_kaze_analyzer = HOGKAZEAnalyzer(self.directorio_resultados)
        self.surf_orb_analyzer = SURFORBAnalyzer(self.directorio_resultados)
        self.advanced_analyzer = AdvancedAnalyzer(self.directorio_resultados)
        self.comparison_analyzer = ComparisonAnalyzer()
        self.hough_results_saver = HoughResultsSaver(self.directorio_resultados)
        
        # Inicializar handlers de análisis
        self.analysis_handlers = AnalysisHandlers(self)
        
        # Inicializar módulos de preprocesamiento
        self.filtros = Filtros()
        self.operaciones_aritmeticas = OperacionesAritmeticas()
        self.operaciones_geometricas = OperacionesGeometricas()
        self.operaciones_logicas = OperacionesLogicas()
        self.operaciones_morfologicas = OperacionesMorfologicas()
        self.segmentacion = Segmentacion()
        
        print("Sistema de detección vehicular inicializado correctamente")
    
    def verificar_directorios(self):
        """Verifica y crea directorios necesarios."""
        directorios = [
            self.directorio_imagenes,
            self.directorio_resultados,
            os.path.join(self.directorio_resultados, "llantas"),
            os.path.join(self.directorio_resultados, "senales"),
            os.path.join(self.directorio_resultados, "semaforos"),
            os.path.join(self.directorio_resultados, "reportes"),
            os.path.join(self.directorio_resultados, "hough_analysis"),
            os.path.join(self.directorio_resultados, "hough_analysis", "individual"),
            os.path.join(self.directorio_resultados, "hough_analysis", "batch"),
            os.path.join(self.directorio_resultados, "hough_analysis", "visualizations"),
            os.path.join(self.directorio_resultados, "texture_analysis"),
            os.path.join(self.directorio_resultados, "advanced_analysis"),
            os.path.join(self.directorio_resultados, "surf_orb_analysis"),
            os.path.join(self.directorio_resultados, "logs"),
            os.path.join(self.directorio_resultados, "reportes", "comparativos"),
            os.path.join(self.directorio_resultados, "reportes", "semaforos"),
        ]
        
        for directorio in directorios:
            os.makedirs(directorio, exist_ok=True)
    
    def asegurar_directorio_existe(self, ruta_archivo):
        """Asegura que el directorio para un archivo existe antes de guardarlo."""
        directorio = os.path.dirname(ruta_archivo)
        os.makedirs(directorio, exist_ok=True)
    
    def mostrar_encabezado(self):
        """Muestra el encabezado del sistema."""
        print("\n" + "="*70)
        print("SISTEMA DE DETECCIÓN DE OBJETOS VEHICULARES")
        print("="*70)
        print(" Universidad del Quindío - Visión Artificial")
        print(" Detección de Llantas, Señales de Tráfico y Semáforos")
        print(" Algoritmos: Hough, AKAZE, FREAK, LoG, GrabCut, Color HSV")
        print("="*70)
    
    def mostrar_menu_principal(self):
        """Muestra el menú principal reestructurado."""
        print("\nMENÚ PRINCIPAL - ANÁLISIS DE TRÁFICO VEHICULAR")
        print("="*60)
        print("1. Gestión de Imágenes")
        print("2. Preprocesamiento de Imágenes")
        print("3. Extracción de Características")
        print("4. Detección de Objetos Específicos")
        print("5. Procesamiento por Lotes")
        print("6. Estadísticas y Reportes")
        print("7. Configuración")
        print("8. Ayuda")
        print("0. Salir")
        print("="*60)
    
    def mostrar_menu_imagenes(self):
        """Muestra el menú de gestión de imágenes."""
        print("\nGESTIÓN DE IMÁGENES")
        print("-" * 40)
        print("1. Listar imágenes disponibles")
        print("2. Cargar imagen específica")
        print("3. Visualizar imagen actual")
        print("4. Información de imagen actual")
        print("5. Volver al menú principal")
        print("-" * 40)
    
    def mostrar_menu_preprocesamiento(self):
        """Muestra el menú de preprocesamiento reorganizado con submódulos."""
        print("\n" + "="*70)
        print("PREPROCESAMIENTO DE IMÁGENES".center(70))
        print("="*70)
        print("Técnicas de Procesamiento Organizadas por Categorías")
        print("-" * 70)
        print("1. Filtros de Imagen")
        print("2. Operaciones Aritméticas") 
        print("3. Operaciones Geométricas")
        print("4. Operaciones Lógicas")
        print("5. Operaciones Morfológicas")
        print("6. Segmentación")
        print("7. Preprocesamiento Automático")
        print("8. Vista Previa Comparativa")
        print("-" * 70)
        print("GESTIÓN DE PREPROCESAMIENTO:")
        print("91. Ver Historial de Operaciones")
        print("92. Resetear a Imagen Original")
        print("93. Guardar Estado Actual")
        print("0.  Volver al menú principal")
        print("-" * 70)
    
    def mostrar_menu_extraccion_caracteristicas(self):
        """Muestra el menú de extracción de características."""
        print("\n EXTRACCIÓN DE CARACTERÍSTICAS")
        print("-" * 50)
        print("1. Descriptores de Textura")
        print("2. Detección de Bordes")
        print("3. Detección de Formas")
        print("4. Métodos Avanzados de Características")
        print("5. Comparación de Algoritmos")
        print("6. Análisis Completo")
        print("0. Volver al menú principal")
        print("-" * 50)
    
    def mostrar_menu_descriptores_textura(self):
        """Muestra el menú de descriptores de textura."""
        print("\n DESCRIPTORES DE TEXTURA")
        print("-" * 40)
        print("1. Estadísticas de Primer Orden")
        print("2. Estadísticas de Segundo Orden (GLCM)")
        print("3. Análisis Completo de Texturas")
        print("4. Comparación de Regiones")
        print("0. Volver")
        print("-" * 40)
    
    def mostrar_menu_deteccion_bordes(self):
        """Muestra el menú de detección de bordes."""
        print("\nDETECCIÓN DE BORDES")
        print("-" * 40)
        print("1. Filtro Canny")
        print("2. Filtro Sobel")
        print("3. Laplaciano de Gauss (LoG)")
        print("4. Análisis de Gradientes")
        print("5. Comparación de Métodos")
        print("0. Volver")
        print("-" * 40)
    
    def mostrar_menu_deteccion_formas(self):
        """Muestra el menú de detección de formas."""
        print("\nDETECCIÓN DE FORMAS")
        print("-" * 40)
        print("1. Transformada de Hough - Líneas")
        print("2. Transformada de Hough - Círculos") 
        print("3. Momentos Geométricos")
        print("4. Análisis Completo de Formas")
        print("5. Análisis Completo de Hough (Imagen Individual)")
        print("6. Análisis de Hough por Lotes (Carpeta)")
        print("0. Volver")
        print("-" * 40)
    
    def mostrar_menu_metodos_avanzados(self):
        """Muestra el menú de métodos avanzados."""
        print("\nMÉTODOS AVANZADOS DE CARACTERÍSTICAS")
        print("-" * 60)
        print("1. SURF (Speeded Up Robust Features)")
        print("2. ORB (Oriented FAST and Rotated BRIEF)")
        print("3. HOG (Histogram of Oriented Gradients)")
        print("4. KAZE")
        print("5. AKAZE")
        print("6. FREAK (Fast Retina Keypoint)")
        print("7. GrabCut Segmentation")
        print("8. Optical Flow (Método Original)")
        print("9. Optical Flow (Método Predeterminado)")
        print("10. Análisis de Secuencias en Carpeta")
        print("11. Análisis Comparativo HOG + KAZE")
        print("12. Análisis Combinado Avanzado")
        print("=" * 60)
        print("ANÁLISIS AUTOMÁTICO:")
        print("99. Ejecutar TODOS los métodos de características (1-7)")
        print("    (SURF, ORB, HOG, KAZE, AKAZE, FREAK, GrabCut)")
        print("0. Volver")
        print("-" * 60)
    
    def mostrar_menu_deteccion_objetos(self):
        """Muestra el menú de detección de objetos específicos."""
        print("\nDETECCIÓN DE OBJETOS ESPECÍFICOS")
        print("-" * 50)
        print("1. Detectar Llantas")
        print("2. Detectar Señales de Tráfico")
        print("3. Detectar Semáforos")
        print("4. Detección Completa (Todos los objetos)")
        print("0. Volver al menú principal")
        print("-" * 50)
    
    def mostrar_menu_metodos(self, tipo_objeto):
        """Muestra menú de métodos para un tipo de objeto."""
        print(f"\nMÉTODOS PARA {tipo_objeto.upper()}")
        print("-" * 40)
        
        if tipo_objeto == "llantas":
            print("1. Hough Circles")
            print("2. AKAZE")
            print("3. Análisis de Texturas")
            print("4. Combinado (Recomendado)")
            print("5. TODOS los métodos (Análisis completo)")
        
        elif tipo_objeto == "señales":
            print("1. Hough Circles")
            print("2. FREAK/ORB")
            print("3. Análisis de Color")
            print("4. Laplaciano de Gauss (LoG)")
            print("5. Combinado (Recomendado)")
            print("6. TODOS los métodos (Análisis completo)")
        
        elif tipo_objeto == "semáforos":
            print("1. Análisis de Color")
            print("2. Análisis de Estructura")
            print("3. GrabCut")
            print("4. Combinado (Recomendado)")
            print("5. Multi-Algoritmo Avanzado")
            print("6. TODOS los métodos (Análisis completo)")
        
        print("0. Volver")
        print("-" * 40)
    
    def ejecutar_sistema(self):
        """Ejecuta el bucle principal del sistema."""
        self.mostrar_encabezado()
        
        while True:
            try:
                self.mostrar_menu_principal()
                opcion = input("\nSeleccione una opción: ").strip()
                
                if opcion == '1':
                    self.gestionar_imagenes()
                elif opcion == '2':
                    self.menu_preprocesamiento()
                elif opcion == '3':
                    self.menu_extraccion_caracteristicas()
                elif opcion == '4':
                    self.menu_deteccion_objetos()
                elif opcion == '5':
                    self.procesamiento_lotes()
                elif opcion == '6':
                    self.ver_estadisticas()
                elif opcion == '7':
                    self.configuracion()
                elif opcion == '8':
                    self.mostrar_ayuda()
                elif opcion == '0':
                    print("\n¡Gracias por usar el sistema de detección vehicular!")
                    break
                else:
                    print("Opción no válida. Por favor, intente de nuevo.")
                
            except KeyboardInterrupt:
                print("\n\n¡Hasta luego!")
                break
            except Exception as e:
                print(f"\nError inesperado: {e}")
                print("Continuando...")
    
    def gestionar_imagenes(self):
        """Gestiona la carga y visualización de imágenes."""
        while True:
            self.mostrar_menu_imagenes()
            opcion = input("\nSeleccione una opción: ").strip()
            
            if opcion == '1':
                self.listar_imagenes()
            elif opcion == '2':
                self.cargar_imagen()
            elif opcion == '3':
                self.visualizar_imagen_actual()
            elif opcion == '4':
                self.mostrar_info_imagen()
            elif opcion == '5':
                break
            else:
                print("Opción no válida.")
    
    def listar_imagenes(self):
        """Lista las imágenes disponibles."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        
        if not imagenes:
            print(f"\nNo se encontraron imágenes en: {self.directorio_imagenes}")
            print("Coloque sus imágenes en la carpeta 'images'")
            return
        
        print(f"\nImágenes disponibles en {self.directorio_imagenes}:")
        print("-" * 60)
        for i, imagen in enumerate(imagenes, 1):
            nombre = os.path.basename(imagen)
            print(f"{i:2d}. {nombre}")
        print("-" * 60)
        print(f"Total: {len(imagenes)} imágenes")
    
    def cargar_imagen(self):
        """Carga una imagen específica."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        
        if not imagenes:
            print(f"\nNo hay imágenes disponibles en {self.directorio_imagenes}")
            return
        
        self.listar_imagenes()
        
        try:
            seleccion = input("\nNúmero de imagen a cargar (0 para cancelar): ").strip()
            
            if seleccion == '0':
                return
            
            indice = int(seleccion) - 1
            
            if 0 <= indice < len(imagenes):
                ruta_imagen = imagenes[indice]
                imagen = ImageHandler.cargar_imagen(ruta_imagen)
                
                if imagen is not None:
                    self.imagen_actual = imagen
                    self.ruta_imagen_actual = ruta_imagen
                    # Inicializar sistema de preprocesamiento
                    self.imagen_original = imagen.copy()
                    self.imagen_preprocesada = imagen.copy()
                    self.historial_preprocesamiento = []
                    nombre = os.path.basename(ruta_imagen)
                    print(f"Imagen cargada: {nombre}")
                    print(f"Dimensiones: {imagen.shape[1]}x{imagen.shape[0]}")
                    print("Sistema de preprocesamiento inicializado")
                else:
                    print("Error al cargar la imagen")
            else:
                print("Número de imagen no válido")
                
        except ValueError:
            print("Por favor, ingrese un número válido")
    
    def seleccionar_imagen_secundaria(self):
        """Selecciona una segunda imagen para operaciones que requieren dos imágenes."""
        print("\nSELECCIÓN DE IMAGEN SECUNDARIA")
        print("=" * 50)
        
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        
        if not imagenes:
            print(f"No hay imágenes disponibles en {self.directorio_imagenes}")
            return False
        
        print("Imágenes disponibles:")
        print("-" * 60)
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre = os.path.basename(ruta_imagen)
            # Marcar la imagen actual
            marca = " (ACTUAL)" if ruta_imagen == self.ruta_imagen_actual else ""
            print(f"{i:2d}. {nombre}{marca}")
        print("-" * 60)
        
        try:
            seleccion = input("\nNúmero de imagen secundaria (0 para cancelar): ").strip()
            
            if seleccion == '0':
                return False
            
            indice = int(seleccion) - 1
            
            if 0 <= indice < len(imagenes):
                ruta_imagen = imagenes[indice]
                imagen = ImageHandler.cargar_imagen(ruta_imagen)
                
                if imagen is not None:
                    self.imagen_secundaria = imagen
                    self.ruta_imagen_secundaria = ruta_imagen
                    nombre = os.path.basename(ruta_imagen)
                    print(f"Imagen secundaria cargada: {nombre}")
                    print(f"Dimensiones: {imagen.shape[1]}x{imagen.shape[0]}")
                    return True
                else:
                    print("Error al cargar la imagen secundaria")
                    return False
            else:
                print("Número de imagen no válido")
                return False
                
        except ValueError:
            print("Por favor, ingrese un número válido")
            return False
    
    def aplicar_a_imagen_activa(self, imagen_resultado, nombre_operacion):
        """Pregunta si aplicar el resultado a la imagen activa y actualiza el estado."""
        print(f"\nResultado de: {nombre_operacion}")
        print("-" * 50)
        print("¿Desea aplicar este preprocesamiento a la imagen activa?")
        print("Esto mantendrá los cambios acumulativos.")
        print()
        print("1. Sí, aplicar a imagen activa (cambios acumulativos)")
        print("2. Solo ver resultado (no aplicar)")
        print("3. Resetear a imagen original y aplicar")
        print("0. Cancelar")
        
        opcion = input("\nSeleccione opción: ").strip()
        
        if opcion == '1':
            # Aplicar a imagen activa (acumulativo)
            self.imagen_actual = imagen_resultado.copy()
            self.imagen_preprocesada = imagen_resultado.copy()
            self.historial_preprocesamiento.append(nombre_operacion)
            print(f"{nombre_operacion} aplicado a la imagen activa")
            print(f"Operaciones aplicadas: {len(self.historial_preprocesamiento)}")
            return True
            
        elif opcion == '2':
            # Solo mostrar sin aplicar
            print("Mostrando resultado sin aplicar cambios")
            return False
            
        elif opcion == '3':
            # Resetear y aplicar
            self.imagen_actual = self.imagen_original.copy()
            resultado_final = self._aplicar_operacion_a_imagen(self.imagen_actual, nombre_operacion)
            if resultado_final is not None:
                self.imagen_actual = resultado_final.copy()
                self.imagen_preprocesada = resultado_final.copy()
                self.historial_preprocesamiento = [nombre_operacion]
                print(f"Imagen reseteada y {nombre_operacion} aplicado")
                return True
            return False
            
        else:
            print("Operación cancelada")
            return False
    
    def mostrar_historial_preprocesamiento(self):
        """Muestra el historial de operaciones de preprocesamiento aplicadas."""
        if not self.historial_preprocesamiento:
            print("No hay operaciones de preprocesamiento aplicadas")
            return
        
        print("\nHISTORIAL DE PREPROCESAMIENTO")
        print("=" * 50)
        for i, operacion in enumerate(self.historial_preprocesamiento, 1):
            print(f"{i:2d}. {operacion}")
        print("=" * 50)
        print(f"Total de operaciones: {len(self.historial_preprocesamiento)}")
    
    def resetear_preprocesamiento(self):
        """Resetea la imagen a su estado original."""
        if self.imagen_original is None:
            print("No hay imagen original disponible")
            return False
        
        confirmacion = input("¿Confirma resetear todos los cambios? (s/N): ").strip().lower()
        if confirmacion in ['s', 'sí', 'si', 'yes', 'y']:
            self.imagen_actual = self.imagen_original.copy()
            self.imagen_preprocesada = self.imagen_original.copy()
            self.historial_preprocesamiento = []
            print("Imagen reseteada al estado original")
            return True
        else:
            print("Operación cancelada")
            return False
    
    def guardar_estado_actual(self):
        """Guarda el estado actual de la imagen preprocesada."""
        if self.imagen_actual is None:
            print("No hay imagen cargada")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_base = "estado_preprocesado" if not self.historial_preprocesamiento else "_".join(self.historial_preprocesamiento[:3])
        nombre_archivo = f"{nombre_base}_{timestamp}.jpg"
        ruta_guardar = os.path.join(self.directorio_resultados, nombre_archivo)
        
        try:
            self.asegurar_directorio_existe(ruta_guardar)
            cv2.imwrite(ruta_guardar, self.imagen_actual)
            print(f"Estado actual guardado: {ruta_guardar}")
            
            # Guardar historial en archivo de texto
            if self.historial_preprocesamiento:
                ruta_historial = ruta_guardar.replace('.jpg', '_historial.txt')
                self.asegurar_directorio_existe(ruta_historial)
                with open(ruta_historial, 'w', encoding='utf-8') as f:
                    f.write("HISTORIAL DE PREPROCESAMIENTO\n")
                    f.write("=" * 40 + "\n")
                    f.write(f"Imagen original: {self.ruta_imagen_actual}\n")
                    f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total operaciones: {len(self.historial_preprocesamiento)}\n\n")
                    f.write("Operaciones aplicadas:\n")
                    for i, operacion in enumerate(self.historial_preprocesamiento, 1):
                        f.write(f"{i:2d}. {operacion}\n")
                print(f"Historial guardado: {ruta_historial}")
            
        except Exception as e:
            print(f"Error guardando estado: {e}")
    
    def _aplicar_operacion_a_imagen(self, imagen, nombre_operacion):
        """Método auxiliar para reaplicar operaciones específicas."""
        # Este método se puede expandir para manejar la reaplicación de operaciones específicas
        return imagen
    
    def visualizar_imagen_actual(self):
        """Visualiza la imagen actualmente cargada."""
        if self.imagen_actual is None:
            print("\nNo hay imagen cargada. Cargue una imagen primero.")
            return
        
        print("Mostrando imagen actual...")
        nombre = os.path.basename(self.ruta_imagen_actual) if self.ruta_imagen_actual else "Imagen"
        
        # Redimensionar para visualización si es muy grande
        imagen_display = ImageHandler.redimensionar_imagen(self.imagen_actual, 800)
        ImageHandler.mostrar_imagen(imagen_display, f"Imagen Actual: {nombre}")
    
    def mostrar_info_imagen(self):
        """Muestra información de la imagen actual."""
        if self.imagen_actual is None:
            print("\nNo hay imagen cargada.")
            return
        
        print("\nINFORMACIÓN DE LA IMAGEN")
        print("-" * 40)
        print(f"Archivo: {os.path.basename(self.ruta_imagen_actual)}")
        print(f"Ruta: {self.ruta_imagen_actual}")
        print(f"Dimensiones: {self.imagen_actual.shape[1]}x{self.imagen_actual.shape[0]}")
        print(f"Canales: {self.imagen_actual.shape[2] if len(self.imagen_actual.shape) > 2 else 1}")
        print(f"Tipo de datos: {self.imagen_actual.dtype}")
        print(f"Tamaño en memoria: {self.imagen_actual.nbytes / 1024:.1f} KB")
    
    def verificar_imagen_cargada(self):
        """Verifica si hay una imagen cargada."""
        if self.imagen_actual is None:
            print("\nNo hay imagen cargada.")
            print("Vaya a 'Gestión de Imágenes' y cargue una imagen primero.")
            return False
        return True
    
    def detectar_llantas(self):
        """Ejecuta detección de llantas."""
        if not self.verificar_imagen_cargada():
            return
        
        print("\nConfiguración de detección de llantas:")
        print("1. CONFIG_PRECISION_ALTA - Hough multiescala + Contornos + Color + AKAZE")
        print("2. CONFIG_ROBUSTA - Contornos circulares + Textura + Color + Validación geométrica")
        print("3. CONFIG_ADAPTATIVA - Hough adaptativo + Análisis textural + Color multirrango")
        print("4. CONFIG_BALANCED - Combinación equilibrada (recomendada)")
        print("5. Ejecutar TODAS las configuraciones")
        print("0. Volver al menú anterior")
        
        opcion = input("\nSeleccione configuración: ").strip()
        
        configuraciones = {
            '1': 'CONFIG_PRECISION_ALTA',
            '2': 'CONFIG_ROBUSTA', 
            '3': 'CONFIG_ADAPTATIVA',
            '4': 'CONFIG_BALANCED'
        }
        
        if opcion == '0':
            return
        
        if opcion == '5':
            # Ejecutar TODAS las configuraciones
            print("Ejecutando TODAS las configuraciones de detección de llantas...")
            try:
                resultado = self.detector_llantas.detectar_llantas_todos_metodos(
                    self.imagen_actual, visualizar=True, guardar=True,
                    ruta_base=self.directorio_resultados
                )
                
                if resultado:
                    print(f"\nAnálisis completo terminado:")
                    total_detecciones = 0
                    for config, res in resultado.items():
                        if 'error' not in res:
                            num = len(res.get('llantas_detectadas', []))
                            total_detecciones += num
                            tiempo = res.get('tiempo_ejecucion', 0)
                            confianza = res.get('confianza_promedio', 0)
                            print(f"   {config}: {num} llantas (tiempo: {tiempo:.3f}s, confianza: {confianza:.3f})")
                        else:
                            print(f"   {config}: ERROR - {res.get('error', 'Desconocido')}")
                    
                    print(f"Total de detecciones: {total_detecciones}")
                    print(f"Reportes guardados en: {self.directorio_resultados}/reportes/")
                else:
                    print("Error en el análisis completo")
            except Exception as e:
                print(f"Error durante el análisis completo: {e}")
        else:
            # Ejecutar configuración individual
            configuracion = configuraciones.get(opcion, 'CONFIG_BALANCED')
            print(f"\nDetectando llantas con configuración: {configuracion}")
            
            try:
                resultado = self.detector_llantas.detectar_llantas(
                    self.imagen_actual, configuracion=configuracion, visualizar=True,
                    guardar=True, ruta_salida=os.path.join(self.directorio_resultados, "llantas")
                )
                
                if resultado:
                    print(f"\nDetección completada:")
                    print(f"   Configuración: {resultado['configuracion']}")
                    print(f"   Llantas detectadas: {resultado['num_llantas']}")
                    print(f"   Candidatos iniciales: {resultado['candidatos_iniciales']}")
                    print(f"   Confianza promedio: {resultado['confianza_promedio']:.3f}")
                else:
                    print("Error en la detección")
                    
            except Exception as e:
                print(f"Error durante la detección: {e}")
    
    def detectar_senales(self):
        """Ejecuta detección de señales de tráfico con selección de forma."""
        if not self.verificar_imagen_cargada():
            return
        
        # Primero seleccionar forma
        print("\nSelección de forma de señal:")
        print("1. CIRCULAR - Señales circulares (prohibición, obligación)")
        print("2. RECTANGULAR - Señales rectangulares (informativas)")
        print("3. TRIANGULAR - Señales triangulares (advertencia)")
        print("4. OCTAGONAL - Señales octagonales (STOP)")
        print("5. TODAS - Todas las formas")
        print("0. Volver al menú anterior")
        
        opcion_forma = input("\nSeleccione forma: ").strip()
        
        if opcion_forma == '0':
            return
        elif opcion_forma not in ['1', '2', '3', '4', '5']:
            print("Opción no válida")
            return
        
        # Mapeo de opciones a formas
        formas = {
            '1': 'CIRCULAR',
            '2': 'RECTANGULAR',
            '3': 'TRIANGULAR',
            '4': 'OCTAGONAL',
            '5': 'TODAS'
        }
        
        forma_elegida = formas[opcion_forma]
        
        print(f"\nConfiguración de detección de señales {forma_elegida.lower()}:")
        print("1. CONFIG_PRECISION_ALTA - Hough multiescala + Color HSV + Validación morfológica")
        print("2. CONFIG_ROBUSTA - Contornos + Color + Validación geométrica")
        print("3. CONFIG_ADAPTATIVA - Hough adaptativo + Análisis textural + Color multirrango")
        print("4. CONFIG_BALANCED - Combinación equilibrada (recomendada)")
        print("5. Ejecutar TODAS las configuraciones")
        print("0. Volver al menú anterior")
        
        opcion = input("\nSeleccione configuración: ").strip()
        
        configuraciones = {
            '1': 'CONFIG_PRECISION_ALTA',
            '2': 'CONFIG_ROBUSTA',
            '3': 'CONFIG_ADAPTATIVA',
            '4': 'CONFIG_BALANCED'
        }
        
        if opcion == '0':
            return
        
        if opcion == '5':
            # Ejecutar TODAS las configuraciones
            print(f"Ejecutando TODAS las configuraciones de detección de señales {forma_elegida.lower()}...")
            try:
                resultado = self.detector_senales.detectar_senales_todos_metodos(
                    self.imagen_actual, forma=forma_elegida, visualizar=True, guardar=True,
                    ruta_base=self.directorio_resultados
                )
                
                if resultado:
                    print(f"\nAnálisis completo terminado:")
                    total_detecciones = 0
                    for config, res in resultado.items():
                        if 'error' not in res:
                            num = len(res.get('senales_detectadas', []))
                            total_detecciones += num
                            tiempo = res.get('tiempo_ejecucion', 0)
                            confianza = res.get('confianza_promedio', 0)
                            print(f"   {config}: {num} señales (tiempo: {tiempo:.3f}s, confianza: {confianza:.3f})")
                        else:
                            print(f"   {config}: ERROR - {res.get('error', 'Desconocido')}")
                    
                    print(f"Total de detecciones: {total_detecciones}")
                    print(f"Reportes guardados en: {self.directorio_resultados}/reportes/")
                else:
                    print("Error en el análisis completo")
            except Exception as e:
                print(f"Error durante el análisis completo: {e}")
        else:
            # Ejecutar configuración individual
            configuracion = configuraciones.get(opcion, 'CONFIG_BALANCED')
            print(f"\nDetectando señales {forma_elegida.lower()} con configuración: {configuracion}")
            
            try:
                resultado = self.detector_senales.detectar_senales(
                    self.imagen_actual, configuracion=configuracion, forma=forma_elegida,
                    visualizar=True, guardar=True, 
                    ruta_salida=os.path.join(self.directorio_resultados, "senales")
                )
                
                if resultado:
                    print(f"\nDetección completada:")
                    print(f"   Forma: {forma_elegida}")
                    print(f"   Configuración: {resultado['configuracion']}")
                    print(f"   Señales detectadas: {resultado['num_senales']}")
                    print(f"   Confianza promedio: {resultado['confianza_promedio']:.3f}")
                    print(f"   Candidatos por forma: {resultado.get('candidatos_forma', 0)}")
                    print(f"   Candidatos Hough: {resultado.get('candidatos_hough', 0)}")
                    print(f"   Candidatos contornos: {resultado.get('candidatos_contornos', 0)}")
                    
                    # Mostrar estadísticas por tipo si están disponibles
                    if 'estadisticas' in resultado and 'por_tipo' in resultado['estadisticas']:
                        print("   Estadísticas por tipo:")
                        for tipo, cantidad in resultado['estadisticas']['por_tipo'].items():
                            if cantidad > 0:
                                print(f"     - {tipo}: {cantidad} detecciones")
                else:
                    print("Error en la detección")
                    
            except Exception as e:
                print(f"Error durante la detección: {e}")
    
    def detectar_semaforos(self):
        """Ejecuta detección de semáforos."""
        if not self.verificar_imagen_cargada():
            return
        
        print("\nConfiguración de detección de semáforos:")
        print("1. CONFIG_PRECISION_ALTA - Color HSV + Estructura + Morfología + Validación")
        print("2. CONFIG_ROBUSTA - Contornos + Color + GrabCut + Validación geométrica")
        print("3. CONFIG_ADAPTATIVA - Color multirrango + Textura + Hough + AKAZE")
        print("4. CONFIG_BALANCED - Combinación equilibrada (recomendada)")
        print("5. Ejecutar TODAS las configuraciones")
        print("0. Volver al menú anterior")
        
        opcion = input("\nSeleccione configuración: ").strip()
        
        configuraciones = {
            '1': 'CONFIG_PRECISION_ALTA',
            '2': 'CONFIG_ROBUSTA',
            '3': 'CONFIG_ADAPTATIVA',
            '4': 'CONFIG_BALANCED'
        }
        
        if opcion == '0':
            return
        
        if opcion == '5':
            # Ejecutar TODAS las configuraciones
            print("Ejecutando TODAS las configuraciones de detección de semáforos...")
            try:
                resultado = self.detector_semaforos.detectar_semaforos_todos_metodos(
                    self.imagen_actual, visualizar=True, guardar=True,
                    ruta_base=self.directorio_resultados
                )
                
                if resultado:
                    print(f"\nAnálisis completo terminado:")
                    total_detecciones = 0
                    for config, res in resultado.items():
                        if 'error' not in res:
                            num = len(res.get('semaforos_detectados', []))
                            total_detecciones += num
                            tiempo = res.get('tiempo_ejecucion', 0)
                            confianza = res.get('confianza_promedio', 0)
                            print(f"   {config}: {num} semáforos (tiempo: {tiempo:.3f}s, confianza: {confianza:.3f})")
                        else:
                            print(f"   {config}: ERROR - {res.get('error', 'Desconocido')}")
                    
                    print(f"Total de detecciones: {total_detecciones}")
                    print(f"Reportes guardados en: {self.directorio_resultados}/reportes/")
                else:
                    print("Error en el análisis completo")
            except Exception as e:
                print(f"Error durante el análisis completo: {e}")
        else:
            # Ejecutar configuración individual
            configuracion = configuraciones.get(opcion, 'CONFIG_BALANCED')
            print(f"\nDetectando semáforos con configuración: {configuracion}")
            
            try:
                resultado = self.detector_semaforos.detectar_semaforos(
                    self.imagen_actual, configuracion=configuracion, visualizar=True,
                    guardar=True, ruta_salida=os.path.join(self.directorio_resultados, "semaforos")
                )
                
                if resultado:
                    print(f"\nDetección completada:")
                    print(f"   Configuración: {resultado['configuracion']}")
                    print(f"   Semáforos detectados: {resultado['num_semaforos']}")
                    print(f"   Candidatos iniciales: {resultado['candidatos_iniciales']}")
                    print(f"   Confianza promedio: {resultado['confianza_promedio']:.3f}")
                    
                    # Mostrar detalles de semáforos detectados
                    if resultado.get('semaforos_detectados'):
                        print("   Detalles de detección:")
                        for i, semaforo in enumerate(resultado['semaforos_detectados'], 1):
                            print(f"     Semáforo {i}: Centro({semaforo[0]}, {semaforo[1]}), Tamaño({semaforo[2]}x{semaforo[3]})")
                            if len(semaforo) > 4:
                                print(f"       Color: {semaforo[4]}, Confianza: {semaforo[5]:.3f}")
                else:
                    print("Error en la detección")
                    
            except Exception as e:
                print(f"Error durante la detección: {e}")
    
    def deteccion_completa(self):
        """Ejecuta detección de todos los tipos de objetos."""
        if not self.verificar_imagen_cargada():
            return
        
        print("\nDETECCIÓN COMPLETA DE OBJETOS")
        print("Se ejecutarán todos los detectores con métodos combinados...")
        
        resultados = {}
        
        # Detectar llantas
        print("\n1️Detectando llantas...")
        try:
            resultado_llantas = self.detector_llantas.detectar_llantas(
                self.imagen_actual, metodo='combinado', visualizar=False, guardar=True,
                ruta_salida=os.path.join(self.directorio_resultados, "llantas")
            )
            resultados['llantas'] = resultado_llantas
        except Exception as e:
            print(f"Error detectando llantas: {e}")
        
        # Detectar señales
        print("Detectando señales...")
        try:
            resultado_senales = self.detector_senales.detectar_senales(
                self.imagen_actual, metodo='combinado', visualizar=False, guardar=True,
                ruta_salida=os.path.join(self.directorio_resultados, "senales")
            )
            resultados['senales'] = resultado_senales
        except Exception as e:
            print(f"Error detectando señales: {e}")
        
        # Detectar semáforos
        print("Detectando semáforos...")
        try:
            resultado_semaforos = self.detector_semaforos.detectar_semaforos(
                self.imagen_actual, metodo='combinado', visualizar=False, guardar=True,
                ruta_salida=os.path.join(self.directorio_resultados, "semaforos")
            )
            resultados['semaforos'] = resultado_semaforos
        except Exception as e:
            print(f"Error detectando semáforos: {e}")
        
        # Mostrar resumen
        print("\nRESUMEN DE DETECCIÓN COMPLETA")
        print("="*50)
        
        total_objetos = 0
        for tipo, resultado in resultados.items():
            if resultado:
                if tipo == 'llantas':
                    num = resultado.get('num_llantas', 0)
                elif tipo == 'senales':
                    num = resultado.get('num_senales', 0)
                elif tipo == 'semaforos':
                    num = resultado.get('num_semaforos', 0)
                else:
                    num = 0
                
                total_objetos += num
                confianza = resultado.get('confianza_promedio', 0)
                print(f"🔹 {tipo.capitalize()}: {num} objetos (confianza: {confianza:.3f})")
        
        print(f"\nTotal de objetos detectados: {total_objetos}")
        
        # Crear imagen combinada
        if resultados:
            self.crear_imagen_combinada(resultados)
    
    def crear_imagen_combinada(self, resultados):
        """Crea una imagen con todas las detecciones combinadas."""
        imagen_resultado = self.imagen_actual.copy()
        
        # Dibujar llantas (azul)
        if 'llantas' in resultados and resultados['llantas']:
            for llanta in resultados['llantas'].get('llantas', []):
                if len(llanta) >= 3:
                    x, y, r = llanta[:3]
                    cv2.circle(imagen_resultado, (int(x), int(y)), int(r), (255, 0, 0), 2)
                    cv2.putText(imagen_resultado, "Llanta", (int(x-30), int(y-r-10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Dibujar señales (rojo)
        if 'senales' in resultados and resultados['senales']:
            for senal in resultados['senales'].get('senales', []):
                if len(senal) >= 3:
                    x, y, r = senal[:3]
                    tipo = senal[3] if len(senal) > 3 else 'Señal'
                    cv2.circle(imagen_resultado, (int(x), int(y)), int(r), (0, 0, 255), 2)
                    cv2.putText(imagen_resultado, tipo, (int(x-30), int(y-r-10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Dibujar semáforos (verde)
        if 'semaforos' in resultados and resultados['semaforos']:
            for i, semaforo in enumerate(resultados['semaforos'].get('semaforos', [])):
                if 'bbox' in semaforo:
                    x1, y1, x2, y2 = semaforo['bbox']
                    cv2.rectangle(imagen_resultado, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(imagen_resultado, f"Semaforo {i+1}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Guardar imagen combinada
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_imagen = os.path.splitext(os.path.basename(self.ruta_imagen_actual))[0]
        ruta_salida = os.path.join(self.directorio_resultados, 
                                 f"{nombre_imagen}_completo_{timestamp}.jpg")
        
        self.asegurar_directorio_existe(ruta_salida)
        cv2.imwrite(ruta_salida, imagen_resultado)
        print(f"Imagen combinada guardada: {ruta_salida}")
        
        # Mostrar imagen
        ImageHandler.mostrar_imagen(imagen_resultado, "Detección Completa")
    
    def procesamiento_lotes(self):
        """Ejecuta procesamiento por lotes."""
        print("\nPROCESAMIENTO POR LOTES")
        print("="*70)
        print("Análisis automatizado para múltiples imágenes")
        print("="*70)
        
        # Mostrar menú de opciones
        print("\nTIPOS DE ANÁLISIS DISPONIBLES:")
        print("1. Detección de Objetos (Llantas, Señales, Semáforos)")
        print("2. Análisis de Texturas")
        print("3. Momentos Geométricos")
        print("4. Transformada de Hough")
        print("5. HOG")
        print("6. KAZE")
        print("7. SURF")
        print("8. ORB")
        print("9. FREAK")
        print("10. AKAZE")
        print("11. GrabCut Segmentación")
        print("12. Laplaciano de Gauss (LoG)")
        print("=" * 70)
        print("ANÁLISIS COMBINADOS:")
        print("50. HOG + KAZE")
        print("51. SURF + ORB")
        print("=" * 70)
        print("ANÁLISIS MASIVOS:")
        print("99. TODOS los métodos de características (2-12)")
        print("0. Volver al menú principal")
        print("=" * 70)
        
        seleccion = input("Seleccione tipo de análisis: ").strip()
        
        if seleccion == '0':
            return
        elif seleccion == '1':
            self._procesamiento_deteccion_objetos()
        elif seleccion == '2':
            self._procesamiento_texturas()
        elif seleccion == '3':
            self._procesamiento_momentos()
        elif seleccion == '4':
            self._procesamiento_hough()
        elif seleccion == '5':
            self._procesamiento_hog()
        elif seleccion == '6':
            self._procesamiento_kaze()
        elif seleccion == '7':
            self._procesamiento_surf()
        elif seleccion == '8':
            self._procesamiento_orb()
        elif seleccion == '9':
            self._procesamiento_freak()
        elif seleccion == '10':
            self._procesamiento_akaze()
        elif seleccion == '11':
            self._procesamiento_grabcut()
        elif seleccion == '12':
            self._procesamiento_log()
        elif seleccion == '50':
            self._procesamiento_hog_kaze()
        elif seleccion == '51':
            self._procesamiento_surf_orb()
        elif seleccion == '99':
            self._procesamiento_todos_metodos()
        else:
            print("Opción no válida")
            
    def _procesamiento_deteccion_objetos(self):
        """Procesamiento por lotes para detección de objetos."""
        print("\nPROCESAMIENTO: DETECCIÓN DE OBJETOS")
        print("="*50)
        
        # Seleccionar carpeta
        carpeta = input(f"Carpeta de imágenes (Enter para '{self.directorio_imagenes}'): ").strip()
        if not carpeta:
            carpeta = self.directorio_imagenes
        
        if not os.path.exists(carpeta):
            print(f"La carpeta no existe: {carpeta}")
            return
        
        # Seleccionar tipos de detección
        print("\nTipos de detección:")
        print("1. Solo llantas")
        print("2. Solo señales")
        print("3. Solo semáforos")
        print("4. Llantas + Señales")
        print("5. Todos los objetos")
        
        seleccion = input("Seleccione opción (5 por defecto): ").strip()
        
        tipos_map = {
            '1': ['llantas'],
            '2': ['senales'],
            '3': ['semaforos'],
            '4': ['llantas', 'senales'],
            '5': ['llantas', 'senales', 'semaforos']
        }
        
        tipos_deteccion = tipos_map.get(seleccion, ['llantas', 'senales', 'semaforos'])
        
        # Seleccionar modo de procesamiento
        print("\nModo de procesamiento:")
        print("1. Método combinado por tipo (Rápido)")
        print("2. TODOS los métodos por tipo (Análisis completo)")
        
        modo = input("Seleccione modo (1 por defecto): ").strip()
        usar_todos_metodos = modo == '2'
        
        # Confirmar procesamiento
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
        
        print(f"\nResumen del procesamiento:")
        print(f"   Carpeta: {carpeta}")
        print(f"   Imágenes encontradas: {len(imagenes)}")
        print(f"   Tipos de detección: {', '.join(tipos_deteccion)}")
        print(f"   Modo: {'TODOS los métodos' if usar_todos_metodos else 'Método combinado'}")
        print(f"   Directorio de salida: {self.directorio_resultados}")
        
        confirmar = input("\n❓ ¿Continuar con el procesamiento? (s/N): ").strip().lower()
        if confirmar not in ['s', 'si', 'sí', 'y', 'yes']:
            print("Procesamiento cancelado")
            return
        
        # Ejecutar procesamiento
        try:
            if usar_todos_metodos:
                # Usar el nuevo procesamiento con todos los métodos
                print("Iniciando procesamiento con TODOS los métodos...")
                estadisticas = self.procesador_lotes.procesar_carpeta_todos_metodos(
                    carpeta, tipos_deteccion=tipos_deteccion,
                    guardar_imagenes=True, generar_reporte=True
                )
            else:
                # Usar el procesamiento tradicional
                print("Iniciando procesamiento con método combinado...")
                estadisticas = self.procesador_lotes.procesar_carpeta(
                    carpeta, tipos_deteccion=tipos_deteccion,
                    guardar_imagenes=True, generar_reporte=True
                )
            
            if estadisticas:
                print("\n✅ Procesamiento por lotes completado exitosamente!")
            
        except Exception as e:
            print(f"❌ Error en procesamiento por lotes: {e}")
    
    def _procesamiento_texturas(self):
        """Procesamiento por lotes para análisis de texturas."""
        print("\nPROCESAMIENTO: ANÁLISIS DE TEXTURAS")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("Análisis de Texturas", len(imagenes), carpeta):
            return
            
        print("🔍 Iniciando análisis de texturas en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\n📊 Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"❌ Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis completo de texturas con guardado de imagen
                resultado = self.texture_analyzer.procesar_imagen_completa(
                    ruta_imagen,  # Pasar la ruta directamente
                    nombre_imagen=f"batch_{nombre_imagen}"
                )
                
                # Guardar imagen procesada adicional si es necesario
                if resultado and 'imagen_procesada' in resultado:
                    output_path = os.path.join(self.texture_analyzer.results_dir, f"batch_{nombre_imagen}_texturas.png")
                    cv2.imwrite(output_path, resultado['imagen_procesada'])
                    print(f"💾 Imagen guardada: {output_path}")
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    print(f"✅ {nombre_imagen} completado")
                    
            except Exception as e:
                print(f"❌ Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("texturas", resultados_totales, carpeta)
        print(f"\n🎯 Análisis de texturas completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_momentos(self):
        """Procesamiento por lotes para momentos geométricos."""
        print("\nPROCESAMIENTO: MOMENTOS GEOMÉTRICOS")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("Momentos Geométricos", len(imagenes), carpeta):
            return
            
        print("🔍 Iniciando análisis de momentos geométricos en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\n📊 Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"❌ Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis de momentos geométricos
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
                
                # Calcular momentos directamente
                momentos = cv2.moments(gray)
                momentos_hu = cv2.HuMoments(momentos).flatten()
                
                # Crear imagen con visualización de momentos
                imagen_momentos = imagen.copy()
                
                # Calcular centro de masa
                if momentos['m00'] != 0:
                    cx = int(momentos['m10'] / momentos['m00'])
                    cy = int(momentos['m01'] / momentos['m00'])
                    cv2.circle(imagen_momentos, (cx, cy), 10, (0, 255, 0), -1)
                    cv2.putText(imagen_momentos, f"Centro: ({cx},{cy})", (cx+15, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Guardar imagen procesada
                output_path = os.path.join(self.directorio_resultados, "texture_analysis", f"batch_{nombre_imagen}_momentos.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, imagen_momentos)
                print(f"Imagen guardada: {output_path}")
                
                resultado = {
                    'momentos_espaciales': momentos,
                    'momentos_hu': momentos_hu,
                    'centro_x': cx if momentos['m00'] != 0 else 0,
                    'centro_y': cy if momentos['m00'] != 0 else 0,
                    'imagen_path': ruta_imagen,
                    'imagen_nombre': nombre_imagen,
                    'output_path': output_path
                }
                resultados_totales.append(resultado)
                print(f"{nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("momentos", resultados_totales, carpeta)
        print(f"\nAnálisis de momentos completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_hough(self):
        """Procesamiento por lotes para Transformada de Hough."""
        print("\nPROCESAMIENTO: TRANSFORMADA DE HOUGH")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("Transformada de Hough", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis de Hough en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\nProcesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis completo de Hough
                resultado = self.hough_analyzer.analisis_completo(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}"
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    print(f"{nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("hough", resultados_totales, carpeta)
        print(f"\nAnálisis de Hough completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_hog(self):
        """Procesamiento por lotes para HOG únicamente."""
        print("\nPROCESAMIENTO: HOG (Histogram of Oriented Gradients)")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("HOG", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis HOG en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\nProcesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis HOG sin visualización para lotes
                resultado = self.hog_kaze_analyzer.extraer_caracteristicas_hog(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,  # Sin mostrar en consola para lotes
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}_hog"
                )
                
                # Crear la misma visualización que el método individual
                import matplotlib.pyplot as plt
                from skimage.feature import hog
                
                # Convertir a escala de grises y normalizar como en el método original
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
                imagen_norm = exposure.equalize_hist(gray)
                
                # Extraer HOG con visualización
                _, hog_image = hog(imagen_norm,
                                 orientations=9,
                                 pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2),
                                 block_norm='L2-Hys',
                                 transform_sqrt=True,
                                 visualize=True,
                                 feature_vector=True)
                
                # Crear la visualización combinada (3 imágenes: original, normalizada, HOG)
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Imagen original (en escala de grises)
                axes[0].imshow(gray, cmap='gray')
                axes[0].set_title('Imagen Original')
                axes[0].axis('off')
                
                # Imagen normalizada
                axes[1].imshow(imagen_norm, cmap='gray')
                axes[1].set_title('Imagen Normalizada')
                axes[1].axis('off')
                
                # Visualización HOG
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                axes[2].imshow(hog_image_rescaled, cmap='hot')
                num_features = resultado.get('num_features', 0) if resultado else 0
                axes[2].set_title(f'HOG Features ({num_features} características)')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Guardar imagen procesada con características HOG
                output_path = os.path.join(self.directorio_resultados, "hog_kaze_analysis", f"batch_{nombre_imagen}_hog.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()  # Cerrar para evitar acumulación de memoria
                print(f"Imagen guardada: {output_path}")
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultado['output_path'] = output_path
                    resultados_totales.append(resultado)
                    print(f"{nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("hog", resultados_totales, carpeta)
        print(f"\nAnálisis HOG completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_kaze(self):
        """Procesamiento por lotes para KAZE únicamente."""
        print("\nPROCESAMIENTO: KAZE")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("KAZE", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis KAZE en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\n📊 Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis KAZE
                resultado = self.hog_kaze_analyzer.extraer_caracteristicas_kaze(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}_kaze"
                )
                
                # Crear imagen con puntos clave KAZE
                imagen_procesada = imagen.copy()
                
                if resultado and 'keypoints' in resultado:
                    keypoints = resultado['keypoints']
                    if keypoints:
                        imagen_procesada = cv2.drawKeypoints(imagen_procesada, keypoints, None, 
                                                           color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Guardar imagen procesada
                output_path = os.path.join(self.directorio_resultados, "hog_kaze_analysis", f"batch_{nombre_imagen}_kaze.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, imagen_procesada)
                print(f"Imagen guardada: {output_path}")
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultado['output_path'] = output_path
                    resultados_totales.append(resultado)
                    print(f"✅ {nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("kaze", resultados_totales, carpeta)
        print(f"\nAnálisis KAZE completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_surf(self):
        """Procesamiento por lotes para SURF únicamente."""
        print("\nPROCESAMIENTO: SURF (Speeded-Up Robust Features)")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("SURF", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis SURF en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\nProcesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis SURF
                resultado = self.surf_orb_analyzer.extraer_caracteristicas_surf(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}_surf"
                )
                
                # Crear imagen con puntos clave SURF
                imagen_procesada = imagen.copy()
                
                if resultado and 'keypoints' in resultado:
                    keypoints = resultado['keypoints']
                    if keypoints:
                        imagen_procesada = cv2.drawKeypoints(imagen_procesada, keypoints, None, 
                                                           color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Guardar imagen procesada
                output_path = os.path.join(self.directorio_resultados, "surf_orb_analysis", f"batch_{nombre_imagen}_surf.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, imagen_procesada)
                print(f"Imagen guardada: {output_path}")
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultado['output_path'] = output_path
                    resultados_totales.append(resultado)
                    print(f"{nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("surf", resultados_totales, carpeta)
        print(f"\nAnálisis SURF completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_orb(self):
        """Procesamiento por lotes para ORB únicamente."""
        print("\nPROCESAMIENTO: ORB (Oriented FAST and Rotated BRIEF)")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("ORB", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis ORB en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\nProcesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis ORB
                resultado = self.surf_orb_analyzer.extraer_caracteristicas_orb(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}_orb"
                )
                
                # Crear imagen con puntos clave ORB
                imagen_procesada = imagen.copy()
                
                if resultado and 'keypoints' in resultado:
                    keypoints = resultado['keypoints']
                    if keypoints:
                        imagen_procesada = cv2.drawKeypoints(imagen_procesada, keypoints, None, 
                                                           color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Guardar imagen procesada
                output_path = os.path.join(self.directorio_resultados, "surf_orb_analysis", f"batch_{nombre_imagen}_orb.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, imagen_procesada)
                print(f"Imagen guardada: {output_path}")
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultado['output_path'] = output_path
                    resultados_totales.append(resultado)
                    print(f"{nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("orb", resultados_totales, carpeta)
        print(f"\nAnálisis ORB completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")

    def _procesamiento_hog_kaze(self):
        """Procesamiento por lotes para HOG + KAZE."""
        print("\nPROCESAMIENTO: HOG + KAZE")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("HOG + KAZE", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis HOG + KAZE en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\nProcesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis HOG
                resultado_hog = self.hog_kaze_analyzer.extraer_caracteristicas_hog(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}_hog"
                )
                
                # Análisis KAZE
                resultado_kaze = self.hog_kaze_analyzer.extraer_caracteristicas_kaze(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}_kaze"
                )
                
                # Crear imagen combinada con HOG y puntos clave KAZE
                imagen_procesada = imagen.copy()
                
                # Si hay puntos clave KAZE, dibujarlos
                if resultado_kaze and 'keypoints' in resultado_kaze:
                    keypoints = resultado_kaze['keypoints']
                    if keypoints:
                        imagen_procesada = cv2.drawKeypoints(imagen_procesada, keypoints, None, 
                                                           color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Guardar imagen procesada
                output_path = os.path.join(self.directorio_resultados, "hog_kaze_analysis", f"batch_{nombre_imagen}_hog_kaze.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, imagen_procesada)
                print(f"Imagen guardada: {output_path}")
                
                if resultado_hog and resultado_kaze:
                    resultado_combinado = {
                        'hog': resultado_hog,
                        'kaze': resultado_kaze,
                        'imagen_path': ruta_imagen,
                        'imagen_nombre': nombre_imagen,
                        'output_path': output_path
                    }
                    resultados_totales.append(resultado_combinado)
                    print(f"{nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("hog_kaze", resultados_totales, carpeta)
        print(f"\nAnálisis HOG + KAZE completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_surf_orb(self):
        """Procesamiento por lotes para SURF + ORB."""
        print("\nPROCESAMIENTO: SURF + ORB")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("SURF + ORB", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis SURF + ORB en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\nProcesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis SURF
                resultado_surf = self.surf_orb_analyzer.extraer_caracteristicas_surf(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}_surf"
                )
                
                # Análisis ORB
                resultado_orb = self.surf_orb_analyzer.extraer_caracteristicas_orb(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}_orb"
                )
                
                # Crear imagen combinada con puntos clave SURF y ORB
                imagen_procesada = imagen.copy()
                
                # Dibujar puntos clave SURF en verde
                if resultado_surf and 'keypoints' in resultado_surf:
                    keypoints_surf = resultado_surf['keypoints']
                    if keypoints_surf:
                        imagen_procesada = cv2.drawKeypoints(imagen_procesada, keypoints_surf, None, 
                                                           color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Dibujar puntos clave ORB en azul
                if resultado_orb and 'keypoints' in resultado_orb:
                    keypoints_orb = resultado_orb['keypoints']
                    if keypoints_orb:
                        imagen_procesada = cv2.drawKeypoints(imagen_procesada, keypoints_orb, imagen_procesada, 
                                                           color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Guardar imagen procesada
                output_path = os.path.join(self.directorio_resultados, "surf_orb_analysis", f"batch_{nombre_imagen}_surf_orb.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, imagen_procesada)
                print(f"Imagen guardada: {output_path}")
                
                if resultado_surf and resultado_orb:
                    resultado_combinado = {
                        'surf': resultado_surf,
                        'orb': resultado_orb,
                        'imagen_path': ruta_imagen,
                        'imagen_nombre': nombre_imagen,
                        'output_path': output_path
                    }
                    resultados_totales.append(resultado_combinado)
                    print(f"{nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("surf_orb", resultados_totales, carpeta)
        print(f"\nAnálisis SURF + ORB completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_freak(self):
        """Procesamiento por lotes para FREAK."""
        print("\nPROCESAMIENTO: FREAK")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("FREAK", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis FREAK en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\nProcesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis FREAK
                resultado = self.advanced_analyzer.extraer_caracteristicas_freak(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}"
                )
                
                # Crear imagen con puntos clave FREAK si están disponibles
                imagen_procesada = imagen.copy()
                
                if resultado and 'keypoints' in resultado:
                    keypoints = resultado['keypoints']
                    if keypoints:
                        imagen_procesada = cv2.drawKeypoints(imagen_procesada, keypoints, None, 
                                                           color=(255, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Guardar imagen procesada
                output_path = os.path.join(self.directorio_resultados, "advanced_analysis", f"batch_{nombre_imagen}_freak.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, imagen_procesada)
                print(f"Imagen guardada: {output_path}")
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultado['output_path'] = output_path
                    resultados_totales.append(resultado)
                    print(f"{nombre_imagen} completado")
                    
            except Exception as e:
                print(f"❌ Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("freak", resultados_totales, carpeta)
        print(f"\nAnálisis FREAK completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_akaze(self):
        """Procesamiento por lotes para AKAZE."""
        print("\nPROCESAMIENTO: AKAZE")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("AKAZE", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis AKAZE en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\nProcesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis AKAZE
                resultado = self.advanced_analyzer.extraer_caracteristicas_akaze(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}"
                )
                
                # Crear imagen con puntos clave AKAZE si están disponibles
                imagen_procesada = imagen.copy()
                
                if resultado and 'keypoints' in resultado:
                    keypoints = resultado['keypoints']
                    if keypoints:
                        imagen_procesada = cv2.drawKeypoints(imagen_procesada, keypoints, None, 
                                                           color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Guardar imagen procesada
                output_path = os.path.join(self.directorio_resultados, "advanced_analysis", f"batch_{nombre_imagen}_akaze.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, imagen_procesada)
                print(f"Imagen guardada: {output_path}")
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultado['output_path'] = output_path
                    resultados_totales.append(resultado)
                    print(f"{nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("akaze", resultados_totales, carpeta)
        print(f"\nAnálisis AKAZE completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_grabcut(self):
        """Procesamiento por lotes para GrabCut."""
        print("\nPROCESAMIENTO: GRABCUT SEGMENTACIÓN")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("GrabCut Segmentación", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis GrabCut en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\nProcesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis GrabCut
                resultado = self.advanced_analyzer.analizar_grabcut_segmentation(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}"
                )
                
                # Guardar imagen procesada (GrabCut genera su propia imagen segmentada)
                output_path = os.path.join(self.directorio_resultados, "advanced_analysis", f"batch_{nombre_imagen}_grabcut.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # GrabCut produce imagen segmentada, la guardamos
                if resultado and 'mask' in resultado:
                    # Crear imagen de resultado basada en la máscara
                    mask = resultado['mask']
                    imagen_segmentada = imagen.copy()
                    imagen_segmentada[mask == 0] = [0, 0, 0]  # Fondo negro
                    cv2.imwrite(output_path, imagen_segmentada)
                    print(f"Imagen guardada: {output_path}")
                else:
                    # Si no hay máscara, guardar imagen original
                    cv2.imwrite(output_path, imagen)
                    print(f"Imagen guardada: {output_path}")
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultado['output_path'] = output_path
                    resultados_totales.append(resultado)
                    print(f"✅ {nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("grabcut", resultados_totales, carpeta)
        print(f"\nAnálisis GrabCut completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_log(self):
        """Procesamiento por lotes para Laplaciano de Gauss."""
        print("\nPROCESAMIENTO: LAPLACIANO DE GAUSS (LoG)")
        print("="*50)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        if not self._confirmar_procesamiento("Laplaciano de Gauss", len(imagenes), carpeta):
            return
            
        print("Iniciando análisis LoG en lote...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"\nProcesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error cargando imagen: {ruta_imagen}")
                    continue
                    
                # Análisis LoG
                resultado = self.advanced_analyzer.analizar_log_detector(
                    imagen, 
                    visualizar=False,  # Sin visualización para procesamiento por lotes
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}"
                )
                
                # Crear imagen con detección LoG
                imagen_procesada = imagen.copy()
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
                
                # Aplicar Laplaciano de Gauss
                log_filtered = cv2.Laplacian(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_64F)
                log_filtered = np.uint8(np.absolute(log_filtered))
                
                # Convertir a color para visualización
                log_colored = cv2.applyColorMap(log_filtered, cv2.COLORMAP_JET)
                
                # Guardar imagen procesada
                output_path = os.path.join(self.directorio_resultados, "advanced_analysis", f"batch_{nombre_imagen}_log.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, log_colored)
                print(f"Imagen guardada: {output_path}")
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultado['output_path'] = output_path
                    resultados_totales.append(resultado)
                    print(f"{nombre_imagen} completado")
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("log", resultados_totales, carpeta)
        print(f"\nAnálisis LoG completado: {len(resultados_totales)}/{len(imagenes)} imágenes procesadas")
    
    def _procesamiento_todos_metodos(self):
        """Procesamiento por lotes con TODOS los métodos de características."""
        print("\nPROCESAMIENTO: TODOS LOS MÉTODOS DE CARACTERÍSTICAS")
        print("="*70)
        print("Ejecutará: Texturas, Momentos, Hough, HOG, KAZE, SURF, ORB, FREAK, AKAZE, GrabCut, LoG")
        print("="*70)
        
        carpeta = self._seleccionar_carpeta()
        if not carpeta:
            return
            
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta)
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta}")
            return
            
        print(f"\nADVERTENCIA: Análisis masivo de {len(imagenes)} imágenes con 11 métodos diferentes")
        print(f"   Tiempo estimado: {len(imagenes) * 3} - {len(imagenes) * 7} minutos")
        print(f"   Se generará UN archivo consolidado (CSV + TXT) por cada método")
        
        confirmar = input("\n❓ ¿Continuar con el procesamiento masivo? (s/N): ").strip().lower()
        if confirmar not in ['s', 'si', 'sí', 'y', 'yes']:
            return
            
        print("Iniciando análisis masivo...")
        
        # Guardar carpeta original para restaurar después
        carpeta_original = self.directorio_imagenes
        self.directorio_imagenes = carpeta
        
        metodos_ejecutados = []
        
        try:
            # Ejecutar cada método de procesamiento individualmente
            metodos = [
                ("Texturas", self._procesamiento_texturas_automatico),
                ("Momentos", self._procesamiento_momentos_automatico),
                ("Hough", self._procesamiento_hough_automatico),
                ("HOG", self._procesamiento_hog_automatico),
                ("KAZE", self._procesamiento_kaze_automatico),
                ("SURF", self._procesamiento_surf_automatico),
                ("ORB", self._procesamiento_orb_automatico),
                ("FREAK", self._procesamiento_freak_automatico),
                ("AKAZE", self._procesamiento_akaze_automatico),
                ("GrabCut", self._procesamiento_grabcut_automatico),
                ("LoG", self._procesamiento_log_automatico)
            ]
            
            for nombre_metodo, metodo in metodos:
                print(f"\n{'='*60}")
                print(f"EJECUTANDO: {nombre_metodo}")
                print(f"{'='*60}")
                
                try:
                    metodo()  # Ejecutar el método individual
                    metodos_ejecutados.append(nombre_metodo)
                    print(f"{nombre_metodo} completado")
                except Exception as e:
                    print(f"Error en {nombre_metodo}: {e}")
        
        finally:
            # Restaurar carpeta original
            self.directorio_imagenes = carpeta_original
        
        print(f"\n🎯 ANÁLISIS MASIVO COMPLETADO")
        print(f"Métodos ejecutados: {len(metodos_ejecutados)}/9")
        print(f"Métodos completados: {', '.join(metodos_ejecutados)}")
        print(f"Resultados guardados en: {self.directorio_resultados}")

    # Métodos automáticos para procesamiento masivo (sin interacción del usuario)
    def _procesamiento_texturas_automatico(self):
        """Procesamiento automático de texturas sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            print(f"No se encontraron imágenes en: {self.directorio_imagenes}")
            return
            
        print(f"Iniciando análisis de texturas en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                # Usar el método correcto del texture_analyzer
                resultado = self.texture_analyzer.procesar_imagen_completa(
                    ruta_imagen, 
                    nombre_imagen
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("texturas", resultados_totales, self.directorio_imagenes)

    def _procesamiento_momentos_automatico(self):
        """Procesamiento automático de momentos sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de momentos en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    continue
                    
                # Calcular momentos geométricos directamente
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                momentos = cv2.moments(gray)
                
                # Calcular momentos de Hu
                momentos_hu = cv2.HuMoments(momentos)
                
                resultado = {
                    'imagen_path': ruta_imagen,
                    'imagen_nombre': nombre_imagen,
                    'momentos_espaciales': momentos,
                    'momentos_hu': momentos_hu.flatten()
                }
                resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("momentos", resultados_totales, self.directorio_imagenes)

    def _procesamiento_hough_automatico(self):
        """Procesamiento automático de Hough sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de Hough en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                # Usar el método correcto del hough_analyzer
                resultado = self.hough_analyzer.analisis_completo_hough(
                    ruta_imagen, 
                    nombre_imagen
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"❌ Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("hough", resultados_totales, self.directorio_imagenes)

    def _procesamiento_hog_automatico(self):
        """Procesamiento automático de HOG sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de HOG en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    continue
                    
                resultado = self.hog_kaze_analyzer.extraer_caracteristicas_hog(
                    imagen, 
                    visualizar=False,
                    mostrar_descriptores=False,
                    guardar_resultados=False,
                    nombre_imagen=f"auto_{nombre_imagen}_hog"
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("hog", resultados_totales, self.directorio_imagenes)

    def _procesamiento_kaze_automatico(self):
        """Procesamiento automático de KAZE sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de KAZE en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    continue
                    
                resultado = self.hog_kaze_analyzer.extraer_caracteristicas_kaze(
                    imagen, 
                    visualizar=False,
                    mostrar_descriptores=False,
                    guardar_resultados=False,
                    nombre_imagen=f"auto_{nombre_imagen}_kaze"
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("kaze", resultados_totales, self.directorio_imagenes)

    def _procesamiento_surf_automatico(self):
        """Procesamiento automático de SURF sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de SURF en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    continue
                    
                resultado = self.surf_orb_analyzer.extraer_caracteristicas_surf(
                    imagen, 
                    visualizar=False,
                    mostrar_descriptores=False,
                    guardar_resultados=False,
                    nombre_imagen=f"auto_{nombre_imagen}_surf"
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("surf", resultados_totales, self.directorio_imagenes)

    def _procesamiento_orb_automatico(self):
        """Procesamiento automático de ORB sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de ORB en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    continue
                    
                resultado = self.surf_orb_analyzer.extraer_caracteristicas_orb(
                    imagen, 
                    visualizar=False,
                    mostrar_descriptores=False,
                    guardar_resultados=False,
                    nombre_imagen=f"auto_{nombre_imagen}_orb"
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("orb", resultados_totales, self.directorio_imagenes)

    def _procesamiento_hog_kaze_automatico(self):
        """Procesamiento automático de HOG+KAZE sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de HOG+KAZE en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                # Usar el método correcto del hog_kaze_analyzer
                resultado = self.hog_kaze_analyzer.analisis_combinado_hog_kaze(
                    ruta_imagen
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("hog_kaze", resultados_totales, self.directorio_imagenes)

    def _procesamiento_surf_orb_automatico(self):
        """Procesamiento automático de SURF+ORB sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de SURF+ORB en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                # Usar el método correcto del surf_orb_analyzer
                resultado = self.surf_orb_analyzer.analisis_combinado_surf_orb(
                    ruta_imagen
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("surf_orb", resultados_totales, self.directorio_imagenes)

    def _procesamiento_freak_automatico(self):
        """Procesamiento automático de FREAK sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de FREAK en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    continue
                    
                # Usar el método correcto del advanced_analyzer
                resultado = self.advanced_analyzer.extraer_caracteristicas_freak(
                    imagen, 
                    visualizar=False,
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}"
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("freak", resultados_totales, self.directorio_imagenes)

    def _procesamiento_akaze_automatico(self):
        """Procesamiento automático de AKAZE sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de AKAZE en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"📊 Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    continue
                    
                # Usar el método correcto del advanced_analyzer
                resultado = self.advanced_analyzer.extraer_caracteristicas_akaze(
                    imagen, 
                    visualizar=False,
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}"
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"❌ Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("akaze", resultados_totales, self.directorio_imagenes)

    def _procesamiento_grabcut_automatico(self):
        """Procesamiento automático de GrabCut sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de GrabCut en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    continue
                    
                # Usar el método correcto del advanced_analyzer
                resultado = self.advanced_analyzer.analizar_grabcut_segmentation(
                    imagen, 
                    visualizar=False,
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}"
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("grabcut", resultados_totales, self.directorio_imagenes)

    def _procesamiento_log_automatico(self):
        """Procesamiento automático de LoG sin interacción del usuario."""
        imagenes = ImageHandler.obtener_imagenes_carpeta(self.directorio_imagenes)
        if not imagenes:
            return
            
        print(f"Iniciando análisis de LoG en {len(imagenes)} imágenes...")
        resultados_totales = []
        
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
            print(f"Procesando {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    continue
                    
                # Usar el método correcto del advanced_analyzer
                resultado = self.advanced_analyzer.analizar_log_detector(
                    imagen, 
                    visualizar=False,
                    mostrar_descriptores=False,
                    guardar_resultados=True,
                    nombre_imagen=f"batch_{nombre_imagen}"
                )
                
                if resultado:
                    resultado['imagen_path'] = ruta_imagen
                    resultado['imagen_nombre'] = nombre_imagen
                    resultados_totales.append(resultado)
                    
            except Exception as e:
                print(f"Error procesando {nombre_imagen}: {e}")
                
        self._generar_reporte_batch("log", resultados_totales, self.directorio_imagenes)

    # Métodos auxiliares para procesamiento por lotes
    def _seleccionar_carpeta(self):
        """Selecciona la carpeta para procesamiento."""
        carpeta = input(f"Carpeta de imágenes (Enter para '{self.directorio_imagenes}'): ").strip()
        if not carpeta:
            carpeta = self.directorio_imagenes
        
        if not os.path.exists(carpeta):
            print(f"La carpeta no existe: {carpeta}")
            return None
            
        return carpeta
    
    def _confirmar_procesamiento(self, tipo_analisis, num_imagenes, carpeta):
        """Confirma el procesamiento por lotes."""
        print(f"\nRESUMEN DEL PROCESAMIENTO:")
        print(f"   Tipo: {tipo_analisis}")
        print(f"   Carpeta: {carpeta}")
        print(f"   Imágenes: {num_imagenes}")
        print(f"   Salida: {self.directorio_resultados}")
        
        confirmar = input("\n¿Continuar con el procesamiento? (s/N): ").strip().lower()
        return confirmar in ['s', 'si', 'sí', 'y', 'yes']
    
    def _generar_reporte_batch(self, tipo_metodo, resultados, carpeta):
        """Genera reporte consolidado del procesamiento por lotes."""
        if not resultados:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_reporte = f"reporte_batch_{tipo_metodo}_{timestamp}"
        
        # Crear directorio de reportes batch
        batch_dir = os.path.join(self.directorio_resultados, "reportes", "batch")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Archivo CSV con estadísticas
        csv_path = os.path.join(batch_dir, f"{nombre_reporte}.csv")
        txt_path = os.path.join(batch_dir, f"{nombre_reporte}.txt")
        
        try:
            # Generar CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                if resultados:
                    primer_resultado = resultados[0]
                    
                    # Determinar headers basado en el tipo de método
                    headers = ['imagen_nombre', 'imagen_path']
                    
                    # Agregar headers específicos del método
                    if tipo_metodo == "texturas":
                        headers.extend(['media', 'desviacion', 'varianza', 'entropia', 'energia'])
                    elif tipo_metodo == "momentos":
                        headers.extend(['area', 'centroide_x', 'centroide_y', 'momento_central'])
                    elif tipo_metodo == "hough":
                        headers.extend(['num_lineas', 'num_circulos', 'lineas_strength', 'circulos_radius_avg'])
                    elif tipo_metodo in ["hog_kaze", "surf_orb"]:
                        for metodo in primer_resultado.keys():
                            if metodo not in ['imagen_path', 'imagen_nombre']:
                                headers.extend([f'{metodo}_keypoints', f'{metodo}_descriptors'])
                    else:
                        # Para métodos individuales (FREAK, AKAZE, GrabCut, LoG)
                        for key in primer_resultado.keys():
                            if key not in ['imagen_path', 'imagen_nombre', 'keypoints', 'descriptors', 'gray_image']:
                                headers.append(key)
                    
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    writer.writeheader()
                    
                    for resultado in resultados:
                        row = {
                            'imagen_nombre': resultado.get('imagen_nombre', ''),
                            'imagen_path': resultado.get('imagen_path', '')
                        }
                        
                        # Agregar datos específicos del método
                        if tipo_metodo == "texturas":
                            stats = resultado.get('statistics', {})
                            row.update({
                                'media': stats.get('mean', 0),
                                'desviacion': stats.get('std', 0),
                                'varianza': stats.get('variance', 0),
                                'entropia': stats.get('entropy', 0),
                                'energia': stats.get('energy', 0)
                            })
                        elif tipo_metodo in ["hog_kaze", "surf_orb"]:
                            for metodo in resultado.keys():
                                if metodo not in ['imagen_path', 'imagen_nombre']:
                                    data = resultado[metodo]
                                    row[f'{metodo}_keypoints'] = len(data.get('keypoints', []))
                                    row[f'{metodo}_descriptors'] = data.get('num_descriptors', 0)
                        else:
                            # Para otros métodos, agregar estadísticas disponibles
                            for key, value in resultado.items():
                                if key not in ['imagen_path', 'imagen_nombre', 'keypoints', 'descriptors', 'gray_image']:
                                    if isinstance(value, (int, float, str)):
                                        row[key] = value
                        
                        writer.writerow(row)
            
            # Generar TXT
            with open(txt_path, 'w', encoding='utf-8') as txtfile:
                txtfile.write(f"REPORTE DE PROCESAMIENTO POR LOTES\n")
                txtfile.write(f"{'='*60}\n")
                txtfile.write(f"Método: {tipo_metodo.upper()}\n")
                txtfile.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                txtfile.write(f"Carpeta procesada: {carpeta}\n")
                txtfile.write(f"Imágenes procesadas: {len(resultados)}\n")
                txtfile.write(f"{'='*60}\n\n")
                
                for i, resultado in enumerate(resultados, 1):
                    txtfile.write(f"{i}. {resultado.get('imagen_nombre', 'imagen_desconocida')}\n")
                    txtfile.write(f"   Ruta: {resultado.get('imagen_path', 'N/A')}\n")
                    
                    if tipo_metodo == "texturas":
                        stats = resultado.get('statistics', {})
                        txtfile.write(f"   Media: {stats.get('mean', 0):.4f}\n")
                        txtfile.write(f"   Desviación: {stats.get('std', 0):.4f}\n")
                        txtfile.write(f"   Entropía: {stats.get('entropy', 0):.4f}\n")
                    elif tipo_metodo in ["hog_kaze", "surf_orb"]:
                        for metodo in resultado.keys():
                            if metodo not in ['imagen_path', 'imagen_nombre']:
                                data = resultado[metodo]
                                txtfile.write(f"   {metodo.upper()}: {len(data.get('keypoints', []))} puntos clave\n")
                    else:
                        # Para otros métodos
                        for key, value in resultado.items():
                            if key not in ['imagen_path', 'imagen_nombre', 'keypoints', 'descriptors', 'gray_image']:
                                if isinstance(value, (int, float)):
                                    txtfile.write(f"   {key}: {value:.4f}\n")
                                elif isinstance(value, str):
                                    txtfile.write(f"   {key}: {value}\n")
                    
                    txtfile.write("\n")
            
            print(f"Reporte generado: {csv_path}")
            print(f"Reporte generado: {txt_path}")
            
        except Exception as e:
            print(f"Error generando reporte: {e}")
    
    def _generar_reporte_global(self, resultados_globales, carpeta):
        """Genera reporte global del análisis masivo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_reporte = f"reporte_analisis_masivo_{timestamp}"
        
        batch_dir = os.path.join(self.directorio_resultados, "reportes", "batch")
        os.makedirs(batch_dir, exist_ok=True)
        
        txt_path = os.path.join(batch_dir, f"{nombre_reporte}.txt")
        
        try:
            with open(txt_path, 'w', encoding='utf-8') as txtfile:
                txtfile.write(f"REPORTE DE ANÁLISIS MASIVO\n")
                txtfile.write(f"{'='*80}\n")
                txtfile.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                txtfile.write(f"Carpeta procesada: {carpeta}\n")
                txtfile.write(f"{'='*80}\n\n")
                
                total_imagenes = 0
                for metodo, resultados in resultados_globales.items():
                    txtfile.write(f"{metodo.upper()}\n")
                    txtfile.write(f"   Imágenes procesadas: {len(resultados)}\n")
                    txtfile.write(f"   Estado: {'✅ Completado' if resultados else '❌ Falló'}\n\n")
                    total_imagenes += len(resultados)
                
                txtfile.write(f"RESUMEN GLOBAL\n")
                txtfile.write(f"   Total de análisis exitosos: {total_imagenes}\n")
                txtfile.write(f"   Métodos ejecutados: {len(resultados_globales)}\n")
                txtfile.write(f"   Directorio de resultados: {self.directorio_resultados}\n")
            
            print(f"Reporte global generado: {txt_path}")
            
        except Exception as e:
            print(f"Error generando reporte global: {e}")
    
    # Métodos auxiliares para ejecutar análisis individuales en lote
    def _ejecutar_texturas_batch(self, imagenes):
        """Ejecuta análisis de texturas en lote."""
        resultados = []
        for ruta_imagen in imagenes:
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
                    resultado = self.texture_analyzer.analisis_completo_texturas(
                        imagen, visualizar=False, guardar_resultados=True,
                        nombre_imagen=f"batch_{nombre_imagen}"
                    )
                    if resultado:
                        resultado['imagen_path'] = ruta_imagen
                        resultado['imagen_nombre'] = nombre_imagen
                        resultados.append(resultado)
            except Exception as e:
                print(f"Error en {os.path.basename(ruta_imagen)}: {e}")
        return resultados
    
    def _ejecutar_momentos_batch(self, imagenes):
        """Ejecuta análisis de momentos en lote."""
        resultados = []
        for ruta_imagen in imagenes:
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
                    resultado = self.texture_analyzer.calcular_momentos_geometricos(
                        imagen, visualizar=False, guardar_resultados=True,
                        nombre_imagen=f"batch_{nombre_imagen}"
                    )
                    if resultado:
                        resultado['imagen_path'] = ruta_imagen
                        resultado['imagen_nombre'] = nombre_imagen
                        resultados.append(resultado)
            except Exception as e:
                print(f"Error en {os.path.basename(ruta_imagen)}: {e}")
        return resultados
    
    def _ejecutar_hough_batch(self, imagenes):
        """Ejecuta análisis de Hough en lote."""
        resultados = []
        for ruta_imagen in imagenes:
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
                    resultado = self.hough_analyzer.analisis_completo(
                        imagen, visualizar=False, guardar_resultados=True,
                        nombre_imagen=f"batch_{nombre_imagen}"
                    )
                    if resultado:
                        resultado['imagen_path'] = ruta_imagen
                        resultado['imagen_nombre'] = nombre_imagen
                        resultados.append(resultado)
            except Exception as e:
                print(f"Error en {os.path.basename(ruta_imagen)}: {e}")
        return resultados
    
    def _ejecutar_hog_kaze_batch(self, imagenes):
        """Ejecuta análisis de HOG+KAZE en lote."""
        resultados = []
        for ruta_imagen in imagenes:
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
                    
                    resultado_hog = self.hog_kaze_analyzer.extraer_caracteristicas_hog(
                        imagen, visualizar=False, mostrar_descriptores=False,
                        guardar_resultados=True, nombre_imagen=f"batch_{nombre_imagen}_hog"
                    )
                    
                    resultado_kaze = self.hog_kaze_analyzer.extraer_caracteristicas_kaze(
                        imagen, visualizar=False, mostrar_descriptores=False,
                        guardar_resultados=True, nombre_imagen=f"batch_{nombre_imagen}_kaze"
                    )
                    
                    if resultado_hog and resultado_kaze:
                        resultado_combinado = {
                            'hog': resultado_hog,
                            'kaze': resultado_kaze,
                            'imagen_path': ruta_imagen,
                            'imagen_nombre': nombre_imagen
                        }
                        resultados.append(resultado_combinado)
            except Exception as e:
                print(f"Error en {os.path.basename(ruta_imagen)}: {e}")
        return resultados
    
    def _ejecutar_surf_orb_batch(self, imagenes):
        """Ejecuta análisis de SURF+ORB en lote."""
        resultados = []
        for ruta_imagen in imagenes:
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
                    
                    resultado_surf = self.surf_orb_analyzer.extraer_caracteristicas_surf(
                        imagen, visualizar=False, mostrar_descriptores=False,
                        guardar_resultados=True, nombre_imagen=f"batch_{nombre_imagen}_surf"
                    )
                    
                    resultado_orb = self.surf_orb_analyzer.extraer_caracteristicas_orb(
                        imagen, visualizar=False, mostrar_descriptores=False,
                        guardar_resultados=True, nombre_imagen=f"batch_{nombre_imagen}_orb"
                    )
                    
                    if resultado_surf and resultado_orb:
                        resultado_combinado = {
                            'surf': resultado_surf,
                            'orb': resultado_orb,
                            'imagen_path': ruta_imagen,
                            'imagen_nombre': nombre_imagen
                        }
                        resultados.append(resultado_combinado)
            except Exception as e:
                print(f"Error en {os.path.basename(ruta_imagen)}: {e}")
        return resultados
    
    def _ejecutar_freak_batch(self, imagenes):
        """Ejecuta análisis de FREAK en lote."""
        resultados = []
        for ruta_imagen in imagenes:
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
                    resultado = self.advanced_analyzer.extraer_caracteristicas_freak(
                        imagen, visualizar=False, mostrar_descriptores=False,
                        guardar_resultados=True, nombre_imagen=f"batch_{nombre_imagen}"
                    )
                    if resultado:
                        resultado['imagen_path'] = ruta_imagen
                        resultado['imagen_nombre'] = nombre_imagen
                        resultados.append(resultado)
            except Exception as e:
                print(f"Error en {os.path.basename(ruta_imagen)}: {e}")
        return resultados
    
    def _ejecutar_akaze_batch(self, imagenes):
        """Ejecuta análisis de AKAZE en lote."""
        resultados = []
        for ruta_imagen in imagenes:
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
                    resultado = self.advanced_analyzer.extraer_caracteristicas_akaze(
                        imagen, visualizar=False, mostrar_descriptores=False,
                        guardar_resultados=True, nombre_imagen=f"batch_{nombre_imagen}"
                    )
                    if resultado:
                        resultado['imagen_path'] = ruta_imagen
                        resultado['imagen_nombre'] = nombre_imagen
                        resultados.append(resultado)
            except Exception as e:
                print(f"Error en {os.path.basename(ruta_imagen)}: {e}")
        return resultados
    
    def _ejecutar_grabcut_batch(self, imagenes):
        """Ejecuta análisis de GrabCut en lote."""
        resultados = []
        for ruta_imagen in imagenes:
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
                    resultado = self.advanced_analyzer.analizar_grabcut_segmentation(
                        imagen, visualizar=False, mostrar_descriptores=False,
                        guardar_resultados=True, nombre_imagen=f"batch_{nombre_imagen}"
                    )
                    if resultado:
                        resultado['imagen_path'] = ruta_imagen
                        resultado['imagen_nombre'] = nombre_imagen
                        resultados.append(resultado)
            except Exception as e:
                print(f"❌ Error en {os.path.basename(ruta_imagen)}: {e}")
        return resultados
    
    def _ejecutar_log_batch(self, imagenes):
        """Ejecuta análisis de LoG en lote."""
        resultados = []
        for ruta_imagen in imagenes:
            try:
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]
                    resultado = self.advanced_analyzer.analizar_log_detector(
                        imagen, visualizar=False, mostrar_descriptores=False,
                        guardar_resultados=True, nombre_imagen=f"batch_{nombre_imagen}"
                    )
                    if resultado:
                        resultado['imagen_path'] = ruta_imagen
                        resultado['imagen_nombre'] = nombre_imagen
                        resultados.append(resultado)
            except Exception as e:
                print(f"Error en {os.path.basename(ruta_imagen)}: {e}")
        return resultados
    
    def ver_estadisticas(self):
        """Muestra estadísticas y reportes."""
        print("\nESTADÍSTICAS Y REPORTES")
        print("="*50)
        
        # Verificar si hay reportes
        carpeta_reportes = os.path.join(self.directorio_resultados, "reportes")
        
        if not os.path.exists(carpeta_reportes):
            print("No se encontraron reportes")
            return
        
        # Listar reportes disponibles
        reportes = []
        for archivo in os.listdir(carpeta_reportes):
            if archivo.endswith(('.json', '.txt', '.csv')):
                reportes.append(archivo)
        
        if not reportes:
            print("No hay reportes disponibles")
            return
        
        print("Reportes disponibles:")
        for i, reporte in enumerate(reportes, 1):
            print(f"{i:2d}. {reporte}")
        
        # Mostrar estadísticas básicas de directorios
        print(f"\nEstadísticas de directorios:")
        
        for tipo in ['llantas', 'senales', 'semaforos']:
            directorio = os.path.join(self.directorio_resultados, tipo)
            if os.path.exists(directorio):
                archivos = [f for f in os.listdir(directorio) if f.endswith(('.jpg', '.png'))]
                print(f"   {tipo.capitalize()}: {len(archivos)} imágenes procesadas")
    
    def configuracion(self):
        """Muestra y permite cambiar configuraciones."""
        print("\nCONFIGURACIÓN DEL SISTEMA")
        print("="*50)
        print(f"Directorio de imágenes: {self.directorio_imagenes}")
        print(f"Directorio de resultados: {self.directorio_resultados}")
        
        if self.imagen_actual is not None:
            nombre = os.path.basename(self.ruta_imagen_actual)
            print(f"Imagen actual: {nombre}")
        else:
            print("Imagen actual: Ninguna cargada")
        
        print("\nOpciones:")
        print("1. Cambiar directorio de imágenes")
        print("2. Cambiar directorio de resultados")
        print("3. Limpiar imagen actual")
        print("0. Volver")
        
        opcion = input("\nSeleccione opción: ").strip()
        
        if opcion == '1':
            nuevo_dir = input("Nuevo directorio de imágenes: ").strip()
            if os.path.exists(nuevo_dir):
                self.directorio_imagenes = nuevo_dir
                print(f"Directorio actualizado: {nuevo_dir}")
            else:
                print("El directorio no existe")
        
        elif opcion == '2':
            nuevo_dir = input("Nuevo directorio de resultados: ").strip()
            self.directorio_resultados = nuevo_dir
            self.verificar_directorios()
            print(f"Directorio actualizado: {nuevo_dir}")
        
        elif opcion == '3':
            self.imagen_actual = None
            self.ruta_imagen_actual = None
            print("Imagen actual eliminada de memoria")

    # =========================================================================
    # NUEVOS MENÚS Y FUNCIONALIDADES
    # =========================================================================
    
    def menu_preprocesamiento(self):
        """Maneja el menú de preprocesamiento reestructurado con submódulos."""
        if self.imagen_actual is None:
            print("Primero debe cargar una imagen")
            return
        
        while True:
            self.mostrar_menu_preprocesamiento()
            opcion = input("\nSeleccione categoría de preprocesamiento: ").strip()
            
            if opcion == '1':
                self.menu_filtros()
            elif opcion == '2':
                self.menu_operaciones_aritmeticas()
            elif opcion == '3':
                self.menu_operaciones_geometricas()
            elif opcion == '4':
                self.menu_operaciones_logicas()
            elif opcion == '5':
                self.menu_operaciones_morfologicas()
            elif opcion == '6':
                self.menu_segmentacion()
            elif opcion == '7':
                self.preprocesamiento_automatico()
            elif opcion == '8':
                self.comparar_preprocesamiento()
            elif opcion == '91':
                self.mostrar_historial_preprocesamiento()
            elif opcion == '92':
                self.resetear_preprocesamiento()
            elif opcion == '93':
                self.guardar_estado_actual()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")
    
    def menu_extraccion_caracteristicas(self):
        """Maneja el menú de extracción de características."""
        if self.imagen_actual is None:
            print("Primero debe cargar una imagen")
            return
        
        while True:
            self.mostrar_menu_extraccion_caracteristicas()
            opcion = input("\nSeleccione tipo de análisis: ").strip()
            
            if opcion == '1':
                self.menu_descriptores_textura()
            elif opcion == '2':
                self.menu_deteccion_bordes()
            elif opcion == '3':
                self.menu_deteccion_formas()
            elif opcion == '4':
                self.menu_metodos_avanzados()
            elif opcion == '5':
                self.comparar_algoritmos()
            elif opcion == '6':
                self.analisis_completo_caracteristicas()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")
    
    def menu_deteccion_objetos(self):
        """Maneja el menú de detección de objetos específicos."""
        while True:
            self.mostrar_menu_deteccion_objetos()
            opcion = input("\nSeleccione tipo de detección: ").strip()
            
            if opcion == '1':
                self.detectar_llantas()
            elif opcion == '2':
                self.detectar_senales()
            elif opcion == '3':
                self.detectar_semaforos()
            elif opcion == '4':
                self.deteccion_completa()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")

    def menu_descriptores_textura(self):
        """Maneja el análisis de descriptores de textura."""
        while True:
            self.mostrar_menu_descriptores_textura()
            opcion = input("\nSeleccione análisis de textura: ").strip()
            
            if opcion == '1':
                self.estadisticas_primer_orden()
            elif opcion == '2':
                self.estadisticas_segundo_orden()
            elif opcion == '3':
                self.analisis_texturas_completo()
            elif opcion == '4':
                self.comparar_regiones_textura()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")

    def menu_deteccion_bordes(self):
        """Maneja la detección de bordes."""
        while True:
            self.mostrar_menu_deteccion_bordes()
            opcion = input("\nSeleccione método de detección: ").strip()
            
            if opcion == '1':
                self.detectar_bordes_canny()
            elif opcion == '2':
                self.detectar_bordes_sobel()
            elif opcion == '3':
                self.detectar_bordes_log()
            elif opcion == '4':
                self.analizar_gradientes()
            elif opcion == '5':
                self.comparar_metodos_bordes()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")

    def menu_deteccion_formas(self):
        """Maneja la detección de formas."""
        while True:
            self.mostrar_menu_deteccion_formas()
            opcion = input("\nSeleccione análisis de formas: ").strip()
            
            if opcion == '1':
                self.detectar_lineas_hough()
            elif opcion == '2':
                self.detectar_circulos_hough()
            elif opcion == '3':
                self.calcular_momentos_geometricos()
            elif opcion == '4':
                self.analisis_formas_completo()
            elif opcion == '5':
                self.analisis_hough_imagen_individual()
            elif opcion == '6':
                self.analisis_hough_por_lotes()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")

    def menu_metodos_avanzados(self):
        """Maneja los métodos avanzados de características."""
        while True:
            self.mostrar_menu_metodos_avanzados()
            opcion = input("\nSeleccione método avanzado: ").strip()
            
            if opcion == '1':
                self.extraer_surf()
            elif opcion == '2':
                self.extraer_orb()
            elif opcion == '3':
                self.extraer_hog()
            elif opcion == '4':
                self.extraer_kaze()
            elif opcion == '5':
                self.extraer_akaze()
            elif opcion == '6':
                self.extraer_freak()
            elif opcion == '7':
                self.segmentacion_grabcut()
            elif opcion == '8':
                self.analisis_optical_flow()
            elif opcion == '9':
                self.analisis_optical_flow_profesora()
            elif opcion == '10':
                self.analisis_secuencias_carpeta()
            elif opcion == '11':
                self.analisis_comparativo_hog_kaze()
            elif opcion == '12':
                self.analisis_avanzado_combinado()
            elif opcion == '99':
                self.ejecutar_todos_metodos_caracteristicas()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")

    # =========================================================================
    # MENÚS DE PREPROCESAMIENTO POR SUBMÓDULOS
    # =========================================================================
    
    def menu_filtros(self):
        """Menú de filtros de imagen."""
        while True:
            print("\nFILTROS DE IMAGEN")
            print("-" * 50)
            print("1. Filtro de Desenfoque (Blur)")
            print("2. Filtro Gaussiano")
            print("3. Filtro de Mediana")
            print("4. Filtro de Nitidez")
            print("5. Filtro Bilateral")
            print("6. Detección de Bordes (Canny)")
            print("7. Ecualización de Histograma")
            print("8. Filtro Sobel")
            print("9. Filtro Laplaciano")
            print("0. Volver")
            print("-" * 50)
            
            opcion = input("\nSeleccione filtro: ").strip()
            
            if opcion == '1':
                self.aplicar_filtro_desenfoque()
            elif opcion == '2':
                self.aplicar_filtro_gaussiano_nuevo()
            elif opcion == '3':
                self.aplicar_filtro_mediana()
            elif opcion == '4':
                self.aplicar_filtro_nitidez()
            elif opcion == '5':
                self.aplicar_filtro_bilateral_nuevo()
            elif opcion == '6':
                self.aplicar_canny()
            elif opcion == '7':
                self.aplicar_ecualizacion_histograma()
            elif opcion == '8':
                self.aplicar_filtro_sobel()
            elif opcion == '9':
                self.aplicar_filtro_laplaciano()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")
    
    def menu_operaciones_aritmeticas(self):
        """Menú de operaciones aritméticas."""
        while True:
            print("\nOPERACIONES ARITMÉTICAS")
            print("-" * 50)
            print("1. Suma de Imágenes")
            print("2. Resta de Imágenes")
            print("3. Multiplicación de Imágenes")
            print("4. División de Imágenes")
            print("5. Ajustar Brillo")
            print("6. Ajustar Contraste")
            print("7. Corrección Gamma")
            print("8. Mezclar Imágenes")
            print("0. Volver")
            print("-" * 50)
            
            opcion = input("\nSeleccione operación: ").strip()
            
            if opcion == '1':
                self.suma_imagenes()
            elif opcion == '2':
                self.resta_imagenes()
            elif opcion == '3':
                self.multiplicacion_imagenes()
            elif opcion == '4':
                self.division_imagenes()
            elif opcion == '5':
                self.ajustar_brillo()
            elif opcion == '6':
                self.ajustar_contraste()
            elif opcion == '7':
                self.aplicar_gamma()
            elif opcion == '8':
                self.mezclar_imagenes()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")
    
    def menu_operaciones_geometricas(self):
        """Menú de operaciones geométricas."""
        while True:
            print("\nOPERACIONES GEOMÉTRICAS")
            print("-" * 50)
            print("1. Redimensionar Imagen")
            print("2. Rotar Imagen")
            print("3. Voltear Imagen")
            print("4. Trasladar Imagen")
            print("5. Recortar Imagen")
            print("6. Transformación de Perspectiva")
            print("7. Escalar Imagen")
            print("8. Transformación Afín")
            print("9. Corrección de Distorsión")
            print("0. Volver")
            print("-" * 50)
            
            opcion = input("\nSeleccione operación: ").strip()
            
            if opcion == '1':
                self.redimensionar_imagen_nueva()
            elif opcion == '2':
                self.rotar_imagen()
            elif opcion == '3':
                self.voltear_imagen()
            elif opcion == '4':
                self.trasladar_imagen()
            elif opcion == '5':
                self.recortar_imagen()
            elif opcion == '6':
                self.transformacion_perspectiva()
            elif opcion == '7':
                self.escalar_imagen()
            elif opcion == '8':
                self.transformacion_afin()
            elif opcion == '9':
                self.corregir_distorsion()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")
    
    def menu_operaciones_logicas(self):
        """Menú de operaciones lógicas."""
        while True:
            print("\nOPERACIONES LÓGICAS")
            print("-" * 50)
            print("1. Operación AND")
            print("2. Operación OR")
            print("3. Operación NOT")
            print("4. Operación XOR")
            print("5. Crear Máscara Rectangular")
            print("6. Crear Máscara Circular")
            print("7. Aplicar Máscara")
            print("8. Segmentación por Color")
            print("9. Combinar Máscaras")
            print("0. Volver")
            print("-" * 50)
            
            opcion = input("\nSeleccione operación: ").strip()
            
            if opcion == '1':
                self.operacion_and()
            elif opcion == '2':
                self.operacion_or()
            elif opcion == '3':
                self.operacion_not()
            elif opcion == '4':
                self.operacion_xor()
            elif opcion == '5':
                self.crear_mascara_rectangular()
            elif opcion == '6':
                self.crear_mascara_circular()
            elif opcion == '7':
                self.aplicar_mascara()
            elif opcion == '8':
                self.segmentar_por_color()
            elif opcion == '9':
                self.combinar_mascaras()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")
    
    def menu_operaciones_morfologicas(self):
        """Menú de operaciones morfológicas."""
        while True:
            print("\n🧬 OPERACIONES MORFOLÓGICAS")
            print("-" * 50)
            print("1. Erosión")
            print("2. Dilatación")
            print("3. Apertura")
            print("4. Cierre")
            print("5. Gradiente Morfológico")
            print("6. Top Hat")
            print("7. Black Hat")
            print("8. Eliminar Ruido Binario")
            print("9. Extraer Contornos")
            print("10. Esqueletización")
            print("11. Rellenar Huecos")
            print("12. Limpiar Bordes")
            print("0. Volver")
            print("-" * 50)
            
            opcion = input("\nSeleccione operación: ").strip()
            
            if opcion == '1':
                self.aplicar_erosion()
            elif opcion == '2':
                self.aplicar_dilatacion()
            elif opcion == '3':
                self.aplicar_apertura()
            elif opcion == '4':
                self.aplicar_cierre()
            elif opcion == '5':
                self.aplicar_gradiente_morfologico()
            elif opcion == '6':
                self.aplicar_top_hat()
            elif opcion == '7':
                self.aplicar_black_hat()
            elif opcion == '8':
                self.eliminar_ruido_binario()
            elif opcion == '9':
                self.extraer_contornos_morfologicos()
            elif opcion == '10':
                self.aplicar_esqueletizacion()
            elif opcion == '11':
                self.rellenar_huecos()
            elif opcion == '12':
                self.limpiar_bordes()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")
    
    def menu_segmentacion(self):
        """Menú de técnicas de segmentación."""
        while True:
            print("\nTÉCNICAS DE SEGMENTACIÓN")
            print("-" * 50)
            print("1. Umbralización Simple")
            print("2. Umbralización Adaptativa")
            print("3. Umbralización Otsu")
            print("4. Detección de Bordes Canny")
            print("5. Detección de Contornos")
            print("6. Segmentación K-means")
            print("7. Segmentación Watershed")
            print("8. Segmentación por Color HSV")
            print("9. Crecimiento de Regiones")
            print("10. Segmentación por Textura")
            print("11. Segmentación GrabCut")
            print("0. Volver")
            print("-" * 50)
            
            opcion = input("\nSeleccione técnica: ").strip()
            
            if opcion == '1':
                self.aplicar_umbral_simple()
            elif opcion == '2':
                self.aplicar_umbral_adaptativo()
            elif opcion == '3':
                self.aplicar_umbral_otsu()
            elif opcion == '4':
                self.aplicar_canny_segmentacion()
            elif opcion == '5':
                self.detectar_contornos_segmentacion()
            elif opcion == '6':
                self.aplicar_kmeans()
            elif opcion == '7':
                self.aplicar_watershed()
            elif opcion == '8':
                self.segmentar_color_hsv()
            elif opcion == '9':
                self.aplicar_crecimiento_regiones()
            elif opcion == '10':
                self.segmentar_por_textura()
            elif opcion == '11':
                self.aplicar_grabcut()
            elif opcion == '0':
                break
            else:
                print("Opción no válida")

    # =========================================================================
    # IMPLEMENTACIONES DE PREPROCESAMIENTO
    # =========================================================================
    
    def aplicar_filtro_gaussiano(self):
        """Aplica filtro Gaussiano."""
        try:
            kernel_size = int(input("Tamaño del kernel (impar, ej: 5): ") or "5")
            sigma = float(input("Sigma (ej: 1.0): ") or "1.0")
            
            resultado = self.preprocessor.apply_gaussian_blur(self.imagen_actual, kernel_size, sigma)
            self._mostrar_resultado_preprocesamiento("Filtro Gaussiano", resultado)
        except Exception as e:
            print(f"Error aplicando filtro Gaussiano: {e}")

    def aplicar_normalizacion(self):
        """Aplica normalización."""
        try:
            resultado = self.preprocessor.normalize_image(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Normalización", resultado)
        except Exception as e:
            print(f"Error aplicando normalización: {e}")

    def aplicar_clahe(self):
        """Aplica CLAHE."""
        try:
            clip_limit = float(input("Límite de clip (ej: 2.0): ") or "2.0")
            tile_size = int(input("Tamaño de tile (ej: 8): ") or "8")
            
            resultado = self.preprocessor.apply_clahe(self.imagen_actual, clip_limit, (tile_size, tile_size))
            self._mostrar_resultado_preprocesamiento("CLAHE", resultado)
        except Exception as e:
            print(f"Error aplicando CLAHE: {e}")

    def corregir_iluminacion(self):
        """Corrige iluminación."""
        try:
            resultado = self.preprocessor.correct_illumination(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Corrección de Iluminación", resultado)
        except Exception as e:
            print(f"Error corrigiendo iluminación: {e}")

    def reducir_ruido_bilateral(self):
        """Reduce ruido con filtro bilateral."""
        try:
            d = int(input("Diámetro del vecindario (ej: 9): ") or "9")
            sigma_color = float(input("Sigma color (ej: 75): ") or "75")
            sigma_space = float(input("Sigma espacial (ej: 75): ") or "75")
            
            resultado = self.preprocessor.reduce_noise_bilateral(self.imagen_actual, d, sigma_color, sigma_space)
            self._mostrar_resultado_preprocesamiento("Filtro Bilateral", resultado)
        except Exception as e:
            print(f"Error aplicando filtro bilateral: {e}")

    def afilar_imagen(self):
        """Afila la imagen."""
        try:
            amount = float(input("Cantidad de afilado (ej: 1.0): ") or "1.0")
            resultado = self.preprocessor.sharpen_image(self.imagen_actual, amount)
            self._mostrar_resultado_preprocesamiento("Afilado", resultado)
        except Exception as e:
            print(f"Error afilando imagen: {e}")

    def redimensionar_imagen(self):
        """Redimensiona la imagen."""
        try:
            width = int(input("Ancho (ej: 512): ") or "512")
            height = int(input("Alto (ej: 512): ") or "512")
            mantener_ratio = input("¿Mantener ratio? (s/n): ").lower().startswith('s')
            
            resultado = self.preprocessor.resize_image(self.imagen_actual, (width, height), mantener_ratio)
            self._mostrar_resultado_preprocesamiento("Redimensionamiento", resultado)
        except Exception as e:
            print(f"Error redimensionando imagen: {e}")

    def preprocesamiento_automatico(self):
        """Aplica preprocesamiento automático."""
        try:
            print("Aplicando preprocesamiento automático...")
            resultado = self.preprocessor.auto_preprocessing(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Preprocesamiento Automático", resultado)
        except Exception as e:
            print(f"Error en preprocesamiento automático: {e}")

    def comparar_preprocesamiento(self):
        """Compara imagen original vs preprocesada."""
        try:
            resultado = self.preprocessor.auto_preprocessing(self.imagen_actual)
            
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].imshow(cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Imagen Original')
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Imagen Preprocesada')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Mostrar historial de procesamiento
            historial = self.preprocessor.get_preprocessing_history()
            print("\nTécnicas aplicadas:")
            for i, tecnica in enumerate(historial, 1):
                print(f"   {i}. {tecnica}")
                
        except Exception as e:
            print(f"❌ Error comparando preprocesamiento: {e}")

    # Función auxiliar para mostrar resultados de preprocesamiento
    def _mostrar_resultado_preprocesamiento(self, nombre_tecnica, imagen_resultado):
        """Muestra el resultado del preprocesamiento con el nuevo sistema avanzado."""
        if imagen_resultado is None:
            print(f"❌ Error aplicando {nombre_tecnica}")
            return
        
        print(f"{nombre_tecnica} procesado correctamente")
        
        # Mostrar resultado primero
        self._mostrar_comparacion_imagenes(self.imagen_actual, imagen_resultado, "Imagen Actual", f"Con {nombre_tecnica}")
        
        # Usar el nuevo sistema de aplicación a imagen activa
        aplicado = self.aplicar_a_imagen_activa(imagen_resultado, nombre_tecnica)
        
        # Opción de guardar (independiente de si se aplicó o no)
        guardar = input("\n¿Guardar imagen procesada? (s/N): ").strip().lower()
        if guardar in ['s', 'sí', 'si', 'yes', 'y']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"{nombre_tecnica.lower().replace(' ', '_')}_{timestamp}.jpg"
            ruta_guardar = os.path.join(self.directorio_resultados, nombre_archivo)
            self.asegurar_directorio_existe(ruta_guardar)
            cv2.imwrite(ruta_guardar, imagen_resultado)
            print(f"Imagen guardada: {ruta_guardar}")
        
        # Mostrar historial si se aplicó
        if aplicado:
            self.mostrar_historial_preprocesamiento()
    
    def _mostrar_comparacion_imagenes(self, imagen1, imagen2, titulo1, titulo2):
        """Muestra comparación entre dos imágenes."""
        opcion = input("¿Mostrar comparación visual? (s/N): ").strip().lower()
        if opcion in ['s', 'sí', 'si', 'yes', 'y']:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 5))
                
                # Imagen 1
                plt.subplot(1, 2, 1)
                if len(imagen1.shape) == 3:
                    plt.imshow(cv2.cvtColor(imagen1, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(imagen1, cmap='gray')
                plt.title(titulo1)
                plt.axis('off')
                
                # Imagen 2
                plt.subplot(1, 2, 2)
                if len(imagen2.shape) == 3:
                    plt.imshow(cv2.cvtColor(imagen2, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(imagen2, cmap='gray')
                plt.title(titulo2)
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("matplotlib no disponible para visualización")
            except Exception as e:
                print(f"Error mostrando imágenes: {e}")

    def estadisticas_primer_orden(self):
        """Delegada al handler."""
        self.analysis_handlers.estadisticas_primer_orden()

    def estadisticas_segundo_orden(self):
        """Delegada al handler."""
        self.analysis_handlers.estadisticas_segundo_orden()

    def analisis_texturas_completo(self):
        """Delegada al handler."""
        self.analysis_handlers.analisis_texturas_completo()

    def comparar_regiones_textura(self):
        """Delegada al handler."""
        self.analysis_handlers.comparar_regiones_textura()

    def detectar_bordes_canny(self):
        """Delegada al handler."""
        self.analysis_handlers.detectar_bordes_canny()

    def detectar_bordes_sobel(self):
        """Delegada al handler."""
        self.analysis_handlers.detectar_bordes_sobel()

    def detectar_bordes_log(self):
        """Delegada al handler."""
        self.analysis_handlers.detectar_bordes_log()

    def analizar_gradientes(self):
        """Delegada al handler."""
        self.analysis_handlers.analizar_gradientes()

    def comparar_metodos_bordes(self):
        """Delegada al handler."""
        self.analysis_handlers.comparar_metodos_bordes()

    def detectar_lineas_hough(self):
        """Delegada al handler."""
        self.analysis_handlers.detectar_lineas_hough()

    def detectar_circulos_hough(self):
        """Delegada al handler."""
        self.analysis_handlers.detectar_circulos_hough()

    def calcular_momentos_geometricos(self):
        """Delegada al handler."""
        self.analysis_handlers.calcular_momentos_geometricos()

    def analisis_formas_completo(self):
        """Delegada al handler."""
        self.analysis_handlers.analisis_formas_completo()

    def extraer_surf(self):
        """Delegada al handler."""
        self.analysis_handlers.extraer_surf()

    def extraer_freak(self):
        """Delegada al handler.""" 
        self.analysis_handlers.extraer_freak()

    def extraer_akaze(self):
        """Delegada al handler."""
        self.analysis_handlers.extraer_akaze_avanzado()

    def segmentacion_grabcut(self):
        """Delegada al handler."""
        self.analysis_handlers.analizar_grabcut()

    def analisis_optical_flow(self):
        """Delegada al handler."""
        self.analysis_handlers.analizar_optical_flow()

    def analisis_avanzado_combinado(self):
        """Delegada al handler."""
        self.analysis_handlers.analisis_avanzado_completo()
        self.analysis_handlers.extraer_surf()

    def extraer_orb(self):
        """Delegada al handler."""
        self.analysis_handlers.extraer_orb()

    def extraer_hog(self):
        """Delegada al handler."""
        self.analysis_handlers.extraer_hog()

    def extraer_kaze(self):
        """Delegada al handler."""
        self.analysis_handlers.extraer_kaze()

    def extraer_akaze(self):
        """Delegada al handler."""
        self.analysis_handlers.extraer_akaze()

    def extraer_freak(self):
        """Delegada al handler."""
        self.analysis_handlers.extraer_freak()

    def segmentacion_grabcut(self):
        """Delegada al handler."""
        self.analysis_handlers.segmentacion_grabcut()

    def analisis_optical_flow(self):
        """Delegada al handler."""
        self.analysis_handlers.analisis_optical_flow()

    def analisis_optical_flow_profesora(self):
        """Análisis de optical flow usando método de la profesora."""
        if not self.verificar_imagen_cargada():
            return
        
        print("\nANÁLISIS OPTICAL FLOW - MÉTODO PREDETERMINADO")
        print("=" * 60)
        
        # Solicitar segunda imagen
        print("Necesita una segunda imagen para calcular el flujo óptico.")
        print("1. Seleccionar imagen de la carpeta")
        print("2. Usar imagen desplazada automáticamente")
        
        opcion = input("\nSeleccione opción (1/2): ").strip()
        
        imagen2 = None
        if opcion == '1':
            self.seleccionar_imagen_secundaria()
            imagen2 = self.imagen_secundaria
        
        try:
            # Llamar al método de la profesora específico
            resultados = self.advanced_analyzer.analizar_optical_flow_profesora(
                self.imagen_actual, 
                imagen2, 
                visualizar=True, 
                mostrar_descriptores=True, 
                guardar_resultados=True,
                nombre_imagen=f"optical_flow_profesora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            print(f"Análisis completado con método predeterminado")
            print(f"Magnitud promedio: {resultados['optical_flow_mean_magnitude']:.4f}")
            print(f"Magnitud máxima: {resultados['optical_flow_max_magnitude']:.4f}")
            
        except Exception as e:
            print(f"Error en análisis optical flow predeterminado: {e}")

    def analisis_secuencias_carpeta(self):
        """Análisis de secuencias de imágenes en carpeta."""
        print("\nANÁLISIS DE SECUENCIAS EN CARPETA")
        print("=" * 60)
        print("Este análisis procesa una carpeta completa de imágenes")
        print("para detectar patrones de movimiento y cambios temporales.")
        print()
        
        # Solicitar carpeta
        carpeta = input("Ingrese la ruta de la carpeta (o Enter para usar 'images'): ").strip()
        if not carpeta:
            carpeta = self.directorio_imagenes
        
        # Verificar que la carpeta existe
        if not os.path.exists(carpeta):
            print(f"La carpeta '{carpeta}' no existe.")
            return
        
        # Solicitar patrón de archivos
        patron = input("Ingrese patrón de archivos (*.jpg, *.png, etc.) [Enter=*.jpg]: ").strip()
        if not patron:
            patron = "*.jpg"
        
        try:
            # Llamar al análisis de secuencias
            resultados = self.advanced_analyzer.analizar_secuencia_imagenes_carpeta(
                carpeta_path=carpeta,
                patron_archivos=patron,
                visualizar=True,
                guardar_resultados=True,
                nombre_secuencia=f"secuencia_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            print(f"Análisis de secuencia completado")
            print(f"Archivos procesados: {resultados['num_transiciones']}")
            print(f"Cambios detectados: {len(resultados['cambios_movimiento'])}")
            
        except Exception as e:
            print(f"Error en análisis de secuencias: {e}")

    def analisis_comparativo_hog_kaze(self):
        """Delegada al handler."""
        self.analysis_handlers.analisis_comparativo_hog_kaze()

    def analisis_avanzado_combinado(self):
        """Delegada al handler."""
        self.analysis_handlers.analisis_avanzado_combinado()

    def comparar_algoritmos(self):
        """Delegada al handler."""
        self.analysis_handlers.comparar_algoritmos()

    def analisis_completo_caracteristicas(self):
        """Delegada al handler."""
        self.analysis_handlers.analisis_completo_caracteristicas()

    def analisis_hough_imagen_individual(self):
        """Realiza análisis completo de Hough para imagen individual."""
        if not self.verificar_imagen_cargada():
            return
        
        print("\nANÁLISIS COMPLETO DE HOUGH - IMAGEN INDIVIDUAL")
        print("=" * 60)
        
        try:
            nombre_imagen = os.path.basename(self.ruta_imagen_actual)
            
            # Procesar imagen con análisis completo de Hough
            print("Calculando momentos de Hough y transformadas...")
            resultados, datos_viz = self.hough_results_saver.procesar_imagen_individual(
                self.imagen_actual, nombre_imagen
            )
            
            # Preguntar si guardar procesamiento individual
            guardar = input("\n¿Desea guardar las imágenes del procesamiento? (s/N): ").strip().lower()
            if guardar in ['s', 'si', 'sí', 'y', 'yes']:
                ruta_proc = self.hough_results_saver.guardar_procesamiento_imagen_individual(
                    self.ruta_imagen_actual, datos_viz
                )
                print(f"Procesamiento guardado en: {ruta_proc}")
            
            # Guardar resultados en formato CSV/TXT/Excel
            self.hough_results_saver.guardar_resultados_csv_txt_excel(
                [resultados], f"hough_individual_{os.path.splitext(nombre_imagen)[0]}"
            )
            
            # Mostrar resumen de resultados
            print("\nRESUMEN DE RESULTADOS:")
            print("-" * 40)
            print(f"Imagen: {resultados['Imagen']}")
            print(f"Líneas detectadas: {resultados['num_lineas_detectadas']}")
            print(f"Círculos detectados: {resultados['num_circulos_detectados']}")
            print(f"Bordes detectados: {resultados['bordes_detectados']}")
            print(f"Centroide: ({resultados['centroide_x']:.1f}, {resultados['centroide_y']:.1f})")
            print(f"Media ángulos líneas: {resultados['momento_media_angulos']:.2f}°")
            print(f"Media radios círculos: {resultados['momento_media_radios']:.2f} px")
            print(f"Energía Hough líneas: {resultados['momento_energia_hough']:.6f}")
            print(f"Energía Hough círculos: {resultados['momento_energia_hough_circ']:.6f}")
            
        except Exception as e:
            print(f"Error en análisis de Hough: {str(e)}")
    
    def analisis_hough_por_lotes(self):
        """Realiza análisis de Hough por lotes para una carpeta de imágenes."""
        print("\nANÁLISIS DE HOUGH POR LOTES")
        print("=" * 60)
        
        # Seleccionar carpeta
        carpeta = input(f"Carpeta de imágenes (Enter para '{self.directorio_imagenes}'): ").strip()
        if not carpeta:
            carpeta = self.directorio_imagenes
        
        if not os.path.exists(carpeta):
            print(f"La carpeta {carpeta} no existe")
            return
        
        # Verificar imágenes disponibles
        import glob
        patrones = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
        archivos = []
        for patron in patrones:
            archivos.extend(glob.glob(os.path.join(carpeta, patron)))
        
        if not archivos:
            print(f"No se encontraron imágenes en {carpeta}")
            return
        
        print(f"Encontradas {len(archivos)} imágenes para procesar")
        
        # Preguntar sobre procesamiento individual
        print("\n¿Desea guardar el procesamiento individual de cada imagen?")
        print("1. Sí - Guardar imágenes del procesamiento de cada imagen")
        print("2. No - Solo generar resultados generales y comparación")
        
        try:
            opcion = int(input("Seleccione opción (2 por defecto): ").strip())
        except (ValueError, EOFError):
            opcion = 2
        
        if opcion not in [1, 2]:
            opcion = 2
            print("Opción no válida. Se usará la opción 2 (solo resultados generales).")
        
        guardar_individual = (opcion == 1)
        
        # Confirmar procesamiento
        print(f"\nResumen del procesamiento:")
        print(f"   Carpeta: {carpeta}")
        print(f"   Imágenes: {len(archivos)}")
        print(f"   Procesamiento individual: {'Sí' if guardar_individual else 'No'}")
        print(f"   Directorio de salida: {self.hough_results_saver.directorio_hough}")
        
        confirmar = input("\n¿Continuar con el procesamiento? (s/N): ").strip().lower()
        if confirmar not in ['s', 'si', 'sí', 'y', 'yes']:
            print("Procesamiento cancelado")
            return
        
        # Ejecutar procesamiento por lotes
        try:
            print("\nIniciando procesamiento por lotes...")
            resultado = self.hough_results_saver.procesar_carpeta_imagenes(
                carpeta, 
                patron="*.png,*.jpg,*.jpeg,*.tif,*.tiff,*.bmp",
                guardar_procesamiento_individual=guardar_individual
            )
            
            if resultado:
                print(f"\nPROCESAMIENTO COMPLETADO EXITOSAMENTE")
                print("=" * 60)
                print(f"Imágenes procesadas: {resultado['num_imagenes']}")
                print(f"Archivo CSV: {os.path.basename(resultado['csv'])}")
                print(f"Archivo Excel: {os.path.basename(resultado['excel'])}")
                print(f"Archivo TXT: {os.path.basename(resultado['txt'])}")
                print(f"Gráfico comparativo: {os.path.basename(resultado['comparacion'])}")
                
                if guardar_individual and 'procesamiento_individual' in resultado:
                    print(f"Procesamientos individuales: {len(resultado['procesamiento_individual'])}")
                
                print(f"\nTodos los archivos guardados en:")
                print(f"   {self.hough_results_saver.directorio_hough}")
                
                # Mostrar estadísticas del DataFrame
                df = resultado['dataframe']
                print(f"\nESTADÍSTICAS GENERALES:")
                print(f"   Promedio líneas detectadas: {df['num_lineas_detectadas'].mean():.2f}")
                print(f"   Promedio círculos detectados: {df['num_circulos_detectados'].mean():.2f}")
                print(f"   Promedio bordes detectados: {df['bordes_detectados'].mean():.0f}")
                print(f"   Promedio intensidad: {df['intensidad_promedio'].mean():.2f}")
            else:
                print("Error en el procesamiento por lotes")
                
        except Exception as e:
            print(f"Error en procesamiento por lotes: {str(e)}")

    def mostrar_ayuda(self):
        """Muestra ayuda del sistema."""
        print("\nAYUDA DEL SISTEMA DE ANÁLISIS DE TRÁFICO VEHICULAR")
        print("="*70)
        print("""
DESCRIPCIÓN:
   Sistema completo para análisis de imágenes de tráfico vehicular con
   capacidades avanzadas de procesamiento, extracción de características
   y detección de objetos específicos.

ESTRUCTURA DEL MENÚ PRINCIPAL:

1. GESTIÓN DE IMÁGENES
   - Cargar y visualizar imágenes
   - Listar imágenes disponibles
   - Información detallada de imágenes

2. PREPROCESAMIENTO DE IMÁGENES
   - Filtro Gaussiano para suavizado
   - Normalización y ecualización de histograma
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Corrección de iluminación
   - Reducción de ruido bilateral
   - Afilado de imagen
   - Redimensionamiento adaptativo
   - Preprocesamiento automático optimizado

3. EXTRACCIÓN DE CARACTERÍSTICAS
   
   3.1 DESCRIPTORES DE TEXTURA:
       - Estadísticas de primer orden (media, varianza, entropía)
       - Estadísticas de segundo orden (GLCM)
       - Análisis completo de texturas vehiculares
       - Comparación de regiones de textura
   
   3.2 DETECCIÓN DE BORDES:
       - Filtro Canny
       - Filtro Sobel (X, Y, magnitud)
       - Laplaciano de Gauss (LoG)
       - Análisis de gradientes direccionales
       - Comparación de métodos de detección
   
   3.3 DETECCIÓN DE FORMAS:
       - Transformada de Hough para líneas (carriles)
       - Transformada de Hough para círculos (señales, llantas)
       - Cálculo de momentos geométricos
       - Análisis completo de formas geométricas
   
   3.4 MÉTODOS AVANZADOS:
       - SURF (Speeded Up Robust Features)
       - ORB (Oriented FAST and Rotated BRIEF)
       - HOG (Histogram of Oriented Gradients)
       - KAZE (Nonlinear Scale Space)
       - AKAZE (Accelerated KAZE)
       - FREAK (Fast Retina Keypoint)
       - GrabCut Segmentation
       - Optical Flow Analysis

4. DETECCIÓN DE OBJETOS ESPECÍFICOS
   - Llantas de vehículos (Hough + AKAZE + texturas)
   - Señales de tráfico circulares (Hough + FREAK + color + LoG)
   - Semáforos (análisis de color + estructura + GrabCut)
   - Detección completa (todos los objetos)

5. PROCESAMIENTO POR LOTES
   - Procesamiento automático de carpetas completas
   - Generación de reportes estadísticos
   - Exportación en múltiples formatos (JSON, CSV, TXT)

🔧 ALGORITMOS IMPLEMENTADOS:
   
   TRADICIONALES:
   - Transformada de Hough (líneas y círculos)
   - Filtros de convolución (Gaussiano, Sobel, Laplaciano)
   - Análisis de color en espacios HSV y LAB
   - Morfología matemática
   
   AVANZADOS:
   - Detectores de puntos clave: SURF, ORB, AKAZE, KAZE, FREAK
   - Descriptores: HOG, LBP, GLCM
   - Segmentación: GrabCut
   - Análisis de movimiento: Optical Flow
   - Clustering: DBSCAN para agrupación de detecciones

ESTRUCTURA DE ARCHIVOS:
   - ./images/: Imágenes de entrada
   - ./resultados_deteccion/: Resultados del procesamiento
     ├── llantas/: Detecciones de llantas
     ├── senales/: Detecciones de señales  
     ├── semaforos/: Detecciones de semáforos
     ├── reportes/: Reportes estadísticos
     ├── texture_analysis/: Análisis de texturas
     ├── hough_analysis/: Análisis de formas
     └── advanced_analysis/: Análisis avanzados

RECOMENDACIONES DE USO:

1. FLUJO BÁSICO:
   - Cargar imagen → Preprocesamiento → Extracción → Detección

2. PARA ANÁLISIS ACADÉMICO:
   - Use "Extracción de Características" para estudiar algoritmos
   - Compare métodos con "Comparación de Algoritmos"
   - Analice texturas antes de detectar objetos

3. PARA DETECCIÓN PRÁCTICA:
   - Use "Detección de Objetos" directamente
   - Los métodos "Combinados" ofrecen mejor robustez
   - Procesamiento por lotes para múltiples imágenes

INICIO RÁPIDO:
   1. Coloque imágenes en ./images/
   2. Ejecute: python main_deteccion_vehicular.py
   3. Opción 1 → Cargar imagen
   4. Seleccione el tipo de análisis deseado
   5. Revise resultados en ./resultados_deteccion/
        """)
    
    # =========================================================================
    # IMPLEMENTACIONES DE MÉTODOS DE PREPROCESAMIENTO CON NUEVOS MÓDULOS
    # =========================================================================
    
    # FILTROS
    def aplicar_filtro_desenfoque(self):
        """Aplica filtro de desenfoque."""
        try:
            kernel_size = int(input("Tamaño del kernel (ej: 5): ") or "5")
            resultado = self.filtros.aplicar_filtro_desenfoque(self.imagen_actual, (kernel_size, kernel_size))
            self._mostrar_resultado_preprocesamiento("Filtro de Desenfoque", resultado)
        except Exception as e:
            print(f"Error aplicando filtro: {e}")
    
    def aplicar_filtro_gaussiano_nuevo(self):
        """Aplica filtro gaussiano usando el nuevo módulo."""
        try:
            kernel_size = int(input("Tamaño del kernel (impar, ej: 5): ") or "5")
            sigma = float(input("Sigma (ej: 1.0): ") or "1.0")
            resultado = self.filtros.aplicar_filtro_gaussiano(self.imagen_actual, (kernel_size, kernel_size), sigma)
            self._mostrar_resultado_preprocesamiento("Filtro Gaussiano", resultado)
        except Exception as e:
            print(f"Error aplicando filtro: {e}")
    
    def aplicar_filtro_mediana(self):
        """Aplica filtro de mediana."""
        try:
            kernel_size = int(input("Tamaño del kernel (ej: 5): ") or "5")
            resultado = self.filtros.aplicar_filtro_mediana(self.imagen_actual, kernel_size)
            self._mostrar_resultado_preprocesamiento("Filtro de Mediana", resultado)
        except Exception as e:
            print(f"Error aplicando filtro: {e}")
    
    # OPERACIONES ARITMÉTICAS
    def ajustar_brillo(self):
        """Ajusta el brillo de la imagen."""
        try:
            factor = float(input("Factor de brillo (>1 aumenta, <1 disminuye, ej: 1.2): ") or "1.2")
            resultado = self.operaciones_aritmeticas.ajustar_brillo(self.imagen_actual, factor)
            self._mostrar_resultado_preprocesamiento("Ajuste de Brillo", resultado)
        except Exception as e:
            print(f"Error ajustando brillo: {e}")
    
    def ajustar_contraste(self):
        """Ajusta el contraste de la imagen."""
        try:
            factor = float(input("Factor de contraste (>1 aumenta, <1 disminuye, ej: 1.2): ") or "1.2")
            resultado = self.operaciones_aritmeticas.ajustar_contraste(self.imagen_actual, factor)
            self._mostrar_resultado_preprocesamiento("Ajuste de Contraste", resultado)
        except Exception as e:
            print(f"Error ajustando contraste: {e}")
    
    # OPERACIONES GEOMÉTRICAS  
    def redimensionar_imagen_nueva(self):
        """Redimensiona la imagen usando el nuevo módulo."""
        try:
            ancho = input("Nuevo ancho (Enter para mantener proporción): ").strip()
            alto = input("Nuevo alto (Enter para mantener proporción): ").strip()
            
            ancho = int(ancho) if ancho else None
            alto = int(alto) if alto else None
            
            resultado = self.operaciones_geometricas.redimensionar_imagen(self.imagen_actual, ancho, alto)
            self._mostrar_resultado_preprocesamiento("Redimensionamiento", resultado)
        except Exception as e:
            print(f"Error redimensionando: {e}")

    def rotar_imagen(self):
        """Rota la imagen."""
        try:
            angulo = float(input("Ángulo de rotación en grados (ej: 45): ") or "45")
            resultado = self.operaciones_geometricas.rotar_imagen(self.imagen_actual, angulo)
            self._mostrar_resultado_preprocesamiento("Rotación", resultado)
        except Exception as e:
            print(f"Error rotando imagen: {e}")
    
    # SEGMENTACIÓN
    def aplicar_umbral_simple(self):
        """Aplica umbralización simple."""
        try:
            umbral = int(input("Valor de umbral (0-255, ej: 127): ") or "127")
            resultado = self.segmentacion.umbral_simple(self.imagen_actual, umbral)
            self._mostrar_resultado_preprocesamiento("Umbralización Simple", resultado)
        except Exception as e:
            print(f"Error aplicando umbral: {e}")
    
    def aplicar_umbral_otsu(self):
        """Aplica umbralización Otsu."""
        try:
            resultado = self.segmentacion.umbral_otsu(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Umbralización Otsu", resultado)
        except Exception as e:
            print(f"Error aplicando Otsu: {e}")
    
    def aplicar_kmeans(self):
        """Aplica segmentación K-means."""
        try:
            k = int(input("Número de clusters (ej: 3): ") or "3")
            resultado = self.segmentacion.kmeans_segmentacion(self.imagen_actual, k)
            self._mostrar_resultado_preprocesamiento("Segmentación K-means", resultado)
        except Exception as e:
            print(f"Error aplicando K-means: {e}")
    
    # OPERACIONES MORFOLÓGICAS
    def aplicar_erosion(self):
        """Aplica erosión morfológica."""
        try:
            kernel_size = int(input("Tamaño del kernel (ej: 5): ") or "5")
            iteraciones = int(input("Número de iteraciones (ej: 1): ") or "1")
            resultado = self.operaciones_morfologicas.erosion(self.imagen_actual, kernel_size, iteraciones)
            self._mostrar_resultado_preprocesamiento("Erosión", resultado)
        except Exception as e:
            print(f"Error aplicando erosión: {e}")
    
    def aplicar_dilatacion(self):
        """Aplica dilatación morfológica."""
        try:
            kernel_size = int(input("Tamaño del kernel (ej: 5): ") or "5") 
            iteraciones = int(input("Número de iteraciones (ej: 1): ") or "1")
            resultado = self.operaciones_morfologicas.dilatacion(self.imagen_actual, kernel_size, iteraciones)
            self._mostrar_resultado_preprocesamiento("Dilatación", resultado)
        except Exception as e:
            print(f"Error aplicando dilatación: {e}")
    
    # Métodos placeholder para mantener compatibilidad del menú
    def aplicar_filtro_nitidez(self):
        """Aplica filtro de nitidez."""
        try:
            resultado = self.filtros.aplicar_filtro_nitidez(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Filtro de Nitidez", resultado)
        except Exception as e:
            print(f"Error aplicando filtro: {e}")
    
    def aplicar_filtro_bilateral_nuevo(self):
        """Aplica filtro bilateral."""
        try:
            resultado = self.filtros.aplicar_filtro_bilateral(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Filtro Bilateral", resultado)
        except Exception as e:
            print(f"Error aplicando filtro: {e}")
    
    def aplicar_canny(self):
        """Aplica detección Canny."""
        try:
            resultado = self.filtros.detectar_bordes_canny(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Detección Canny", resultado)
        except Exception as e:
            print(f"Error aplicando Canny: {e}")
    
    def aplicar_ecualizacion_histograma(self):
        """Aplica ecualización de histograma."""
        try:
            if len(self.imagen_actual.shape) > 2:
                imagen_gris = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                imagen_gris = self.imagen_actual
            resultado = self.filtros.ecualizar_histograma(imagen_gris)
            self._mostrar_resultado_preprocesamiento("Ecualización", resultado)
        except Exception as e:
            print(f"Error aplicando ecualización: {e}")
    
    # Métodos simples para mantener funcionalidad básica
    def aplicar_filtro_sobel(self):
        """Aplica filtro Sobel para detección de bordes."""
        try:
            if not self.verificar_imagen_cargada():
                return
            
            print("\nAPLICANDO FILTRO SOBEL")
            print("=" * 30)
            direccion = input("Dirección (x/y/ambas) [ambas]: ").strip().lower() or "ambas"
            
            resultado = self.filtros.aplicar_filtro_sobel(self.imagen_actual, direccion)
            self._mostrar_resultado_preprocesamiento(f"Filtro Sobel ({direccion})", resultado)
            
        except Exception as e:
            print(f"❌ Error aplicando filtro Sobel: {e}")
    
    def aplicar_filtro_laplaciano(self):
        """Aplica filtro Laplaciano para detección de bordes."""
        try:
            if not self.verificar_imagen_cargada():
                return
            
            print("\nAPLICANDO FILTRO LAPLACIANO")
            print("=" * 30)
            
            resultado = self.filtros.aplicar_filtro_laplaciano(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Filtro Laplaciano", resultado)
            
        except Exception as e:
            print(f"❌ Error aplicando filtro Laplaciano: {e}")
    
    def suma_imagenes(self):
        """Suma dos imágenes."""
        try:
            print("\nSUMA DE IMÁGENES")
            print("=" * 40)
            print("Esta operación requiere una segunda imagen.")
            
            if not self.seleccionar_imagen_secundaria():
                return
            
            print("\nProcesando suma de imágenes...")
            resultado = self.operaciones_aritmeticas.suma_imagenes(self.imagen_actual, self.imagen_secundaria)
            self._mostrar_resultado_preprocesamiento("Suma de Imágenes", resultado)
            
        except Exception as e:
            print(f"Error en suma de imágenes: {e}")
    
    def resta_imagenes(self):
        """Resta dos imágenes."""
        try:
            print("\nRESTA DE IMÁGENES")
            print("=" * 40)
            print("Esta operación requiere una segunda imagen.")
            
            if not self.seleccionar_imagen_secundaria():
                return
            
            print("\nProcesando resta de imágenes...")
            resultado = self.operaciones_aritmeticas.resta_imagenes(self.imagen_actual, self.imagen_secundaria)
            self._mostrar_resultado_preprocesamiento("Resta de Imágenes", resultado)
            
        except Exception as e:
            print(f"Error en resta de imágenes: {e}")
    
    def multiplicacion_imagenes(self):
        """Multiplica dos imágenes."""
        try:
            print("\nMULTIPLICACIÓN DE IMÁGENES")
            print("=" * 40)
            print("Esta operación requiere una segunda imagen.")
            
            if not self.seleccionar_imagen_secundaria():
                return
            
            print("\nProcesando multiplicación de imágenes...")
            resultado = self.operaciones_aritmeticas.multiplicacion_imagenes(self.imagen_actual, self.imagen_secundaria)
            self._mostrar_resultado_preprocesamiento("Multiplicación de Imágenes", resultado)
            
        except Exception as e:
            print(f"Error en multiplicación de imágenes: {e}")
    
    def division_imagenes(self):
        """Divide dos imágenes."""
        try:
            print("\nDIVISIÓN DE IMÁGENES")
            print("=" * 40)
            print("Esta operación requiere una segunda imagen.")
            
            if not self.seleccionar_imagen_secundaria():
                return
            
            print("\nProcesando división de imágenes...")
            resultado = self.operaciones_aritmeticas.division_imagenes(self.imagen_actual, self.imagen_secundaria)
            self._mostrar_resultado_preprocesamiento("División de Imágenes", resultado)
            
        except Exception as e:
            print(f"Error en división de imágenes: {e}")
    
    def aplicar_gamma(self):
        """Aplica corrección gamma."""
        try:
            gamma = float(input("Valor gamma (ej: 0.7): ") or "0.7")
            resultado = self.operaciones_aritmeticas.ajustar_gamma(self.imagen_actual, gamma)
            self._mostrar_resultado_preprocesamiento("Corrección Gamma", resultado)
        except Exception as e:
            print(f"Error aplicando gamma: {e}")
    
    def mezclar_imagenes(self):
        """Mezcla dos imágenes con transparencia."""
        try:
            print("\nMEZCLA DE IMÁGENES")
            print("=" * 40)
            print("Esta operación requiere una segunda imagen.")
            
            if not self.seleccionar_imagen_secundaria():
                return
            
            alpha = float(input("Factor de transparencia (0.0-1.0, ej: 0.5): ") or "0.5")
            if not (0.0 <= alpha <= 1.0):
                print("El factor debe estar entre 0.0 y 1.0")
                return
            
            print(f"\nProcesando mezcla con alpha={alpha}...")
            resultado = self.operaciones_aritmeticas.mezclar_imagenes(self.imagen_actual, self.imagen_secundaria, alpha)
            self._mostrar_resultado_preprocesamiento("Mezcla de Imágenes", resultado)
            
        except Exception as e:
            print(f"Error en mezcla de imágenes: {e}")
    
    def voltear_imagen(self):
        """Voltea imagen."""
        try:
            resultado = self.operaciones_geometricas.voltear_imagen(self.imagen_actual, 1)
            self._mostrar_resultado_preprocesamiento("Volteo Horizontal", resultado)
        except Exception as e:
            print(f"Error volteando: {e}")
    
    def trasladar_imagen(self):
        """Traslada la imagen en coordenadas X e Y."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nTRASLADAR IMAGEN")
            print("=" * 30)
            dx = int(input("Desplazamiento en X: "))
            dy = int(input("Desplazamiento en Y: "))
            
            resultado = self.operaciones_geometricas.trasladar_imagen(self.imagen_actual, dx, dy)
            self._mostrar_resultado_preprocesamiento(f"Traslación ({dx}, {dy})", resultado)
            
        except Exception as e:
            print(f"Error trasladando imagen: {e}")
    
    def recortar_imagen(self):
        """Recorta una región rectangular de la imagen."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nRECORTAR IMAGEN")
            print("=" * 30)
            h, w = self.imagen_actual.shape[:2]
            print(f"Tamaño actual: {w}x{h}")
            
            x1 = int(input("Coordenada X inicial: "))
            y1 = int(input("Coordenada Y inicial: "))
            x2 = int(input("Coordenada X final: "))
            y2 = int(input("Coordenada Y final: "))
            
            # Validar coordenadas
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            resultado = self.operaciones_geometricas.recortar_imagen(self.imagen_actual, x1, y1, x2, y2)
            self._mostrar_resultado_preprocesamiento(f"Recorte ({x1},{y1},{x2},{y2})", resultado)
            
        except Exception as e:
            print(f"Error recortando imagen: {e}")
    
    def transformacion_perspectiva(self):
        """Aplica transformación de perspectiva con puntos predefinidos."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nTRANSFORMACIÓN DE PERSPECTIVA")
            print("=" * 40)
            print("Aplicando transformación de perspectiva con puntos de ejemplo...")
            
            h, w = self.imagen_actual.shape[:2]
            
            # Puntos origen (esquinas de la imagen)
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            
            # Puntos destino (perspectiva trapezoidal)
            offset = w // 6
            pts2 = np.float32([[offset, 0], [w-offset, 0], [0, h], [w, h]])
            
            resultado = self.operaciones_geometricas.aplicar_transformacion_perspectiva(
                self.imagen_actual, pts1, pts2)
            self._mostrar_resultado_preprocesamiento("Transformación de Perspectiva", resultado)
            
        except Exception as e:
            print(f"Error aplicando transformación de perspectiva: {e}")
    
    def escalar_imagen(self):
        """Escala la imagen por factores específicos."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nESCALAR IMAGEN")
            print("=" * 30)
            factor_x = float(input("Factor de escala X [1.0]: ") or "1.0")
            factor_y_input = input("Factor de escala Y (Enter para usar mismo que X): ").strip()
            factor_y = float(factor_y_input) if factor_y_input else None
            
            resultado = self.operaciones_geometricas.escalar_imagen(self.imagen_actual, factor_x, factor_y)
            self._mostrar_resultado_preprocesamiento(f"Escalado ({factor_x}, {factor_y or factor_x})", resultado)
            
        except Exception as e:
            print(f"Error escalando imagen: {e}")
        
    def transformacion_afin(self):
        """Aplica transformación afín con puntos predefinidos."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nTRANSFORMACIÓN AFÍN")
            print("=" * 30)
            print("Aplicando transformación afín con puntos de ejemplo...")
            
            h, w = self.imagen_actual.shape[:2]
            
            # Tres puntos origen
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            
            # Tres puntos destino (ligera deformación)
            pts2 = np.float32([[60, 40], [190, 60], [40, 210]])
            
            resultado = self.operaciones_geometricas.aplicar_transformacion_afin(
                self.imagen_actual, pts1, pts2)
            self._mostrar_resultado_preprocesamiento("Transformación Afín", resultado)
            
        except Exception as e:
            print(f"Error aplicando transformación afín: {e}")
        
    def corregir_distorsion(self):
        """Corrige distorsión de barril en la imagen."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nCORRECCIÓN DE DISTORSIÓN")
            print("=" * 40)
            k1 = float(input("Coeficiente k1 [-0.2]: ") or "-0.2")
            k2 = float(input("Coeficiente k2 [0.0]: ") or "0.0")
            k3 = float(input("Coeficiente k3 [0.0]: ") or "0.0")
            
            resultado = self.operaciones_geometricas.corregir_distorsion_barril(
                self.imagen_actual, k1, k2, k3)
            self._mostrar_resultado_preprocesamiento(f"Corrección Distorsión (k1={k1})", resultado)
            
        except Exception as e:
            print(f"Error corrigiendo distorsión: {e}")
    
    # Operaciones lógicas
    def operacion_and(self):
        """Operación lógica AND entre dos imágenes."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nOPERACIÓN LÓGICA AND")
            print("=" * 30)
            print("Esta operación requiere una segunda imagen.")
            
            if not self.seleccionar_imagen_secundaria():
                return
            
            resultado = self.operaciones_logicas.operacion_and(self.imagen_actual, self.imagen_secundaria)
            self._mostrar_resultado_preprocesamiento("Operación AND", resultado)
            
        except Exception as e:
            print(f"Error en operación AND: {e}")
    
    def operacion_or(self):
        """Operación lógica OR entre dos imágenes."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nOPERACIÓN LÓGICA OR")
            print("=" * 30)
            print("Esta operación requiere una segunda imagen.")
            
            if not self.seleccionar_imagen_secundaria():
                return
            
            resultado = self.operaciones_logicas.operacion_or(self.imagen_actual, self.imagen_secundaria)
            self._mostrar_resultado_preprocesamiento("Operación OR", resultado)
            
        except Exception as e:
            print(f"Error en operación OR: {e}")
    
    def operacion_not(self):
        """Operación lógica NOT (inversión) de la imagen."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nOPERACIÓN LÓGICA NOT")
            print("=" * 30)
            
            resultado = self.operaciones_logicas.operacion_not(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Operación NOT", resultado)
            
        except Exception as e:
            print(f"Error en operación NOT: {e}")
    
    def operacion_xor(self):
        """Operación lógica XOR entre dos imágenes."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nOPERACIÓN LÓGICA XOR")
            print("=" * 30)
            print("Esta operación requiere una segunda imagen.")
            
            if not self.seleccionar_imagen_secundaria():
                return
            
            resultado = self.operaciones_logicas.operacion_xor(self.imagen_actual, self.imagen_secundaria)
            self._mostrar_resultado_preprocesamiento("Operación XOR", resultado)
            
        except Exception as e:
            print(f"Error en operación XOR: {e}")
    
    def crear_mascara_rectangular(self):
        """Crea una máscara rectangular y la aplica a la imagen."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nCREAR MÁSCARA RECTANGULAR")
            print("=" * 40)
            h, w = self.imagen_actual.shape[:2]
            print(f"Tamaño de imagen: {w}x{h}")
            
            x = int(input("Posición X: "))
            y = int(input("Posición Y: "))
            ancho = int(input("Ancho de la máscara: "))
            alto = int(input("Alto de la máscara: "))
            
            mascara = self.operaciones_logicas.crear_mascara_rectangular(
                self.imagen_actual, x, y, ancho, alto)
            resultado = self.operaciones_logicas.aplicar_mascara(self.imagen_actual, mascara)
            self._mostrar_resultado_preprocesamiento(f"Máscara Rectangular ({x},{y},{ancho},{alto})", resultado)
            
        except Exception as e:
            print(f"Error creando máscara rectangular: {e}")
    
    def crear_mascara_circular(self):
        """Crea una máscara circular y la aplica a la imagen."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nCREAR MÁSCARA CIRCULAR")
            print("=" * 30)
            h, w = self.imagen_actual.shape[:2]
            print(f"Tamaño de imagen: {w}x{h}")
            
            centro_x = int(input("Centro X: "))
            centro_y = int(input("Centro Y: "))
            radio = int(input("Radio: "))
            
            mascara = self.operaciones_logicas.crear_mascara_circular(
                self.imagen_actual, centro_x, centro_y, radio)
            resultado = self.operaciones_logicas.aplicar_mascara(self.imagen_actual, mascara)
            self._mostrar_resultado_preprocesamiento(f"Máscara Circular ({centro_x},{centro_y},r={radio})", resultado)
            
        except Exception as e:
            print(f"Error creando máscara circular: {e}")
    
    def aplicar_mascara(self):
        """Aplica la imagen secundaria como máscara a la imagen actual."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nAPLICAR MÁSCARA")
            print("=" * 30)
            print("Esta operación usa la segunda imagen como máscara.")
            
            if not self.seleccionar_imagen_secundaria():
                return
            
            resultado = self.operaciones_logicas.aplicar_mascara(self.imagen_actual, self.imagen_secundaria)
            self._mostrar_resultado_preprocesamiento("Máscara Aplicada", resultado)
            
        except Exception as e:
            print(f"Error aplicando máscara: {e}")
    
    def segmentar_por_color(self):
        """Segmenta la imagen por rango de color."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nSEGMENTACIÓN POR COLOR")
            print("=" * 30)
            print("Seleccione espacio de color:")
            print("1. BGR")
            print("2. HSV")
            print("3. LAB")
            
            opcion = input("Opción [1]: ").strip() or "1"
            espacios = {"1": "BGR", "2": "HSV", "3": "LAB"}
            espacio = espacios.get(opcion, "BGR")
            
            print(f"\nEspacio de color: {espacio}")
            print("Ingrese límites (valores 0-255 para BGR/LAB, H:0-179, S/V:0-255 para HSV):")
            
            if espacio == "HSV":
                limite_inf = np.array([
                    int(input("H mínimo [0]: ") or "0"),
                    int(input("S mínimo [50]: ") or "50"),
                    int(input("V mínimo [50]: ") or "50")
                ])
                limite_sup = np.array([
                    int(input("H máximo [179]: ") or "179"),
                    int(input("S máximo [255]: ") or "255"),
                    int(input("V máximo [255]: ") or "255")
                ])
            else:
                limite_inf = np.array([
                    int(input("Canal 1 mínimo [0]: ") or "0"),
                    int(input("Canal 2 mínimo [0]: ") or "0"),
                    int(input("Canal 3 mínimo [0]: ") or "0")
                ])
                limite_sup = np.array([
                    int(input("Canal 1 máximo [255]: ") or "255"),
                    int(input("Canal 2 máximo [255]: ") or "255"),
                    int(input("Canal 3 máximo [255]: ") or "255")
                ])
            
            mascara = self.operaciones_logicas.crear_mascara_por_rango_color(
                self.imagen_actual, limite_inf, limite_sup, espacio)
            
            # Aplicar la máscara
            resultado = self.operaciones_logicas.aplicar_mascara(self.imagen_actual, mascara)
            self._mostrar_resultado_preprocesamiento(f"Segmentación Color {espacio}", resultado)
            
        except Exception as e:
            print(f"Error segmentando por color: {e}")
    
    def combinar_mascaras(self):
        """Combina múltiples máscaras (funcionalidad simplificada)."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nCOMBINAR MÁSCARAS")
            print("=" * 30)
            print("Esta función combina la imagen actual con la segunda imagen como máscaras.")
            
            if not self.seleccionar_imagen_secundaria():
                return
                
            print("Operación de combinación:")
            print("1. OR")
            print("2. AND") 
            print("3. XOR")
            
            opcion = input("Seleccione operación [1]: ").strip() or "1"
            operaciones = {"1": "OR", "2": "AND", "3": "XOR"}
            operacion = operaciones.get(opcion, "OR")
            
            # Convertir a escala de grises si es necesario para usar como máscaras
            mask1 = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY) if len(self.imagen_actual.shape) > 2 else self.imagen_actual
            mask2 = cv2.cvtColor(self.imagen_secundaria, cv2.COLOR_BGR2GRAY) if len(self.imagen_secundaria.shape) > 2 else self.imagen_secundaria
            
            resultado = self.operaciones_logicas.combinar_mascaras([mask1, mask2], operacion)
            self._mostrar_resultado_preprocesamiento(f"Máscaras Combinadas ({operacion})", resultado)
            
        except Exception as e:
            print(f"Error combinando máscaras: {e}")
    
    # Más operaciones morfológicas
    def aplicar_apertura(self):
        """Aplica apertura."""
        try:
            resultado = self.operaciones_morfologicas.apertura(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Apertura", resultado)
        except Exception as e:
            print(f"Error aplicando apertura: {e}")
    
    def aplicar_cierre(self):
        """Aplica cierre."""
        try:
            resultado = self.operaciones_morfologicas.cierre(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Cierre", resultado)
        except Exception as e:
            print(f"Error aplicando cierre: {e}")
    
    def aplicar_gradiente_morfologico(self):
        """Aplica gradiente morfológico (dilatación - erosión)."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nGRADIENTE MORFOLÓGICO")
            print("=" * 30)
            kernel_size = int(input("Tamaño del kernel [5]: ") or "5")
            
            resultado = self.operaciones_morfologicas.gradiente_morfologico(
                self.imagen_actual, kernel_size)
            self._mostrar_resultado_preprocesamiento(f"Gradiente Morfológico (k={kernel_size})", resultado)
            
        except Exception as e:
            print(f"Error aplicando gradiente morfológico: {e}")
    
    def aplicar_top_hat(self):
        """Aplica transformación Top Hat."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nTOP HAT")
            print("=" * 30)
            kernel_size = int(input("Tamaño del kernel [5]: ") or "5")
            
            resultado = self.operaciones_morfologicas.top_hat(
                self.imagen_actual, kernel_size)
            self._mostrar_resultado_preprocesamiento(f"Top Hat (k={kernel_size})", resultado)
            
        except Exception as e:
            print(f"Error aplicando Top Hat: {e}")
    
    def aplicar_black_hat(self):
        """Aplica transformación Black Hat."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nBLACK HAT")
            print("=" * 30)
            kernel_size = int(input("Tamaño del kernel [5]: ") or "5")
            
            resultado = self.operaciones_morfologicas.black_hat(
                self.imagen_actual, kernel_size)
            self._mostrar_resultado_preprocesamiento(f"Black Hat (k={kernel_size})", resultado)
            
        except Exception as e:
            print(f"Error aplicando Black Hat: {e}")
    
    def eliminar_ruido_binario(self):
        """Elimina ruido en imagen binaria usando morfología."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nELIMINAR RUIDO BINARIO")
            print("=" * 30)
            print("Método:")
            print("1. Apertura")
            print("2. Cierre")
            
            metodo_opcion = input("Seleccione método [1]: ").strip() or "1"
            metodo = "apertura" if metodo_opcion == "1" else "cierre"
            
            kernel_size = int(input("Tamaño del kernel [5]: ") or "5")
            
            resultado = self.operaciones_morfologicas.eliminar_ruido_binaria(
                self.imagen_actual, metodo, kernel_size)
            self._mostrar_resultado_preprocesamiento(f"Eliminación Ruido ({metodo})", resultado)
            
        except Exception as e:
            print(f"Error eliminando ruido: {e}")
    
    def extraer_contornos_morfologicos(self):
        """Extrae contornos usando operaciones morfológicas."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nEXTRAER CONTORNOS MORFOLÓGICOS")
            print("=" * 40)
            kernel_size = int(input("Tamaño del kernel [3]: ") or "3")
            
            resultado = self.operaciones_morfologicas.extraer_contornos_morfologicos(
                self.imagen_actual, kernel_size)
            self._mostrar_resultado_preprocesamiento(f"Contornos Morfológicos (k={kernel_size})", resultado)
            
        except Exception as e:
            print(f"Error extrayendo contornos: {e}")
    
    def aplicar_esqueletizacion(self):
        """Aplica esqueletización a la imagen binaria."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nESQUELETIZACIÓN")
            print("=" * 30)
            print("Aplicando esqueletización...")
            
            resultado = self.operaciones_morfologicas.esqueletizacion(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Esqueletización", resultado)
            
        except Exception as e:
            print(f"Error aplicando esqueletización: {e}")
    
    def rellenar_huecos(self):
        """Rellena huecos en imagen binaria."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nRELLENAR HUECOS")
            print("=" * 30)
            print("Rellenando huecos en imagen binaria...")
            
            resultado = self.operaciones_morfologicas.rellenar_huecos(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Relleno de Huecos", resultado)
            
        except Exception as e:
            print(f"Error rellenando huecos: {e}")
    
    def limpiar_bordes(self):
        """Limpia objetos que tocan los bordes de la imagen."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nLIMPIAR BORDES")
            print("=" * 30)
            print("Conectividad:")
            print("1. 4-conectividad")
            print("2. 8-conectividad")
            
            conectividad_opcion = input("Seleccione conectividad [1]: ").strip() or "1"
            conectividad = 4 if conectividad_opcion == "1" else 8
            
            resultado = self.operaciones_morfologicas.limpiar_bordes(
                self.imagen_actual, conectividad)
            self._mostrar_resultado_preprocesamiento(f"Limpieza Bordes ({conectividad}-conn)", resultado)
            
        except Exception as e:
            print(f"Error limpiando bordes: {e}")
    
    # Más segmentación
    def aplicar_umbral_adaptativo(self):
        """Aplica umbralización adaptativa."""
        try:
            resultado = self.segmentacion.umbral_adaptativo(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Umbral Adaptativo", resultado)
        except Exception as e:
            print(f"Error aplicando umbral adaptativo: {e}")
    
    def aplicar_canny_segmentacion(self):
        """Aplica Canny para segmentación."""
        try:
            resultado = self.segmentacion.detector_canny(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Segmentación Canny", resultado)
        except Exception as e:
            print(f"Error aplicando Canny: {e}")
    
    def detectar_contornos_segmentacion(self):
        """Detecta contornos."""
        try:
            resultado = self.segmentacion.deteccion_contornos(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Detección de Contornos", resultado)
        except Exception as e:
            print(f"Error detectando contornos: {e}")
    
    def aplicar_watershed(self):
        """Aplica Watershed."""
        try:
            resultado = self.segmentacion.watershed_segmentacion(self.imagen_actual)
            self._mostrar_resultado_preprocesamiento("Watershed", resultado)
        except Exception as e:
            print(f"Error aplicando Watershed: {e}")
    
    def segmentar_color_hsv(self):
        """Segmenta imagen por color en espacio HSV."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nSEGMENTACIÓN COLOR HSV")
            print("=" * 30)
            print("Ingrese rangos HSV (H: 0-179, S/V: 0-255):")
            
            hue_min = int(input("Hue mínimo [0]: ") or "0")
            hue_max = int(input("Hue máximo [179]: ") or "179")
            sat_min = int(input("Saturación mínima [50]: ") or "50")
            sat_max = int(input("Saturación máxima [255]: ") or "255")
            val_min = int(input("Valor mínimo [50]: ") or "50")
            val_max = int(input("Valor máximo [255]: ") or "255")
            
            resultado = self.segmentacion.segmentar_color_hsv(
                self.imagen_actual, hue_min, hue_max, sat_min, val_min, sat_max, val_max)
            self._mostrar_resultado_preprocesamiento(f"Segmentación HSV (H:{hue_min}-{hue_max})", resultado)
            
        except Exception as e:
            print(f"Error segmentando por color HSV: {e}")
    
    def aplicar_crecimiento_regiones(self):
        """Aplica algoritmo de crecimiento de regiones."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nCRECIMIENTO DE REGIONES")
            print("=" * 30)
            print("Configuración:")
            umbral = int(input("Umbral de similitud [20]: ") or "20")
            
            # Para simplificar, usaremos puntos semilla automáticos
            print("Usando puntos semilla automáticos (centro y esquinas)...")
            
            resultado = self.segmentacion.crecimiento_regiones(self.imagen_actual, None, umbral)
            self._mostrar_resultado_preprocesamiento(f"Crecimiento de Regiones (umbral={umbral})", resultado)
            
        except Exception as e:
            print(f"Error aplicando crecimiento de regiones: {e}")
    
    def segmentar_por_textura(self):
        """Segmenta imagen basándose en características de textura."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nSEGMENTACIÓN POR TEXTURA")
            print("=" * 40)
            tamano_ventana = int(input("Tamaño de ventana [15]: ") or "15")
            
            resultado = self.segmentacion.segmentacion_por_textura(
                self.imagen_actual, tamano_ventana)
            self._mostrar_resultado_preprocesamiento(f"Segmentación Textura (ventana={tamano_ventana})", resultado)
            
        except Exception as e:
            print(f"Error segmentando por textura: {e}")
    
    def aplicar_grabcut(self):
        """Aplica algoritmo GrabCut para segmentación interactiva."""
        try:
            if not self.verificar_imagen_cargada():
                return
                
            print("\nGRABCUT SEGMENTACIÓN")
            print("=" * 30)
            h, w = self.imagen_actual.shape[:2]
            print(f"Tamaño de imagen: {w}x{h}")
            print("Definiendo región de interés automáticamente...")
            
            # Usar rectángulo predeterminado (centro de la imagen)
            margen_x, margen_y = w//4, h//4
            rect = (margen_x, margen_y, w//2, h//2)
            
            iteraciones = int(input("Número de iteraciones [5]: ") or "5")
            
            mascara, resultado = self.segmentacion.grabcut_segmentacion(
                self.imagen_actual, rect, iteraciones)
            self._mostrar_resultado_preprocesamiento(f"GrabCut (iter={iteraciones})", resultado)
            
        except Exception as e:
            print(f"Error en GrabCut: {e}")

    def ejecutar_todos_metodos_caracteristicas(self):
        """
        Ejecuta todos los métodos avanzados de características con opciones por defecto
        y guarda los resultados automáticamente.
        """
        if not self.verificar_imagen_cargada():
            return
            
        print("\n" + "="*80)
        print("ANÁLISIS COMPLETO DE CARACTERÍSTICAS AVANZADAS")
        print("="*80)
        print("Ejecutando todos los métodos de características con opciones por defecto...")
        print("Los resultados se guardarán automáticamente.")
        print("="*80)
        
        # Obtener nombre base de la imagen para los archivos de resultados
        nombre_imagen = "imagen_analisis"
        if self.ruta_imagen_actual:
            nombre_imagen = os.path.splitext(os.path.basename(self.ruta_imagen_actual))[0]
        
        metodos_ejecutados = []
        errores = []
        
        # 1. SURF (Speeded Up Robust Features)
        print("\n1/7 - Ejecutando SURF...")
        try:
            resultado_surf = self.surf_orb_analyzer.extraer_caracteristicas_surf(
                self.imagen_actual, 
                visualizar=True, 
                mostrar_descriptores=True, 
                guardar_resultados=True,
                nombre_imagen=f"{nombre_imagen}_surf"
            )
            metodos_ejecutados.append("SURF")
            print("SURF completado")
        except Exception as e:
            error_msg = f"❌ Error en SURF: {e}"
            print(error_msg)
            errores.append(error_msg)
        
        # 2. ORB (Oriented FAST and Rotated BRIEF)
        print("\n2/7 - Ejecutando ORB...")
        try:
            resultado_orb = self.surf_orb_analyzer.extraer_caracteristicas_orb(
                self.imagen_actual, 
                visualizar=True, 
                mostrar_descriptores=True, 
                guardar_resultados=True,
                nombre_imagen=f"{nombre_imagen}_orb"
            )
            metodos_ejecutados.append("ORB")
            print("ORB completado")
        except Exception as e:
            error_msg = f"❌ Error en ORB: {e}"
            print(error_msg)
            errores.append(error_msg)
        
        # 3. HOG (Histogram of Oriented Gradients)
        print("\n3/7 - Ejecutando HOG...")
        try:
            resultado_hog = self.hog_kaze_analyzer.extraer_caracteristicas_hog(
                self.imagen_actual, 
                visualizar=True, 
                mostrar_descriptores=True, 
                guardar_resultados=True,
                nombre_imagen=f"{nombre_imagen}_hog"
            )
            metodos_ejecutados.append("HOG")
            print("HOG completado")
        except Exception as e:
            error_msg = f"❌ Error en HOG: {e}"
            print(error_msg)
            errores.append(error_msg)
        
        # 4. KAZE
        print("\n4/7 - Ejecutando KAZE...")
        try:
            resultado_kaze = self.hog_kaze_analyzer.extraer_caracteristicas_kaze(
                self.imagen_actual, 
                visualizar=True, 
                mostrar_descriptores=True, 
                guardar_resultados=True,
                nombre_imagen=f"{nombre_imagen}_kaze"
            )
            metodos_ejecutados.append("KAZE")
            print("KAZE completado")
        except Exception as e:
            error_msg = f"❌ Error en KAZE: {e}"
            print(error_msg)
            errores.append(error_msg)
        
        # 5. AKAZE
        print("\n5/7 - Ejecutando AKAZE...")
        try:
            resultado_akaze = self.advanced_analyzer.extraer_caracteristicas_akaze(
                self.imagen_actual, 
                visualizar=True, 
                mostrar_descriptores=True, 
                guardar_resultados=True,
                nombre_imagen=f"{nombre_imagen}_akaze"
            )
            metodos_ejecutados.append("AKAZE")
            print("AKAZE completado")
        except Exception as e:
            error_msg = f"❌ Error en AKAZE: {e}"
            print(error_msg)
            errores.append(error_msg)
        
        # 6. FREAK (Fast Retina Keypoint)
        print("\n6/7 - Ejecutando FREAK...")
        try:
            resultado_freak = self.advanced_analyzer.extraer_caracteristicas_freak(
                self.imagen_actual, 
                visualizar=True, 
                mostrar_descriptores=True, 
                guardar_resultados=True,
                nombre_imagen=f"{nombre_imagen}_freak"
            )
            metodos_ejecutados.append("FREAK")
            print("FREAK completado")
        except Exception as e:
            error_msg = f"❌ Error en FREAK: {e}"
            print(error_msg)
            errores.append(error_msg)
        
        # 7. GrabCut Segmentation
        print("\n7/7 - Ejecutando GrabCut...")
        try:
            resultado_grabcut = self.advanced_analyzer.analizar_grabcut_segmentation(
                self.imagen_actual, 
                visualizar=True, 
                mostrar_descriptores=True, 
                guardar_resultados=True,
                nombre_imagen=f"{nombre_imagen}_grabcut"
            )
            metodos_ejecutados.append("GrabCut")
            print("GrabCut completado")
        except Exception as e:
            error_msg = f"❌ Error en GrabCut: {e}"
            print(error_msg)
            errores.append(error_msg)
        
        # Resumen final
        print("\n" + "="*80)
        print("RESUMEN DEL ANÁLISIS COMPLETO")
        print("="*80)
        print(f"Métodos ejecutados exitosamente ({len(metodos_ejecutados)}/7):")
        for metodo in metodos_ejecutados:
            print(f"   ✓ {metodo}")
        
        if errores:
            print(f"\n❌ Errores encontrados ({len(errores)}):")
            for error in errores:
                print(f"   • {error}")
        
        print(f"\nLos resultados se han guardado en: {self.directorio_resultados}")
        print("   • Archivos CSV con estadísticas detalladas")
        print("   • Archivos TXT con reportes textuales")
        print("   • Imágenes con visualizaciones (si están habilitadas)")
        
        print("\nAnálisis completo de características terminado")
        print("="*80)

def main():
    """Función principal."""
    try:
        sistema = SistemaDeteccionVehicular()
        sistema.ejecutar_sistema()
    except KeyboardInterrupt:
        print("\n\n¡Sistema interrumpido por el usuario!")
    except Exception as e:
        print(f"\nError crítico del sistema: {e}")
        print("Verifique que todas las dependencias estén instaladas correctamente")

if __name__ == "__main__":
    main()