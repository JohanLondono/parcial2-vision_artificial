#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Procesamiento por Lotes
==================================

Procesamiento automático de múltiples imágenes con guardado
de resultados y generación de reportes.
"""

import cv2
import numpy as np
import os
import json
import csv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# Importar detectores
from .detector_llantas import DetectorLlantas
from .detector_senales import DetectorSenales
from .detector_semaforos import DetectorSemaforos

# Importar utilidades
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import ImageHandler

class ProcesadorLotes:
    """Sistema para procesar múltiples imágenes automáticamente."""
    
    def __init__(self, directorio_salida="./resultados_deteccion"):
        """
        Inicializar el procesador por lotes.
        
        Args:
            directorio_salida (str): Directorio donde guardar resultados
        """
        self.directorio_salida = directorio_salida
        self.setup_directorios()
        
        # Inicializar detectores
        self.detector_llantas = DetectorLlantas()
        self.detector_senales = DetectorSenales()
        self.detector_semaforos = DetectorSemaforos()
        
        # Estadísticas del procesamiento
        self.estadisticas = {
            'imagenes_procesadas': 0,
            'imagenes_con_error': 0,
            'total_llantas': 0,
            'total_senales': 0,
            'total_semaforos': 0,
            'tiempo_total': 0,
            'errores': []
        }
    
    def setup_directorios(self):
        """Crear estructura de directorios para resultados."""
        directorios = [
            self.directorio_salida,
            os.path.join(self.directorio_salida, "llantas"),
            os.path.join(self.directorio_salida, "senales"),
            os.path.join(self.directorio_salida, "semaforos"),
            os.path.join(self.directorio_salida, "reportes"),
            os.path.join(self.directorio_salida, "logs")
        ]
        
        for directorio in directorios:
            os.makedirs(directorio, exist_ok=True)
        
        print(f"Directorios de salida creados en: {self.directorio_salida}")
    
    def procesar_carpeta(self, carpeta_imagenes, tipos_deteccion=None, metodos=None, 
                        guardar_imagenes=True, generar_reporte=True):
        """
        Procesa todas las imágenes de una carpeta.
        
        Args:
            carpeta_imagenes (str): Ruta de la carpeta con imágenes
            tipos_deteccion (list): Lista de tipos ['llantas', 'senales', 'semaforos']
            metodos (dict): Métodos por tipo {'llantas': 'hough', 'senales': 'color'}
            guardar_imagenes (bool): Si guardar imágenes procesadas
            generar_reporte (bool): Si generar reporte final
            
        Returns:
            dict: Resumen del procesamiento
        """
        if tipos_deteccion is None:
            tipos_deteccion = ['llantas', 'senales', 'semaforos']
        
        if metodos is None:
            metodos = {
                'llantas': 'combinado',
                'senales': 'combinado', 
                'semaforos': 'combinado'
            }
        
        print(f"\nIniciando procesamiento por lotes")
        print(f"Carpeta: {carpeta_imagenes}")
        print(f"Tipos de detección: {tipos_deteccion}")
        print(f"Métodos: {metodos}")
        print(f"Guardar imágenes: {guardar_imagenes}")
        
        # Obtener lista de imágenes
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta_imagenes)
        
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta_imagenes}")
            return None
        
        print(f"Encontradas {len(imagenes)} imágenes para procesar")
        
        # Reiniciar estadísticas
        self.estadisticas = {
            'imagenes_procesadas': 0,
            'imagenes_con_error': 0,
            'total_llantas': 0,
            'total_senales': 0,
            'total_semaforos': 0,
            'tiempo_total': 0,
            'errores': [],
            'resultados_detallados': []
        }
        
        tiempo_inicio = datetime.now()
        
        # Procesar cada imagen
        for i, ruta_imagen in enumerate(imagenes):
            print(f"\nProcesando imagen {i+1}/{len(imagenes)}: {os.path.basename(ruta_imagen)}")
            
            try:
                resultado_imagen = self.procesar_imagen_individual(
                    ruta_imagen, tipos_deteccion, metodos, guardar_imagenes
                )
                
                if resultado_imagen:
                    self.estadisticas['resultados_detallados'].append(resultado_imagen)
                    self.estadisticas['imagenes_procesadas'] += 1
                    
                    # Acumular conteos
                    if 'llantas' in resultado_imagen:
                        self.estadisticas['total_llantas'] += resultado_imagen['llantas']['num_detecciones']
                    if 'senales' in resultado_imagen:
                        self.estadisticas['total_senales'] += resultado_imagen['senales']['num_detecciones']
                    if 'semaforos' in resultado_imagen:
                        self.estadisticas['total_semaforos'] += resultado_imagen['semaforos']['num_detecciones']
                else:
                    self.estadisticas['imagenes_con_error'] += 1
                    
            except Exception as e:
                error_msg = f"Error procesando {os.path.basename(ruta_imagen)}: {str(e)}"
                print(f"{error_msg}")
                self.estadisticas['errores'].append(error_msg)
                self.estadisticas['imagenes_con_error'] += 1
            
            # Mostrar progreso
            progreso = (i + 1) / len(imagenes) * 100
            print(f"Progreso: {progreso:.1f}% ({i+1}/{len(imagenes)})")
        
        # Calcular tiempo total
        tiempo_fin = datetime.now()
        self.estadisticas['tiempo_total'] = (tiempo_fin - tiempo_inicio).total_seconds()
        
        # Generar reporte si se solicita
        if generar_reporte:
            self.generar_reporte_final()
        
        # Mostrar resumen
        self.mostrar_resumen()
        
        return self.estadisticas
    
    def procesar_imagen_individual(self, ruta_imagen, tipos_deteccion, metodos, guardar_imagenes):
        """
        Procesa una imagen individual con los detectores especificados.
        
        Args:
            ruta_imagen (str): Ruta de la imagen
            tipos_deteccion (list): Tipos de detección a realizar
            metodos (dict): Métodos por tipo
            guardar_imagenes (bool): Si guardar resultados
            
        Returns:
            dict: Resultados de la detección
        """
        # Cargar imagen
        imagen = ImageHandler.cargar_imagen(ruta_imagen)
        if imagen is None:
            return None
        
        nombre_archivo = os.path.splitext(os.path.basename(ruta_imagen))[0]
        resultado_imagen = {
            'nombre_archivo': nombre_archivo,
            'ruta_original': ruta_imagen,
            'timestamp': datetime.now().isoformat()
        }
        
        # Procesar cada tipo de detección
        for tipo in tipos_deteccion:
            print(f"  🔍 Detectando {tipo}...")
            
            metodo = metodos.get(tipo, 'combinado')
            
            try:
                if tipo == 'llantas':
                    resultado = self.detector_llantas.detectar_llantas(
                        imagen, metodo=metodo, visualizar=False, 
                        guardar=False, ruta_salida=None
                    )
                    
                    if resultado and guardar_imagenes:
                        ruta_salida = os.path.join(self.directorio_salida, "llantas", 
                                                 f"{nombre_archivo}_llantas_{metodo}.jpg")
                        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
                        cv2.imwrite(ruta_salida, resultado['imagen_resultado'])
                    
                elif tipo == 'senales':
                    resultado = self.detector_senales.detectar_senales(
                        imagen, metodo=metodo, visualizar=False,
                        guardar=False, ruta_salida=None
                    )
                    
                    if resultado and guardar_imagenes:
                        ruta_salida = os.path.join(self.directorio_salida, "senales",
                                                 f"{nombre_archivo}_senales_{metodo}.jpg")
                        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
                        cv2.imwrite(ruta_salida, resultado['imagen_resultado'])
                    
                elif tipo == 'semaforos':
                    resultado = self.detector_semaforos.detectar_semaforos(
                        imagen, metodo=metodo, visualizar=False,
                        guardar=False, ruta_salida=None
                    )
                    
                    if resultado and guardar_imagenes:
                        ruta_salida = os.path.join(self.directorio_salida, "semaforos",
                                                 f"{nombre_archivo}_semaforos_{metodo}.jpg")
                        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
                        cv2.imwrite(ruta_salida, resultado['imagen_resultado'])
                
                # Procesar resultado
                if resultado:
                    resultado_procesado = self.procesar_resultado_deteccion(resultado, tipo)
                    resultado_imagen[tipo] = resultado_procesado
                    print(f"{tipo}: {resultado_procesado['num_detecciones']} detecciones")
                else:
                    resultado_imagen[tipo] = {'num_detecciones': 0, 'error': 'Sin resultado'}
                    print(f"{tipo}: Sin detecciones")
                    
            except Exception as e:
                error_msg = f"Error en detección de {tipo}: {str(e)}"
                resultado_imagen[tipo] = {'num_detecciones': 0, 'error': error_msg}
                print(f"{tipo}: Error - {error_msg}")
        
        return resultado_imagen
    
    def procesar_resultado_deteccion(self, resultado, tipo):
        """
        Procesa y limpia un resultado de detección para almacenamiento.
        
        Args:
            resultado (dict): Resultado crudo del detector
            tipo (str): Tipo de detección
            
        Returns:
            dict: Resultado procesado
        """
        resultado_limpio = {
            'metodo': resultado.get('metodo', 'desconocido'),
            'num_detecciones': 0,
            'confianza_promedio': resultado.get('confianza_promedio', 0.0),
            'detecciones': []
        }
        
        # Extraer detecciones según el tipo
        if tipo == 'llantas':
            llantas = resultado.get('llantas', [])
            resultado_limpio['num_detecciones'] = len(llantas)
            
            for llanta in llantas:
                if len(llanta) >= 3:
                    x, y, r = llanta[:3]
                    resultado_limpio['detecciones'].append({
                        'centro_x': int(x),
                        'centro_y': int(y),
                        'radio': int(r),
                        'tipo': 'llanta'
                    })
        
        elif tipo == 'senales':
            senales = resultado.get('senales', [])
            resultado_limpio['num_detecciones'] = len(senales)
            
            for senal in senales:
                if len(senal) >= 3:
                    x, y, r = senal[:3]
                    tipo_senal = senal[3] if len(senal) > 3 else 'desconocida'
                    resultado_limpio['detecciones'].append({
                        'centro_x': int(x),
                        'centro_y': int(y),
                        'radio': int(r),
                        'tipo': tipo_senal
                    })
        
        elif tipo == 'semaforos':
            semaforos = resultado.get('semaforos', [])
            resultado_limpio['num_detecciones'] = len(semaforos)
            
            for semaforo in semaforos:
                if 'bbox' in semaforo:
                    x_min, y_min, x_max, y_max = semaforo['bbox']
                    deteccion = {
                        'bbox_x': int(x_min),
                        'bbox_y': int(y_min),
                        'bbox_width': int(x_max - x_min),
                        'bbox_height': int(y_max - y_min),
                        'tipo': 'semaforo'
                    }
                    
                    if 'num_luces' in semaforo:
                        deteccion['num_luces'] = semaforo['num_luces']
                    if 'colores_detectados' in semaforo:
                        deteccion['colores'] = semaforo['colores_detectados']
                    
                    resultado_limpio['detecciones'].append(deteccion)
        
        return resultado_limpio
    
    def generar_reporte_final(self):
        """Genera reportes finales en múltiples formatos."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Reporte JSON completo
        ruta_json = os.path.join(self.directorio_salida, "reportes", f"reporte_completo_{timestamp}.json")
        with open(ruta_json, 'w', encoding='utf-8') as f:
            json.dump(self.estadisticas, f, indent=2, ensure_ascii=False)
        
        # Reporte CSV resumido
        ruta_csv = os.path.join(self.directorio_salida, "reportes", f"resumen_{timestamp}.csv")
        self.generar_csv_resumen(ruta_csv)
        
        # Reporte de texto
        ruta_txt = os.path.join(self.directorio_salida, "reportes", f"resumen_{timestamp}.txt")
        self.generar_txt_resumen(ruta_txt)
        
        # Gráficos estadísticos
        self.generar_graficos_estadisticos(timestamp)
        
        print(f"\n📋 Reportes generados:")
        print(f"  - JSON completo: {ruta_json}")
        print(f"  - CSV resumido: {ruta_csv}")
        print(f"  - TXT resumido: {ruta_txt}")
    
    def generar_csv_resumen(self, ruta_csv):
        """Genera archivo CSV con resumen por imagen."""
        with open(ruta_csv, 'w', newline='', encoding='utf-8') as csvfile:
            campos = [
                'nombre_archivo', 'llantas_detectadas', 'senales_detectadas', 
                'semaforos_detectados', 'confianza_llantas', 'confianza_senales', 
                'confianza_semaforos', 'timestamp'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=campos)
            writer.writeheader()
            
            for resultado in self.estadisticas.get('resultados_detallados', []):
                fila = {
                    'nombre_archivo': resultado.get('nombre_archivo', ''),
                    'llantas_detectadas': resultado.get('llantas', {}).get('num_detecciones', 0),
                    'senales_detectadas': resultado.get('senales', {}).get('num_detecciones', 0),
                    'semaforos_detectados': resultado.get('semaforos', {}).get('num_detecciones', 0),
                    'confianza_llantas': resultado.get('llantas', {}).get('confianza_promedio', 0),
                    'confianza_senales': resultado.get('senales', {}).get('confianza_promedio', 0),
                    'confianza_semaforos': resultado.get('semaforos', {}).get('confianza_promedio', 0),
                    'timestamp': resultado.get('timestamp', '')
                }
                writer.writerow(fila)
    
    def generar_txt_resumen(self, ruta_txt):
        """Genera archivo de texto con resumen del procesamiento."""
        with open(ruta_txt, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE PROCESAMIENTO POR LOTES\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tiempo total de procesamiento: {self.estadisticas['tiempo_total']:.2f} segundos\n\n")
            
            f.write("ESTADÍSTICAS GENERALES:\n")
            f.write(f"  - Imágenes procesadas: {self.estadisticas['imagenes_procesadas']}\n")
            f.write(f"  - Imágenes con error: {self.estadisticas['imagenes_con_error']}\n")
            f.write(f"  - Total llantas detectadas: {self.estadisticas['total_llantas']}\n")
            f.write(f"  - Total señales detectadas: {self.estadisticas['total_senales']}\n")
            f.write(f"  - Total semáforos detectados: {self.estadisticas['total_semaforos']}\n\n")
            
            if self.estadisticas['errores']:
                f.write("ERRORES ENCONTRADOS:\n")
                for error in self.estadisticas['errores']:
                    f.write(f"  - {error}\n")
                f.write("\n")
            
            f.write("DETALLE POR IMAGEN:\n")
            for resultado in self.estadisticas.get('resultados_detallados', []):
                f.write(f"\n{resultado['nombre_archivo']}:\n")
                
                if 'llantas' in resultado:
                    f.write(f"  Llantas: {resultado['llantas']['num_detecciones']}\n")
                if 'senales' in resultado:
                    f.write(f"  Señales: {resultado['senales']['num_detecciones']}\n")
                if 'semaforos' in resultado:
                    f.write(f"  Semáforos: {resultado['semaforos']['num_detecciones']}\n")
    
    def generar_graficos_estadisticos(self, timestamp):
        """Genera gráficos estadísticos del procesamiento."""
        if not self.estadisticas.get('resultados_detallados'):
            return
        
        # Preparar datos
        nombres = []
        llantas = []
        senales = []
        semaforos = []
        
        for resultado in self.estadisticas['resultados_detallados']:
            nombres.append(resultado['nombre_archivo'][:15])  # Truncar nombres largos
            llantas.append(resultado.get('llantas', {}).get('num_detecciones', 0))
            senales.append(resultado.get('senales', {}).get('num_detecciones', 0))
            semaforos.append(resultado.get('semaforos', {}).get('num_detecciones', 0))
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Estadísticas de Detección por Lotes', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Detecciones por imagen
        x_pos = np.arange(len(nombres))
        width = 0.25
        
        axes[0, 0].bar(x_pos - width, llantas, width, label='Llantas', color='blue', alpha=0.7)
        axes[0, 0].bar(x_pos, senales, width, label='Señales', color='red', alpha=0.7)
        axes[0, 0].bar(x_pos + width, semaforos, width, label='Semáforos', color='green', alpha=0.7)
        axes[0, 0].set_xlabel('Imágenes')
        axes[0, 0].set_ylabel('Número de Detecciones')
        axes[0, 0].set_title('Detecciones por Imagen')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(nombres, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Gráfico 2: Totales por tipo
        tipos = ['Llantas', 'Señales', 'Semáforos']
        totales = [
            self.estadisticas['total_llantas'],
            self.estadisticas['total_senales'],
            self.estadisticas['total_semaforos']
        ]
        colores = ['blue', 'red', 'green']
        
        axes[0, 1].bar(tipos, totales, color=colores, alpha=0.7)
        axes[0, 1].set_ylabel('Total de Detecciones')
        axes[0, 1].set_title('Totales por Tipo de Objeto')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for i, total in enumerate(totales):
            axes[0, 1].text(i, total + max(totales)*0.01, str(total), 
                           ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 3: Distribución de éxito
        labels = ['Procesadas', 'Con Error']
        sizes = [self.estadisticas['imagenes_procesadas'], self.estadisticas['imagenes_con_error']]
        colors = ['lightgreen', 'lightcoral']
        
        if sum(sizes) > 0:
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Tasa de Éxito en Procesamiento')
        
        # Gráfico 4: Estadísticas de tiempo
        axes[1, 1].bar(['Tiempo Total'], [self.estadisticas['tiempo_total']], 
                      color='orange', alpha=0.7)
        axes[1, 1].set_ylabel('Tiempo (segundos)')
        axes[1, 1].set_title('Tiempo de Procesamiento')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Añadir información adicional
        if self.estadisticas['imagenes_procesadas'] > 0:
            tiempo_promedio = self.estadisticas['tiempo_total'] / self.estadisticas['imagenes_procesadas']
            axes[1, 1].text(0, self.estadisticas['tiempo_total'] * 0.5, 
                           f"Promedio por imagen:\n{tiempo_promedio:.2f}s", 
                           ha='center', va='center', fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Guardar gráfico
        ruta_grafico = os.path.join(self.directorio_salida, "reportes", f"estadisticas_{timestamp}.png")
        os.makedirs(os.path.dirname(ruta_grafico), exist_ok=True)
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Gráficos: {ruta_grafico}")
    
    def mostrar_resumen(self):
        """Muestra resumen del procesamiento en consola."""
        print(f"\n📊 RESUMEN DEL PROCESAMIENTO")
        print("="*50)
        print(f"Tiempo total: {self.estadisticas['tiempo_total']:.2f} segundos")
        print(f"Imágenes procesadas: {self.estadisticas['imagenes_procesadas']}")
        print(f"Imágenes con error: {self.estadisticas['imagenes_con_error']}")
        print(f"Total llantas: {self.estadisticas['total_llantas']}")
        print(f"Total señales: {self.estadisticas['total_senales']}")
        print(f"Total semáforos: {self.estadisticas['total_semaforos']}")
        
        if self.estadisticas['imagenes_procesadas'] > 0:
            tiempo_promedio = self.estadisticas['tiempo_total'] / self.estadisticas['imagenes_procesadas']
            print(f"⚡ Tiempo promedio por imagen: {tiempo_promedio:.2f} segundos")
        
        if self.estadisticas['errores']:
            print(f"\nErrores encontrados: {len(self.estadisticas['errores'])}")
        
        print("="*50)

# Función de utilidad
def procesar_carpeta_imagenes(carpeta_imagenes, tipos_deteccion=None, metodos=None,
                            directorio_salida="./resultados_deteccion", 
                            guardar_imagenes=True, generar_reporte=True):
    """
    Función de conveniencia para procesar una carpeta de imágenes.
    
    Args:
        carpeta_imagenes (str): Ruta de la carpeta con imágenes
        tipos_deteccion (list): Tipos de detección ['llantas', 'senales', 'semaforos']
        metodos (dict): Métodos por tipo
        directorio_salida (str): Directorio de salida
        guardar_imagenes (bool): Si guardar imágenes procesadas
        generar_reporte (bool): Si generar reporte
        
    Returns:
        dict: Estadísticas del procesamiento
    """
    procesador = ProcesadorLotes(directorio_salida)
    return procesador.procesar_carpeta(
        carpeta_imagenes, tipos_deteccion, metodos, 
        guardar_imagenes, generar_reporte
    )