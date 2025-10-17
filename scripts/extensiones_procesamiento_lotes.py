#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extensión del Procesador de Lotes - Métodos Múltiples
======================================================

Extensión que permite procesar lotes de imágenes utilizando
TODOS los métodos de detección disponibles para cada tipo de objeto.
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
import time

def agregar_procesamiento_multiples_metodos():
    """
    Agrega capacidades de procesamiento con todos los métodos al ProcesadorLotes.
    """
    from detectores.procesador_lotes import ProcesadorLotes
    
    # Importar las extensiones de métodos múltiples
    from .extensiones_multiples_llantas import agregar_metodos_multiples_llantas
    from .extensiones_multiples_senales import agregar_metodos_multiples_senales
    from .extensiones_multiples_semaforos import agregar_metodos_multiples_semaforos
    
    def procesar_carpeta_todos_metodos(self, carpeta_imagenes, tipos_deteccion=None, 
                                     guardar_imagenes=True, generar_reporte=True):
        """
        Procesa todas las imágenes de una carpeta usando TODOS los métodos disponibles.
        
        Args:
            carpeta_imagenes (str): Ruta de la carpeta con imágenes
            tipos_deteccion (list): Lista de tipos ['llantas', 'senales', 'semaforos']
            guardar_imagenes (bool): Si guardar imágenes procesadas
            generar_reporte (bool): Si generar reporte final
            
        Returns:
            dict: Resumen del procesamiento con todos los métodos
        """
        if tipos_deteccion is None:
            tipos_deteccion = ['llantas', 'senales', 'semaforos']
        
        print(f"\nIniciando procesamiento por lotes con TODOS los métodos")
        print(f"Carpeta: {carpeta_imagenes}")
        print(f"Tipos de detección: {', '.join(tipos_deteccion)}")
        print(f"Modo: TODOS los métodos por tipo de objeto")
        
        # Asegurar que las extensiones estén cargadas
        agregar_metodos_multiples_llantas()
        agregar_metodos_multiples_senales()
        agregar_metodos_multiples_semaforos()
        
        # Obtener lista de imágenes
        from utils.image_utils import ImageHandler
        imagenes = ImageHandler.obtener_imagenes_carpeta(carpeta_imagenes)
        
        if not imagenes:
            print(f"No se encontraron imágenes en: {carpeta_imagenes}")
            return None
        
        print(f"Encontradas {len(imagenes)} imágenes para procesar")
        
        # Inicializar estadísticas mejoradas
        estadisticas_completas = {
            'imagenes_procesadas': 0,
            'imagenes_con_error': 0,
            'tiempo_total': 0,
            'timestamp_inicio': datetime.now().isoformat(),
            'resultados_por_imagen': {},
            'resumen_por_metodo': {},
            'errores': []
        }
        
        # Inicializar resumen por método
        if 'llantas' in tipos_deteccion:
            for metodo in ['hough', 'akaze', 'textura', 'combinado']:
                estadisticas_completas['resumen_por_metodo'][f'llantas_{metodo}'] = {
                    'total_detecciones': 0,
                    'imagenes_exitosas': 0,
                    'tiempo_total': 0,
                    'errores': 0
                }
        
        if 'senales' in tipos_deteccion:
            for metodo in ['hough', 'freak', 'color', 'log', 'combinado']:
                estadisticas_completas['resumen_por_metodo'][f'senales_{metodo}'] = {
                    'total_detecciones': 0,
                    'imagenes_exitosas': 0,
                    'tiempo_total': 0,
                    'errores': 0
                }
        
        if 'semaforos' in tipos_deteccion:
            for metodo in ['color', 'estructura', 'grabcut', 'combinado']:
                estadisticas_completas['resumen_por_metodo'][f'semaforos_{metodo}'] = {
                    'total_detecciones': 0,
                    'imagenes_exitosas': 0,
                    'tiempo_total': 0,
                    'errores': 0
                }
        
        tiempo_inicio_total = time.time()
        
        # Procesar cada imagen
        for i, ruta_imagen in enumerate(imagenes, 1):
            nombre_imagen = os.path.basename(ruta_imagen)
            print(f"\nProcesando imagen {i}/{len(imagenes)}: {nombre_imagen}")
            
            try:
                # Cargar imagen
                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
                
                # Inicializar resultados para esta imagen
                resultados_imagen = {
                    'nombre_archivo': nombre_imagen,
                    'ruta_original': ruta_imagen,
                    'timestamp': datetime.now().isoformat(),
                    'dimensiones': {'width': imagen.shape[1], 'height': imagen.shape[0]},
                    'resultados_por_tipo': {}
                }
                
                # Procesar cada tipo de detección
                for tipo in tipos_deteccion:
                    print(f"Procesando {tipo}...")
                    
                    try:
                        if tipo == 'llantas':
                            resultado = self.detector_llantas.detectar_llantas_todos_metodos(
                                imagen, visualizar=False, guardar=guardar_imagenes,
                                ruta_base=self.directorio_salida
                            )
                            
                        elif tipo == 'senales':
                            resultado = self.detector_senales.detectar_senales_todos_metodos(
                                imagen, visualizar=False, guardar=guardar_imagenes,
                                ruta_base=self.directorio_salida
                            )
                            
                        elif tipo == 'semaforos':
                            resultado = self.detector_semaforos.detectar_semaforos_todos_metodos(
                                imagen, visualizar=False, guardar=guardar_imagenes,
                                ruta_base=self.directorio_salida
                            )
                        
                        if resultado:
                            resultados_imagen['resultados_por_tipo'][tipo] = resultado
                            
                            # Actualizar estadísticas por método
                            for metodo, res in resultado.items():
                                clave_metodo = f'{tipo}_{metodo}'
                                if clave_metodo in estadisticas_completas['resumen_por_metodo']:
                                    stats = estadisticas_completas['resumen_por_metodo'][clave_metodo]
                                    
                                    if 'error' not in res:
                                        # Determinar clave de objetos detectados según el tipo
                                        if tipo == 'llantas':
                                            objetos_detectados = res.get('llantas_detectadas', [])
                                        elif tipo == 'senales':
                                            objetos_detectados = res.get('senales_detectadas', [])
                                        elif tipo == 'semaforos':
                                            objetos_detectados = res.get('semaforos_detectados', [])
                                        else:
                                            objetos_detectados = []
                                        
                                        stats['total_detecciones'] += len(objetos_detectados)
                                        stats['imagenes_exitosas'] += 1
                                        stats['tiempo_total'] += res.get('tiempo_ejecucion', 0)
                                        
                                        print(f"{metodo}: {len(objetos_detectados)} {tipo} detectados")
                                    else:
                                        stats['errores'] += 1
                                        print(f"{metodo}: Error - {res.get('error', 'Desconocido')}")
                        else:
                            print(f"Error procesando {tipo}")
                            resultados_imagen['resultados_por_tipo'][tipo] = {'error': 'Falló el procesamiento'}
                            
                    except Exception as e:
                        error_msg = f"Error procesando {tipo} en {nombre_imagen}: {e}"
                        print(f"{error_msg}")
                        estadisticas_completas['errores'].append(error_msg)
                        resultados_imagen['resultados_por_tipo'][tipo] = {'error': str(e)}
                
                # Guardar resultados de esta imagen
                estadisticas_completas['resultados_por_imagen'][nombre_imagen] = resultados_imagen
                estadisticas_completas['imagenes_procesadas'] += 1
                
                print(f"Imagen {nombre_imagen} procesada exitosamente")
                
            except Exception as e:
                error_msg = f"Error general procesando {nombre_imagen}: {e}"
                print(f"{error_msg}")
                estadisticas_completas['errores'].append(error_msg)
                estadisticas_completas['imagenes_con_error'] += 1
        
        # Finalizar estadísticas
        estadisticas_completas['tiempo_total'] = time.time() - tiempo_inicio_total
        estadisticas_completas['timestamp_fin'] = datetime.now().isoformat()
        
        # Generar reporte si se solicita
        if generar_reporte:
            self._generar_reporte_completo_multiples_metodos(estadisticas_completas)
        
        # Mostrar resumen
        self._mostrar_resumen_procesamiento_completo(estadisticas_completas)
        
        return estadisticas_completas
    
    def _generar_reporte_completo_multiples_metodos(self, estadisticas):
        """
        Genera un reporte completo del procesamiento con múltiples métodos.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Crear directorio de reportes si no existe
            dir_reportes = os.path.join(self.directorio_salida, "reportes")
            os.makedirs(dir_reportes, exist_ok=True)
            
            # Generar reporte JSON detallado
            nombre_json = f"procesamiento_completo_multiples_metodos_{timestamp}.json"
            ruta_json = os.path.join(dir_reportes, nombre_json)
            
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(estadisticas, f, indent=2, ensure_ascii=False)
            
            print(f"Reporte JSON guardado: {ruta_json}")
            
            # Generar reporte de texto legible
            nombre_txt = f"resumen_procesamiento_multiples_metodos_{timestamp}.txt"
            ruta_txt = os.path.join(dir_reportes, nombre_txt)
            
            with open(ruta_txt, 'w', encoding='utf-8') as f:
                f.write("REPORTE DE PROCESAMIENTO POR LOTES - MÚLTIPLES MÉTODOS\n")
                f.write("=" * 70 + "\n")
                f.write(f"Fecha de procesamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Tiempo total de procesamiento: {estadisticas['tiempo_total']:.2f} segundos\n")
                f.write(f"Imágenes procesadas: {estadisticas['imagenes_procesadas']}\n")
                f.write(f"Imágenes con error: {estadisticas['imagenes_con_error']}\n\n")
                
                # Resumen por método
                f.write("RESUMEN POR MÉTODO:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Método':<20} {'Detecciones':<12} {'Éxito':<8} {'Tiempo (s)':<12} {'Errores':<8}\n")
                f.write("-" * 50 + "\n")
                
                for metodo, stats in estadisticas['resumen_por_metodo'].items():
                    f.write(f"{metodo:<20} {stats['total_detecciones']:<12} "
                           f"{stats['imagenes_exitosas']:<8} {stats['tiempo_total']:<12.3f} "
                           f"{stats['errores']:<8}\n")
                
                # Detalles por imagen
                f.write(f"\n\nDETALLES POR IMAGEN:\n")
                f.write("-" * 30 + "\n")
                
                for nombre_imagen, resultado in estadisticas['resultados_por_imagen'].items():
                    f.write(f"\nImagen: {nombre_imagen}\n")
                    f.write(f"  Dimensiones: {resultado['dimensiones']['width']}x{resultado['dimensiones']['height']}\n")
                    
                    for tipo, resultados_tipo in resultado['resultados_por_tipo'].items():
                        f.write(f"  {tipo.upper()}:\n")
                        if 'error' in resultados_tipo:
                            f.write(f"    ERROR: {resultados_tipo['error']}\n")
                        else:
                            for metodo, res in resultados_tipo.items():
                                if 'error' not in res:
                                    if tipo == 'llantas':
                                        num_obj = len(res.get('llantas_detectadas', []))
                                    elif tipo == 'senales':
                                        num_obj = len(res.get('senales_detectadas', []))
                                    elif tipo == 'semaforos':
                                        num_obj = len(res.get('semaforos_detectados', []))
                                    else:
                                        num_obj = 0
                                    
                                    f.write(f"    {metodo}: {num_obj} objetos ({res.get('tiempo_ejecucion', 0):.3f}s)\n")
                                else:
                                    f.write(f"    {metodo}: ERROR\n")
                
                # Errores
                if estadisticas['errores']:
                    f.write(f"\n\nERRORES ENCONTRADOS:\n")
                    f.write("-" * 20 + "\n")
                    for error in estadisticas['errores']:
                        f.write(f"  - {error}\n")
            
            print(f"Reporte de texto guardado: {ruta_txt}")
            
        except Exception as e:
            print(f"Error generando reporte: {e}")
    
    def _mostrar_resumen_procesamiento_completo(self, estadisticas):
        """
        Muestra un resumen en consola del procesamiento completo.
        """
        print(f"\nPROCESAMIENTO COMPLETO FINALIZADO")
        print("=" * 60)
        print(f"Tiempo total: {estadisticas['tiempo_total']:.2f} segundos")
        print(f"Imágenes procesadas: {estadisticas['imagenes_procesadas']}")
        print(f"Imágenes con error: {estadisticas['imagenes_con_error']}")
        
        print(f"\nRESUMEN POR MÉTODO:")
        print("-" * 50)
        print(f"{'Método':<20} {'Detecciones':<12} {'Éxito':<8} {'Errores':<8}")
        print("-" * 50)
        
        for metodo, stats in estadisticas['resumen_por_metodo'].items():
            print(f"{metodo:<20} {stats['total_detecciones']:<12} "
                  f"{stats['imagenes_exitosas']:<8} {stats['errores']:<8}")
        
        # Método más efectivo
        mejor_detecciones = max(estadisticas['resumen_por_metodo'].items(), 
                               key=lambda x: x[1]['total_detecciones'])
        mejor_tiempo = min(estadisticas['resumen_por_metodo'].items(),
                          key=lambda x: x[1]['tiempo_total'] if x[1]['tiempo_total'] > 0 else float('inf'))
        
        print(f"\n🏆 DESTACADOS:")
        print(f"   Más detecciones: {mejor_detecciones[0]} ({mejor_detecciones[1]['total_detecciones']} objetos)")
        print(f"   Más rápido: {mejor_tiempo[0]} ({mejor_tiempo[1]['tiempo_total']:.3f}s)")
        
        print(f"\nResultados guardados en: {self.directorio_salida}/")
        print(f"Reportes detallados en: {self.directorio_salida}/reportes/")
    
    # Agregar el método a la clase ProcesadorLotes
    ProcesadorLotes.procesar_carpeta_todos_metodos = procesar_carpeta_todos_metodos
    ProcesadorLotes._generar_reporte_completo_multiples_metodos = _generar_reporte_completo_multiples_metodos
    ProcesadorLotes._mostrar_resumen_procesamiento_completo = _mostrar_resumen_procesamiento_completo
    
    print("Extensión de procesamiento con múltiples métodos agregada")

if __name__ == "__main__":
    # Prueba del sistema
    agregar_procesamiento_multiples_metodos()
    print("Sistema de procesamiento por lotes con múltiples métodos listo.")