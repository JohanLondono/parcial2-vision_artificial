#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extensión del Detector de Llantas - Métodos Múltiples
=====================================================

Sistema que ejecuta TODOS los métodos de detección de llantas
y guarda resultados detallados para cada método individual.
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time

def agregar_metodos_multiples_llantas():
    """
    Agrega métodos para ejecutar todos los algoritmos de detección de llantas.
    Esta función extiende la clase DetectorLlantas existente.
    """
    from detectores.detector_llantas import DetectorLlantas
    
    def detectar_llantas_todos_metodos(self, imagen, visualizar=False, guardar=True, ruta_base=None):
        """
        Ejecuta TODOS los métodos de detección de llantas y guarda resultados por separado.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imágenes resultado
            ruta_base (str): Ruta base donde guardar resultados
            
        Returns:
            dict: Resultados de todos los métodos
        """
        print("🔍 Ejecutando TODOS los métodos de detección de llantas...")
        
        # Definir métodos individuales (excluir combinado para evitar redundancia al final)
        metodos = ['hough', 'akaze', 'textura']
        resultados_completos = {}
        
        for metodo in metodos:
            print(f"\n  🔧 Ejecutando método: {metodo.upper()}")
            
            # Crear ruta de salida específica para este método
            if ruta_base and guardar:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"llantas_{metodo}_{timestamp}.jpg"
                ruta_salida = os.path.join(ruta_base, "llantas", nombre_archivo)
                os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
            else:
                ruta_salida = None
            
            # Ejecutar método específico
            try:
                inicio_tiempo = time.time()
                
                if metodo == 'hough':
                    resultado = self._detectar_llantas_hough(imagen, visualizar, guardar, ruta_salida)
                elif metodo == 'akaze':
                    resultado = self._detectar_llantas_akaze(imagen, visualizar, guardar, ruta_salida)
                elif metodo == 'textura':
                    resultado = self._detectar_llantas_textura(imagen, visualizar, guardar, ruta_salida)
                
                tiempo_ejecucion = time.time() - inicio_tiempo
                
                if resultado:
                    # Agregar información adicional
                    resultado['tiempo_ejecucion'] = tiempo_ejecucion
                    resultado['metodo_utilizado'] = metodo
                    resultado['imagen_info'] = {
                        'width': imagen.shape[1],
                        'height': imagen.shape[0],
                        'channels': imagen.shape[2] if len(imagen.shape) == 3 else 1
                    }
                    
                    resultados_completos[metodo] = resultado
                    print(f"    ✅ {metodo.upper()}: {len(resultado.get('llantas_detectadas', []))} llantas detectadas")
                    print(f"    ⏱️  Tiempo: {tiempo_ejecucion:.3f} segundos")
                    
                    # Guardar información detallada del método
                    if guardar and ruta_base:
                        self._guardar_info_deteccion_extendida(resultado, metodo, ruta_base)
                else:
                    resultados_completos[metodo] = {'error': 'Falló la detección', 'tiempo_ejecucion': tiempo_ejecucion}
                    print(f"    ❌ {metodo.upper()}: Error en detección")
                    
            except Exception as e:
                print(f"    ❌ {metodo.upper()}: Error - {e}")
                resultados_completos[metodo] = {'error': str(e)}
        
        # Ejecutar método combinado al final
        print(f"\n  🚀 Ejecutando método: COMBINADO")
        try:
            if ruta_base and guardar:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"llantas_combinado_{timestamp}.jpg"
                ruta_salida = os.path.join(ruta_base, "llantas", nombre_archivo)
            else:
                ruta_salida = None
                
            inicio_tiempo = time.time()
            resultado_combinado = self._detectar_llantas_combinado(imagen, visualizar, guardar, ruta_salida)
            tiempo_ejecucion = time.time() - inicio_tiempo
            
            if resultado_combinado:
                resultado_combinado['tiempo_ejecucion'] = tiempo_ejecucion
                resultado_combinado['metodo_utilizado'] = 'combinado'
                resultado_combinado['imagen_info'] = {
                    'width': imagen.shape[1],
                    'height': imagen.shape[0],
                    'channels': imagen.shape[2] if len(imagen.shape) == 3 else 1
                }
                
                resultados_completos['combinado'] = resultado_combinado
                print(f"    ✅ COMBINADO: {len(resultado_combinado.get('llantas_detectadas', []))} llantas detectadas")
                print(f"    ⏱️  Tiempo: {tiempo_ejecucion:.3f} segundos")
                
                if guardar and ruta_base:
                    self._guardar_info_deteccion_extendida(resultado_combinado, 'combinado', ruta_base)
            else:
                resultados_completos['combinado'] = {'error': 'Falló la detección combinada', 'tiempo_ejecucion': tiempo_ejecucion}
                print(f"    ❌ COMBINADO: Error en detección")
                
        except Exception as e:
            print(f"    ❌ COMBINADO: Error - {e}")
            resultados_completos['combinado'] = {'error': str(e)}
        
        # Generar reporte comparativo
        if guardar and ruta_base:
            self._generar_reporte_comparativo(resultados_completos, ruta_base)
        
        print(f"\n🎉 Detección completa de llantas finalizada. {len(resultados_completos)} métodos ejecutados.")
        return resultados_completos
    
    def _guardar_info_deteccion_extendida(self, resultado, metodo, ruta_base):
        """
        Guarda información detallada de la detección con análisis extendido.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Crear directorio para reportes
            dir_reportes = os.path.join(ruta_base, "reportes", "llantas")
            os.makedirs(dir_reportes, exist_ok=True)
            
            # Crear archivo de reporte
            nombre_reporte = f"deteccion_llantas_{metodo}_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write(f"REPORTE DETALLADO DE DETECCIÓN DE LLANTAS\n")
                f.write(f"MÉTODO: {metodo.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Método utilizado: {metodo}\n")
                f.write(f"Número de llantas detectadas: {len(resultado.get('llantas_detectadas', []))}\n")
                f.write(f"Tiempo de ejecución: {resultado.get('tiempo_ejecucion', 0):.3f} segundos\n\n")
                
                # Información de la imagen
                if 'imagen_info' in resultado:
                    info = resultado['imagen_info']
                    f.write("INFORMACIÓN DE LA IMAGEN:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  Dimensiones: {info.get('width', 'N/A')} x {info.get('height', 'N/A')} píxeles\n")
                    f.write(f"  Canales: {info.get('channels', 'N/A')}\n")
                    f.write(f"  Área total: {info.get('width', 0) * info.get('height', 0):,} píxeles\n\n")
                
                # Información específica del método
                if metodo == 'hough':
                    f.write("PARÁMETROS DE HOUGH CIRCLES:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  dp: {self.config['hough_circles']['dp']}\n")
                    f.write(f"  min_dist: {self.config['hough_circles']['min_dist']}\n")
                    f.write(f"  param1: {self.config['hough_circles']['param1']}\n")
                    f.write(f"  param2: {self.config['hough_circles']['param2']}\n")
                    f.write(f"  min_radius: {self.config['hough_circles']['min_radius']}\n")
                    f.write(f"  max_radius: {self.config['hough_circles']['max_radius']}\n\n")
                
                elif metodo == 'akaze':
                    f.write("PARÁMETROS DE AKAZE:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  threshold: {self.config['akaze']['threshold']}\n")
                    f.write(f"  nOctaves: {self.config['akaze']['nOctaves']}\n")
                    f.write(f"  nOctaveLayers: {self.config['akaze']['nOctaveLayers']}\n")
                    if 'num_keypoints' in resultado:
                        f.write(f"  Keypoints detectados: {resultado['num_keypoints']}\n")
                    if 'num_clusters' in resultado:
                        f.write(f"  Clusters formados: {resultado['num_clusters']}\n\n")
                
                elif metodo == 'textura':
                    f.write("ANÁLISIS DE TEXTURAS:\n")
                    f.write("-" * 30 + "\n")
                    if 'num_regiones' in resultado:
                        f.write(f"  Regiones analizadas: {resultado['num_regiones']}\n")
                    if 'criterio_textura' in resultado:
                        f.write(f"  Criterio aplicado: {resultado['criterio_textura']}\n\n")
                
                # Detalles de llantas detectadas
                if resultado.get('llantas_detectadas'):
                    f.write(f"DETALLES DE LLANTAS DETECTADAS:\n")
                    f.write("-" * 40 + "\n")
                    for i, llanta in enumerate(resultado['llantas_detectadas'], 1):
                        f.write(f"  Llanta {i}:\n")
                        f.write(f"    Centro: ({llanta[0]:.1f}, {llanta[1]:.1f})\n")
                        f.write(f"    Radio: {llanta[2]:.1f} píxeles\n")
                        if len(llanta) > 3:
                            f.write(f"    Confianza: {llanta[3]:.3f}\n")
                        
                        # Calcular área de la llanta
                        area = np.pi * (llanta[2] ** 2)
                        f.write(f"    Área: {area:.1f} píxeles²\n")
                        f.write("\n")
                
                # Estadísticas adicionales
                if resultado.get('llantas_detectadas'):
                    llantas = resultado['llantas_detectadas']
                    radios = [l[2] for l in llantas]
                    
                    f.write("ESTADÍSTICAS DE DETECCIÓN:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  Radio promedio: {np.mean(radios):.1f} píxeles\n")
                    f.write(f"  Radio mínimo: {np.min(radios):.1f} píxeles\n")
                    f.write(f"  Radio máximo: {np.max(radios):.1f} píxeles\n")
                    f.write(f"  Desviación estándar de radios: {np.std(radios):.1f} píxeles\n")
                
                # Información de máscara de segmentación (si está disponible)
                if 'mask' in resultado:
                    f.write(f"\nINFORMACIÓN DE SEGMENTACIÓN:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  Máscara generada: Sí\n")
                    f.write(f"  Píxeles segmentados: {np.count_nonzero(resultado['mask'])}\n")
                    f.write(f"  Porcentaje de imagen segmentada: {np.count_nonzero(resultado['mask']) / resultado['mask'].size * 100:.2f}%\n")
                
            print(f"    📄 Reporte detallado guardado: {ruta_reporte}")
            
        except Exception as e:
            print(f"    ⚠️  Error guardando reporte: {e}")
    
    def _generar_reporte_comparativo(self, resultados_completos, ruta_base):
        """
        Genera un reporte comparativo de todos los métodos ejecutados.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Crear directorio para reportes
            dir_reportes = os.path.join(ruta_base, "reportes", "comparativos")
            os.makedirs(dir_reportes, exist_ok=True)
            
            # Crear archivo de reporte comparativo
            nombre_reporte = f"comparativo_llantas_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write("REPORTE COMPARATIVO - DETECCIÓN DE LLANTAS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Métodos ejecutados: {len(resultados_completos)}\n\n")
                
                # Tabla resumen
                f.write("RESUMEN COMPARATIVO:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Método':<12} {'Llantas':<8} {'Tiempo (s)':<12} {'Estado':<10}\n")
                f.write("-" * 50 + "\n")
                
                for metodo, resultado in resultados_completos.items():
                    if 'error' in resultado:
                        f.write(f"{metodo.upper():<12} {'ERROR':<8} {'N/A':<12} {'Error':<10}\n")
                    else:
                        num_llantas = len(resultado.get('llantas_detectadas', []))
                        tiempo = resultado.get('tiempo_ejecucion', 0)
                        f.write(f"{metodo.upper():<12} {num_llantas:<8} {tiempo:<12.3f} {'OK':<10}\n")
                
                f.write("\n")
                
                # Análisis detallado por método
                for metodo, resultado in resultados_completos.items():
                    f.write(f"\nDETALLES - MÉTODO {metodo.upper()}:\n")
                    f.write("-" * 30 + "\n")
                    
                    if 'error' in resultado:
                        f.write(f"  Estado: ERROR - {resultado['error']}\n")
                    else:
                        llantas = resultado.get('llantas_detectadas', [])
                        f.write(f"  Llantas detectadas: {len(llantas)}\n")
                        f.write(f"  Tiempo de ejecución: {resultado.get('tiempo_ejecucion', 0):.3f} segundos\n")
                        
                        if llantas:
                            radios = [l[2] for l in llantas]
                            f.write(f"  Radio promedio: {np.mean(radios):.1f} píxeles\n")
                            f.write(f"  Rango de radios: {np.min(radios):.1f} - {np.max(radios):.1f} píxeles\n")
                
                # Recomendaciones
                f.write(f"\nRECOMENDACIONES:\n")
                f.write("-" * 20 + "\n")
                
                # Encontrar el método más rápido
                tiempos_validos = {m: r.get('tiempo_ejecucion', float('inf')) 
                                 for m, r in resultados_completos.items() 
                                 if 'error' not in r}
                if tiempos_validos:
                    metodo_rapido = min(tiempos_validos, key=tiempos_validos.get)
                    f.write(f"  Método más rápido: {metodo_rapido.upper()} ({tiempos_validos[metodo_rapido]:.3f}s)\n")
                
                # Encontrar el método con más detecciones
                detecciones = {m: len(r.get('llantas_detectadas', []))
                             for m, r in resultados_completos.items() 
                             if 'error' not in r}
                if detecciones:
                    metodo_mas_detecciones = max(detecciones, key=detecciones.get)
                    f.write(f"  Método con más detecciones: {metodo_mas_detecciones.upper()} ({detecciones[metodo_mas_detecciones]} llantas)\n")
                
            print(f"📊 Reporte comparativo guardado: {ruta_reporte}")
            
        except Exception as e:
            print(f"⚠️  Error generando reporte comparativo: {e}")
    
    # Agregar los métodos a la clase DetectorLlantas
    DetectorLlantas.detectar_llantas_todos_metodos = detectar_llantas_todos_metodos
    DetectorLlantas._guardar_info_deteccion_extendida = _guardar_info_deteccion_extendida
    DetectorLlantas._generar_reporte_comparativo = _generar_reporte_comparativo
    
    print("✅ Métodos múltiples agregados al DetectorLlantas")

# Función de utilidad para usar desde el sistema principal
def detectar_llantas_imagen_todos_metodos(ruta_imagen, ruta_salida="./resultados_deteccion"):
    """
    Función de utilidad para detectar llantas con todos los métodos.
    
    Args:
        ruta_imagen (str): Ruta de la imagen
        ruta_salida (str): Directorio de salida
        
    Returns:
        dict: Resultados de todos los métodos
    """
    from detectores.detector_llantas import DetectorLlantas
    
    # Asegurarse de que los métodos estén disponibles
    agregar_metodos_multiples_llantas()
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"❌ Error: No se pudo cargar la imagen: {ruta_imagen}")
        return None
    
    # Crear detector y ejecutar todos los métodos
    detector = DetectorLlantas()
    return detector.detectar_llantas_todos_metodos(imagen, visualizar=False, guardar=True, ruta_base=ruta_salida)

if __name__ == "__main__":
    # Prueba del sistema
    agregar_metodos_multiples_llantas()
    print("Sistema de métodos múltiples para llantas listo.")