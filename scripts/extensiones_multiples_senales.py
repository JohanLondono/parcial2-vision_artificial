#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extensión del Detector de Señales - Métodos Múltiples
=====================================================

Sistema que ejecuta TODOS los métodos de detección de señales
y guarda resultados detallados para cada método individual.
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time

def agregar_metodos_multiples_senales():
    """
    Agrega métodos para ejecutar todos los algoritmos de detección de señales.
    Esta función extiende la clase DetectorSenales existente.
    """
    from detectores.detector_senales import DetectorSenales
    
    def detectar_senales_todos_metodos(self, imagen, visualizar=False, guardar=True, ruta_base=None):
        """
        Ejecuta TODOS los métodos de detección de señales y guarda resultados por separado.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imágenes resultado
            ruta_base (str): Ruta base donde guardar resultados
            
        Returns:
            dict: Resultados de todos los métodos
        """
        print("🔍 Ejecutando TODOS los métodos de detección de señales...")
        
        # Definir métodos individuales (excluir combinado para evitar redundancia al final)
        metodos = ['hough', 'freak', 'color', 'log']
        resultados_completos = {}
        
        for metodo in metodos:
            print(f"\n  🔧 Ejecutando método: {metodo.upper()}")
            
            # Crear ruta de salida específica para este método
            if ruta_base and guardar:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"senales_{metodo}_{timestamp}.jpg"
                ruta_salida = os.path.join(ruta_base, "senales", nombre_archivo)
                os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
            else:
                ruta_salida = None
            
            # Ejecutar método específico
            try:
                inicio_tiempo = time.time()
                
                if metodo == 'hough':
                    resultado = self._detectar_senales_hough(imagen, visualizar, guardar, ruta_salida)
                elif metodo == 'freak':
                    resultado = self._detectar_senales_freak(imagen, visualizar, guardar, ruta_salida)
                elif metodo == 'color':
                    resultado = self._detectar_senales_color(imagen, visualizar, guardar, ruta_salida)
                elif metodo == 'log':
                    resultado = self._detectar_senales_log(imagen, visualizar, guardar, ruta_salida)
                
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
                    print(f"    ✅ {metodo.upper()}: {len(resultado.get('senales_detectadas', []))} señales detectadas")
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
                nombre_archivo = f"senales_combinado_{timestamp}.jpg"
                ruta_salida = os.path.join(ruta_base, "senales", nombre_archivo)
            else:
                ruta_salida = None
                
            inicio_tiempo = time.time()
            resultado_combinado = self._detectar_senales_combinado(imagen, visualizar, guardar, ruta_salida)
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
                print(f"    ✅ COMBINADO: {len(resultado_combinado.get('senales_detectadas', []))} señales detectadas")
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
        
        print(f"\n🎉 Detección completa de señales finalizada. {len(resultados_completos)} métodos ejecutados.")
        return resultados_completos
    
    def _guardar_info_deteccion_extendida(self, resultado, metodo, ruta_base):
        """
        Guarda información detallada de la detección con análisis extendido.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Crear directorio para reportes
            dir_reportes = os.path.join(ruta_base, "reportes", "senales")
            os.makedirs(dir_reportes, exist_ok=True)
            
            # Crear archivo de reporte
            nombre_reporte = f"deteccion_senales_{metodo}_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write(f"REPORTE DETALLADO DE DETECCIÓN DE SEÑALES\n")
                f.write(f"MÉTODO: {metodo.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Método utilizado: {metodo}\n")
                f.write(f"Número de señales detectadas: {len(resultado.get('senales_detectadas', []))}\n")
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
                    f.write("PARÁMETROS DE HOUGH:\n")
                    f.write("-" * 30 + "\n")
                    f.write("  Detección de líneas y formas geométricas\n")
                    f.write("  Análisis de contornos cerrados\n\n")
                
                elif metodo == 'freak':
                    f.write("PARÁMETROS DE FREAK:\n")
                    f.write("-" * 30 + "\n")
                    f.write("  Descriptor de características binario\n")
                    f.write("  Análisis de patrones retinales\n")
                    if 'num_keypoints' in resultado:
                        f.write(f"  Keypoints detectados: {resultado['num_keypoints']}\n")
                    if 'num_matches' in resultado:
                        f.write(f"  Coincidencias: {resultado['num_matches']}\n\n")
                
                elif metodo == 'color':
                    f.write("ANÁLISIS DE COLOR:\n")
                    f.write("-" * 30 + "\n")
                    f.write("  Espacio de color HSV\n")
                    f.write("  Detección de colores característicos\n")
                    if 'colores_detectados' in resultado:
                        f.write(f"  Colores encontrados: {resultado['colores_detectados']}\n\n")
                
                elif metodo == 'log':
                    f.write("FILTRO LOG (Laplaciano de Gaussiano):\n")
                    f.write("-" * 30 + "\n")
                    f.write("  Detección de bordes y características\n")
                    f.write("  Análisis de cambios de intensidad\n")
                    if 'sigma' in resultado:
                        f.write(f"  Sigma utilizado: {resultado['sigma']}\n\n")
                
                # Detalles de señales detectadas
                if resultado.get('senales_detectadas'):
                    f.write(f"DETALLES DE SEÑALES DETECTADAS:\n")
                    f.write("-" * 40 + "\n")
                    for i, senal in enumerate(resultado['senales_detectadas'], 1):
                        f.write(f"  Señal {i}:\n")
                        if len(senal) >= 4:  # Formato [x, y, w, h]
                            f.write(f"    Posición: ({senal[0]:.1f}, {senal[1]:.1f})\n")
                            f.write(f"    Tamaño: {senal[2]:.1f} x {senal[3]:.1f} píxeles\n")
                            area = senal[2] * senal[3]
                            f.write(f"    Área: {area:.1f} píxeles²\n")
                        if len(senal) > 4:
                            f.write(f"    Confianza: {senal[4]:.3f}\n")
                        f.write("\n")
                
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
            nombre_reporte = f"comparativo_senales_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write("REPORTE COMPARATIVO - DETECCIÓN DE SEÑALES\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Métodos ejecutados: {len(resultados_completos)}\n\n")
                
                # Tabla resumen
                f.write("RESUMEN COMPARATIVO:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'Método':<12} {'Señales':<8} {'Tiempo (s)':<12} {'Estado':<10}\n")
                f.write("-" * 50 + "\n")
                
                for metodo, resultado in resultados_completos.items():
                    if 'error' in resultado:
                        f.write(f"{metodo.upper():<12} {'ERROR':<8} {'N/A':<12} {'Error':<10}\n")
                    else:
                        num_senales = len(resultado.get('senales_detectadas', []))
                        tiempo = resultado.get('tiempo_ejecucion', 0)
                        f.write(f"{metodo.upper():<12} {num_senales:<8} {tiempo:<12.3f} {'OK':<10}\n")
                
                f.write("\n")
                
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
                detecciones = {m: len(r.get('senales_detectadas', []))
                             for m, r in resultados_completos.items() 
                             if 'error' not in r}
                if detecciones:
                    metodo_mas_detecciones = max(detecciones, key=detecciones.get)
                    f.write(f"  Método con más detecciones: {metodo_mas_detecciones.upper()} ({detecciones[metodo_mas_detecciones]} señales)\n")
                
            print(f"📊 Reporte comparativo guardado: {ruta_reporte}")
            
        except Exception as e:
            print(f"⚠️  Error generando reporte comparativo: {e}")
    
    # Agregar los métodos a la clase DetectorSenales
    DetectorSenales.detectar_senales_todos_metodos = detectar_senales_todos_metodos
    DetectorSenales._guardar_info_deteccion_extendida = _guardar_info_deteccion_extendida
    DetectorSenales._generar_reporte_comparativo = _generar_reporte_comparativo
    
    print("✅ Métodos múltiples agregados al DetectorSenales")

if __name__ == "__main__":
    # Prueba del sistema
    agregar_metodos_multiples_senales()
    print("Sistema de métodos múltiples para señales listo.")