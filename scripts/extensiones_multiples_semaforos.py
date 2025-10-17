#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extensión del Detector de Semáforos - Métodos Múltiples
=======================================================

Sistema que ejecuta TODOS los métodos de detección de semáforos
y guarda resultados detallados para cada método individual.
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time

def agregar_metodos_multiples_semaforos():
    """
    Agrega métodos para ejecutar todos los algoritmos de detección de semáforos.
    Esta función extiende la clase DetectorSemaforos existente.
    """
    from detectores.detector_semaforos import DetectorSemaforos
    
    def detectar_semaforos_todos_metodos(self, imagen, visualizar=False, guardar=True, ruta_base=None):
        """
        Ejecuta TODOS los métodos de detección de semáforos y guarda resultados por separado.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imágenes resultado
            ruta_base (str): Ruta base donde guardar resultados
            
        Returns:
            dict: Resultados de todos los métodos
        """
        print("Ejecutando TODOS los métodos de detección de semáforos...")
        
        # Definir métodos individuales (excluir combinado para evitar redundancia al final)
        metodos = ['color', 'estructura', 'grabcut']
        resultados_completos = {}
        
        for metodo in metodos:
            print(f"\nEjecutando método: {metodo.upper()}")
            
            # Crear ruta de salida específica para este método
            if ruta_base and guardar:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"semaforos_{metodo}_{timestamp}.jpg"
                ruta_salida = os.path.join(ruta_base, "semaforos", nombre_archivo)
                os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
            else:
                ruta_salida = None
            
            # Ejecutar método específico
            try:
                inicio_tiempo = time.time()
                
                if metodo == 'color':
                    resultado = self._detectar_semaforos_color(imagen, visualizar, guardar, ruta_salida)
                elif metodo == 'estructura':
                    resultado = self._detectar_semaforos_estructura(imagen, visualizar, guardar, ruta_salida)
                elif metodo == 'grabcut':
                    resultado = self._detectar_semaforos_grabcut(imagen, visualizar, guardar, ruta_salida)
                
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
                    print(f"{metodo.upper()}: {len(resultado.get('semaforos_detectados', []))} semáforos detectados")
                    print(f"Tiempo: {tiempo_ejecucion:.3f} segundos")
                    
                    # Guardar información detallada del método
                    if guardar and ruta_base:
                        self._guardar_info_deteccion_extendida(resultado, metodo, ruta_base)
                else:
                    resultados_completos[metodo] = {'error': 'Falló la detección', 'tiempo_ejecucion': tiempo_ejecucion}
                    print(f"{metodo.upper()}: Error en detección")
                    
            except Exception as e:
                print(f"{metodo.upper()}: Error - {e}")
                resultados_completos[metodo] = {'error': str(e)}
        
        # Ejecutar método combinado al final
        print(f"\nEjecutando método: COMBINADO")
        try:
            if ruta_base and guardar:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"semaforos_combinado_{timestamp}.jpg"
                ruta_salida = os.path.join(ruta_base, "semaforos", nombre_archivo)
            else:
                ruta_salida = None
                
            inicio_tiempo = time.time()
            resultado_combinado = self._detectar_semaforos_combinado(imagen, visualizar, guardar, ruta_salida)
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
                print(f"COMBINADO: {len(resultado_combinado.get('semaforos_detectados', []))} semáforos detectados")
                print(f"Tiempo: {tiempo_ejecucion:.3f} segundos")
                
                if guardar and ruta_base:
                    self._guardar_info_deteccion_extendida(resultado_combinado, 'combinado', ruta_base)
            else:
                resultados_completos['combinado'] = {'error': 'Falló la detección combinada', 'tiempo_ejecucion': tiempo_ejecucion}
                print(f"COMBINADO: Error en detección")
                
        except Exception as e:
            print(f"COMBINADO: Error - {e}")
            resultados_completos['combinado'] = {'error': str(e)}
        
        # Generar reporte comparativo
        if guardar and ruta_base:
            self._generar_reporte_comparativo(resultados_completos, ruta_base)
        
        print(f"\nDetección completa de semáforos finalizada. {len(resultados_completos)} métodos ejecutados.")
        return resultados_completos
    
    def _guardar_info_deteccion_extendida(self, resultado, metodo, ruta_base):
        """
        Guarda información detallada de la detección con análisis extendido.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Crear directorio para reportes
            dir_reportes = os.path.join(ruta_base, "reportes", "semaforos")
            os.makedirs(dir_reportes, exist_ok=True)
            
            # Crear archivo de reporte
            nombre_reporte = f"deteccion_semaforos_{metodo}_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write(f"REPORTE DETALLADO DE DETECCIÓN DE SEMÁFOROS\n")
                f.write(f"MÉTODO: {metodo.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Método utilizado: {metodo}\n")
                f.write(f"Número de semáforos detectados: {len(resultado.get('semaforos_detectados', []))}\n")
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
                if metodo == 'color':
                    f.write("ANÁLISIS DE COLOR:\n")
                    f.write("-" * 30 + "\n")
                    f.write("  Detección por rangos HSV\n")
                    f.write("  Colores objetivo: Rojo, Amarillo, Verde\n")
                    if 'colores_detectados' in resultado:
                        f.write(f"  Colores encontrados: {resultado['colores_detectados']}\n")
                    if 'distribucion_colores' in resultado:
                        f.write(f"  Distribución de colores: {resultado['distribucion_colores']}\n\n")
                
                elif metodo == 'estructura':
                    f.write("ANÁLISIS ESTRUCTURAL:\n")
                    f.write("-" * 30 + "\n")
                    f.write("  Detección de formas rectangulares\n")
                    f.write("  Análisis de proporciones y disposición\n")
                    if 'num_contornos' in resultado:
                        f.write(f"  Contornos analizados: {resultado['num_contornos']}\n")
                    if 'formas_validas' in resultado:
                        f.write(f"  Formas válidas: {resultado['formas_validas']}\n\n")
                
                elif metodo == 'grabcut':
                    f.write("SEGMENTACIÓN GRABCUT:\n")
                    f.write("-" * 30 + "\n")
                    f.write("  Segmentación de primer plano/fondo\n")
                    f.write("  Refinamiento iterativo\n")
                    if 'iteraciones' in resultado:
                        f.write(f"  Iteraciones realizadas: {resultado['iteraciones']}\n")
                    if 'calidad_segmentacion' in resultado:
                        f.write(f"  Calidad de segmentación: {resultado['calidad_segmentacion']:.3f}\n\n")
                
                # Detalles de semáforos detectados
                if resultado.get('semaforos_detectados'):
                    f.write(f"DETALLES DE SEMÁFOROS DETECTADOS:\n")
                    f.write("-" * 40 + "\n")
                    for i, semaforo in enumerate(resultado['semaforos_detectados'], 1):
                        f.write(f"  Semáforo {i}:\n")
                        if len(semaforo) >= 4:  # Formato [x, y, w, h]
                            f.write(f"    Posición: ({semaforo[0]:.1f}, {semaforo[1]:.1f})\n")
                            f.write(f"    Tamaño: {semaforo[2]:.1f} x {semaforo[3]:.1f} píxeles\n")
                            area = semaforo[2] * semaforo[3]
                            f.write(f"    Área: {area:.1f} píxeles²\n")
                            
                            # Calcular relación de aspecto
                            aspect_ratio = semaforo[2] / semaforo[3] if semaforo[3] > 0 else 0
                            f.write(f"    Relación de aspecto: {aspect_ratio:.2f}\n")
                        
                        if len(semaforo) > 4:
                            f.write(f"    Confianza: {semaforo[4]:.3f}\n")
                        
                        # Información adicional específica del método
                        if metodo == 'color' and len(semaforo) > 5:
                            f.write(f"    Estado detectado: {semaforo[5]}\n")
                        
                        f.write("\n")
                
                # Estadísticas adicionales
                if resultado.get('semaforos_detectados'):
                    semaforos = resultado['semaforos_detectados']
                    areas = [s[2] * s[3] for s in semaforos if len(s) >= 4]
                    
                    if areas:
                        f.write("ESTADÍSTICAS DE DETECCIÓN:\n")
                        f.write("-" * 30 + "\n")
                        f.write(f"  Área promedio: {np.mean(areas):.1f} píxeles²\n")
                        f.write(f"  Área mínima: {np.min(areas):.1f} píxeles²\n")
                        f.write(f"  Área máxima: {np.max(areas):.1f} píxeles²\n")
                        f.write(f"  Desviación estándar de áreas: {np.std(areas):.1f} píxeles²\n")
                
                # Información de máscara de segmentación (si está disponible)
                if 'mask' in resultado:
                    f.write(f"\nINFORMACIÓN DE SEGMENTACIÓN:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  Máscara generada: Sí\n")
                    f.write(f"  Píxeles segmentados: {np.count_nonzero(resultado['mask'])}\n")
                    f.write(f"  Porcentaje de imagen segmentada: {np.count_nonzero(resultado['mask']) / resultado['mask'].size * 100:.2f}%\n")
                
            print(f"Reporte detallado guardado: {ruta_reporte}")
            
        except Exception as e:
            print(f"Error guardando reporte: {e}")
    
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
            nombre_reporte = f"comparativo_semaforos_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write("REPORTE COMPARATIVO - DETECCIÓN DE SEMÁFOROS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Métodos ejecutados: {len(resultados_completos)}\n\n")
                
                # Tabla resumen
                f.write("RESUMEN COMPARATIVO:\n")
                f.write("-" * 54 + "\n")
                f.write(f"{'Método':<12} {'Semáforos':<10} {'Tiempo (s)':<12} {'Estado':<10}\n")
                f.write("-" * 54 + "\n")
                
                for metodo, resultado in resultados_completos.items():
                    if 'error' in resultado:
                        f.write(f"{metodo.upper():<12} {'ERROR':<10} {'N/A':<12} {'Error':<10}\n")
                    else:
                        num_semaforos = len(resultado.get('semaforos_detectados', []))
                        tiempo = resultado.get('tiempo_ejecucion', 0)
                        f.write(f"{metodo.upper():<12} {num_semaforos:<10} {tiempo:<12.3f} {'OK':<10}\n")
                
                f.write("\n")
                
                # Análisis de efectividad por método
                f.write("ANÁLISIS DE EFECTIVIDAD:\n")
                f.write("-" * 30 + "\n")
                for metodo, resultado in resultados_completos.items():
                    if 'error' not in resultado:
                        f.write(f"  {metodo.upper()}:\n")
                        semaforos = resultado.get('semaforos_detectados', [])
                        f.write(f"    - Detecciones: {len(semaforos)}\n")
                        f.write(f"    - Tiempo: {resultado.get('tiempo_ejecucion', 0):.3f}s\n")
                        
                        if metodo == 'color':
                            f.write(f"    - Especialidad: Identificación por color de luces\n")
                        elif metodo == 'estructura':
                            f.write(f"    - Especialidad: Análisis de forma y proporción\n")
                        elif metodo == 'grabcut':
                            f.write(f"    - Especialidad: Segmentación precisa\n")
                        elif metodo == 'combinado':
                            f.write(f"    - Especialidad: Enfoque integral\n")
                        f.write("\n")
                
                # Recomendaciones
                f.write(f"RECOMENDACIONES:\n")
                f.write("-" * 20 + "\n")
                
                # Encontrar el método más rápido
                tiempos_validos = {m: r.get('tiempo_ejecucion', float('inf')) 
                                 for m, r in resultados_completos.items() 
                                 if 'error' not in r}
                if tiempos_validos:
                    metodo_rapido = min(tiempos_validos, key=tiempos_validos.get)
                    f.write(f"  Método más rápido: {metodo_rapido.upper()} ({tiempos_validos[metodo_rapido]:.3f}s)\n")
                
                # Encontrar el método con más detecciones
                detecciones = {m: len(r.get('semaforos_detectados', []))
                             for m, r in resultados_completos.items() 
                             if 'error' not in r}
                if detecciones:
                    metodo_mas_detecciones = max(detecciones, key=detecciones.get)
                    f.write(f"  Método con más detecciones: {metodo_mas_detecciones.upper()} ({detecciones[metodo_mas_detecciones]} semáforos)\n")
                
                # Recomendaciones específicas
                f.write(f"\n  Recomendaciones específicas:\n")
                f.write(f"    - COLOR: Mejor para condiciones de buena iluminación\n")
                f.write(f"    - ESTRUCTURA: Mejor para semáforos tradicionales\n")
                f.write(f"    - GRABCUT: Mejor para segmentación precisa\n")
                f.write(f"    - COMBINADO: Enfoque más robusto general\n")
                
            print(f"Reporte comparativo guardado: {ruta_reporte}")
            
        except Exception as e:
            print(f"Error generando reporte comparativo: {e}")
    
    # Agregar los métodos a la clase DetectorSemaforos
    DetectorSemaforos.detectar_semaforos_todos_metodos = detectar_semaforos_todos_metodos
    DetectorSemaforos._guardar_info_deteccion_extendida = _guardar_info_deteccion_extendida
    DetectorSemaforos._generar_reporte_comparativo = _generar_reporte_comparativo
    
    print("Métodos múltiples agregados al DetectorSemaforos")

if __name__ == "__main__":
    # Prueba del sistema
    agregar_metodos_multiples_semaforos()
    print("Sistema de métodos múltiples para semáforos listo.")