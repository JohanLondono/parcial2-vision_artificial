#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extensión del Detector de Semáforos - Configuraciones Múltiples
==============================================================

Sistema que ejecuta TODAS las configuraciones de detección de semáforos
y guarda resultados detallados para cada configuración individual.
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

def agregar_metodos_multiples_semaforos():
    """
    Agrega métodos para ejecutar todas las configuraciones de detección de semáforos.
    Esta función extiende la clase DetectorSemaforos existente.
    """
    from detectores.detector_semaforos_nuevo import DetectorSemaforos
    
    def detectar_semaforos_todos_metodos(self, imagen, visualizar=False, guardar=True, ruta_base=None):
        """
        Ejecuta TODAS las configuraciones de detección de semáforos y guarda resultados por separado.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imágenes resultado
            ruta_base (str): Ruta base donde guardar resultados
            
        Returns:
            dict: Resultados de todas las configuraciones
        """
        print("Ejecutando TODAS las configuraciones de detección de semáforos...")
        
        # Definir configuraciones disponibles
        configuraciones = ['CONFIG_PRECISION_ALTA', 'CONFIG_ROBUSTA', 'CONFIG_ADAPTATIVA', 'CONFIG_BALANCED']
        resultados_completos = {}
        
        for config in configuraciones:
            print(f"\nEjecutando configuración: {config}")
            
            # Crear ruta de salida específica para esta configuración
            if ruta_base and guardar:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"semaforos_{config.lower()}_{timestamp}.jpg"
                ruta_salida = os.path.join(ruta_base, "semaforos", nombre_archivo)
                os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
            else:
                ruta_salida = None
            
            # Ejecutar configuración específica
            try:
                inicio_tiempo = time.time()
                
                resultado = self.detectar_semaforos(imagen, config, visualizar=False, guardar=guardar, ruta_salida=ruta_salida)
                
                tiempo_ejecucion = time.time() - inicio_tiempo
                
                if resultado:
                    # Agregar información adicional
                    resultado['tiempo_ejecucion'] = tiempo_ejecucion
                    resultado['metodo_utilizado'] = config
                    resultado['imagen_info'] = {
                        'width': imagen.shape[1],
                        'height': imagen.shape[0],
                        'channels': imagen.shape[2] if len(imagen.shape) == 3 else 1
                    }
                    
                    resultados_completos[config] = resultado
                    print(f"{config}: {len(resultado.get('semaforos_detectados', []))} semáforos detectados")
                    print(f"Tiempo: {tiempo_ejecucion:.3f} segundos")
                    
                    # Guardar información detallada de la configuración
                    if guardar and ruta_base:
                        self._guardar_info_deteccion_extendida(resultado, config, ruta_base)
                else:
                    resultados_completos[config] = {'error': 'Falló la detección', 'tiempo_ejecucion': tiempo_ejecucion}
                    print(f"{config}: Error en detección")
                    
            except Exception as e:
                print(f"{config}: Error - {e}")
                resultados_completos[config] = {'error': str(e)}
        
        # Generar reporte comparativo
        if guardar and ruta_base:
            self._generar_reporte_comparativo(resultados_completos, ruta_base)
        
        print(f"\nDetección completa de semáforos finalizada. {len(resultados_completos)} configuraciones ejecutadas.")
        return resultados_completos
    
    def _guardar_info_deteccion_extendida(self, resultado, configuracion, ruta_base):
        """
        Guarda información detallada de la detección con análisis extendido.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Crear directorio para reportes
            dir_reportes = os.path.join(ruta_base, "reportes", "semaforos")
            os.makedirs(dir_reportes, exist_ok=True)
            
            # Crear archivo de reporte
            nombre_reporte = f"deteccion_semaforos_{configuracion}_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write("REPORTE DETALLADO DE DETECCIÓN DE SEMÁFOROS\n")
                f.write(f"CONFIGURACIÓN: {configuracion.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Configuración utilizada: {configuracion}\n")
                f.write(f"Número de semáforos detectados: {len(resultado.get('semaforos_detectados', []))}\n")
                f.write(f"Tiempo de ejecución: {resultado.get('tiempo_ejecucion', 0):.3f} segundos\n\n")
                
                # Información de la imagen
                if 'imagen_info' in resultado:
                    f.write("INFORMACIÓN DE LA IMAGEN:\n")
                    f.write(f"  Dimensiones: {resultado['imagen_info']['width']} x {resultado['imagen_info']['height']}\n")
                    f.write(f"  Canales: {resultado['imagen_info']['channels']}\n")
                    f.write(f"  Píxeles totales: {resultado['imagen_info']['width'] * resultado['imagen_info']['height']}\n\n")
                
                # Información específica de la configuración
                if 'configuracion' in resultado:
                    f.write("DETALLES DE LA CONFIGURACIÓN:\n")
                    f.write(f"  Nombre: {resultado['configuracion']}\n")
                    f.write(f"  Candidatos iniciales: {resultado.get('candidatos_iniciales', 0)}\n")
                    f.write(f"  Confianza promedio: {resultado.get('confianza_promedio', 0):.3f}\n\n")
                
                # Información específica por configuración
                if configuracion == 'CONFIG_PRECISION_ALTA':
                    f.write("DETALLES ESPECÍFICOS - PRECISIÓN ALTA:\n")
                    f.write("  - Algoritmos: Color HSV + Estructura + Morfología + Validación\n")
                    f.write("  - Enfoque: Máxima precisión con validación estricta\n")
                    f.write("  - Umbral de validación: 0.65\n\n")
                
                elif configuracion == 'CONFIG_ROBUSTA':
                    f.write("DETALLES ESPECÍFICOS - ROBUSTA:\n")
                    f.write("  - Algoritmos: Contornos + Color + GrabCut + Validación\n")
                    f.write("  - Enfoque: Robustez en condiciones difíciles\n")
                    f.write("  - Umbral de validación: 0.60\n\n")
                
                elif configuracion == 'CONFIG_ADAPTATIVA':
                    f.write("DETALLES ESPECÍFICOS - ADAPTATIVA:\n")
                    f.write("  - Algoritmos: Color multirrango + Textura + Hough + AKAZE\n")
                    f.write("  - Enfoque: Adaptación a diferentes condiciones\n")
                    f.write("  - Umbral de validación: 0.55\n\n")
                
                elif configuracion == 'CONFIG_BALANCED':
                    f.write("DETALLES ESPECÍFICOS - EQUILIBRADA:\n")
                    f.write("  - Algoritmos: Color + Estructura + Geometría + Consistencia\n")
                    f.write("  - Enfoque: Balance entre precisión y robustez\n")
                    f.write("  - Umbral de validación: 0.58\n\n")
                
                # Detalles de semáforos detectados
                if resultado.get('semaforos_detectados'):
                    f.write("DETALLES DE SEMÁFOROS DETECTADOS:\n")
                    for i, semaforo in enumerate(resultado['semaforos_detectados']):
                        f.write(f"  Semáforo {i+1}:\n")
                        f.write(f"    Centro: ({semaforo[0]}, {semaforo[1]})\n")
                        f.write(f"    Dimensiones: {semaforo[2]} x {semaforo[3]}\n")
                        if len(semaforo) > 4:
                            f.write(f"    Color detectado: {semaforo[4]}\n")
                        if len(semaforo) > 5:
                            f.write(f"    Confianza: {semaforo[5]:.3f}\n")
                        f.write("\n")
                
                # Estadísticas adicionales
                if resultado.get('semaforos_detectados'):
                    semaforos = resultado['semaforos_detectados']
                    areas = [s[2] * s[3] for s in semaforos if len(s) >= 4]
                    
                    if areas:
                        f.write("ESTADÍSTICAS:\n")
                        f.write(f"  Área promedio: {np.mean(areas):.2f}\n")
                        f.write(f"  Área mínima: {np.min(areas):.2f}\n")
                        f.write(f"  Área máxima: {np.max(areas):.2f}\n")
                        f.write(f"  Desviación estándar: {np.std(areas):.2f}\n\n")
                
                # Análisis de colores detectados
                if resultado.get('semaforos_detectados'):
                    colores = [s[4] for s in resultado['semaforos_detectados'] if len(s) > 4]
                    if colores:
                        conteo_colores = {}
                        for color in colores:
                            conteo_colores[color] = conteo_colores.get(color, 0) + 1
                        
                        f.write("ANÁLISIS DE COLORES:\n")
                        for color, cantidad in conteo_colores.items():
                            f.write(f"  {color}: {cantidad} detecciones\n")
                        f.write("\n")
                
            print(f"Reporte detallado guardado: {ruta_reporte}")
            
        except Exception as e:
            print(f"Error guardando reporte: {e}")
    
    def _generar_reporte_comparativo(self, resultados_completos, ruta_base):
        """
        Genera un reporte comparativo de todas las configuraciones ejecutadas.
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
                f.write(f"Configuraciones ejecutadas: {len(resultados_completos)}\n\n")
                
                # Tabla resumen
                f.write("RESUMEN COMPARATIVO:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Configuración':<20} {'Semáforos':<10} {'Tiempo (s)':<12} {'Confianza':<12} {'Estado':<10}\n")
                f.write("-" * 80 + "\n")
                
                for config, resultado in resultados_completos.items():
                    if 'error' not in resultado:
                        num_semaforos = len(resultado.get('semaforos_detectados', []))
                        tiempo = resultado.get('tiempo_ejecucion', 0)
                        confianza = resultado.get('confianza_promedio', 0)
                        estado = "OK"
                    else:
                        num_semaforos = 0
                        tiempo = resultado.get('tiempo_ejecucion', 0)
                        confianza = 0
                        estado = "ERROR"
                    
                    f.write(f"{config:<20} {num_semaforos:<10} {tiempo:<12.3f} {confianza:<12.3f} {estado:<10}\n")
                
                f.write("\n")
                
                # Análisis detallado por configuración
                f.write("ANÁLISIS DETALLADO POR CONFIGURACIÓN:\n")
                f.write("-" * 50 + "\n")
                
                for config, resultado in resultados_completos.items():
                    f.write(f"\n{config}:\n")
                    if 'error' not in resultado:
                        f.write(f"  - Semáforos detectados: {len(resultado.get('semaforos_detectados', []))}\n")
                        f.write(f"  - Tiempo de ejecución: {resultado.get('tiempo_ejecucion', 0):.3f} segundos\n")
                        f.write(f"  - Confianza promedio: {resultado.get('confianza_promedio', 0):.3f}\n")
                        f.write(f"  - Candidatos iniciales: {resultado.get('candidatos_iniciales', 0)}\n")
                        
                        if resultado.get('semaforos_detectados'):
                            areas = [s[2] * s[3] for s in resultado['semaforos_detectados'] if len(s) >= 4]
                            if areas:
                                f.write(f"  - Área promedio: {np.mean(areas):.2f} px²\n")
                    else:
                        f.write(f"  - Error: {resultado['error']}\n")
                
                # Recomendaciones
                f.write("\nRECOMENDACIONES:\n")
                f.write("-" * 20 + "\n")
                
                # Encontrar la configuración más rápida
                tiempos_validos = {c: r.get('tiempo_ejecucion', float('inf')) 
                                 for c, r in resultados_completos.items() 
                                 if 'error' not in r}
                if tiempos_validos:
                    config_rapida = min(tiempos_validos, key=tiempos_validos.get)
                    f.write(f"- Configuración más rápida: {config_rapida} ({tiempos_validos[config_rapida]:.3f}s)\n")
                
                # Encontrar la configuración con más detecciones
                detecciones = {c: len(r.get('semaforos_detectados', []))
                             for c, r in resultados_completos.items() 
                             if 'error' not in r}
                if detecciones:
                    config_detectora = max(detecciones, key=detecciones.get)
                    f.write(f"- Configuración con más detecciones: {config_detectora} ({detecciones[config_detectora]} semáforos)\n")
                
                # Encontrar la configuración con mayor confianza
                confianzas = {c: r.get('confianza_promedio', 0)
                            for c, r in resultados_completos.items() 
                            if 'error' not in r}
                if confianzas:
                    config_confiable = max(confianzas, key=confianzas.get)
                    f.write(f"- Configuración más confiable: {config_confiable} (confianza: {confianzas[config_confiable]:.3f})\n")
                
                # Recomendaciones específicas
                f.write("\nRECOMENDACIONES POR ESCENARIO:\n")
                f.write("- CONFIG_PRECISION_ALTA: Mejor para imágenes de alta calidad con buena iluminación\n")
                f.write("- CONFIG_ROBUSTA: Mejor para imágenes con ruido o condiciones adversas\n")
                f.write("- CONFIG_ADAPTATIVA: Mejor para variedad de condiciones y tipos de semáforos\n")
                f.write("- CONFIG_BALANCED: Mejor opción general, balance entre precisión y robustez\n")
                
            print(f"Reporte comparativo guardado: {ruta_reporte}")
            
        except Exception as e:
            print(f"Error generando reporte comparativo: {e}")
    
    # Agregar los métodos a la clase DetectorSemaforos
    DetectorSemaforos.detectar_semaforos_todos_metodos = detectar_semaforos_todos_metodos
    DetectorSemaforos._guardar_info_deteccion_extendida = _guardar_info_deteccion_extendida
    DetectorSemaforos._generar_reporte_comparativo = _generar_reporte_comparativo
    
    print("Métodos múltiples agregados al DetectorSemaforos")

# Función de utilidad para usar desde el sistema principal
def detectar_semaforos_imagen_todos_metodos(ruta_imagen, ruta_salida="./resultados_deteccion"):
    """
    Función de utilidad para detectar semáforos con todas las configuraciones.
    
    Args:
        ruta_imagen (str): Ruta de la imagen
        ruta_salida (str): Directorio de salida
        
    Returns:
        dict: Resultados de todas las configuraciones
    """
    from detectores.detector_semaforos_nuevo import DetectorSemaforos
    
    # Asegurarse de que los métodos estén disponibles
    agregar_metodos_multiples_semaforos()
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
        return None
    
    # Crear detector y ejecutar todas las configuraciones
    detector = DetectorSemaforos()
    return detector.detectar_semaforos_todos_metodos(imagen, visualizar=False, guardar=True, ruta_base=ruta_salida)

if __name__ == "__main__":
    # Prueba del sistema
    agregar_metodos_multiples_semaforos()
    print("Sistema de configuraciones múltiples para semáforos listo.")