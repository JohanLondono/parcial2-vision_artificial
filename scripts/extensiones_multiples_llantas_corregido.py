#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extensión del Detector de Llantas - Configuraciones Múltiples
============================================================

Sistema que ejecuta TODAS las configuraciones de detección de llantas
y guarda resultados detallados para cada configuración individual.
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

def agregar_metodos_multiples_llantas():
    """
    Agrega métodos para ejecutar todas las configuraciones de detección de llantas.
    Esta función extiende la clase DetectorLlantas existente.
    """
    from detectores.detector_llantas import DetectorLlantas
    
    def detectar_llantas_todos_metodos(self, imagen, visualizar=False, guardar=True, ruta_base=None):
        """
        Ejecuta TODAS las configuraciones de detección de llantas y guarda resultados por separado.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imágenes resultado
            ruta_base (str): Ruta base donde guardar resultados
            
        Returns:
            dict: Resultados de todas las configuraciones
        """
        print("Ejecutando TODAS las configuraciones de detección de llantas...")
        
        # Definir configuraciones disponibles
        configuraciones = ['CONFIG_PRECISION_ALTA', 'CONFIG_ROBUSTA', 'CONFIG_ADAPTATIVA', 'CONFIG_BALANCED']
        resultados_completos = {}
        
        for config in configuraciones:
            print(f"\nEjecutando configuración: {config}")
            
            # Crear ruta de salida específica para esta configuración
            if ruta_base and guardar:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"llantas_{config.lower()}_{timestamp}.jpg"
                ruta_salida = os.path.join(ruta_base, "llantas", nombre_archivo)
                os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
            else:
                ruta_salida = None
            
            # Ejecutar configuración específica
            try:
                inicio_tiempo = time.time()
                
                resultado = self.detectar_llantas(imagen, config, visualizar=False, guardar=guardar, ruta_salida=ruta_salida)
                
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
                    print(f"{config}: {len(resultado.get('llantas_detectadas', []))} llantas detectadas")
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
        
        print(f"\nDetección completa de llantas finalizada. {len(resultados_completos)} configuraciones ejecutadas.")
        return resultados_completos
    
    def _guardar_info_deteccion_extendida(self, resultado, configuracion, ruta_base):
        """
        Guarda información detallada de la detección con análisis extendido.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Crear directorio para reportes
            dir_reportes = os.path.join(ruta_base, "reportes", "llantas")
            os.makedirs(dir_reportes, exist_ok=True)
            
            # Crear archivo de reporte
            nombre_reporte = f"deteccion_llantas_{configuracion}_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write("REPORTE DETALLADO DE DETECCIÓN DE LLANTAS\n")
                f.write(f"CONFIGURACIÓN: {configuracion.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Configuración utilizada: {configuracion}\n")
                f.write(f"Número de llantas detectadas: {len(resultado.get('llantas_detectadas', []))}\n")
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
                
                # Detalles de llantas detectadas
                if resultado.get('llantas_detectadas'):
                    f.write("DETALLES DE LLANTAS DETECTADAS:\n")
                    for i, llanta in enumerate(resultado['llantas_detectadas']):
                        f.write(f"  Llanta {i+1}:\n")
                        f.write(f"    Centro: ({llanta[0]}, {llanta[1]})\n")
                        f.write(f"    Radio: {llanta[2]}\n")
                        if len(llanta) > 3:
                            f.write(f"    Confianza: {llanta[3]:.3f}\n")
                        f.write("\n")
                
                # Estadísticas adicionales
                if resultado.get('llantas_detectadas'):
                    radios = [llanta[2] for llanta in resultado['llantas_detectadas']]
                    f.write("ESTADÍSTICAS:\n")
                    f.write(f"  Radio promedio: {np.mean(radios):.2f}\n")
                    f.write(f"  Radio mínimo: {np.min(radios):.2f}\n")
                    f.write(f"  Radio máximo: {np.max(radios):.2f}\n")
                    f.write(f"  Desviación estándar: {np.std(radios):.2f}\n\n")
                
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
            nombre_reporte = f"comparativo_llantas_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write("REPORTE COMPARATIVO - DETECCIÓN DE LLANTAS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Configuraciones ejecutadas: {len(resultados_completos)}\n\n")
                
                # Tabla resumen
                f.write("RESUMEN COMPARATIVO:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Configuración':<20} {'Llantas':<8} {'Tiempo (s)':<12} {'Confianza':<12} {'Estado':<10}\n")
                f.write("-" * 80 + "\n")
                
                for config, resultado in resultados_completos.items():
                    if 'error' not in resultado:
                        num_llantas = len(resultado.get('llantas_detectadas', []))
                        tiempo = resultado.get('tiempo_ejecucion', 0)
                        confianza = resultado.get('confianza_promedio', 0)
                        estado = "OK"
                    else:
                        num_llantas = 0
                        tiempo = resultado.get('tiempo_ejecucion', 0)
                        confianza = 0
                        estado = "ERROR"
                    
                    f.write(f"{config:<20} {num_llantas:<8} {tiempo:<12.3f} {confianza:<12.3f} {estado:<10}\n")
                
                f.write("\n")
                
                # Análisis detallado por configuración
                f.write("ANÁLISIS DETALLADO POR CONFIGURACIÓN:\n")
                f.write("-" * 50 + "\n")
                
                for config, resultado in resultados_completos.items():
                    f.write(f"\n{config}:\n")
                    if 'error' not in resultado:
                        f.write(f"  - Llantas detectadas: {len(resultado.get('llantas_detectadas', []))}\n")
                        f.write(f"  - Tiempo de ejecución: {resultado.get('tiempo_ejecucion', 0):.3f} segundos\n")
                        f.write(f"  - Confianza promedio: {resultado.get('confianza_promedio', 0):.3f}\n")
                        f.write(f"  - Candidatos iniciales: {resultado.get('candidatos_iniciales', 0)}\n")
                        
                        if resultado.get('llantas_detectadas'):
                            radios = [llanta[2] for llanta in resultado['llantas_detectadas']]
                            f.write(f"  - Radio promedio: {np.mean(radios):.2f} px\n")
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
                detecciones = {c: len(r.get('llantas_detectadas', []))
                             for c, r in resultados_completos.items() 
                             if 'error' not in r}
                if detecciones:
                    config_detectora = max(detecciones, key=detecciones.get)
                    f.write(f"- Configuración con más detecciones: {config_detectora} ({detecciones[config_detectora]} llantas)\n")
                
                # Encontrar la configuración con mayor confianza
                confianzas = {c: r.get('confianza_promedio', 0)
                            for c, r in resultados_completos.items() 
                            if 'error' not in r}
                if confianzas:
                    config_confiable = max(confianzas, key=confianzas.get)
                    f.write(f"- Configuración más confiable: {config_confiable} (confianza: {confianzas[config_confiable]:.3f})\n")
                
            print(f"Reporte comparativo guardado: {ruta_reporte}")
            
        except Exception as e:
            print(f"Error generando reporte comparativo: {e}")
    
    # Agregar los métodos a la clase DetectorLlantas
    DetectorLlantas.detectar_llantas_todos_metodos = detectar_llantas_todos_metodos
    DetectorLlantas._guardar_info_deteccion_extendida = _guardar_info_deteccion_extendida
    DetectorLlantas._generar_reporte_comparativo = _generar_reporte_comparativo
    
    print("Métodos múltiples agregados al DetectorLlantas")

# Función de utilidad para usar desde el sistema principal
def detectar_llantas_imagen_todos_metodos(ruta_imagen, ruta_salida="./resultados_deteccion"):
    """
    Función de utilidad para detectar llantas con todas las configuraciones.
    
    Args:
        ruta_imagen (str): Ruta de la imagen
        ruta_salida (str): Directorio de salida
        
    Returns:
        dict: Resultados de todas las configuraciones
    """
    from detectores.detector_llantas import DetectorLlantas
    
    # Asegurarse de que los métodos estén disponibles
    agregar_metodos_multiples_llantas()
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
        return None
    
    # Crear detector y ejecutar todas las configuraciones
    detector = DetectorLlantas()
    return detector.detectar_llantas_todos_metodos(imagen, visualizar=False, guardar=True, ruta_base=ruta_salida)

if __name__ == "__main__":
    # Prueba del sistema
    agregar_metodos_multiples_llantas()
    print("Sistema de configuraciones múltiples para llantas listo.")