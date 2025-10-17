#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extensión del Detector de Señales - Configuraciones Múltiples
============================================================

Sistema que ejecuta TODAS las configuraciones de detección de señales
y guarda resultados detallados para cada configuración individual.
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

def agregar_metodos_multiples_senales():
    """
    Agrega métodos para ejecutar todas las configuraciones de detección de señales.
    Esta función extiende la clase DetectorSenales existente.
    """
    from detectores.detector_senales import DetectorSenales
    
    def detectar_senales_todos_metodos(self, imagen, forma='CIRCULAR', visualizar=False, guardar=True, ruta_base=None):
        """
        Ejecuta TODAS las configuraciones de detección de señales y guarda resultados por separado.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            forma (str): Forma de señal a detectar ('CIRCULAR', 'RECTANGULAR', 'TRIANGULAR', 'OCTAGONAL', 'TODAS')
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imágenes resultado
            ruta_base (str): Ruta base donde guardar resultados
            
        Returns:
            dict: Resultados de todas las configuraciones
        """
        print(f"Ejecutando TODAS las configuraciones de detección de señales para forma: {forma}")
        
        # Definir configuraciones disponibles
        configuraciones = ['CONFIG_PRECISION_ALTA', 'CONFIG_ROBUSTA', 'CONFIG_ADAPTATIVA', 'CONFIG_BALANCED']
        resultados_completos = {}
        
        for config in configuraciones:
            print(f"\nEjecutando configuración: {config}")
            
            # Crear ruta de salida específica para esta configuración
            if ruta_base and guardar:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"senales_{config.lower()}_{timestamp}.jpg"
                ruta_salida = os.path.join(ruta_base, "senales", nombre_archivo)
                os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
            else:
                ruta_salida = None
            
            # Ejecutar configuración específica
            try:
                inicio_tiempo = time.time()
                
                resultado = self.detectar_senales(imagen, config, forma, visualizar=False, guardar=guardar, ruta_salida=ruta_salida)
                
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
                    print(f"{config}: {len(resultado.get('senales_detectadas', []))} señales detectadas")
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
        
        print(f"\nDetección completa de señales finalizada. {len(resultados_completos)} configuraciones ejecutadas.")
        return resultados_completos
    
    def _guardar_info_deteccion_extendida(self, resultado, configuracion, ruta_base):
        """
        Guarda información detallada de la detección con análisis extendido.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Crear directorio para reportes
            dir_reportes = os.path.join(ruta_base, "reportes", "senales")
            os.makedirs(dir_reportes, exist_ok=True)
            
            # Crear archivo de reporte
            nombre_reporte = f"deteccion_senales_{configuracion}_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write("REPORTE DETALLADO DE DETECCIÓN DE SEÑALES\n")
                f.write(f"CONFIGURACIÓN: {configuracion.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Configuración utilizada: {configuracion}\n")
                f.write(f"Número de señales detectadas: {len(resultado.get('senales_detectadas', []))}\n")
                f.write(f"Tiempo de ejecución: {resultado.get('tiempo_ejecucion', 0):.3f} segundos\n\n")
                
                # Información de la imagen
                if 'imagen_info' in resultado:
                    f.write("INFORMACIÓN DE LA IMAGEN:\n")
                    f.write(f"  Dimensiones: {resultado['imagen_info']['width']} x {resultado['imagen_info']['height']}\n")
                    f.write(f"  Canales: {resultado['imagen_info']['channels']}\n")
                    f.write(f"  Píxeles totales: {resultado['imagen_info']['width'] * resultado['imagen_info']['height']}\n\n")
                
                # Información específica de la configuración
                if 'metodo' in resultado:
                    f.write("DETALLES DE LA CONFIGURACIÓN:\n")
                    f.write(f"  Método: {resultado['metodo']}\n")
                    f.write(f"  Confianza promedio: {resultado.get('confianza_promedio', 0):.3f}\n")
                    f.write(f"  Candidatos Hough: {resultado.get('candidatos_hough', 0)}\n")
                    f.write(f"  Candidatos contornos: {resultado.get('candidatos_contornos', 0)}\n\n")
                
                # Detalles de señales detectadas
                if resultado.get('senales_detectadas'):
                    f.write("DETALLES DE SEÑALES DETECTADAS:\n")
                    for i, senal in enumerate(resultado['senales_detectadas']):
                        f.write(f"  Señal {i+1}:\n")
                        f.write(f"    Centro: ({senal[0]}, {senal[1]})\n")
                        f.write(f"    Radio: {senal[2]}\n")
                        if len(senal) > 3:
                            f.write(f"    Tipo: {senal[3] if len(senal) > 3 else 'N/A'}\n")
                        if len(senal) > 4:
                            f.write(f"    Confianza: {senal[4]:.3f}\n")
                        f.write("\n")
                
                # Estadísticas básicas de detecciones
                if resultado.get('senales_detectadas'):
                    senales = resultado['senales_detectadas']
                    if len(senales) > 0 and len(senales[0]) >= 3:
                        radios = [senal[2] for senal in senales if len(senal) >= 3]
                        if radios:
                            f.write("ESTADÍSTICAS BÁSICAS:\n")
                            f.write(f"  Radio promedio: {np.mean(radios):.2f}\n")
                            f.write(f"  Radio mínimo: {np.min(radios):.2f}\n")
                            f.write(f"  Radio máximo: {np.max(radios):.2f}\n")
                            f.write(f"  Desviación estándar: {np.std(radios):.2f}\n\n")
                
                # Estadísticas por tipo si están disponibles
                if 'estadisticas' in resultado and 'por_tipo' in resultado['estadisticas']:
                    f.write("ESTADÍSTICAS POR TIPO:\n")
                    por_tipo = resultado['estadisticas']['por_tipo']
                    for tipo, cantidad in por_tipo.items():
                        f.write(f"  {tipo}: {cantidad} detecciones\n")
                    f.write("\n")
                    
                    # Agregar estadísticas adicionales
                    stats = resultado['estadisticas']
                    if 'radios' in stats:
                        f.write("ESTADÍSTICAS DE RADIOS:\n")
                        f.write(f"  Promedio: {stats['radios']['promedio']:.2f}\n")
                        f.write(f"  Mínimo: {stats['radios']['min']:.2f}\n")
                        f.write(f"  Máximo: {stats['radios']['max']:.2f}\n")
                    
                    if 'confianzas' in stats:
                        f.write("ESTADÍSTICAS DE CONFIANZAS:\n")
                        f.write(f"  Promedio: {stats['confianzas']['promedio']:.3f}\n")
                        f.write(f"  Mínimo: {stats['confianzas']['min']:.3f}\n")
                        f.write(f"  Máximo: {stats['confianzas']['max']:.3f}\n")
                
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
            nombre_reporte = f"comparativo_senales_{timestamp}.txt"
            ruta_reporte = os.path.join(dir_reportes, nombre_reporte)
            
            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                f.write("REPORTE COMPARATIVO - DETECCIÓN DE SEÑALES\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Configuraciones ejecutadas: {len(resultados_completos)}\n\n")
                
                # Tabla resumen
                f.write("RESUMEN COMPARATIVO:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Configuración':<20} {'Señales':<8} {'Tiempo (s)':<12} {'Confianza':<12} {'Estado':<10}\n")
                f.write("-" * 80 + "\n")
                
                for config, resultado in resultados_completos.items():
                    if 'error' not in resultado:
                        num_senales = len(resultado.get('senales_detectadas', []))
                        tiempo = resultado.get('tiempo_ejecucion', 0)
                        confianza = resultado.get('confianza_promedio', 0)
                        estado = "OK"
                    else:
                        num_senales = 0
                        tiempo = resultado.get('tiempo_ejecucion', 0)
                        confianza = 0
                        estado = "ERROR"
                    
                    f.write(f"{config:<20} {num_senales:<8} {tiempo:<12.3f} {confianza:<12.3f} {estado:<10}\n")
                
                f.write("\n")
                
                # Análisis detallado por configuración
                f.write("ANÁLISIS DETALLADO POR CONFIGURACIÓN:\n")
                f.write("-" * 50 + "\n")
                
                for config, resultado in resultados_completos.items():
                    f.write(f"\n{config}:\n")
                    if 'error' not in resultado:
                        f.write(f"  - Señales detectadas: {len(resultado.get('senales_detectadas', []))}\n")
                        f.write(f"  - Tiempo de ejecución: {resultado.get('tiempo_ejecucion', 0):.3f} segundos\n")
                        f.write(f"  - Confianza promedio: {resultado.get('confianza_promedio', 0):.3f}\n")
                        f.write(f"  - Candidatos Hough: {resultado.get('candidatos_hough', 0)}\n")
                        f.write(f"  - Candidatos contornos: {resultado.get('candidatos_contornos', 0)}\n")
                        
                        if resultado.get('senales_detectadas'):
                            radios = [senal[2] for senal in resultado['senales_detectadas']]
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
                detecciones = {c: len(r.get('senales_detectadas', []))
                             for c, r in resultados_completos.items() 
                             if 'error' not in r}
                if detecciones:
                    config_detectora = max(detecciones, key=detecciones.get)
                    f.write(f"- Configuración con más detecciones: {config_detectora} ({detecciones[config_detectora]} señales)\n")
                
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
    
    # Agregar los métodos a la clase DetectorSenales
    DetectorSenales.detectar_senales_todos_metodos = detectar_senales_todos_metodos
    DetectorSenales._guardar_info_deteccion_extendida = _guardar_info_deteccion_extendida
    DetectorSenales._generar_reporte_comparativo = _generar_reporte_comparativo
    
    print("Métodos múltiples agregados al DetectorSenales")

# Función de utilidad para usar desde el sistema principal
def detectar_senales_imagen_todos_metodos(ruta_imagen, forma='CIRCULAR', ruta_salida="./resultados_deteccion"):
    """
    Función de utilidad para detectar señales con todas las configuraciones.
    
    Args:
        ruta_imagen (str): Ruta de la imagen
        forma (str): Forma de señal a detectar
        ruta_salida (str): Directorio de salida
        
    Returns:
        dict: Resultados de todas las configuraciones
    """
    from detectores.detector_senales import DetectorSenales
    
    # Asegurarse de que los métodos estén disponibles
    agregar_metodos_multiples_senales()
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen: {ruta_imagen}")
        return None
    
    # Crear detector y ejecutar todas las configuraciones
    detector = DetectorSenales()
    return detector.detectar_senales_todos_metodos(imagen, forma, visualizar=False, guardar=True, ruta_base=ruta_salida)

if __name__ == "__main__":
    # Prueba del sistema
    agregar_metodos_multiples_senales()
    print("Sistema de configuraciones múltiples para señales listo.")