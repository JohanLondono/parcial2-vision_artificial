#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de Llantas en Imágenes de Tráfico Vehicular
===================================================

Sistema especializado para detectar llantas de vehículos utilizando
múltiples algoritmos de visión por computadora.

Algoritmos utilizados:
- Transformada de Hough para círculos
- AKAZE para puntos característicos
- Análisis de texturas y bordes
- Filtrado por color y forma
"""

import cv2
import numpy as np
import os
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

class DetectorLlantas:
    """Detector especializado de llantas en vehículos."""
    
    def __init__(self):
        """Inicializar el detector de llantas."""
        self.config = {
            'hough_circles': {
                'dp': 1,
                'min_dist': 50,
                'param1': 100,
                'param2': 30,
                'min_radius': 15,
                'max_radius': 200
            },
            'akaze': {
                'threshold': 0.003,
                'nOctaves': 4,
                'nOctaveLayers': 4,
                'diffusivity': cv2.KAZE_DIFF_PM_G2
            },
            'color_ranges': {
                'negro_bajo': [0, 0, 0],
                'negro_alto': [180, 255, 80],
                'gris_bajo': [0, 0, 50],
                'gris_alto': [180, 50, 200]
            }
        }
    
    def detectar_llantas(self, imagen, metodo='combinado', visualizar=True, guardar=False, ruta_salida=None):
        """
        Detecta llantas en la imagen usando el método especificado.
        
        Args:
            imagen (np.ndarray): Imagen de entrada
            metodo (str): Método a usar ('hough', 'akaze', 'textura', 'combinado')
            visualizar (bool): Si mostrar resultados
            guardar (bool): Si guardar imagen resultado
            ruta_salida (str): Ruta donde guardar resultado
            
        Returns:
            dict: Resultados de la detección
        """
        print(f"🔍 Detectando llantas usando método: {metodo}")
        
        if metodo == 'hough':
            return self._detectar_llantas_hough(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'akaze':
            return self._detectar_llantas_akaze(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'textura':
            return self._detectar_llantas_textura(imagen, visualizar, guardar, ruta_salida)
        elif metodo == 'combinado':
            return self._detectar_llantas_combinado(imagen, visualizar, guardar, ruta_salida)
        else:
            print(f"❌ Método no reconocido: {metodo}")
            return None
    
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
        
        # Definir métodos individuales (excluir combinado para evitar redundancia)
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
                if metodo == 'hough':
                    resultado = self._detectar_llantas_hough(imagen, visualizar, guardar, ruta_salida)
                elif metodo == 'akaze':
                    resultado = self._detectar_llantas_akaze(imagen, visualizar, guardar, ruta_salida)
                elif metodo == 'textura':
                    resultado = self._detectar_llantas_textura(imagen, visualizar, guardar, ruta_salida)
                
                if resultado:
                    resultados_completos[metodo] = resultado
                    print(f"    ✅ {metodo.upper()}: {len(resultado.get('llantas_detectadas', []))} llantas detectadas")
                    
                    # Guardar información detallada del método
                    if guardar and ruta_base:
                        self._guardar_info_deteccion(resultado, metodo, ruta_base)
                else:
                    resultados_completos[metodo] = {'error': 'Falló la detección'}
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
                
            resultado_combinado = self._detectar_llantas_combinado(imagen, visualizar, guardar, ruta_salida)
            if resultado_combinado:
                resultados_completos['combinado'] = resultado_combinado
                print(f"    ✅ COMBINADO: {len(resultado_combinado.get('llantas_detectadas', []))} llantas detectadas")
                
                if guardar and ruta_base:
                    self._guardar_info_deteccion(resultado_combinado, 'combinado', ruta_base)
            else:
                resultados_completos['combinado'] = {'error': 'Falló la detección combinada'}
                print(f"    ❌ COMBINADO: Error en detección")
                
        except Exception as e:
            print(f"    ❌ COMBINADO: Error - {e}")
            resultados_completos['combinado'] = {'error': str(e)}
        
        print(f"\n🎉 Detección completa de llantas finalizada. {len(resultados_completos)} métodos ejecutados.")
        return resultados_completos
    
    def _detectar_llantas_hough(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Detecta llantas usando Transformada de Hough para círculos."""
        # Preprocesar imagen
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (9, 9), 2)
        
        # Detectar círculos con Hough
        circulos = cv2.HoughCircles(
            gris,
            cv2.HOUGH_GRADIENT,
            dp=self.config['hough_circles']['dp'],
            minDist=self.config['hough_circles']['min_dist'],
            param1=self.config['hough_circles']['param1'],
            param2=self.config['hough_circles']['param2'],
            minRadius=self.config['hough_circles']['min_radius'],
            maxRadius=self.config['hough_circles']['max_radius']
        )
        
        llantas_detectadas = []
        imagen_resultado = imagen.copy()
        
        if circulos is not None:
            circulos = np.round(circulos[0, :]).astype("int")
            
            for (x, y, r) in circulos:
                # Validar si es realmente una llanta
                if self._validar_llanta_hough(imagen, x, y, r):
                    llantas_detectadas.append((x, y, r))
                    
                    # Dibujar círculo detectado
                    cv2.circle(imagen_resultado, (x, y), r, (0, 255, 0), 3)
                    cv2.circle(imagen_resultado, (x, y), 2, (0, 255, 0), -1)
                    cv2.putText(imagen_resultado, "Llanta", (x-30, y-r-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calcular estadísticas
        resultado = {
            'metodo': 'Hough Circles',
            'num_llantas': len(llantas_detectadas),
            'llantas': llantas_detectadas,
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': self._calcular_confianza_hough(llantas_detectadas)
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Llantas - Hough")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "hough")
        
        return resultado
    
    def _detectar_llantas_akaze(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Detecta llantas usando detector AKAZE."""
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Crear detector AKAZE
        detector_akaze = cv2.AKAZE_create(
            threshold=self.config['akaze']['threshold'],
            nOctaves=self.config['akaze']['nOctaves'],
            nOctaveLayers=self.config['akaze']['nOctaveLayers'],
            diffusivity=self.config['akaze']['diffusivity']
        )
        
        # Detectar puntos clave
        keypoints, _ = detector_akaze.detectAndCompute(gris, None)
        
        # Agrupar keypoints en regiones circulares potenciales
        llantas_detectadas = self._agrupar_keypoints_circulares(keypoints)
        
        # Dibujar resultados
        imagen_resultado = imagen.copy()
        
        # Dibujar keypoints
        imagen_keypoints = cv2.drawKeypoints(imagen_resultado, keypoints, None, 
                                           color=(255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Dibujar llantas detectadas
        for (x, y, r) in llantas_detectadas:
            cv2.circle(imagen_keypoints, (x, y), r, (0, 255, 0), 3)
            cv2.putText(imagen_keypoints, "Llanta", (x-30, y-r-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        resultado = {
            'metodo': 'AKAZE',
            'num_llantas': len(llantas_detectadas),
            'llantas': llantas_detectadas,
            'keypoints': len(keypoints),
            'imagen_resultado': imagen_keypoints,
            'confianza_promedio': self._calcular_confianza_akaze(keypoints, llantas_detectadas)
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Llantas - AKAZE")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_keypoints, ruta_salida, "akaze")
        
        return resultado
    
    def _detectar_llantas_textura(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Detecta llantas usando análisis de texturas."""
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Análisis de texturas con GLCM
        gris_uint8 = img_as_ubyte(gris)
        
        # Calcular GLCM
        distancias = [1, 2, 3]
        angulos = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(gris_uint8, distancias, angulos, levels=256, symmetric=True, normed=True)
        
        # Propiedades de textura
        contraste = np.mean(graycoprops(glcm, 'contrast'))
        energia = np.mean(graycoprops(glcm, 'energy'))
        homogeneidad = np.mean(graycoprops(glcm, 'homogeneity'))
        correlacion = np.mean(graycoprops(glcm, 'correlation'))
        
        # Detectar bordes para encontrar formas circulares
        bordes = cv2.Canny(gris, 50, 150)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        llantas_detectadas = []
        imagen_resultado = imagen.copy()
        
        for contorno in contornos:
            # Filtrar por área
            area = cv2.contourArea(contorno)
            if area < 500 or area > 50000:
                continue
            
            # Verificar circularidad
            perimetro = cv2.arcLength(contorno, True)
            if perimetro == 0:
                continue
                
            circularidad = 4 * np.pi * area / (perimetro * perimetro)
            
            if circularidad > 0.3:  # Umbral de circularidad
                # Encontrar círculo que mejor ajuste
                (x, y), radio = cv2.minEnclosingCircle(contorno)
                x, y, radio = int(x), int(y), int(radio)
                
                # Validar por textura
                if self._validar_llanta_textura(radio, contraste, energia):
                    llantas_detectadas.append((x, y, radio))
                    cv2.circle(imagen_resultado, (x, y), radio, (0, 255, 0), 3)
                    cv2.putText(imagen_resultado, "Llanta", (x-30, y-radio-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        resultado = {
            'metodo': 'Análisis de Texturas',
            'num_llantas': len(llantas_detectadas),
            'llantas': llantas_detectadas,
            'textura_contraste': contraste,
            'textura_energia': energia,
            'textura_homogeneidad': homogeneidad,
            'textura_correlacion': correlacion,
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': circularidad if llantas_detectadas else 0
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Llantas - Texturas")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "textura")
        
        return resultado
    
    def _detectar_llantas_combinado(self, imagen, visualizar=True, guardar=False, ruta_salida=None):
        """Combina múltiples métodos para detección robusta."""
        # Ejecutar todos los métodos
        resultado_hough = self._detectar_llantas_hough(imagen, False, False)
        resultado_akaze = self._detectar_llantas_akaze(imagen, False, False)
        resultado_textura = self._detectar_llantas_textura(imagen, False, False)
        
        # Fusionar resultados
        todas_llantas = []
        todas_llantas.extend(resultado_hough['llantas'])
        todas_llantas.extend(resultado_akaze['llantas'])
        todas_llantas.extend(resultado_textura['llantas'])
        
        # Eliminar duplicados usando NMS (Non-Maximum Suppression)
        llantas_finales = self._aplicar_nms(todas_llantas, umbral_distancia=30)
        
        # Crear imagen resultado
        imagen_resultado = imagen.copy()
        for i, (x, y, r) in enumerate(llantas_finales):
            cv2.circle(imagen_resultado, (x, y), r, (0, 255, 0), 3)
            cv2.circle(imagen_resultado, (x, y), 2, (255, 0, 0), -1)
            cv2.putText(imagen_resultado, f"Llanta {i+1}", (x-30, y-r-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        resultado = {
            'metodo': 'Combinado (Hough + AKAZE + Texturas)',
            'num_llantas': len(llantas_finales),
            'llantas': llantas_finales,
            'detecciones_hough': len(resultado_hough['llantas']),
            'detecciones_akaze': len(resultado_akaze['llantas']),
            'detecciones_textura': len(resultado_textura['llantas']),
            'imagen_resultado': imagen_resultado,
            'confianza_promedio': (resultado_hough['confianza_promedio'] + 
                                 resultado_akaze['confianza_promedio'] + 
                                 resultado_textura['confianza_promedio']) / 3
        }
        
        if visualizar:
            self._mostrar_resultado(resultado, "Detección de Llantas - Método Combinado")
        
        if guardar and ruta_salida:
            self._guardar_resultado(imagen_resultado, ruta_salida, "combinado")
        
        return resultado
    
    def _validar_llanta_hough(self, imagen, x, y, r):
        """Valida si un círculo detectado es realmente una llanta."""
        # Verificar que el círculo esté dentro de la imagen
        alto, ancho = imagen.shape[:2]
        if x - r < 0 or x + r >= ancho or y - r < 0 or y + r >= alto:
            return False
        
        # Extraer región del círculo
        mask = np.zeros((alto, ancho), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Analizar color en HSV
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        region_hsv = hsv[mask == 255]
        
        if len(region_hsv) == 0:
            return False
        
        # Verificar si predominan colores oscuros (negro/gris)
        media_v = np.mean(region_hsv[:, 2])  # Canal de valor (brillo)
        return media_v < 100  # Umbral para colores oscuros
    
    def _validar_llanta_textura(self, r, contraste, energia):
        """Valida llanta basándose en características de textura."""
        # Verificar dimensiones mínimas/máximas
        if r < 20 or r > 150:
            return False
        
        # Las llantas tienen alta textura (alto contraste, baja energía)
        return contraste > 0.3 and energia < 0.3
    
    def _agrupar_keypoints_circulares(self, keypoints):
        """Agrupa keypoints en regiones circulares potenciales."""
        if not keypoints:
            return []
        
        # Convertir keypoints a coordenadas
        puntos = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        # Usar clustering simple para agrupar puntos
        from sklearn.cluster import DBSCAN
        
        # Aplicar DBSCAN para encontrar grupos de puntos
        clustering = DBSCAN(eps=30, min_samples=5).fit(puntos)
        labels = clustering.labels_
        
        llantas = []
        for label in set(labels):
            if label == -1:  # Ruido
                continue
            
            # Puntos del cluster
            cluster_points = puntos[labels == label]
            
            # Calcular centro y radio promedio
            centro_x = np.mean(cluster_points[:, 0])
            centro_y = np.mean(cluster_points[:, 1])
            
            # Calcular radio como distancia promedio desde el centro
            distancias = np.sqrt((cluster_points[:, 0] - centro_x)**2 + 
                               (cluster_points[:, 1] - centro_y)**2)
            radio = np.mean(distancias)
            
            if 15 <= radio <= 100:  # Filtrar por tamaño razonable
                llantas.append((int(centro_x), int(centro_y), int(radio)))
        
        return llantas
    
    def _aplicar_nms(self, llantas, umbral_distancia=30):
        """Aplica Non-Maximum Suppression para eliminar detecciones duplicadas."""
        if not llantas:
            return []
        
        llantas = np.array(llantas)
        indices_mantener = []
        
        for i in range(len(llantas)):
            mantener = True
            for j in range(len(llantas)):
                if i != j and j in indices_mantener:
                    # Calcular distancia entre centros
                    distancia = np.sqrt((llantas[i][0] - llantas[j][0])**2 + 
                                      (llantas[i][1] - llantas[j][1])**2)
                    if distancia < umbral_distancia:
                        mantener = False
                        break
            
            if mantener:
                indices_mantener.append(i)
        
        return [tuple(llantas[i]) for i in indices_mantener]
    
    def _calcular_confianza_hough(self, llantas):
        """Calcula confianza promedio para detecciones Hough."""
        if not llantas:
            return 0.0
        
        confianzas = []
        for (x, y, r) in llantas:
            # Confianza basada en qué tan circular es la región
            confianza = min(1.0, r / 50.0)  # Normalizar por tamaño típico
            confianzas.append(confianza)
        
        return np.mean(confianzas)
    
    def _calcular_confianza_akaze(self, keypoints, llantas):
        """Calcula confianza para detecciones AKAZE."""
        if not llantas:
            return 0.0
        
        # Confianza basada en densidad de keypoints
        return min(1.0, len(keypoints) / 100.0)
    
    def _mostrar_resultado(self, resultado, titulo):
        """Muestra resultado de la detección."""
        print(f"\n📊 Resultado - {titulo}")
        print(f"Método: {resultado['metodo']}")
        print(f"Llantas detectadas: {resultado['num_llantas']}")
        if 'confianza_promedio' in resultado:
            print(f"Confianza promedio: {resultado['confianza_promedio']:.3f}")
        
        # Mostrar imagen resultado
        plt.figure(figsize=(12, 8))
        imagen_rgb = cv2.cvtColor(resultado['imagen_resultado'], cv2.COLOR_BGR2RGB)
        plt.imshow(imagen_rgb)
        plt.title(titulo)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _guardar_resultado(self, imagen_resultado, ruta_base, metodo):
        """Guarda imagen resultado."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"llantas_{metodo}_{timestamp}.jpg"
        
        if ruta_base:
            directorio = os.path.dirname(ruta_base)
            ruta_completa = os.path.join(directorio, nombre_archivo)
        else:
            ruta_completa = nombre_archivo
        
        os.makedirs(os.path.dirname(ruta_completa), exist_ok=True)
        cv2.imwrite(ruta_completa, imagen_resultado)
        print(f"✅ Resultado guardado: {ruta_completa}")

# Función de utilidad
def detectar_llantas_imagen(ruta_imagen, metodo='combinado', visualizar=True, guardar=False, ruta_salida=None):
    """
    Función de conveniencia para detectar llantas en una imagen.
    
    Args:
        ruta_imagen (str): Ruta de la imagen
        metodo (str): Método de detección
        visualizar (bool): Si mostrar resultados
        guardar (bool): Si guardar resultados
        ruta_salida (str): Ruta donde guardar
        
    Returns:
        dict: Resultados de la detección
    """
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"❌ Error: No se pudo cargar la imagen: {ruta_imagen}")
        return None
    
    # Crear detector y ejecutar
    detector = DetectorLlantas()
    return detector.detectar_llantas(imagen, metodo, visualizar, guardar, ruta_salida)

    def _guardar_info_deteccion(self, resultado, metodo, ruta_base):
        """
        Guarda información detallada de la detección.
        
        Args:
            resultado (dict): Resultado de la detección
            metodo (str): Método utilizado
            ruta_base (str): Ruta base donde guardar
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
                f.write(f"REPORTE DE DETECCIÓN DE LLANTAS - MÉTODO {metodo.upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Método utilizado: {metodo}\n")
                f.write(f"Número de llantas detectadas: {len(resultado.get('llantas_detectadas', []))}\n\n")
                
                # Información de parámetros específicos del método
                if metodo == 'hough':
                    f.write("PARÁMETROS DE HOUGH CIRCLES:\n")
                    f.write("-" * 30 + "\n")
                    config = resultado.get('config_utilizada', {})
                    for param, valor in config.items():
                        f.write(f"  {param}: {valor}\n")
                
                elif metodo == 'akaze':
                    f.write("PARÁMETROS DE AKAZE:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  Keypoints detectados: {resultado.get('num_keypoints', 0)}\n")
                    f.write(f"  Clusters formados: {resultado.get('num_clusters', 0)}\n")
                
                elif metodo == 'textura':
                    f.write("ANÁLISIS DE TEXTURAS:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  Regiones analizadas: {resultado.get('num_regiones', 0)}\n")
                    f.write(f"  Criterio de textura aplicado: {resultado.get('criterio_textura', 'N/A')}\n")
                
                # Detalles de llantas detectadas
                if resultado.get('llantas_detectadas'):
                    f.write(f"\nDETALLES DE LLANTAS DETECTADAS:\n")
                    f.write("-" * 40 + "\n")
                    for i, llanta in enumerate(resultado['llantas_detectadas'], 1):
                        f.write(f"  Llanta {i}:\n")
                        f.write(f"    Centro: ({llanta[0]:.1f}, {llanta[1]:.1f})\n")
                        f.write(f"    Radio: {llanta[2]:.1f} píxeles\n")
                        if len(llanta) > 3:
                            f.write(f"    Confianza: {llanta[3]:.3f}\n")
                        f.write("\n")
                
                # Métricas de rendimiento si están disponibles
                if 'tiempo_ejecucion' in resultado:
                    f.write(f"MÉTRICAS DE RENDIMIENTO:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  Tiempo de ejecución: {resultado['tiempo_ejecucion']:.3f} segundos\n")
                
                # Información de la imagen procesada
                if 'imagen_info' in resultado:
                    info = resultado['imagen_info']
                    f.write(f"\nINFORMACIÓN DE LA IMAGEN:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  Dimensiones: {info.get('width', 'N/A')} x {info.get('height', 'N/A')}\n")
                    f.write(f"  Canales: {info.get('channels', 'N/A')}\n")
                
            print(f"    📄 Reporte guardado: {ruta_reporte}")
            
        except Exception as e:
            print(f"    ⚠️  Error guardando reporte: {e}")


# Función de utilidad para detección con todos los métodos
def detectar_llantas_imagen_todos_metodos(ruta_imagen, ruta_salida="./resultados_deteccion"):
    """
    Función de utilidad para detectar llantas con todos los métodos.
    
    Args:
        ruta_imagen (str): Ruta de la imagen
        ruta_salida (str): Directorio de salida
        
    Returns:
        dict: Resultados de todos los métodos
    """
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"❌ Error: No se pudo cargar la imagen: {ruta_imagen}")
        return None
    
    # Crear detector y ejecutar todos los métodos
    detector = DetectorLlantas()
    return detector.detectar_llantas_todos_metodos(imagen, visualizar=False, guardar=True, ruta_base=ruta_salida)