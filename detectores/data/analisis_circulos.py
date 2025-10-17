import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime

class AnalizadorCirculos:
    def __init__(self, carpeta_imagenes='images/', carpeta_resultados='resultados/'):
        self.carpeta_imagenes = carpeta_imagenes
        self.carpeta_resultados = carpeta_resultados
        
        # Crear carpeta de resultados si no existe
        if not os.path.exists(self.carpeta_resultados):
            os.makedirs(self.carpeta_resultados)
            
        self.resultados_df = pd.DataFrame(columns=[
            'Nombre_Imagen', 'Formato', 'Tamaño', 'Num_Circulos', 
            'Area_Media', 'Area_Max', 'Area_Min',
            'Perimetro_Medio', 'Perimetro_Max', 'Perimetro_Min',
            'Radio_Medio', 'Radio_Max', 'Radio_Min',
            'Filtro_Usado', 'Metodo_Deteccion'
        ])
        
        # Atributos para almacenar resultados de análisis
        self.imagen_original = None
        self.imagen_rgb = None
        self.imagen_gris = None
        self.nombre_imagen = None
        self.formato_imagen = None
        self.tamaño_imagen = None
        self.circulos = None
        self.circulos_contornos = None
        self.areas = []
        self.perimetros = []
        self.radios = []
            
    def cargar_imagen(self, ruta_imagen):
        """Carga una imagen desde la ruta especificada."""
        self.imagen_original = cv2.imread(ruta_imagen)
        if self.imagen_original is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        
        # Convertir BGR a RGB para visualización con matplotlib
        self.imagen_rgb = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB)
        self.nombre_imagen = os.path.basename(ruta_imagen)
        self.formato_imagen = os.path.splitext(ruta_imagen)[1]
        self.tamaño_imagen = self.imagen_original.shape
        
        return self.imagen_rgb
    
    def convertir_escala_grises(self, imagen=None):
        """Convierte una imagen a escala de grises."""
        if imagen is None:
            imagen = self.imagen_original
            
        if len(imagen.shape) == 2:  # Ya es escala de grises
            self.imagen_gris = imagen
            return self.imagen_gris
            
        self.imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return self.imagen_gris
    
    def detectar_circulos_hough(self, imagen=None, metodo=cv2.HOUGH_GRADIENT, 
                                dp=1, minDist=50, param1=50, param2=30, 
                                minRadius=0, maxRadius=0):
        """
        Detecta círculos usando la transformada de Hough.
        
        Args:
            imagen: Imagen en escala de grises (si es None, usa la imagen cargada)
            metodo: Método de detección (por defecto cv2.HOUGH_GRADIENT)
            dp: Resolución del acumulador
            minDist: Distancia mínima entre círculos
            param1: Umbral para el detector de bordes
            param2: Umbral para la detección de centros
            minRadius: Radio mínimo a detectar
            maxRadius: Radio máximo a detectar
            
        Returns:
            Tupla (imagen_con_circulos, circulos_detectados)
        """
        if imagen is None:
            if not hasattr(self, 'imagen_gris') or self.imagen_gris is None:
                self.imagen_gris = self.convertir_escala_grises()
            imagen = self.imagen_gris
            
        self.circulos = cv2.HoughCircles(
            imagen, 
            metodo, 
            dp, 
            minDist, 
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )
        
        # Crear una copia de la imagen original para dibujar los círculos
        self.imagen_con_circulos = self.imagen_rgb.copy() if hasattr(self, 'imagen_rgb') else None
        
        # Limpiar métricas anteriores
        self.areas = []
        self.perimetros = []
        self.radios = []
        
        if self.circulos is not None:
            # Convertir los valores a enteros
            self.circulos = np.uint16(np.around(self.circulos))
            
            for circulo in self.circulos[0, :]:
                centro_x, centro_y, radio = circulo[0], circulo[1], circulo[2]
                
                # Calcular área y perímetro
                area = np.pi * radio ** 2
                perimetro = 2 * np.pi * radio
                
                # Guardar métricas
                self.areas.append(area)
                self.perimetros.append(perimetro)
                self.radios.append(radio)
                
                # Dibujar círculo en la imagen
                if self.imagen_con_circulos is not None:
                    cv2.circle(self.imagen_con_circulos, (centro_x, centro_y), radio, (0, 255, 0), 2)
                    cv2.circle(self.imagen_con_circulos, (centro_x, centro_y), 2, (255, 0, 0), 3)
        
        return self.imagen_con_circulos, self.circulos
    
    def detectar_circulos_contornos(self, imagen=None, metodo=cv2.RETR_LIST, 
                              aprox=cv2.CHAIN_APPROX_SIMPLE, min_area=50, 
                              circularidad_min=0.6, invertir=False, debug=False):
        """
        Detecta círculos usando contornos.
        
        Args:
            imagen: Imagen binaria (si es None, usa la imagen binaria cargada)
            metodo: Método de recuperación de contornos (RETR_LIST para detectar todos)
            aprox: Método de aproximación de contornos
            min_area: Área mínima de los círculos a detectar
            circularidad_min: Umbral mínimo de circularidad (0-1, donde 1 es círculo perfecto)
            invertir: Si es True, invierte la imagen binaria antes de buscar contornos
            debug: Si es True, imprime información de depuración
            
        Returns:
            Tupla (imagen_con_contornos, circulos_detectados)
        """
        if imagen is None:
            if not hasattr(self, 'imagen_binaria'):
                if not hasattr(self, 'imagen_gris'):
                    self.imagen_gris = self.convertir_escala_grises()
                # Usar umbral adaptativo en lugar de fijo
                self.imagen_binaria = cv2.adaptiveThreshold(
                    self.imagen_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
            imagen = self.imagen_binaria
    
        # Asegurarnos de que la imagen sea binaria
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            _, imagen = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    
        # Opcionalmente invertir la imagen (útil cuando los círculos son blancos sobre fondo negro)
        if invertir:
            imagen = cv2.bitwise_not(imagen)
        
        # Aplicar operaciones morfológicas para mejorar la detección
        kernel = np.ones((5, 5), np.uint8)
        imagen = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel, iterations=2)
        imagen = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Guardar la imagen binaria procesada para depuración
        self.imagen_binaria_procesada = imagen.copy()
        
        # Encontrar contornos (la función findContours modifica la imagen de entrada)
        imagen_contornos = imagen.copy()  
        contornos, jerarquia = cv2.findContours(imagen_contornos, metodo, aprox)
        
        if debug:
            print(f"Total de contornos encontrados: {len(contornos)}")
        
        # Crear una copia de la imagen original para dibujar
        self.imagen_con_contornos = self.imagen_rgb.copy() if hasattr(self, 'imagen_rgb') else np.zeros((imagen.shape[0], imagen.shape[1], 3), dtype=np.uint8)
        
        # Limpiar métricas anteriores
        self.areas = []
        self.perimetros = []
        self.radios = []
        
        # Filtrar contornos circulares
        self.circulos_contornos = []
        
        for i, contorno in enumerate(contornos):
            # Calcular área y perímetro
            area = cv2.contourArea(contorno)
            
            # Filtrar por área mínima
            if area < min_area:
                continue
                
            perimetro = cv2.arcLength(contorno, True)
            
            # Calcular circularidad (4 * pi * area / perimetro^2)
            # Para un círculo perfecto, la circularidad es 1
            circularidad = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0
            
            if debug:
                print(f"Contorno {i}: Área={area:.2f}, Perímetro={perimetro:.2f}, Circularidad={circularidad:.4f}")
            
            # Filtrar por circularidad
            if circularidad > circularidad_min:
                # Calcular centro y radio aproximado
                (x, y), radio = cv2.minEnclosingCircle(contorno)
                centro = (int(x), int(y))
                radio = int(radio)
                
                self.circulos_contornos.append((centro, radio))
                
                # Guardar métricas
                self.areas.append(area)
                self.perimetros.append(perimetro)
                self.radios.append(radio)
                
                # Dibujar círculo en la imagen
                if self.imagen_con_contornos is not None:
                    cv2.circle(self.imagen_con_contornos, centro, radio, (0, 255, 0), 2)
                    cv2.circle(self.imagen_con_contornos, centro, 2, (255, 0, 0), 3)
                    # Dibujar el contorno original para depuración
                    cv2.drawContours(self.imagen_con_contornos, [contorno], 0, (255, 255, 0), 1)
        
        if debug:
            print(f"Círculos detectados: {len(self.circulos_contornos)}")
        
        return self.imagen_con_contornos, self.circulos_contornos

    def analizar_circulos(self):
        """
        Analiza los círculos detectados y calcula estadísticas.
        
        Returns:
            Diccionario con estadísticas de los círculos
        """
        if not self.areas:
            return {
                "num_circulos": 0,
                "areas": {
                    "media": 0,
                    "max": 0,
                    "min": 0,
                    "desv_std": 0
                },
                "radios": {
                    "media": 0,
                    "max": 0,
                    "min": 0,
                    "desv_std": 0
                },
                "perimetros": {
                    "media": 0,
                    "max": 0,
                    "min": 0,
                    "desv_std": 0
                }
            }
            
        return {
            "num_circulos": len(self.areas),
            "areas": {
                "media": np.mean(self.areas),
                "max": np.max(self.areas),
                "min": np.min(self.areas),
                "desv_std": np.std(self.areas)
            },
            "radios": {
                "media": np.mean(self.radios),
                "max": np.max(self.radios),
                "min": np.min(self.radios),
                "desv_std": np.std(self.radios)
            },
            "perimetros": {
                "media": np.mean(self.perimetros),
                "max": np.max(self.perimetros),
                "min": np.min(self.perimetros),
                "desv_std": np.std(self.perimetros)
            }
        }
    
    def procesar_imagen(self, ruta_imagen, metodo_deteccion='hough'):
        """
        Procesa una imagen completa aplicando una serie de pasos para detectar círculos.
        
        Args:
            ruta_imagen: Ruta a la imagen a procesar
            metodo_deteccion: Método de detección ('hough' o 'contornos')
            
        Returns:
            Tupla (imagen_resultado, circulos_detectados)
        """
        print(f"Procesando imagen: {ruta_imagen}")
        
        # Cargar imagen
        self.cargar_imagen(ruta_imagen)
        
        # Convertir a escala de grises
        imagen_gris = self.convertir_escala_grises()
        
        # Aplicar filtro gaussiano para reducir ruido
        imagen_filtrada = cv2.GaussianBlur(self.imagen_original, (5, 5), 0)
        
        # Convertir imagen filtrada a escala de grises
        imagen_filtrada_gris = self.convertir_escala_grises(imagen_filtrada)
        
        # Preprocesamiento específico según el método
        if metodo_deteccion == 'hough':
            # Para Hough: Binarización simple puede funcionar
            _, imagen_binaria = cv2.threshold(imagen_filtrada_gris, 127, 255, cv2.THRESH_BINARY)
        else:
            # Para contornos: Usar umbral adaptativo para mejor segmentación
            imagen_binaria = cv2.adaptiveThreshold(
                imagen_filtrada_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
        
        # Aplicar operación morfológica para mejorar la detección
        kernel = np.ones((5, 5), np.uint8)
        imagen_morfologica = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel, iterations=2)
        imagen_morfologica = cv2.morphologyEx(imagen_morfologica, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Detectar círculos según el método seleccionado
        if metodo_deteccion == 'hough':
            imagen_resultado, circulos = self.detectar_circulos_hough(
                imagen=imagen_filtrada_gris,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=10,
                maxRadius=100
            )
        else:  # 'contornos'
            # Intentar diferentes configuraciones para mejorar la detección
            imagen_resultado, circulos = self.detectar_circulos_contornos(
                imagen=imagen_morfologica,
                metodo=cv2.RETR_LIST,
                min_area=50,
                circularidad_min=0.6,
                invertir=False,
                debug=True  # Activar modo debug para ver información
            )
        
        # Guardar imágenes procesadas
        nombre_base = os.path.splitext(self.nombre_imagen)[0]
        
        self.guardar_imagen(self.imagen_rgb, f"{nombre_base}_original.jpg")
        self.guardar_imagen(imagen_gris, f"{nombre_base}_gris.jpg")
        self.guardar_imagen(cv2.cvtColor(imagen_filtrada, cv2.COLOR_BGR2RGB), f"{nombre_base}_filtrada.jpg")
        self.guardar_imagen(imagen_binaria, f"{nombre_base}_binaria.jpg")
        self.guardar_imagen(imagen_morfologica, f"{nombre_base}_morfologica.jpg")
        
        # También guardar la imagen binaria procesada si estamos usando contornos
        if metodo_deteccion == 'contornos' and hasattr(self, 'imagen_binaria_procesada'):
            self.guardar_imagen(self.imagen_binaria_procesada, f"{nombre_base}_binaria_procesada.jpg")
        
        self.guardar_imagen(imagen_resultado, f"{nombre_base}_resultado.jpg")
        
        # Añadir resultados al DataFrame
        num_circulos = len(self.areas) if hasattr(self, 'areas') and self.areas else 0
        
        if num_circulos > 0:
            area_media = np.mean(self.areas)
            area_max = np.max(self.areas)
            area_min = np.min(self.areas)
            
            perimetro_medio = np.mean(self.perimetros)
            perimetro_max = np.max(self.perimetros)
            perimetro_min = np.min(self.perimetros)
            
            radio_medio = np.mean(self.radios)
            radio_max = np.max(self.radios)
            radio_min = np.min(self.radios)
        else:
            area_media = area_max = area_min = 0
            perimetro_medio = perimetro_max = perimetro_min = 0
            radio_medio = radio_max = radio_min = 0
        
        nueva_fila = {
            'Nombre_Imagen': self.nombre_imagen,
            'Formato': self.formato_imagen,
            'Tamaño': str(self.tamaño_imagen),
            'Num_Circulos': num_circulos,
            'Area_Media': area_media,
            'Area_Max': area_max,
            'Area_Min': area_min,
            'Perimetro_Medio': perimetro_medio,
            'Perimetro_Max': perimetro_max,
            'Perimetro_Min': perimetro_min,
            'Radio_Medio': radio_medio,
            'Radio_Max': radio_max,
            'Radio_Min': radio_min,
            'Filtro_Usado': 'Gaussiano',
            'Metodo_Deteccion': metodo_deteccion
        }
        
        self.resultados_df = self.resultados_df._append(nueva_fila, ignore_index=True)
        
        return imagen_resultado, circulos
    
    def procesar_multiples_imagenes(self, lista_rutas, metodo_deteccion='hough'):
        """
        Procesa múltiples imágenes y genera un informe consolidado.
        
        Args:
            lista_rutas: Lista de rutas de imágenes a procesar
            metodo_deteccion: Método de detección ('hough' o 'contornos')
            
        Returns:
            DataFrame con resultados del análisis
        """
        # Limpiar DataFrame de resultados antes de empezar
        self.resultados_df = pd.DataFrame(columns=[
            'Nombre_Imagen', 'Formato', 'Tamaño', 'Num_Circulos', 
            'Area_Media', 'Area_Max', 'Area_Min',
            'Perimetro_Medio', 'Perimetro_Max', 'Perimetro_Min',
            'Radio_Medio', 'Radio_Max', 'Radio_Min',
            'Filtro_Usado', 'Metodo_Deteccion'
        ])
        
        # Variables para mantener datos para el PDF
        todas_areas = []
        todos_perimetros = []
        todos_radios = []
        
        for ruta in lista_rutas:
            # Guardar el estado de análisis anterior temporalmente
            areas_ant = self.areas.copy() if hasattr(self, 'areas') and self.areas else []
            perimetros_ant = self.perimetros.copy() if hasattr(self, 'perimetros') and self.perimetros else []
            radios_ant = self.radios.copy() if hasattr(self, 'radios') and self.radios else []
            
            # Limpiar el estado para el nuevo análisis
            self.imagen_original = None
            self.imagen_rgb = None
            self.imagen_gris = None
            self.nombre_imagen = None
            self.formato_imagen = None
            self.tamaño_imagen = None
            self.circulos = None
            self.circulos_contornos = None
            self.areas = []
            self.perimetros = []
            self.radios = []
            self.imagen_con_circulos = None
            self.imagen_con_contornos = None
            self.imagen_binaria = None
            self.imagen_gauss = None
            self.imagen_morfologica = None
            
            # Procesar la imagen actual
            self.procesar_imagen(ruta, metodo_deteccion)
            
            # Acumular datos para el PDF
            todas_areas.extend(self.areas)
            todos_perimetros.extend(self.perimetros)
            todos_radios.extend(self.radios)
        
        # Restaurar datos acumulados para el PDF
        self.areas = todas_areas
        self.perimetros = todos_perimetros
        self.radios = todos_radios
            
        # Guardar resultados en Excel
        ruta_excel = self.guardar_resultados_excel()
        print(f"Resultados guardados en: {ruta_excel}")
        
        return self.resultados_df
    
    def guardar_imagen(self, imagen, nombre_archivo):
        """
        Guarda una imagen procesada.
        
        Args:
            imagen: Imagen a guardar
            nombre_archivo: Nombre del archivo de salida
            
        Returns:
            Ruta completa donde se guardó la imagen
        """
        ruta_completa = os.path.join(self.carpeta_resultados, nombre_archivo)
        
        # Para imágenes que deben guardarse con OpenCV (no RGB)
        if len(imagen.shape) == 3 and imagen.shape[2] == 3:
            # Convertir RGB a BGR para guardar con OpenCV
            imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
            cv2.imwrite(ruta_completa, imagen_bgr)
        else:
            # Para imágenes en escala de grises o binarias
            cv2.imwrite(ruta_completa, imagen)
            
        return ruta_completa
    
    def guardar_resultados_excel(self):
        """
        Guarda los resultados en un archivo Excel.
        
        Returns:
            Ruta completa al archivo Excel generado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"resultados_circulos_{timestamp}.xlsx"
        ruta_completa = os.path.join(self.carpeta_resultados, nombre_archivo)
        
        self.resultados_df.to_excel(ruta_completa, index=False)
        
        return ruta_completa
    
    def visualizar_resultados(self, ruta_imagen=None, metodo_deteccion='hough'):
        """
        Visualiza los resultados del procesamiento en una figura completa.
        
        Args:
            ruta_imagen: Ruta a la imagen (si es None, usa la imagen actual)
            metodo_deteccion: Método de detección ('hough' o 'contornos')
            
        Returns:
            Objeto de figura de matplotlib
        """
        if ruta_imagen:
            # Procesar la imagen
            imagen_resultado, _ = self.procesar_imagen(ruta_imagen, metodo_deteccion)
        elif not hasattr(self, 'imagen_rgb') or self.imagen_rgb is None:
            raise ValueError("No hay imagen cargada para visualizar.")
        
        # Crear figura para visualización
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Mostrar imágenes
        axs[0, 0].imshow(self.imagen_rgb)
        axs[0, 0].set_title('Imagen Original')
        axs[0, 0].axis('off')
        
        if hasattr(self, 'imagen_gris') and self.imagen_gris is not None:
            axs[0, 1].imshow(self.imagen_gris, cmap='gray')
            axs[0, 1].set_title('Escala de Grises')
            axs[0, 1].axis('off')
        
        # Mostrar imagen con filtro gaussiano
        if hasattr(self, 'imagen_gauss') and self.imagen_gauss is not None:
            axs[0, 2].imshow(cv2.cvtColor(self.imagen_gauss, cv2.COLOR_BGR2RGB))
            axs[0, 2].set_title('Filtro Gaussiano')
            axs[0, 2].axis('off')
        
        # Mostrar imagen binaria
        if hasattr(self, 'imagen_binaria') and self.imagen_binaria is not None:
            axs[1, 0].imshow(self.imagen_binaria, cmap='gray')
            axs[1, 0].set_title('Imagen Binaria')
            axs[1, 0].axis('off')
        
        # Mostrar resultado de operaciones morfológicas
        if hasattr(self, 'imagen_morfologica') and self.imagen_morfologica is not None:
            axs[1, 1].imshow(self.imagen_morfologica, cmap='gray')
            axs[1, 1].set_title('Op. Morfológicas')
            axs[1, 1].axis('off')
        
        # Mostrar resultado final con círculos detectados
        if metodo_deteccion == 'hough':
            if hasattr(self, 'imagen_con_circulos') and self.imagen_con_circulos is not None:
                axs[1, 2].imshow(self.imagen_con_circulos)
                axs[1, 2].set_title(f'Círculos Detectados ({metodo_deteccion})')
                axs[1, 2].axis('off')
        else:
            if hasattr(self, 'imagen_con_contornos') and self.imagen_con_contornos is not None:
                axs[1, 2].imshow(self.imagen_con_contornos)
                axs[1, 2].set_title(f'Círculos Detectados ({metodo_deteccion})')
                axs[1, 2].axis('off')
        
        # Mostrar información sobre círculos detectados
        num_circulos = 0
        if metodo_deteccion == 'hough' and hasattr(self, 'circulos') and self.circulos is not None:
            num_circulos = len(self.circulos[0])
        elif metodo_deteccion == 'contornos' and hasattr(self, 'circulos_contornos') and self.circulos_contornos:
            num_circulos = len(self.circulos_contornos)
        
        info_texto = f"Círculos detectados: {num_circulos}"
        if num_circulos > 0:
            info_texto += f"\nRadio medio: {np.mean(self.radios):.2f}px"
        
        plt.figtext(0.5, 0.01, info_texto, ha='center')
        
        plt.tight_layout()
        
        return fig
