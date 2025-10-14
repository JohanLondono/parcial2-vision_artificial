function menu_preprocesamiento_matlab()
    % MENU_PREPROCESAMIENTO_MATLAB
    % Menú principal para el preprocesamiento de imágenes en MATLAB
    % Incluye filtros, operaciones aritméticas, geométricas, lógicas y morfológicas
    
    % Variables globales para mantener el estado
    global imagen_original;        % Imagen original sin modificar
    global imagen_activa;          % Imagen actual (con procesamientos aplicados)
    global imagen_procesada;       % Resultado del último procesamiento
    global nombre_imagen_activa;
    global ruta_guardado;
    global historial_procesamientos;  % Historial de procesamientos aplicados
    global usar_imagen_procesada;     % Flag para usar imagen procesada como base
    
    % Inicializar variables
    imagen_original = [];
    imagen_activa = [];
    imagen_procesada = [];
    nombre_imagen_activa = '';
    ruta_guardado = 'resultados_matlab';
    historial_procesamientos = {};
    usar_imagen_procesada = false;
    
    % Crear directorio de resultados si no existe
    if ~exist(ruta_guardado, 'dir')
        mkdir(ruta_guardado);
    end
    
    % Mostrar menú principal
    while true
        mostrar_menu_principal();
        opcion = input('Seleccione una opción: ', 's');
        
        switch opcion
            case '1'
                cargar_imagen();
            case '2'
                if verificar_imagen_cargada()
                    menu_filtros();
                end
            case '3'
                if verificar_imagen_cargada()
                    menu_operaciones_aritmeticas();
                end
            case '4'
                if verificar_imagen_cargada()
                    menu_operaciones_geometricas();
                end
            case '5'
                if verificar_imagen_cargada()
                    menu_operaciones_logicas();
                end
            case '6'
                if verificar_imagen_cargada()
                    menu_operaciones_morfologicas();
                end
            case '7'
                if verificar_imagen_cargada()
                    menu_segmentacion();
                end
            case '8'
                if verificar_imagen_cargada()
                    mostrar_imagen_actual();
                end
            case '9'
                if verificar_imagen_cargada()
                    aplicar_procesamiento_a_imagen_activa();
                end
            case '10'
                if verificar_imagen_cargada()
                    mostrar_historial_procesamientos();
                end
            case '11'
                if verificar_imagen_cargada()
                    resetear_a_imagen_original();
                end
            case '12'
                if verificar_imagen_cargada()
                    guardar_imagen_procesada();
                end
            case '0'
                fprintf('¡Gracias por usar el programa!\n');
                break;
            otherwise
                fprintf('Opción no válida. Intente nuevamente.\n');
        end
    end
end

function mostrar_menu_principal()
    % Muestra el menú principal de opciones
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('           PREPROCESAMIENTO DE IMÁGENES - MATLAB\n');
    fprintf('%s\n', repmat('=', 1, 60));
    fprintf('1. Cargar imagen\n');
    fprintf('2. Aplicar filtros\n');
    fprintf('3. Operaciones aritméticas\n');
    fprintf('4. Operaciones geométricas\n');
    fprintf('5. Operaciones lógicas\n');
    fprintf('6. Operaciones morfológicas\n');
    fprintf('7. Segmentación de imágenes\n');
    fprintf('8. Mostrar imagen actual\n');
    fprintf('9. Aplicar procesamiento a imagen activa\n');
    fprintf('10. Ver historial de procesamientos\n');
    fprintf('11. Resetear a imagen original\n');
    fprintf('12. Guardar imagen actual\n');
    fprintf('0. Salir\n');
    fprintf('%s\n', repmat('=', 1, 60));
end

function resultado = verificar_imagen_cargada()
    % Verifica si hay una imagen cargada
    global imagen_activa;
    global imagen_original;
    
    if isempty(imagen_activa) || isempty(imagen_original)
        fprintf('Error: No hay ninguna imagen cargada.\n');
        fprintf('Por favor, cargue una imagen primero (opción 1).\n');
        resultado = false;
    else
        resultado = true;
    end
end

function cargar_imagen()
    % Función para cargar una imagen
    global imagen_original;
    global imagen_activa;
    global nombre_imagen_activa;
    global historial_procesamientos;
    global usar_imagen_procesada;
    
    fprintf('\n--- CARGAR IMAGEN ---\n');
    [archivo, ruta] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tiff', 'Archivos de imagen'}, ...
                                'Seleccione una imagen');
    
    if archivo == 0
        fprintf('No se seleccionó ninguna imagen.\n');
        return;
    end
    
    try
        ruta_completa = fullfile(ruta, archivo);
        imagen_cargada = imread(ruta_completa);
        
        % Guardar como imagen original y activa
        imagen_original = imagen_cargada;
        imagen_activa = imagen_cargada;
        nombre_imagen_activa = archivo;
        
        % Reiniciar historial y estado
        historial_procesamientos = {};
        usar_imagen_procesada = false;
        
        fprintf('Imagen cargada exitosamente: %s\n', archivo);
        fprintf('Dimensiones: %dx%d\n', size(imagen_activa, 1), size(imagen_activa, 2));
        
        if size(imagen_activa, 3) == 3
            fprintf('Tipo: Imagen a color (RGB)\n');
        else
            fprintf('Tipo: Imagen en escala de grises\n');
        end
        
        % Mostrar la imagen cargada
        figure('Name', 'Imagen Original');
        imshow(imagen_activa);
        title('Imagen Original');
        
    catch ME
        fprintf('Error al cargar la imagen: %s\n', ME.message);
    end
end

function menu_filtros()
    % Menú para aplicar filtros a la imagen
    global imagen_activa;
    global imagen_procesada;
    
    while true
        fprintf('\n--- FILTROS ---\n');
        fprintf('1. Filtro de desenfoque (promedio)\n');
        fprintf('2. Filtro gaussiano\n');
        fprintf('3. Filtro de mediana\n');
        fprintf('4. Filtro de nitidez\n');
        fprintf('5. Detección de bordes (Sobel)\n');
        fprintf('6. Detección de bordes (Canny)\n');
        fprintf('7. Ecualización de histograma\n');
        fprintf('8. Filtro paso alto\n');
        fprintf('9. Filtro paso bajo\n');
        fprintf('0. Volver al menú principal\n');
        
        opcion = input('Seleccione una opción: ', 's');
        
        switch opcion
            case '1'
                tamano = input('Tamaño del kernel (default 5): ');
                if isempty(tamano)
                    tamano = 5;
                end
                kernel = ones(tamano) / (tamano^2);
                imagen_procesada = imfilter(imagen_activa, kernel);
                mostrar_comparacion('Filtro de Desenfoque');
                
            case '2'
                sigma = input('Valor de sigma (default 1): ');
                if isempty(sigma)
                    sigma = 1;
                end
                imagen_procesada = imgaussfilt(imagen_activa, sigma);
                mostrar_comparacion('Filtro Gaussiano');
                
            case '3'
                tamano = input('Tamaño del kernel (default [3 3]): ');
                if isempty(tamano)
                    tamano = [3 3];
                end
                if size(imagen_activa, 3) == 3
                    imagen_procesada = imagen_activa;
                    for i = 1:3
                        imagen_procesada(:,:,i) = medfilt2(imagen_activa(:,:,i), tamano);
                    end
                else
                    imagen_procesada = medfilt2(imagen_activa, tamano);
                end
                mostrar_comparacion('Filtro de Mediana');
                
            case '4'
                kernel = [-1 -1 -1; -1 9 -1; -1 -1 -1];
                imagen_procesada = imfilter(imagen_activa, kernel);
                mostrar_comparacion('Filtro de Nitidez');
                
            case '5'
                if size(imagen_activa, 3) == 3
                    img_gris = rgb2gray(imagen_activa);
                else
                    img_gris = imagen_activa;
                end
                imagen_procesada = edge(img_gris, 'sobel');
                mostrar_comparacion('Detección de Bordes - Sobel');
                
            case '6'
                if size(imagen_activa, 3) == 3
                    img_gris = rgb2gray(imagen_activa);
                else
                    img_gris = imagen_activa;
                end
                umbral_bajo = input('Umbral bajo (default 0.1): ');
                umbral_alto = input('Umbral alto (default 0.2): ');
                if isempty(umbral_bajo)
                    umbral_bajo = 0.1;
                end
                if isempty(umbral_alto)
                    umbral_alto = 0.2;
                end
                imagen_procesada = edge(img_gris, 'canny', [umbral_bajo umbral_alto]);
                mostrar_comparacion('Detección de Bordes - Canny');
                
            case '7'
                if size(imagen_activa, 3) == 3
                    img_gris = rgb2gray(imagen_activa);
                else
                    img_gris = imagen_activa;
                end
                imagen_procesada = histeq(img_gris);
                mostrar_comparacion('Ecualización de Histograma');
                
            case '8'
                kernel = [-1 -1 -1; -1 8 -1; -1 -1 -1];
                imagen_procesada = imfilter(imagen_activa, kernel);
                mostrar_comparacion('Filtro Paso Alto');
                
            case '9'
                kernel = ones(5) / 25;
                imagen_procesada = imfilter(imagen_activa, kernel);
                mostrar_comparacion('Filtro Paso Bajo');
                
            case '0'
                break;
            otherwise
                fprintf('Opción no válida.\n');
        end
    end
end

function menu_operaciones_aritmeticas()
    % Menú para operaciones aritméticas
    global imagen_activa;
    global imagen_procesada;
    
    while true
        fprintf('\n--- OPERACIONES ARITMÉTICAS ---\n');
        fprintf('1. Suma de imágenes\n');
        fprintf('2. Resta de imágenes\n');
        fprintf('3. Multiplicación de imágenes\n');
        fprintf('4. División de imágenes\n');
        fprintf('5. Ajustar brillo\n');
        fprintf('6. Ajustar contraste\n');
        fprintf('7. Suma con constante\n');
        fprintf('8. Multiplicación por constante\n');
        fprintf('0. Volver al menú principal\n');
        
        opcion = input('Seleccione una opción: ', 's');
        
        switch opcion
            case '1'
                imagen2 = cargar_segunda_imagen();
                if ~isempty(imagen2)
                    imagen_procesada = imadd(imagen_activa, imagen2);
                    mostrar_comparacion('Suma de Imágenes');
                end
                
            case '2'
                imagen2 = cargar_segunda_imagen();
                if ~isempty(imagen2)
                    imagen_procesada = imsubtract(imagen_activa, imagen2);
                    mostrar_comparacion('Resta de Imágenes');
                end
                
            case '3'
                imagen2 = cargar_segunda_imagen();
                if ~isempty(imagen2)
                    imagen_procesada = immultiply(imagen_activa, imagen2);
                    mostrar_comparacion('Multiplicación de Imágenes');
                end
                
            case '4'
                imagen2 = cargar_segunda_imagen();
                if ~isempty(imagen2)
                    imagen_procesada = imdivide(imagen_activa, imagen2);
                    mostrar_comparacion('División de Imágenes');
                end
                
            case '5'
                factor = input('Factor de brillo (1.0 = sin cambio): ');
                if isempty(factor)
                    factor = 1.2;
                end
                imagen_procesada = imadjust(imagen_activa, [], [], factor);
                mostrar_comparacion('Ajuste de Brillo');
                
            case '6'
                low_in = input('Valor mínimo de entrada (0-1, default 0): ');
                high_in = input('Valor máximo de entrada (0-1, default 1): ');
                if isempty(low_in)
                    low_in = 0;
                end
                if isempty(high_in)
                    high_in = 1;
                end
                imagen_procesada = imadjust(imagen_activa, [low_in high_in], []);
                mostrar_comparacion('Ajuste de Contraste');
                
            case '7'
                constante = input('Constante a sumar: ');
                if isempty(constante)
                    constante = 50;
                end
                imagen_procesada = imadd(imagen_activa, constante);
                mostrar_comparacion('Suma con Constante');
                
            case '8'
                constante = input('Constante multiplicadora: ');
                if isempty(constante)
                    constante = 1.5;
                end
                imagen_procesada = immultiply(imagen_activa, constante);
                mostrar_comparacion('Multiplicación por Constante');
                
            case '0'
                break;
            otherwise
                fprintf('Opción no válida.\n');
        end
    end
end

function menu_operaciones_geometricas()
    % Menú para operaciones geométricas
    global imagen_activa;
    global imagen_procesada;
    
    while true
        fprintf('\n--- OPERACIONES GEOMÉTRICAS ---\n');
        fprintf('1. Redimensionar imagen\n');
        fprintf('2. Rotar imagen\n');
        fprintf('3. Voltear horizontalmente\n');
        fprintf('4. Voltear verticalmente\n');
        fprintf('5. Trasladar imagen\n');
        fprintf('6. Recortar imagen\n');
        fprintf('7. Transformación afín\n');
        fprintf('0. Volver al menú principal\n');
        
        opcion = input('Seleccione una opción: ', 's');
        
        switch opcion
            case '1'
                factor = input('Factor de escala (default 0.5): ');
                if isempty(factor)
                    factor = 0.5;
                end
                imagen_procesada = imresize(imagen_activa, factor);
                mostrar_comparacion('Redimensionar');
                
            case '2'
                angulo = input('Ángulo de rotación en grados: ');
                if isempty(angulo)
                    angulo = 45;
                end
                imagen_procesada = imrotate(imagen_activa, angulo);
                mostrar_comparacion('Rotación');
                
            case '3'
                imagen_procesada = flip(imagen_activa, 2);
                mostrar_comparacion('Volteo Horizontal');
                
            case '4'
                imagen_procesada = flip(imagen_activa, 1);
                mostrar_comparacion('Volteo Vertical');
                
            case '5'
                dx = input('Desplazamiento en X: ');
                dy = input('Desplazamiento en Y: ');
                if isempty(dx)
                    dx = 50;
                end
                if isempty(dy)
                    dy = 30;
                end
                tform = affine2d([1 0 0; 0 1 0; dx dy 1]);
                imagen_procesada = imwarp(imagen_activa, tform);
                mostrar_comparacion('Traslación');
                
            case '6'
                fprintf('Seleccione la región a recortar en la imagen...\n');
                figure;
                imshow(imagen_activa);
                title('Seleccione la región a recortar');
                rect = getrect;
                close;
                if ~isempty(rect)
                    x = round(rect(1));
                    y = round(rect(2));
                    w = round(rect(3));
                    h = round(rect(4));
                    imagen_procesada = imagen_activa(y:y+h-1, x:x+w-1, :);
                    mostrar_comparacion('Recorte');
                end
                
            case '7'
                % Transformación afín simple (cizallamiento)
                shear_x = input('Factor de cizallamiento en X (default 0.2): ');
                if isempty(shear_x)
                    shear_x = 0.2;
                end
                tform = affine2d([1 shear_x 0; 0 1 0; 0 0 1]);
                imagen_procesada = imwarp(imagen_activa, tform);
                mostrar_comparacion('Transformación Afín');
                
            case '0'
                break;
            otherwise
                fprintf('Opción no válida.\n');
        end
    end
end

function menu_operaciones_logicas()
    % Menú para operaciones lógicas
    global imagen_activa;
    global imagen_procesada;
    
    while true
        fprintf('\n--- OPERACIONES LÓGICAS ---\n');
        fprintf('1. AND con otra imagen\n');
        fprintf('2. OR con otra imagen\n');
        fprintf('3. XOR con otra imagen\n');
        fprintf('4. NOT (inversión)\n');
        fprintf('5. Binarización simple\n');
        fprintf('6. Binarización de Otsu\n');
        fprintf('0. Volver al menú principal\n');
        
        opcion = input('Seleccione una opción: ', 's');
        
        switch opcion
            case '1'
                imagen2 = cargar_segunda_imagen();
                if ~isempty(imagen2)
                    % Convertir a binario si es necesario
                    img1_bin = convertir_a_binario(imagen_activa);
                    img2_bin = convertir_a_binario(imagen2);
                    imagen_procesada = img1_bin & img2_bin;
                    mostrar_comparacion('AND Lógico');
                end
                
            case '2'
                imagen2 = cargar_segunda_imagen();
                if ~isempty(imagen2)
                    img1_bin = convertir_a_binario(imagen_activa);
                    img2_bin = convertir_a_binario(imagen2);
                    imagen_procesada = img1_bin | img2_bin;
                    mostrar_comparacion('OR Lógico');
                end
                
            case '3'
                imagen2 = cargar_segunda_imagen();
                if ~isempty(imagen2)
                    img1_bin = convertir_a_binario(imagen_activa);
                    img2_bin = convertir_a_binario(imagen2);
                    imagen_procesada = xor(img1_bin, img2_bin);
                    mostrar_comparacion('XOR Lógico');
                end
                
            case '4'
                img_bin = convertir_a_binario(imagen_activa);
                imagen_procesada = ~img_bin;
                mostrar_comparacion('NOT (Inversión)');
                
            case '5'
                umbral = input('Umbral (0-255, default 128): ');
                if isempty(umbral)
                    umbral = 128;
                end
                if size(imagen_activa, 3) == 3
                    img_gris = rgb2gray(imagen_activa);
                else
                    img_gris = imagen_activa;
                end
                imagen_procesada = img_gris > umbral;
                mostrar_comparacion('Binarización Simple');
                
            case '6'
                if size(imagen_activa, 3) == 3
                    img_gris = rgb2gray(imagen_activa);
                else
                    img_gris = imagen_activa;
                end
                umbral = graythresh(img_gris);
                imagen_procesada = imbinarize(img_gris, umbral);
                fprintf('Umbral de Otsu calculado: %.3f\n', umbral);
                mostrar_comparacion('Binarización de Otsu');
                
            case '0'
                break;
            otherwise
                fprintf('Opción no válida.\n');
        end
    end
end

function menu_operaciones_morfologicas()
    % Menú para operaciones morfológicas
    global imagen_activa;
    global imagen_procesada;
    
    while true
        fprintf('\n--- OPERACIONES MORFOLÓGICAS ---\n');
        fprintf('1. Erosión\n');
        fprintf('2. Dilatación\n');
        fprintf('3. Apertura\n');
        fprintf('4. Cierre\n');
        fprintf('5. Gradiente morfológico\n');
        fprintf('6. Top Hat\n');
        fprintf('7. Bottom Hat\n');
        fprintf('8. Esqueletización\n');
        fprintf('0. Volver al menú principal\n');
        
        opcion = input('Seleccione una opción: ', 's');
        
        % Convertir imagen a binaria para operaciones morfológicas
        img_bin = convertir_a_binario(imagen_activa);
        
        % Crear elemento estructurante
        se = crear_elemento_estructurante();
        
        switch opcion
            case '1'
                imagen_procesada = imerode(img_bin, se);
                mostrar_comparacion('Erosión');
                
            case '2'
                imagen_procesada = imdilate(img_bin, se);
                mostrar_comparacion('Dilatación');
                
            case '3'
                imagen_procesada = imopen(img_bin, se);
                mostrar_comparacion('Apertura');
                
            case '4'
                imagen_procesada = imclose(img_bin, se);
                mostrar_comparacion('Cierre');
                
            case '5'
                dilatada = imdilate(img_bin, se);
                erosionada = imerode(img_bin, se);
                imagen_procesada = dilatada - erosionada;
                mostrar_comparacion('Gradiente Morfológico');
                
            case '6'
                apertura = imopen(img_bin, se);
                imagen_procesada = img_bin - apertura;
                mostrar_comparacion('Top Hat');
                
            case '7'
                cierre = imclose(img_bin, se);
                imagen_procesada = cierre - img_bin;
                mostrar_comparacion('Bottom Hat');
                
            case '8'
                imagen_procesada = bwmorph(img_bin, 'skel', Inf);
                mostrar_comparacion('Esqueletización');
                
            case '0'
                break;
            otherwise
                fprintf('Opción no válida.\n');
        end
    end
end

function se = crear_elemento_estructurante()
    % Crea un elemento estructurante para operaciones morfológicas
    fprintf('\nTipo de elemento estructurante:\n');
    fprintf('1. Disco\n');
    fprintf('2. Cuadrado\n');
    fprintf('3. Línea\n');
    
    tipo = input('Seleccione el tipo (default 1): ');
    if isempty(tipo)
        tipo = 1;
    end
    
    switch tipo
        case 1
            radio = input('Radio del disco (default 5): ');
            if isempty(radio)
                radio = 5;
            end
            se = strel('disk', radio);
            
        case 2
            tamano = input('Tamaño del cuadrado (default 5): ');
            if isempty(tamano)
                tamano = 5;
            end
            se = strel('square', tamano);
            
        case 3
            longitud = input('Longitud de la línea (default 5): ');
            angulo = input('Ángulo de la línea (default 0): ');
            if isempty(longitud)
                longitud = 5;
            end
            if isempty(angulo)
                angulo = 0;
            end
            se = strel('line', longitud, angulo);
            
        otherwise
            se = strel('disk', 5);
    end
end

function imagen2 = cargar_segunda_imagen()
    % Carga una segunda imagen para operaciones que requieren dos imágenes
    fprintf('Seleccione la segunda imagen...\n');
    [archivo, ruta] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tiff', 'Archivos de imagen'}, ...
                                'Seleccione la segunda imagen');
    
    if archivo == 0
        fprintf('No se seleccionó ninguna imagen.\n');
        imagen2 = [];
        return;
    end
    
    try
        ruta_completa = fullfile(ruta, archivo);
        imagen2 = imread(ruta_completa);
        
        % Redimensionar la segunda imagen para que coincida con la primera
        global imagen_activa;
        if ~isequal(size(imagen2), size(imagen_activa))
            imagen2 = imresize(imagen2, [size(imagen_activa, 1), size(imagen_activa, 2)]);
            fprintf('Segunda imagen redimensionada para coincidir con la primera.\n');
        end
        
    catch ME
        fprintf('Error al cargar la segunda imagen: %s\n', ME.message);
        imagen2 = [];
    end
end

function img_bin = convertir_a_binario(imagen)
    % Convierte una imagen a binaria
    if size(imagen, 3) == 3
        img_gris = rgb2gray(imagen);
    else
        img_gris = imagen;
    end
    
    if isa(img_gris, 'logical')
        img_bin = img_gris;
    else
        umbral = graythresh(img_gris);
        img_bin = imbinarize(img_gris, umbral);
    end
end

function mostrar_comparacion(titulo)
    % Muestra la imagen activa y procesada lado a lado
    global imagen_activa;
    global imagen_procesada;
    global historial_procesamientos;
    global usar_imagen_procesada;
    
    figure('Name', titulo, 'Position', [100, 100, 1200, 400]);
    
    subplot(1, 2, 1);
    imshow(imagen_activa);
    if length(historial_procesamientos) > 0
        title('Imagen Actual (con procesamientos aplicados)');
    else
        title('Imagen Original');
    end
    
    subplot(1, 2, 2);
    imshow(imagen_procesada);
    title(['Resultado: ' titulo]);
    
    fprintf('Procesamiento completado: %s\n', titulo);
    
    % Agregar al historial
    historial_procesamientos{end+1} = titulo;
    
    % Preguntar si aplicar el procesamiento a la imagen activa
    respuesta = input('\n¿Aplicar este procesamiento a la imagen activa? (s/n): ', 's');
    if strcmpi(respuesta, 's') || strcmpi(respuesta, 'si') || strcmpi(respuesta, 'sí')
        imagen_activa = imagen_procesada;
        usar_imagen_procesada = true;
        fprintf('Procesamiento aplicado a la imagen activa.\n');
        fprintf('Los siguientes procesamientos se aplicarán sobre este resultado.\n');
    else
        fprintf('Procesamiento no aplicado. La imagen activa permanece sin cambios.\n');
    end
end

function mostrar_imagen_actual()
    % Muestra la imagen actualmente cargada
    global imagen_activa;
    global nombre_imagen_activa;
    global historial_procesamientos;
    
    if isempty(imagen_activa)
        fprintf('No hay imagen cargada.\n');
        return;
    end
    
    figure('Name', 'Imagen Actual');
    imshow(imagen_activa);
    
    if length(historial_procesamientos) > 0
        titulo = sprintf('Imagen Actual: %s (con %d procesamientos)', nombre_imagen_activa, length(historial_procesamientos));
    else
        titulo = ['Imagen Actual: ' nombre_imagen_activa ' (original)'];
    end
    
    title(titulo);
    
    % Mostrar información adicional
    fprintf('\n--- INFORMACIÓN DE LA IMAGEN ACTUAL ---\n');
    fprintf('Nombre: %s\n', nombre_imagen_activa);
    fprintf('Dimensiones: %dx%d\n', size(imagen_activa, 1), size(imagen_activa, 2));
    if size(imagen_activa, 3) == 3
        fprintf('Tipo: Imagen a color (RGB)\n');
    else
        fprintf('Tipo: Imagen en escala de grises\n');
    end
    fprintf('Procesamientos aplicados: %d\n', length(historial_procesamientos));
end

function guardar_imagen_procesada()
    % Guarda la imagen actual (con todos los procesamientos aplicados)
    global imagen_activa;
    global nombre_imagen_activa;
    global ruta_guardado;
    global historial_procesamientos;
    
    if isempty(imagen_activa)
        fprintf('No hay imagen para guardar.\n');
        return;
    end
    
    % Generar nombre de archivo con timestamp y número de procesamientos
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    [~, nombre_base, ~] = fileparts(nombre_imagen_activa);
    num_procesamientos = length(historial_procesamientos);
    
    if num_procesamientos > 0
        nombre_salida = sprintf('%s_procesada_%dops_%s.png', nombre_base, num_procesamientos, timestamp);
    else
        nombre_salida = sprintf('%s_original_%s.png', nombre_base, timestamp);
    end
    
    ruta_completa = fullfile(ruta_guardado, nombre_salida);
    
    try
        imwrite(imagen_activa, ruta_completa);
        fprintf('Imagen guardada exitosamente en: %s\n', ruta_completa);
        
        % Guardar también el historial de procesamientos
        if num_procesamientos > 0
            nombre_historial = sprintf('%s_historial_%s.txt', nombre_base, timestamp);
            ruta_historial = fullfile(ruta_guardado, nombre_historial);
            guardar_historial_archivo(ruta_historial);
        end
        
    catch ME
        fprintf('Error al guardar la imagen: %s\n', ME.message);
    end
end

function menu_segmentacion()
    % Menú para técnicas de segmentación de imágenes
    global imagen_activa;
    global imagen_procesada;
    
    while true
        fprintf('\n--- SEGMENTACIÓN DE IMÁGENES ---\n');
        fprintf('1. Umbralización simple\n');
        fprintf('2. Umbralización adaptativa\n');
        fprintf('3. Método de Otsu\n');
        fprintf('4. Detección de bordes Canny\n');
        fprintf('5. Detección de contornos\n');
        fprintf('6. Segmentación por K-means\n');
        fprintf('7. Segmentación Watershed\n');
        fprintf('8. Crecimiento de regiones\n');
        fprintf('9. Segmentación por color HSV\n');
        fprintf('10. Detección de copas de árboles\n');
        fprintf('11. Extracción de copas de árboles\n');
        fprintf('0. Volver al menú principal\n');
        
        opcion = input('Seleccione una opción: ', 's');
        
        switch opcion
            case '1'
                segmentacion_umbral_simple();
                
            case '2'
                segmentacion_umbral_adaptativo();
                
            case '3'
                segmentacion_otsu();
                
            case '4'
                segmentacion_canny();
                
            case '5'
                segmentacion_contornos();
                
            case '6'
                segmentacion_kmeans();
                
            case '7'
                segmentacion_watershed();
                
            case '8'
                segmentacion_crecimiento_regiones();
                
            case '9'
                segmentacion_color_hsv();
                
            case '10'
                deteccion_copas_arboles();
                
            case '11'
                extraccion_copas_arboles();
                
            case '0'
                break;
            otherwise
                fprintf('Opción no válida.\n');
        end
    end
end

function segmentacion_umbral_simple()
    % Aplica umbralización simple
    global imagen_activa;
    global imagen_procesada;
    
    % Convertir a escala de grises si es necesario
    if size(imagen_activa, 3) == 3
        img_gris = rgb2gray(imagen_activa);
    else
        img_gris = imagen_activa;
    end
    
    umbral = input('Umbral (0-1, default 0.5): ');
    if isempty(umbral)
        umbral = 0.5;
    end
    
    imagen_procesada = imbinarize(img_gris, umbral);
    mostrar_comparacion('Umbralización Simple');
end

function segmentacion_umbral_adaptativo()
    % Aplica umbralización adaptativa
    global imagen_activa;
    global imagen_procesada;
    
    % Convertir a escala de grises si es necesario
    if size(imagen_activa, 3) == 3
        img_gris = rgb2gray(imagen_activa);
    else
        img_gris = imagen_activa;
    end
    
    fprintf('Método de umbralización adaptativa:\n');
    fprintf('1. Gaussiano\n');
    fprintf('2. Media\n');
    metodo = input('Seleccione método (default 1): ');
    if isempty(metodo)
        metodo = 1;
    end
    
    sensibilidad = input('Sensibilidad (0-1, default 0.5): ');
    if isempty(sensibilidad)
        sensibilidad = 0.5;
    end
    
    if metodo == 1
        imagen_procesada = imbinarize(img_gris, 'adaptive', 'Sensitivity', sensibilidad, 'ForegroundPolarity', 'bright');
    else
        imagen_procesada = imbinarize(img_gris, 'adaptive', 'Sensitivity', sensibilidad, 'ForegroundPolarity', 'bright');
    end
    
    mostrar_comparacion('Umbralización Adaptativa');
end

function segmentacion_otsu()
    % Aplica método de Otsu
    global imagen_activa;
    global imagen_procesada;
    
    % Convertir a escala de grises si es necesario
    if size(imagen_activa, 3) == 3
        img_gris = rgb2gray(imagen_activa);
    else
        img_gris = imagen_activa;
    end
    
    umbral = graythresh(img_gris);
    imagen_procesada = imbinarize(img_gris, umbral);
    
    fprintf('Umbral de Otsu calculado: %.3f\n', umbral);
    mostrar_comparacion('Método de Otsu');
end

function segmentacion_canny()
    % Aplica detector de bordes Canny
    global imagen_activa;
    global imagen_procesada;
    
    % Convertir a escala de grises si es necesario
    if size(imagen_activa, 3) == 3
        img_gris = rgb2gray(imagen_activa);
    else
        img_gris = imagen_activa;
    end
    
    umbral_bajo = input('Umbral bajo (0-1, default 0.1): ');
    umbral_alto = input('Umbral alto (0-1, default 0.2): ');
    
    if isempty(umbral_bajo)
        umbral_bajo = 0.1;
    end
    if isempty(umbral_alto)
        umbral_alto = 0.2;
    end
    
    imagen_procesada = edge(img_gris, 'canny', [umbral_bajo umbral_alto]);
    mostrar_comparacion('Detector Canny');
end

function segmentacion_contornos()
    % Detección y marcado de contornos
    global imagen_activa;
    global imagen_procesada;
    
    % Convertir a escala de grises si es necesario
    if size(imagen_activa, 3) == 3
        img_gris = rgb2gray(imagen_activa);
    else
        img_gris = imagen_activa;
    end
    
    % Binarizar imagen
    umbral = graythresh(img_gris);
    img_bin = imbinarize(img_gris, umbral);
    
    % Encontrar contornos usando bwboundaries
    [contornos, ~] = bwboundaries(img_bin);
    
    % Crear imagen de resultado
    if size(imagen_activa, 3) == 3
        imagen_procesada = imagen_activa;
    else
        imagen_procesada = cat(3, img_gris, img_gris, img_gris);
    end
    
    % Dibujar contornos en verde
    for k = 1:length(contornos)
        contorno = contornos{k};
        for i = 1:size(contorno, 1)
            y = contorno(i, 1);
            x = contorno(i, 2);
            if y > 0 && y <= size(imagen_procesada, 1) && x > 0 && x <= size(imagen_procesada, 2)
                imagen_procesada(y, x, :) = [0, 255, 0]; % Verde
            end
        end
    end
    
    fprintf('Contornos detectados: %d\n', length(contornos));
    mostrar_comparacion('Detección de Contornos');
end

function segmentacion_kmeans()
    % Segmentación por K-means
    global imagen_activa;
    global imagen_procesada;
    
    k = input('Número de clusters (default 3): ');
    if isempty(k)
        k = 3;
    end
    
    % Convertir imagen a vector de características
    if size(imagen_activa, 3) == 3
        img_double = im2double(imagen_activa);
        datos = reshape(img_double, [], 3);
    else
        img_double = im2double(imagen_activa);
        datos = reshape(img_double, [], 1);
    end
    
    % Aplicar K-means
    [idx, centros] = kmeans(datos, k);
    
    % Reconstruir imagen segmentada
    imagen_segmentada = centros(idx, :);
    
    if size(imagen_activa, 3) == 3
        imagen_procesada = reshape(imagen_segmentada, size(imagen_activa));
    else
        imagen_procesada = reshape(imagen_segmentada, size(imagen_activa));
    end
    
    mostrar_comparacion('Segmentación K-means');
end

function segmentacion_watershed()
    % Segmentación Watershed
    global imagen_activa;
    global imagen_procesada;
    
    % Convertir a escala de grises
    if size(imagen_activa, 3) == 3
        img_gris = rgb2gray(imagen_activa);
    else
        img_gris = imagen_activa;
    end
    
    % Calcular gradiente
    hy = fspecial('sobel');
    hx = hy';
    Iy = imfilter(double(img_gris), hy, 'replicate');
    Ix = imfilter(double(img_gris), hx, 'replicate');
    gradmag = sqrt(Ix.^2 + Iy.^2);
    
    % Aplicar watershed
    L = watershed(gradmag);
    
    % Crear imagen de resultado
    imagen_procesada = imagen_activa;
    if size(imagen_procesada, 3) == 1
        imagen_procesada = cat(3, imagen_procesada, imagen_procesada, imagen_procesada);
    end
    
    % Marcar bordes de watershed en rojo
    imagen_procesada(L == 0) = 255; % Bordes en blanco
    
    mostrar_comparacion('Segmentación Watershed');
end

function segmentacion_crecimiento_regiones()
    % Segmentación por crecimiento de regiones
    global imagen_activa;
    global imagen_procesada;
    
    % Convertir a escala de grises
    if size(imagen_activa, 3) == 3
        img_gris = rgb2gray(imagen_activa);
    else
        img_gris = imagen_activa;
    end
    
    fprintf('Haga clic en la imagen para seleccionar puntos semilla...\n');
    figure;
    imshow(img_gris);
    title('Seleccione puntos semilla (doble clic para terminar)');
    
    % Obtener puntos semilla del usuario
    [x, y] = getpts;
    close;
    
    if isempty(x)
        fprintf('No se seleccionaron puntos semilla.\n');
        return;
    end
    
    umbral = input('Umbral de similitud (0-255, default 20): ');
    if isempty(umbral)
        umbral = 20;
    end
    
    % Crear máscara de resultado
    mascara = false(size(img_gris));
    
    % Para cada punto semilla
    for i = 1:length(x)
        seed_x = round(x(i));
        seed_y = round(y(i));
        
        if seed_x > 0 && seed_x <= size(img_gris, 2) && seed_y > 0 && seed_y <= size(img_gris, 1)
            % Valor del pixel semilla
            valor_semilla = double(img_gris(seed_y, seed_x));
            
            % Crecimiento simple en una ventana alrededor de la semilla
            radio = 30;
            for dy = -radio:radio
                for dx = -radio:radio
                    ny = seed_y + dy;
                    nx = seed_x + dx;
                    
                    if ny > 0 && ny <= size(img_gris, 1) && nx > 0 && nx <= size(img_gris, 2)
                        if abs(double(img_gris(ny, nx)) - valor_semilla) < umbral
                            mascara(ny, nx) = true;
                        end
                    end
                end
            end
        end
    end
    
    % Aplicar máscara a imagen original
    if size(imagen_activa, 3) == 3
        imagen_procesada = imagen_activa;
        for c = 1:3
            canal = imagen_procesada(:,:,c);
            canal(~mascara) = 0;
            imagen_procesada(:,:,c) = canal;
        end
    else
        imagen_procesada = img_gris;
        imagen_procesada(~mascara) = 0;
    end
    
    mostrar_comparacion('Crecimiento de Regiones');
end

function segmentacion_color_hsv()
    % Segmentación por color en espacio HSV
    global imagen_activa;
    global imagen_procesada;
    
    if size(imagen_activa, 3) ~= 3
        fprintf('Esta función requiere una imagen en color.\n');
        return;
    end
    
    % Convertir a HSV
    img_hsv = rgb2hsv(imagen_activa);
    
    fprintf('Rangos de color para segmentación:\n');
    hue_min = input('Tono mínimo (0-1, default 0.2): ');
    hue_max = input('Tono máximo (0-1, default 0.4): ');
    sat_min = input('Saturación mínima (0-1, default 0.2): ');
    sat_max = input('Saturación máxima (0-1, default 1.0): ');
    val_min = input('Valor mínimo (0-1, default 0.2): ');
    val_max = input('Valor máximo (0-1, default 1.0): ');
    
    if isempty(hue_min), hue_min = 0.2; end
    if isempty(hue_max), hue_max = 0.4; end
    if isempty(sat_min), sat_min = 0.2; end
    if isempty(sat_max), sat_max = 1.0; end
    if isempty(val_min), val_min = 0.2; end
    if isempty(val_max), val_max = 1.0; end
    
    % Crear máscara
    mascara = (img_hsv(:,:,1) >= hue_min & img_hsv(:,:,1) <= hue_max) & ...
              (img_hsv(:,:,2) >= sat_min & img_hsv(:,:,2) <= sat_max) & ...
              (img_hsv(:,:,3) >= val_min & img_hsv(:,:,3) <= val_max);
    
    % Aplicar máscara
    imagen_procesada = imagen_activa;
    for c = 1:3
        canal = imagen_procesada(:,:,c);
        canal(~mascara) = 0;
        imagen_procesada(:,:,c) = canal;
    end
    
    mostrar_comparacion('Segmentación por Color HSV');
end

function deteccion_copas_arboles()
    % Detecta copas de árboles usando segmentación por color verde
    global imagen_activa;
    global imagen_procesada;
    
    if size(imagen_activa, 3) ~= 3
        fprintf('Esta función requiere una imagen en color.\n');
        return;
    end
    
    fprintf('Método de detección:\n');
    fprintf('1. Segmentación HSV (verde)\n');
    fprintf('2. K-means\n');
    metodo = input('Seleccione método (default 1): ');
    if isempty(metodo)
        metodo = 1;
    end
    
    if metodo == 1
        % Método HSV para detectar verde
        img_hsv = rgb2hsv(imagen_activa);
        
        % Rangos típicos para verde (copas de árboles)
        hue_min = 0.2;    % Verde típicamente entre 0.2-0.4
        hue_max = 0.4;
        sat_min = 0.1;    % Baja saturación para incluir verdes claros
        sat_max = 1.0;
        val_min = 0.1;    % Valor mínimo para excluir sombras
        val_max = 1.0;
        
        % Crear máscara para verde
        mascara = (img_hsv(:,:,1) >= hue_min & img_hsv(:,:,1) <= hue_max) & ...
                  (img_hsv(:,:,2) >= sat_min & img_hsv(:,:,2) <= sat_max) & ...
                  (img_hsv(:,:,3) >= val_min & img_hsv(:,:,3) <= val_max);
        
        % Operaciones morfológicas para limpiar la máscara
        se = strel('disk', 3);
        mascara = imopen(mascara, se);  % Eliminar ruido
        mascara = imclose(mascara, se); % Rellenar huecos
        
    else
        % Método K-means
        img_double = im2double(imagen_activa);
        datos = reshape(img_double, [], 3);
        
        [idx, centros] = kmeans(datos, 3);
        
        % Identificar cluster más verde
        verdosidad = centros(:,2) ./ (centros(:,1) + centros(:,3) + 0.01);
        [~, cluster_verde] = max(verdosidad);
        
        % Crear máscara
        mascara = reshape(idx == cluster_verde, size(imagen_activa, 1), size(imagen_activa, 2));
    end
    
    % Encontrar contornos de las copas
    [contornos, ~] = bwboundaries(mascara);
    
    % Crear imagen de resultado con contornos marcados
    imagen_procesada = imagen_activa;
    
    % Dibujar contornos en verde
    for k = 1:length(contornos)
        contorno = contornos{k};
        for i = 1:size(contorno, 1)
            y = contorno(i, 1);
            x = contorno(i, 2);
            if y > 1 && y < size(imagen_procesada, 1) && x > 1 && x < size(imagen_procesada, 2)
                % Dibujar contorno más grueso
                for dy = -1:1
                    for dx = -1:1
                        ny = y + dy;
                        nx = x + dx;
                        if ny > 0 && ny <= size(imagen_procesada, 1) && nx > 0 && nx <= size(imagen_procesada, 2)
                            imagen_procesada(ny, nx, :) = [0, 255, 0]; % Verde
                        end
                    end
                end
            end
        end
    end
    
    fprintf('Copas de árboles detectadas: %d\n', length(contornos));
    mostrar_comparacion('Detección de Copas de Árboles');
end

function extraccion_copas_arboles()
    % Extrae solo las copas de árboles, eliminando el resto
    global imagen_activa;
    global imagen_procesada;
    
    if size(imagen_activa, 3) ~= 3
        fprintf('Esta función requiere una imagen en color.\n');
        return;
    end
    
    % Segmentación HSV para detectar verde
    img_hsv = rgb2hsv(imagen_activa);
    
    % Rangos para verde (copas de árboles)
    hue_min = 0.2;
    hue_max = 0.4;
    sat_min = 0.1;
    sat_max = 1.0;
    val_min = 0.1;
    val_max = 1.0;
    
    % Crear máscara para verde
    mascara = (img_hsv(:,:,1) >= hue_min & img_hsv(:,:,1) <= hue_max) & ...
              (img_hsv(:,:,2) >= sat_min & img_hsv(:,:,2) <= sat_max) & ...
              (img_hsv(:,:,3) >= val_min & img_hsv(:,:,3) <= val_max);
    
    % Post-procesamiento morfológico
    se = strel('disk', 5);
    mascara = imclose(mascara, se);  % Rellenar huecos
    mascara = imopen(mascara, se);   % Eliminar ruido pequeño
    
    % Preguntar tipo de fondo
    fprintf('Tipo de fondo:\n');
    fprintf('1. Fondo negro\n');
    fprintf('2. Fondo blanco\n');
    tipo_fondo = input('Seleccione tipo de fondo (default 1): ');
    if isempty(tipo_fondo)
        tipo_fondo = 1;
    end
    
    % Extraer solo las copas
    imagen_procesada = imagen_activa;
    
    if tipo_fondo == 1
        % Fondo negro
        for c = 1:3
            canal = imagen_procesada(:,:,c);
            canal(~mascara) = 0;
            imagen_procesada(:,:,c) = canal;
        end
    else
        % Fondo blanco
        for c = 1:3
            canal = imagen_procesada(:,:,c);
            canal(~mascara) = 255;
            imagen_procesada(:,:,c) = canal;
        end
    end
    
    mostrar_comparacion('Extracción de Copas de Árboles');
end

function aplicar_procesamiento_a_imagen_activa()
    % Aplica el último procesamiento realizado a la imagen activa
    global imagen_activa;
    global imagen_procesada;
    global historial_procesamientos;
    
    if isempty(imagen_procesada)
        fprintf('No hay procesamiento para aplicar.\n');
        fprintf('Primero realice algún procesamiento.\n');
        return;
    end
    
    respuesta = input('¿Confirma aplicar el último procesamiento a la imagen activa? (s/n): ', 's');
    if strcmpi(respuesta, 's') || strcmpi(respuesta, 'si') || strcmpi(respuesta, 'sí')
        imagen_activa = imagen_procesada;
        fprintf('Procesamiento aplicado a la imagen activa.\n');
        fprintf('Total de procesamientos aplicados: %d\n', length(historial_procesamientos));
    else
        fprintf('Operación cancelada.\n');
    end
end

function mostrar_historial_procesamientos()
    % Muestra el historial de procesamientos aplicados
    global historial_procesamientos;
    global nombre_imagen_activa;
    
    fprintf('\n--- HISTORIAL DE PROCESAMIENTOS ---\n');
    fprintf('Imagen: %s\n', nombre_imagen_activa);
    fprintf('Total de procesamientos: %d\n', length(historial_procesamientos));
    fprintf('%s\n', repmat('-', 1, 40));
    
    if isempty(historial_procesamientos)
        fprintf('No se han aplicado procesamientos.\n');
        return;
    end
    
    for i = 1:length(historial_procesamientos)
        fprintf('%2d. %s\n', i, historial_procesamientos{i});
    end
    
    fprintf('%s\n', repmat('-', 1, 40));
end

function resetear_a_imagen_original()
    % Resetea la imagen activa a la imagen original
    global imagen_original;
    global imagen_activa;
    global historial_procesamientos;
    global usar_imagen_procesada;
    
    if isempty(imagen_original)
        fprintf('No hay imagen original cargada.\n');
        return;
    end
    
    respuesta = input('¿Confirma resetear todos los procesamientos? (s/n): ', 's');
    if strcmpi(respuesta, 's') || strcmpi(respuesta, 'si') || strcmpi(respuesta, 'sí')
        imagen_activa = imagen_original;
        historial_procesamientos = {};
        usar_imagen_procesada = false;
        
        fprintf('Imagen reseteada a la versión original.\n');
        fprintf('Historial de procesamientos eliminado.\n');
        
        % Mostrar imagen original
        figure('Name', 'Imagen Reseteada');
        imshow(imagen_activa);
        title('Imagen Original (Reseteada)');
    else
        fprintf('Operación cancelada.\n');
    end
end

function guardar_historial_archivo(ruta_archivo)
    % Guarda el historial de procesamientos en un archivo de texto
    global historial_procesamientos;
    global nombre_imagen_activa;
    
    try
        fid = fopen(ruta_archivo, 'w');
        if fid == -1
            fprintf('Error al crear archivo de historial.\n');
            return;
        end
        
        fprintf(fid, 'HISTORIAL DE PROCESAMIENTOS\n');
        fprintf(fid, '============================\n\n');
        fprintf(fid, 'Imagen original: %s\n', nombre_imagen_activa);
        fprintf(fid, 'Fecha: %s\n', datestr(now));
        fprintf(fid, 'Total de procesamientos: %d\n\n', length(historial_procesamientos));
        
        fprintf(fid, 'PROCESAMIENTOS APLICADOS:\n');
        fprintf(fid, '-------------------------\n');
        
        for i = 1:length(historial_procesamientos)
            fprintf(fid, '%2d. %s\n', i, historial_procesamientos{i});
        end
        
        fclose(fid);
        fprintf('Historial guardado en: %s\n', ruta_archivo);
        
    catch ME
        fprintf('Error al guardar historial: %s\n', ME.message);
        if fid ~= -1
            fclose(fid);
        end
    end
end

% Función principal para iniciar el programa
% Para ejecutar, simplemente escriba: menu_preprocesamiento_matlab()