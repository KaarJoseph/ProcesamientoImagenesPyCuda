import pycuda.autoinit
import pycuda.driver as cuda
from flask import Flask, request, jsonify, render_template
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import io
import base64
import math
import time

app = Flask(__name__)

# Inicialización del dispositivo CUDA
cuda.init()
device = cuda.Device(0)  # Asegúrate de que esté disponible

# Definición de los filtros CUDA
kernels = {
    "sobel": """
        __global__ void sobel_filter(float *input, float *output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                int idx = y * width + x;
                float Gx = input[idx - 1 - width] - input[idx + 1 - width]
                          + 2 * input[idx - 1] - 2 * input[idx + 1]
                          + input[idx - 1 + width] - input[idx + 1 + width];
                float Gy = input[idx - 1 - width] + 2 * input[idx - width] + input[idx + 1 - width]
                          - input[idx - 1 + width] - 2 * input[idx + width] - input[idx + 1 + width];
                output[idx] = sqrtf(Gx * Gx + Gy * Gy);
            } else if (x < width && y < height) {
                output[y * width + x] = input[y * width + x];
            }
        }
    """,
    "highpass": """
        __global__ void highpass_filter(float *input, float *output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                int idx = y * width + x;
                float result = 4 * input[idx] - input[idx - 1] - input[idx + 1] - input[idx - width] - input[idx + width];
                output[idx] = result;
            } else if (x < width && y < height) {
                output[y * width + x] = input[y * width + x];
            }
        }
    """,
    "erosion": """
        __global__ void erosion_filter(float *input, float *output, int width, int height, int maskSize) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int halfMask = maskSize / 2;
            
            if (x >= halfMask && x < width - halfMask && y >= halfMask && y < height - halfMask) {
                float min_val = input[y * width + x];
                
                for (int ky = -halfMask; ky <= halfMask; ky++) {
                    for (int kx = -halfMask; kx <= halfMask; kx++) {
                        int idx = (y + ky) * width + (x + kx);
                        min_val = fminf(min_val, input[idx]);
                    }
                }
                
                output[y * width + x] = min_val;
            } else {
                output[y * width + x] = input[y * width + x];
            }
        }
    """
}

@app.route('/api/procesar_imagen', methods=['POST'])
def procesar_imagen():
    try:
        start_time = time.time()  # Tiempo de inicio del procesamiento

        # Crear contexto de CUDA para asegurar procesamiento único
        context = device.make_context()

        data = request.json
        image_data = data['image']
        filtro = data['filtro']
        
        # Obtener el tamaño de la máscara (si aplica)
        mask_size_str = data.get('maskSize', '3x3')
        mask_size = int(mask_size_str.split('x')[0])
        
        # Obtener threads per block en ambas dimensiones
        threads_per_block_x = int(data['threadsPerBlockX'])
        threads_per_block_y = int(data['threadsPerBlockY'])

        # Validación para no exceder el límite de 1024 hilos por bloque en total
        MAX_HILOS_POR_BLOQUE = 1024
        if threads_per_block_x * threads_per_block_y > MAX_HILOS_POR_BLOQUE:
            return jsonify({"error": f"La cantidad total de hilos por bloque no puede exceder {MAX_HILOS_POR_BLOQUE}."}), 400

        # Procesar imagen y asegurar su forma adecuada
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        imagen_np = np.array(image.convert('L')).astype(np.float32)
        height, width = imagen_np.shape

        # Crear el módulo y el kernel
        mod = SourceModule(kernels[filtro])
        filtro_func = mod.get_function(f"{filtro}_filter")

        # Configuración de memoria en GPU
        input_gpu = cuda.mem_alloc(imagen_np.nbytes)
        output_gpu = cuda.mem_alloc(imagen_np.nbytes)
        cuda.memcpy_htod(input_gpu, imagen_np)

        # Configuración de bloques y grid para asegurar procesamiento en cada píxel
        block = (threads_per_block_x, threads_per_block_y, 1)
        grid = (math.ceil(width / block[0]), math.ceil(height / block[1]))

        # Ejecutar el kernel
        if filtro == "erosion":
            filtro_func(input_gpu, output_gpu, np.int32(width), np.int32(height), np.int32(mask_size), block=block, grid=grid)
        else:
            filtro_func(input_gpu, output_gpu, np.int32(width), np.int32(height), block=block, grid=grid)

        # Obtener resultados y asegurar que estén en escala de grises
        resultado = np.empty_like(imagen_np)
        cuda.memcpy_dtoh(resultado, output_gpu)
        resultado = np.clip(resultado, 0, 255)  # Limitar valores a escala de grises
        resultado_image = Image.fromarray(resultado.astype(np.uint8))

        # Convertir la imagen a PNG y enviarla
        buffered = io.BytesIO()
        resultado_image.save(buffered, format="PNG")
        processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Liberar el contexto CUDA
        context.pop()

        # Tiempo de procesamiento
        processing_time_seconds = time.time() - start_time
        processing_time_milliseconds = processing_time_seconds * 1000

        # Enviar la información de procesamiento junto con la imagen
        return jsonify({
            "processedImageUrl": "data:image/png;base64," + processed_image_str,
            "processingTimeSeconds": round(processing_time_seconds, 4),  # Tiempo en segundos con 4 decimales
            "processingTimeMilliseconds": round(processing_time_milliseconds, 4),  # Tiempo en milisegundos con 4 decimales
            "maskUsed": mask_size_str,
            "threadCount": threads_per_block_x * threads_per_block_y,
            "blockCount": grid[0] * grid[1]
        })
    
    except Exception as e:
        print("Error procesando imagen:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('Pagina.html')

if __name__ == '__main__':
    app.run(debug=True)
