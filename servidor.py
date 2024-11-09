from flask import Flask, request, jsonify, render_template
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import io
import base64
import math

app = Flask(__name__)

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
                        float val = input[(y + ky) * width + (x + kx)];
                        if (val < min_val) min_val = val;
                    }
                }
                output[y * width + x] = min_val;
            }
        }
    """
}

@app.route('/')
def index():
    return render_template("Pagina.html")

@app.route('/api/procesar_imagen', methods=['POST'])
def procesar_imagen():
    # Recibir datos y filtro seleccionado
    data = request.json
    image_data = data['image']
    filtro = data['filtro']
    mask_size = data.get('maskSize', 3)  # Solo para el filtro de erosión

    # Decodificar imagen
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    imagen_np = np.array(image.convert('L')).astype(np.float32)

    # Cargar el kernel CUDA seleccionado
    mod = SourceModule(kernels[filtro])
    filtro_func = mod.get_function(f"{filtro}_filter")

    width, height = imagen_np.shape
    input_gpu = cuda.mem_alloc(imagen_np.nbytes)
    output_gpu = cuda.mem_alloc(imagen_np.nbytes)
    cuda.memcpy_htod(input_gpu, imagen_np)

    # Definir bloque y cuadrícula
    block = (16, 16, 1)
    grid = (math.ceil(width / block[0]), math.ceil(height / block[1]))

    # Ejecutar el filtro CUDA seleccionado
    if filtro == "erosion":
        filtro_func(input_gpu, output_gpu, np.int32(width), np.int32(height), np.int32(mask_size), block=block, grid=grid)
    else:
        filtro_func(input_gpu, output_gpu, np.int32(width), np.int32(height), block=block, grid=grid)

    # Recuperar y procesar imagen de salida
    resultado = np.empty_like(imagen_np)
    cuda.memcpy_dtoh(resultado, output_gpu)
    resultado_image = Image.fromarray(np.uint8(resultado))
    buffered = io.BytesIO()
    resultado_image.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return jsonify({"processedImageUrl": "data:image/png;base64," + processed_image_str})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
