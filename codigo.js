// Mostrar la imagen en el canvas de vista previa
function previewImage(event) {
    const canvas = document.getElementById("previewCanvas");
    const ctx = canvas.getContext("2d");
    const reader = new FileReader();
    
    reader.onload = function() {
        const img = new Image();
        img.onload = function() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = reader.result;
    };
    
    reader.readAsDataURL(event.target.files[0]);
}

// Enviar imagen y opciones de filtro al backend para procesamiento
function procesarImagen() {
    const filtroSeleccionado = document.getElementById("imageSelect").value;
    const maskSize = document.querySelector('input[name="mask"]:checked').value;
    const threadsPerBlock = document.getElementById("threadsPerBlock").value;

    const canvas = document.getElementById("previewCanvas");
    const imageData = canvas.toDataURL("image/png").replace(/^data:image\/(png|jpg);base64,/, "");

    fetch('/api/procesar_imagen', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: imageData,
            filtro: filtroSeleccionado,
            maskSize: maskSize,
            threadsPerBlock: threadsPerBlock
        })
    })
    .then(response => response.json())
    .then(data => {
        const processedCanvas = document.getElementById("processedCanvas");
        const ctx = processedCanvas.getContext("2d");
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, processedCanvas.width, processedCanvas.height);
        };
        img.src = data.processedImageUrl;
    })
    .catch(error => console.error('Error:', error));
}
