import cv2
import numpy as np
from flask import Flask, Response, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Cargar modelo entrenado
model = load_model('crocodile_model.h5')
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Inicializar cámara
camera = cv2.VideoCapture(0)

def procesar_frame(elemento):
    # Redimensionar y normalizar frame (RGB)
    elemento = cv2.resize(elemento, (IMG_HEIGHT, IMG_WIDTH))
    elemento = elemento / 255.0
    elemento = np.expand_dims(elemento, axis=0)
    return elemento

def generar_video():
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: No se pudo leer el frame de la cámara.")
            break
        else:
            # Preprocesar frame para el modelo
            processed_frame = procesar_frame(frame)
            prediction = model.predict(processed_frame, verbose=0)[0][0]
            label = "Cocodrilo" if prediction < 0.3 else "No cocodrilo"  # Invertido para coincidir con clases
            
            # Imprimir predicción para depuración
            print(f"Predicción: {label} ({prediction:.2f})")
            
            # Añadir texto con la predicción
            cv2.putText(frame, f"{label} ({prediction:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convertir frame a formato JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        test_image_coco = cv2.imread('dataset/cocodrilos/cocodrilo1.jpg')
        test_image_no_coco = cv2.imread('dataset/no_cocodrilos/perro1.jpg')
        for img, name in [(test_image_coco, 'cocodrilo'), (test_image_no_coco, 'no_cocodrilo')]:
            if img is not None:
                processed = procesar_frame(img)
                pred = model.predict(processed, verbose=0)[0][0]
                label = "Cocodrilo" if pred < 0.5 else "No cocodrilo"
                print(f"Predicción para {name}: {label} ({pred:.2f})")
    except Exception as e:
        print(f"Error al probar imágenes estáticas: {str(e)}")
    app.run(debug=True)