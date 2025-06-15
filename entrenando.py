import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os
import cv2
from sklearn.utils import class_weight

# Configuración
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 30  # Aumentado para mejor entrenamiento
DATA_DIR = 'dataset'
CLASS_NAMES = ['cocodrilos', 'no_cocodrilos']  # Nombres de las carpetas
MIN_IMAGES_PER_CLASS = 100
USE_TRANSFER_LEARNING = True

# Verificar si el directorio del dataset existe
if not os.path.exists(DATA_DIR):
    print(f"Error: La carpeta '{DATA_DIR}' no existe. Asegúrate de que el dataset esté en la ubicación correcta.")
    exit()

# Verificar si las carpetas de clases existen
for class_name in CLASS_NAMES:
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        print(f"Error: La carpeta '{class_path}' no existe. Verifica los nombres de las carpetas en el dataset.")
        exit()

# Generador de datos con aumento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Cargar datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASS_NAMES,  # cocodrilos=0, no_cocodrilos=1
    subset='training',
    shuffle=True
)

# Cargar datos de validación
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASS_NAMES,
    subset='validation',
    shuffle=True
)

# Verificar número de imágenes
print(f"Imágenes de entrenamiento: {train_generator.samples}")
print(f"Imágenes de validación: {validation_generator.samples}")

# Validar que haya suficientes imágenes
if train_generator.samples < MIN_IMAGES_PER_CLASS or validation_generator.samples < 1:
    print(f"Error: Dataset demasiado pequeño. Se encontraron {train_generator.samples} imágenes de entrenamiento y "
          f"{validation_generator.samples} de validación. Se recomiendan al menos {MIN_IMAGES_PER_CLASS} imágenes por clase.")
    exit()

# Calcular pesos de clases para manejar desbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Pesos de clases: {class_weights_dict}")

# Crear modelo
if USE_TRANSFER_LEARNING:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    # Descongelar las últimas 10 capas para ajuste fino
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
else:
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

# Compilar modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Tasa de aprendizaje baja para ajuste fino
              loss='binary_crossentropy', metrics=['accuracy'])

# Mostrar resumen del modelo
model.summary()

# Entrenar modelo
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        verbose=1
    )
except Exception as e:
    print(f"Error durante el entrenamiento: {str(e)}")
    exit()

# Guardar modelo
model.save('crocodile_model.h5')
print("Modelo entrenado y guardado como 'crocodile_model.h5'")

# Mostrar métricas finales
print(f"Precisión final en entrenamiento: {history.history['accuracy'][-1]:.4f}")
if validation_generator.samples > 0:
    print(f"Precisión final en validación: {history.history['val_accuracy'][-1]:.4f}")

# Evaluar modelo en imágenes de prueba
def evaluate_image(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)[0][0]
    label = "Cocodrilo" if prediction < 0.5 else "No cocodrilo"  # Invertido para coincidir con clases
    return label, prediction

# Probar con una imagen de cada clase
try:
    coco_path = os.path.join(DATA_DIR, 'cocodrilos', os.listdir(os.path.join(DATA_DIR, 'cocodrilos'))[0])
    no_coco_path = os.path.join(DATA_DIR, 'no_cocodrilos', os.listdir(os.path.join(DATA_DIR, 'no_cocodrilos'))[0])
    print(f"Prueba con imagen de cocodrilo ({coco_path}): {evaluate_image(coco_path, model)}")
    print(f"Prueba con imagen de no cocodrilo ({no_coco_path}): {evaluate_image(no_coco_path, model)}")
except Exception as e:
    print(f"Error al evaluar imágenes de prueba: {str(e)}")