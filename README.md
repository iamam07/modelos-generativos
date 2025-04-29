# Conditional VAE Convolucional sobre MNIST

Este proyecto implementa un **Variational Autoencoder Condicionado (CVAE)** con arquitecturas convolucionales para el dataset MNIST, utilizando **PyTorch Lightning** para facilitar el entrenamiento y la gestión del ciclo de vida del modelo.

---

## 📋 Requisitos

- Python 3.8 o superior
- GPU con soporte CUDA (opcional, pero recomendado)
- pip

Contenido de `requeriments.txt`:
```
torch
pytorch-lightning
torchvision
numpy
matplotlib
```

---

## ⚙️ Instalación

1. Clona el repositorio:
   ```bash
git clone https://github.com/iamam07/modelos-generativos.git
cd tu-repo
```

2. Instala dependencias:
   ```bash
pip install -r requeriments.txt
```

---

## 🚀 Uso

Abre la notebook situada en el repositorio (por ejemplo `CVAE_MNIST.ipynb`) con tu entorno de Jupyter (Jupyter Lab, Jupyter Notebook) o desde VS Code si prefieres trabajar con notebooks ahí.

1. Navega a la carpeta del proyecto:
   ```bash
   cd tu-repo
   ```
2. Abre Jupyter Lab:  
   ```bash
   jupyter lab
   ```
3. En el explorador de Jupyter, abre `CVAE_MNIST.ipynb`.
4. Ejecuta las celdas secuencialmente (Cell → Run All) para instalar dependencias, preparar datos, entrenar el modelo y visualizar resultados.

---

## 📐 Estructura del Código

1. **Importación y configuración**
   - Instalación de requerimientos y ajustes de `multiprocessing` y precisión de botones de matriz.

2. **`MNISTDataModule`**
   - Hereda de `LightningDataModule` para descargar y preparar datos de MNIST.
   - Genera `train_dataloader` y `val_dataloader` con `DataLoader` de PyTorch.

3. **Encoder Convolucional (`CVAEEncoderConv`)**
   - Tres capas `Conv2d` con normales por batch y activación ReLU.
   - Cálculo dinámico de la dimensión aplanada usando `get_flattened_size_and_shape`.
   - Producción de `mu` y `logvar` para el espacio latente.

4. **Decoder Convolucional (`CVAEDecoderConv`)**
   - Tres capas `ConvTranspose2d` inversas con BatchNorm y ReLU.
   - Reconstrucción de la imagen final con `sigmoid`.

5. **LightningModule (`ConvCVAE`)**
   - Combina encoder y decoder.
   - Implementa el método de reparametrización, la función de pérdida (BCE + KL), y pasos de entrenamiento/validación.
   - Configura el optimizador Adam.

6. **Entrenamiento y visualización**
   - Función `train_model` que entrena para cada learning rate y muestra grillas de imágenes generadas.

7. **Generación condicionada**
   - Funciones `generate_images` y `generate_for_digit` para producir muestras latentes condicionadas a cada dígito.

---

## 📊 Resultados esperados

- Durante el entrenamiento, el script mostrará en cada época la pérdida de entrenamiento y validación.
- Tras el entrenamiento, generará y visualizará grillas de 10×10 imágenes para cada dígito 0–9.
- Permite generar muestras adicionales para un dígito específico mediante `generate_for_digit`.

---

## 🤝 Contribuciones

1. Haz un fork del proyecto
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Realiza tus cambios y haz commit (`git commit -m "Añade..."`)
4. Haz push a tu rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request para revisión

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.

