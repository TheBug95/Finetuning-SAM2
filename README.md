# Finetuning de SAM2 para Segmentación Médica ✅ ACTUALIZADO

## 🔧 CORRECCIONES APLICADAS

### ✅ Problemas Solucionados:
1. **TypeError ReduceLROnPlateau**: Removido argumento `verbose` no soportado
2. **Modelo actualizado**: Migrado de SAM a SAM2 oficial
3. **Imports corregidos**: Usando `Sam2Model` y `Sam2Processor` de transformers
4. **Modelo por defecto**: Cambiado a `facebook/sam2-hiera-base-plus`

### 🎯 Modelos SAM2 Disponibles:
- `facebook/sam2-hiera-tiny` (más rápido, menos preciso)
- `facebook/sam2-hiera-small` (balance)
- `facebook/sam2-hiera-base-plus` (recomendado, por defecto)
- `facebook/sam2-hiera-large` (mejor rendimiento, más recursos)

---

Este proyecto implementa tres métodos de finetuning para SAM2 (Segment Anything Model 2) aplicado a segmentación médica usando datasets en formato COCO.

## 🎯 Métodos Implementados

1. **Finetuning Clásico**: Entrena todos los parámetros del modelo
2. **LoRA (Low-Rank Adaptation)**: Entrena solo adaptadores de bajo rango
3. **QLoRA (Quantized LoRA)**: Combina quantización 4-bit con LoRA para máxima eficiencia

## 📁 Estructura del Proyecto

```
Finetuning SAM2/
├── utils.py                 # Utilidades comunes (dataset, métricas, visualización)
├── finetune_classic.py      # Finetuning clásico
├── finetune_lora.py         # Finetuning con LoRA
├── finetune_qlora.py        # Finetuning con QLoRA
├── main.py                  # Script principal para ejecutar todos los métodos
├── inference.py             # Script de inferencia y evaluación
├── README.md                # Este archivo
└── data/                    # Datasets
    ├── Cataract COCO Segmentation/
    │   ├── train/
    │   ├── valid/
    │   └── test/
    └── Diabetic-Retinopathy COCO Segmentation/
        ├── train/
        ├── valid/
        └── test/
```

## 🚀 Instalación

### Requisitos
- Python 3.10+
- CUDA (recomendado para GPU)
- 8GB+ RAM
- 4GB+ VRAM (GPU)

### Instalación de dependencias
Las dependencias se instalan automáticamente:
- torch
- transformers
- datasets
- peft
- pycocotools
- opencv-python
- bitsandbytes
- accelerate

## 📊 Datasets

El proyecto usa dos datasets médicos en formato COCO:

1. **Cataract COCO Segmentation**: Segmentación de cataratas
2. **Diabetic-Retinopathy COCO Segmentation**: Segmentación de retinopatía diabética

Cada dataset debe tener la estructura:
```
Dataset/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

## 🏃‍♂️ Uso Rápido

### 1. Ejecutar todos los métodos
```bash
python main.py --cataract_dir "data/Cataract COCO Segmentation" \
               --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
               --run_all
```

### 2. Ejecutar método específico
```bash
# Solo finetuning clásico
python main.py --cataract_dir "data/Cataract COCO Segmentation" \
               --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
               --run_classic

# Solo LoRA
python main.py --cataract_dir "data/Cataract COCO Segmentation" \
               --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
               --run_lora

# Solo QLoRA
python main.py --cataract_dir "data/Cataract COCO Segmentation" \
               --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
               --run_qlora
```

### 3. Prueba rápida (pocas muestras)
```bash
python main.py --cataract_dir "data/Cataract COCO Segmentation" \
               --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
               --run_all \
               --max_samples 10
```

## ⚙️ Configuración Avanzada

### Finetuning Clásico
```bash
python finetune_classic.py \
    --cataract_dir "data/Cataract COCO Segmentation" \
    --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
    --batch_size 2 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --save_dir "checkpoints_classic"
```

### LoRA
```bash
python finetune_lora.py \
    --cataract_dir "data/Cataract COCO Segmentation" \
    --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
    --batch_size 4 \
    --num_epochs 15 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --save_dir "checkpoints_lora" \
    --merge_final
```

### QLoRA
```bash
python finetune_qlora.py \
    --cataract_dir "data/Cataract COCO Segmentation" \
    --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
    --batch_size 2 \
    --num_epochs 20 \
    --learning_rate 2e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --save_dir "checkpoints_qlora"
```

## 🔍 Inferencia y Evaluación

### Inferencia en imagen individual
```bash
python inference.py \
    --model_type lora \
    --model_path "checkpoints_lora/sam2_lora_best" \
    --test_image "test_image.jpg"
```

### Evaluación en dataset
```bash
python inference.py \
    --model_type qlora \
    --model_path "checkpoints_qlora/sam2_qlora_best" \
    --test_dataset "data/Cataract COCO Segmentation" \
    --dataset_split test \
    --max_samples 50
```

### Comparar múltiples modelos
```bash
# Crear archivo de configuración (models_config.json)
echo '[
    {
        "name": "Classic",
        "type": "classic",
        "path": "checkpoints_classic/sam2_classic_best.pth"
    },
    {
        "name": "LoRA",
        "type": "lora", 
        "path": "checkpoints_lora/sam2_lora_best"
    },
    {
        "name": "QLoRA",
        "type": "qlora",
        "path": "checkpoints_qlora/sam2_qlora_best"
    }
]' > models_config.json

# Ejecutar comparación
python inference.py \
    --compare_models models_config.json \
    --test_image "test_image.jpg"
```

## 📈 Resultados y Métricas

Cada método genera:

1. **Checkpoints**: Modelos guardados durante el entrenamiento
2. **Métricas**: Gráficas de loss e IoU por época
3. **Logs**: Progreso detallado del entrenamiento

### Archivos generados:
- `checkpoints_*/`: Modelos entrenados
- `training_metrics_*.png`: Gráficas de entrenamiento
- `evaluation_results_*.json`: Resultados de evaluación
- `model_comparison.png`: Comparación visual de modelos

## 🔧 Personalización

### Modificar hiperparámetros
Edita los valores por defecto en `main.py` o pasa argumentos personalizados:

```bash
python main.py \
    --classic_lr 5e-6 \
    --classic_epochs 20 \
    --lora_r 32 \
    --qlora_lr 3e-4 \
    # ... otros parámetros
```

### Usar modelo diferente
```bash
python main.py \
    --model_name "facebook/sam-vit-large" \
    # ... otros argumentos
```

### Añadir nuevos datasets
Modifica `utils.py` para incluir nuevos datasets en formato COCO.

## 💡 Recomendaciones

### Por tipo de hardware:

**GPU >8GB VRAM:**
- Usar finetuning clásico para mejor rendimiento
- Batch size 4-8

**GPU 4-8GB VRAM:**
- Usar LoRA
- Batch size 2-4

**GPU <4GB VRAM:**
- Usar QLoRA obligatoriamente
- Batch size 1-2

### Por objetivo:

**Máximo rendimiento:**
- Finetuning clásico
- Más épocas
- Learning rate bajo

**Balance eficiencia/rendimiento:**
- LoRA con r=16-32
- Moderate learning rate

**Máxima eficiencia:**
- QLoRA con r=8
- Más épocas para compensar

## 🐛 Troubleshooting

### Error de memoria GPU
```bash
# Reducir batch size
--batch_size 1

# Usar QLoRA
--run_qlora

# Limitar muestras para testing
--max_samples 50
```

### Error de dependencias
```bash
# Reinstalar paquetes
pip install --upgrade torch transformers peft bitsandbytes
```

### Error en datasets
```bash
# Verificar estructura de directorios
ls "data/Cataract COCO Segmentation/train/"
ls "data/Diabetic-Retinopathy COCO Segmentation/train/"
```

## 📚 Referencias

- [SAM2 Paper](https://arxiv.org/abs/2408.00714)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Transformers Library](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)

## 🤝 Contribuciones

Para contribuir:
1. Fork el repositorio
2. Crea una rama feature
3. Implementa mejoras
4. Envía pull request

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo LICENSE para detalles.

---

¡Listo para comenzar el finetuning de SAM2! 🚀
