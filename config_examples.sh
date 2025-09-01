# Configuración de ejemplo para diferentes escenarios de finetuning

## CONFIGURACIÓN PARA PRUEBA RÁPIDA (TESTING)
python main.py \
    --cataract_dir "data/Cataract COCO Segmentation" \
    --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
    --run_qlora \
    --qlora_epochs 3 \
    --qlora_batch_size 1 \
    --max_samples 10

## CONFIGURACIÓN PARA GPU HIGH-END (>8GB VRAM)
python main.py \
    --cataract_dir "data/Cataract COCO Segmentation" \
    --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
    --run_all \
    --classic_batch_size 4 \
    --classic_epochs 15 \
    --classic_lr 1e-5 \
    --lora_batch_size 8 \
    --lora_epochs 20 \
    --lora_lr 1e-4 \
    --lora_r 32 \
    --qlora_batch_size 4 \
    --qlora_epochs 25 \
    --qlora_lr 2e-4

## CONFIGURACIÓN PARA GPU MEDIA (4-8GB VRAM)
python main.py \
    --cataract_dir "data/Cataract COCO Segmentation" \
    --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
    --run_lora --run_qlora \
    --lora_batch_size 4 \
    --lora_epochs 20 \
    --lora_lr 1e-4 \
    --lora_r 16 \
    --qlora_batch_size 2 \
    --qlora_epochs 25 \
    --qlora_lr 2e-4

## CONFIGURACIÓN PARA GPU LOW-END (<4GB VRAM)
python main.py \
    --cataract_dir "data/Cataract COCO Segmentation" \
    --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
    --run_qlora \
    --qlora_batch_size 1 \
    --qlora_epochs 30 \
    --qlora_lr 3e-4 \
    --qlora_r 8 \
    --qlora_alpha 16

## CONFIGURACIÓN PARA MÁXIMO RENDIMIENTO (SIN LÍMITES DE RECURSOS)
python main.py \
    --cataract_dir "data/Cataract COCO Segmentation" \
    --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
    --run_classic \
    --model_name "facebook/sam-vit-large" \
    --classic_batch_size 8 \
    --classic_epochs 25 \
    --classic_lr 5e-6

## CONFIGURACIÓN PARA MÁXIMA EFICIENCIA
python main.py \
    --cataract_dir "data/Cataract COCO Segmentation" \
    --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation" \
    --run_qlora \
    --qlora_batch_size 1 \
    --qlora_epochs 40 \
    --qlora_lr 5e-4 \
    --qlora_r 4 \
    --qlora_alpha 8
