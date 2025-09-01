"""
Finetuning de SAM2 usando QLoRA (Quantized Low-Rank Adaptation)
Combina quantización 4-bit con LoRA para máxima eficiencia de memoria
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import SamModel, SamProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from tqdm import tqdm
import argparse

from utils import (
    create_data_loaders, 
    calculate_iou, 
    print_training_progress,
    MetricsTracker,
    save_model_checkpoint,
    visualize_prediction
)


class SAM2QLoRATrainer:
    """Entrenador para finetuning de SAM2 con QLoRA"""
    
    def __init__(self, 
                 model_name="facebook/sam-vit-base", 
                 learning_rate=2e-4,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 device=None):
        """
        Args:
            model_name: Nombre del modelo SAM pre-entrenado
            learning_rate: Tasa de aprendizaje (puede ser mayor que LoRA)
            lora_r: Rango de LoRA (menor para QLoRA)
            lora_alpha: Parámetro de escalado de LoRA
            lora_dropout: Dropout para LoRA
            device: Dispositivo (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        
        print(f"Inicializando SAM2 QLoRA Trainer en {self.device}")
        print(f"Modelo: {model_name}")
        print(f"QLoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        
        # Configuración de quantización 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        print("Cargando modelo con quantización 4-bit...")
        
        # Cargar procesador
        self.processor = SamProcessor.from_pretrained(model_name)
        
        # Cargar modelo base con quantización
        base_model = SamModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16
        )
        
        # Preparar modelo para k-bit training
        base_model = prepare_model_for_kbit_training(base_model)
        
        # Configurar LoRA (con parámetros más conservadores para QLoRA)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "qkv",  # Attention layers
                "proj", # Projection layers
                "dense" # Dense layers
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        # Aplicar LoRA al modelo quantizado
        self.model = get_peft_model(base_model, lora_config)
        
        # Imprimir estadísticas de parámetros
        self.model.print_trainable_parameters()
        
        # Optimizador especializado para quantización
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-6  # Epsilon más pequeño para estabilidad con quantización
        )
        
        # Scheduler más agresivo para QLoRA
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=5,
            T_mult=2,
            eta_min=learning_rate * 0.1
        )
        
        # Mixed precision training (especialmente importante para QLoRA)
        self.scaler = GradScaler()
        
        # Métricas
        self.metrics = MetricsTracker()
        
        # Calcular uso de memoria
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"Memoria GPU utilizada: {memory_used:.2f} GB")
    
    def compute_loss(self, pred_masks, gt_masks):
        """Calcula loss combinado con estabilización para quantización"""
        # Asegurar que las dimensiones coincidan
        if pred_masks.dim() == 4 and gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)
        elif pred_masks.dim() == 3 and gt_masks.dim() == 4:
            pred_masks = pred_masks.unsqueeze(1)
        
        # Redimensionar si es necesario
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = torch.nn.functional.interpolate(
                gt_masks.float(), 
                size=pred_masks.shape[-2:], 
                mode='nearest'
            )
        
        # Estabilización para quantización
        epsilon = 1e-7
        
        # BCE Loss con label smoothing para estabilidad
        label_smoothing = 0.05
        gt_smooth = gt_masks.float() * (1 - label_smoothing) + label_smoothing * 0.5
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_masks, gt_smooth
        )
        
        # Dice Loss con estabilización
        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_masks).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice_score = (2 * intersection + epsilon) / (union + epsilon)
        dice_loss = 1 - dice_score.mean()
        
        # Loss combinado con pesos ajustados para QLoRA
        total_loss = 0.7 * bce_loss + 0.3 * dice_loss
        
        return total_loss, bce_loss, dice_loss
    
    def train_epoch(self, train_loader):
        """Entrena una época con QLoRA"""
        self.model.train()
        total_loss = 0
        total_iou = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Entrenando (QLoRA)")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            batch_loss = 0
            batch_iou = 0
            batch_size = len(batch)
            
            for sample in batch:
                try:
                    # Mover a device con dtype apropiado
                    pixel_values = sample['pixel_values'].to(self.device)
                    input_points = [points.to(self.device) for points in sample['input_points']]
                    input_labels = [labels.to(self.device) for labels in sample['input_labels']]
                    gt_masks = sample['ground_truth_masks'].to(self.device)
                    
                    with autocast():
                        # Forward pass con QLoRA
                        outputs = self.model(
                            pixel_values=pixel_values,
                            input_points=input_points,
                            input_labels=input_labels
                        )
                        
                        # Calcular loss
                        pred_masks = outputs.pred_masks.squeeze(1)
                        loss, bce_loss, dice_loss = self.compute_loss(pred_masks, gt_masks)
                        
                        # Escalado del loss para quantización
                        loss = loss / batch_size
                        
                        # Calcular IoU
                        with torch.no_grad():
                            iou = calculate_iou(
                                torch.sigmoid(pred_masks), 
                                gt_masks,
                                threshold=0.5
                            )
                    
                    # Acumular gradientes
                    self.scaler.scale(loss).backward()
                    batch_loss += loss.item() * batch_size
                    batch_iou += iou
                    
                except Exception as e:
                    print(f"Error en sample: {e}")
                    continue
            
            if batch_size > 0:
                # Clip gradients para estabilidad con quantización
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                avg_batch_loss = batch_loss / batch_size
                avg_batch_iou = batch_iou / batch_size
                
                total_loss += avg_batch_loss
                total_iou += avg_batch_iou
                num_batches += 1
                
                # Actualizar progress bar
                pbar.set_postfix({
                    'Loss': f'{avg_batch_loss:.4f}',
                    'IoU': f'{avg_batch_iou:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_iou = total_iou / max(num_batches, 1)
        
        # Actualizar scheduler
        self.scheduler.step()
        
        return avg_loss, avg_iou
    
    def validate(self, val_loader):
        """Valida el modelo con QLoRA"""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validando (QLoRA)"):
                batch_loss = 0
                batch_iou = 0
                batch_size = len(batch)
                
                for sample in batch:
                    try:
                        # Mover a device
                        pixel_values = sample['pixel_values'].to(self.device)
                        input_points = [points.to(self.device) for points in sample['input_points']]
                        input_labels = [labels.to(self.device) for labels in sample['input_labels']]
                        gt_masks = sample['ground_truth_masks'].to(self.device)
                        
                        # Forward pass
                        outputs = self.model(
                            pixel_values=pixel_values,
                            input_points=input_points,
                            input_labels=input_labels
                        )
                        
                        # Calcular métricas
                        pred_masks = outputs.pred_masks.squeeze(1)
                        loss, _, _ = self.compute_loss(pred_masks, gt_masks)
                        iou = calculate_iou(
                            torch.sigmoid(pred_masks), 
                            gt_masks,
                            threshold=0.5
                        )
                        
                        batch_loss += loss.item()
                        batch_iou += iou
                        
                    except Exception as e:
                        print(f"Error en validación: {e}")
                        continue
                
                if batch_size > 0:
                    total_loss += batch_loss / batch_size
                    total_iou += batch_iou / batch_size
                    num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_iou = total_iou / max(num_batches, 1)
        
        return avg_loss, avg_iou
    
    def train(self, 
              train_loader, 
              val_loader, 
              num_epochs=20,
              save_dir="checkpoints"):
        """Entrenamiento principal con QLoRA"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"\nIniciando entrenamiento QLoRA por {num_epochs} épocas")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Entrenamiento
            train_loss, train_iou = self.train_epoch(train_loader)
            
            # Validación
            val_loss, val_iou = self.validate(val_loader)
            
            # Actualizar métricas
            self.metrics.update(train_loss, val_loss, train_iou, val_iou)
            
            # Imprimir progreso
            print_training_progress(epoch + 1, train_loss, val_loss, val_iou)
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(save_dir, "sam2_qlora_best")
                # Guardar solo los adaptadores QLoRA
                self.model.save_pretrained(save_path)
                print(f"Mejor modelo QLoRA guardado: {save_path}")
            
            # Guardar checkpoint regular
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"sam2_qlora_epoch_{epoch+1}")
                self.model.save_pretrained(save_path)
            
            # Limpiar caché de CUDA para evitar OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("\n" + "=" * 60)
        print("Entrenamiento QLoRA completado!")
        print(f"Mejor validation loss: {best_val_loss:.4f}")
        
        # Imprimir uso final de memoria
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"Memoria GPU final: {memory_used:.2f} GB")
        
        # Graficar métricas
        self.metrics.plot_metrics(os.path.join(save_dir, "training_metrics_qlora.png"))
        
        return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Finetuning QLoRA de SAM2')
    parser.add_argument('--cataract_dir', type=str, required=True,
                       help='Directorio del dataset de cataratas')
    parser.add_argument('--retinopathy_dir', type=str, required=True,
                       help='Directorio del dataset de retinopatía diabética')
    parser.add_argument('--model_name', type=str, default='facebook/sam-vit-base',
                       help='Nombre del modelo SAM')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Tamaño del batch (menor debido a quantización)')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Número de épocas')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Tasa de aprendizaje')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='Rango de LoRA (menor para QLoRA)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='Alpha de LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='Dropout de LoRA')
    parser.add_argument('--save_dir', type=str, default='checkpoints_qlora',
                       help='Directorio para guardar checkpoints')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Máximo número de muestras por dataset (para testing)')
    
    args = parser.parse_args()
    
    # Verificar soporte para CUDA
    if not torch.cuda.is_available():
        print("Advertencia: QLoRA requiere GPU para máxima eficiencia")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")
    
    # Crear trainer
    trainer = SAM2QLoRATrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device
    )
    
    # Crear data loaders
    print("Cargando datasets...")
    train_loader, val_loader = create_data_loaders(
        cataract_dir=args.cataract_dir,
        retinopathy_dir=args.retinopathy_dir,
        processor=trainer.processor,
        batch_size=args.batch_size,
        max_samples_per_dataset=args.max_samples
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Entrenar
    best_loss = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )
    
    print(f"Entrenamiento QLoRA completado. Mejor loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
