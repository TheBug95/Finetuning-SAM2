"""
Finetuning de SAM2 usando LoRA (Low-Rank Adaptation)
Entrena solo adaptadores de bajo rango para mayor eficiencia
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import Sam2Model, Sam2Processor
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import argparse

from utils import (
    create_data_loaders,
    calculate_iou,
    print_training_progress,
    MetricsTracker,
    save_model_checkpoint,
    visualize_prediction,
    ensure_sam2_attention_dropout
)


class SAM2LoRATrainer:
    """Entrenador para finetuning de SAM2 con LoRA"""
    
    def __init__(self, 
                 model_name="facebook/sam2-hiera-base-plus", 
                 learning_rate=1e-4,
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 device=None):
        """
        Args:
            model_name: Nombre del modelo SAM pre-entrenado
            learning_rate: Tasa de aprendizaje (puede ser mayor que classic)
            lora_r: Rango de LoRA (menor = más eficiente)
            lora_alpha: Parámetro de escalado de LoRA
            lora_dropout: Dropout para LoRA
            device: Dispositivo (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        
        print(f"Inicializando SAM2 LoRA Trainer en {self.device}")
        print(f"Modelo: {model_name}")
        print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        
        # Cargar modelo base y procesador
        self.processor = Sam2Processor.from_pretrained(model_name)
        base_model = Sam2Model.from_pretrained(model_name)

        # Algunas versiones de SAM2 no exponen `dropout_p` en las capas de
        # atención, lo que genera errores durante el forward. Aseguramos que
        # el atributo exista antes de aplicar LoRA.
        ensure_sam2_attention_dropout(base_model)
        
        # Configurar LoRA
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
        
        # Aplicar LoRA al modelo
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)
        
        # Imprimir estadísticas de parámetros
        self.model.print_trainable_parameters()
        
        # Optimizador (solo parámetros LoRA)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Métricas
        self.metrics = MetricsTracker()
    
    def compute_loss(self, pred_masks, gt_masks):
        """Calcula loss combinado (BCE + Dice)"""
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
        
        # BCE Loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_masks, gt_masks.float()
        )
        
        # Dice Loss
        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_masks).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice_score = (2 * intersection + 1e-8) / (union + 1e-8)
        dice_loss = 1 - dice_score.mean()
        
        # Loss combinado
        total_loss = bce_loss + dice_loss
        
        return total_loss, bce_loss, dice_loss
    
    def train_epoch(self, train_loader):
        """Entrena una época"""
        self.model.train()
        total_loss = 0
        total_iou = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Entrenando (LoRA)")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
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
                    
                    with autocast():
                        # Forward pass con LoRA
                        outputs = self.model(
                            pixel_values=pixel_values,
                            input_points=input_points,
                            input_labels=input_labels
                        )
                        
                        # Calcular loss
                        pred_masks = outputs.pred_masks.squeeze(1)
                        loss, bce_loss, dice_loss = self.compute_loss(pred_masks, gt_masks)
                        
                        # Calcular IoU
                        with torch.no_grad():
                            iou = calculate_iou(
                                torch.sigmoid(pred_masks), 
                                gt_masks,
                                threshold=0.5
                            )
                    
                    batch_loss += loss
                    batch_iou += iou
                    
                except Exception as e:
                    print(f"Error en sample: {e}")
                    continue
            
            if batch_size > 0:
                # Promedio del batch
                batch_loss = batch_loss / batch_size
                batch_iou = batch_iou / batch_size
                
                # Backward pass
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += batch_loss.item()
                total_iou += batch_iou
                num_batches += 1
                
                # Actualizar progress bar
                pbar.set_postfix({
                    'Loss': f'{batch_loss.item():.4f}',
                    'IoU': f'{batch_iou:.4f}'
                })
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_iou = total_iou / max(num_batches, 1)
        
        return avg_loss, avg_iou
    
    def validate(self, val_loader):
        """Valida el modelo"""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validando (LoRA)"):
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
                        
                        batch_loss += loss
                        batch_iou += iou
                        
                    except Exception as e:
                        print(f"Error en validación: {e}")
                        continue
                
                if batch_size > 0:
                    total_loss += (batch_loss / batch_size).item()
                    total_iou += batch_iou / batch_size
                    num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_iou = total_iou / max(num_batches, 1)
        
        return avg_loss, avg_iou
    
    def train(self, 
              train_loader, 
              val_loader, 
              num_epochs=15,
              save_dir="checkpoints"):
        """Entrenamiento principal"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"\nIniciando entrenamiento LoRA por {num_epochs} épocas")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Entrenamiento
            train_loss, train_iou = self.train_epoch(train_loader)
            
            # Validación
            val_loss, val_iou = self.validate(val_loader)
            
            # Actualizar scheduler
            self.scheduler.step(val_loss)
            
            # Actualizar métricas
            self.metrics.update(train_loss, val_loss, train_iou, val_iou)
            
            # Imprimir progreso
            print_training_progress(epoch + 1, train_loss, val_loss, val_iou)
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(save_dir, "sam2_lora_best.pth")
                # Guardar solo los adaptadores LoRA
                self.model.save_pretrained(save_path)
                print(f"Mejor modelo LoRA guardado: {save_path}")
            
            # Guardar checkpoint regular
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"sam2_lora_epoch_{epoch+1}")
                self.model.save_pretrained(save_path)
        
        print("\n" + "=" * 60)
        print("Entrenamiento LoRA completado!")
        print(f"Mejor validation loss: {best_val_loss:.4f}")
        
        # Graficar métricas
        self.metrics.plot_metrics(os.path.join(save_dir, "training_metrics_lora.png"))
        
        return best_val_loss
    
    def merge_and_save_model(self, save_path):
        """Fusiona LoRA con el modelo base y guarda"""
        print("Fusionando adaptadores LoRA con modelo base...")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(save_path)
        print(f"Modelo fusionado guardado: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Finetuning LoRA de SAM2')
    parser.add_argument('--cataract_dir', type=str, required=True,
                       help='Directorio del dataset de cataratas')
    parser.add_argument('--retinopathy_dir', type=str, required=True,
                       help='Directorio del dataset de retinopatía diabética')
    parser.add_argument('--model_name', type=str, default='facebook/sam2-hiera-base-plus',
                       help='Nombre del modelo SAM2')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Tamaño del batch (puede ser mayor que classic)')
    parser.add_argument('--num_epochs', type=int, default=15,
                       help='Número de épocas')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Tasa de aprendizaje')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='Rango de LoRA')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='Alpha de LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='Dropout de LoRA')
    parser.add_argument('--save_dir', type=str, default='checkpoints_lora',
                       help='Directorio para guardar checkpoints')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Máximo número de muestras por dataset (para testing)')
    parser.add_argument('--merge_final', action='store_true',
                       help='Fusionar LoRA con modelo base al final')
    
    args = parser.parse_args()
    
    # Configurar device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")
    
    # Crear trainer
    trainer = SAM2LoRATrainer(
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
    
    # Fusionar modelo si se solicita
    if args.merge_final:
        merge_path = os.path.join(args.save_dir, "sam2_lora_merged")
        trainer.merge_and_save_model(merge_path)
    
    print(f"Entrenamiento LoRA completado. Mejor loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
