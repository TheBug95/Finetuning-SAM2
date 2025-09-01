"""
Finetuning clásico de SAM2 para segmentación médica
Entrena todos los parámetros del modelo
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from transformers import Sam2Model, Sam2Processor
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


class SAM2ClassicTrainer:
    """Entrenador para finetuning clásico de SAM2"""
    
    def __init__(self, 
                 model_name="facebook/sam2-hiera-base-plus", 
                 learning_rate=1e-5,
                 device=None):
        """
        Args:
            model_name: Nombre del modelo SAM pre-entrenado
            learning_rate: Tasa de aprendizaje
            device: Dispositivo (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        
        print(f"Inicializando SAM2 Classic Trainer en {self.device}")
        print(f"Modelo: {model_name}")
        
        # Cargar modelo y procesador
        self.processor = Sam2Processor.from_pretrained(model_name)
        self.model = Sam2Model.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Configurar para finetuning completo
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Optimizador
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
        
        print(f"Parámetros totales: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Parámetros entrenables: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
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
        
        pbar = tqdm(train_loader, desc="Entrenando")
        
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
                        # Forward pass
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
            for batch in tqdm(val_loader, desc="Validando"):
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
              num_epochs=10,
              save_dir="checkpoints"):
        """Entrenamiento principal"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"\nIniciando entrenamiento clásico por {num_epochs} épocas")
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
                save_path = os.path.join(save_dir, "sam2_classic_best.pth")
                save_model_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, save_path
                )
            
            # Guardar checkpoint regular
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"sam2_classic_epoch_{epoch+1}.pth")
                save_model_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, save_path
                )
        
        print("\n" + "=" * 60)
        print("Entrenamiento completado!")
        print(f"Mejor validation loss: {best_val_loss:.4f}")
        
        # Graficar métricas
        self.metrics.plot_metrics(os.path.join(save_dir, "training_metrics_classic.png"))
        
        return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Finetuning clásico de SAM2')
    parser.add_argument('--cataract_dir', type=str, required=True,
                       help='Directorio del dataset de cataratas')
    parser.add_argument('--retinopathy_dir', type=str, required=True,
                       help='Directorio del dataset de retinopatía diabética')
    parser.add_argument('--model_name', type=str, default='facebook/sam2-hiera-base-plus',
                       help='Nombre del modelo SAM2')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Tamaño del batch')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Número de épocas')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Tasa de aprendizaje')
    parser.add_argument('--save_dir', type=str, default='checkpoints_classic',
                       help='Directorio para guardar checkpoints')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Máximo número de muestras por dataset (para testing)')
    
    args = parser.parse_args()
    
    # Configurar device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")
    
    # Crear trainer
    trainer = SAM2ClassicTrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
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
    
    print(f"Entrenamiento completado. Mejor loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
