"""
Finetuning clásico de SAM2 para segmentación médica
Entrena todos los parámetros del modelo
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
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
        self.scaler = GradScaler('cuda')
        
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
            valid_samples = 0
            
            for sample in batch:
                try:
                    # Mover a device
                    pixel_values = sample['pixel_values'].to(self.device)
                    
                    # Handle input_points - SAM2 processor returns nested lists
                    input_points = sample['input_points']  # Should be [[[x,y], [x,y], ...]]
                    input_labels = sample['input_labels']  # Should be [[[label], [label], ...]]
                    
                    # Convert nested lists to tensors and move to device
                    if isinstance(input_points, list):
                        # input_points is [batch][object][point][coord]
                        input_points_tensors = []
                        for batch_points in input_points:  # For each image in batch
                            batch_tensors = []
                            for obj_points in batch_points:  # For each object in image
                                if isinstance(obj_points, list):
                                    tensor_points = torch.tensor(obj_points, dtype=torch.float32).to(self.device)
                                else:
                                    tensor_points = obj_points.to(self.device)
                                batch_tensors.append(tensor_points)
                            input_points_tensors.append(batch_tensors)
                        input_points = input_points_tensors
                    else:
                        input_points = input_points.to(self.device)
                    
                    if isinstance(input_labels, list):
                        # input_labels is [batch][object][label]
                        input_labels_tensors = []
                        for batch_labels in input_labels:  # For each image in batch
                            batch_tensors = []
                            for obj_labels in batch_labels:  # For each object in image
                                if isinstance(obj_labels, list):
                                    tensor_labels = torch.tensor(obj_labels, dtype=torch.long).to(self.device)
                                else:
                                    tensor_labels = obj_labels.to(self.device)
                                batch_tensors.append(tensor_labels)
                            input_labels_tensors.append(batch_tensors)
                        input_labels = input_labels_tensors
                    else:
                        input_labels = input_labels.to(self.device)
                    
                    gt_masks = sample['ground_truth_masks'].to(self.device)
                    
                    # Verificar que gt_masks tenga la forma correcta
                    if len(gt_masks.shape) == 3:
                        # Si hay múltiples máscaras, tomar la primera o combinarlas
                        gt_masks = gt_masks.max(dim=0)[0]  # Combinar máscaras superpuestas
                    
                    # Asegurar que gt_masks tenga la forma correcta [H, W]
                    if len(gt_masks.shape) == 1:
                        # Si es 1D, reshape según las dimensiones de la imagen
                        h, w = pixel_values.shape[-2:]
                        gt_masks = gt_masks.view(h, w)
                    
                    with autocast('cuda'):
                        # Forward pass
                        outputs = self.model(
                            pixel_values=pixel_values,
                            input_points=input_points,
                            input_labels=input_labels
                        )
                        
                        # Calcular loss
                        pred_masks = outputs.pred_masks.squeeze(1)
                        
                        # Asegurar que las dimensiones coincidan
                        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                            gt_masks = torch.nn.functional.interpolate(
                                gt_masks.unsqueeze(0).unsqueeze(0), 
                                size=pred_masks.shape[-2:], 
                                mode='nearest'
                            ).squeeze()
                        
                        loss, bce_loss, dice_loss = self.compute_loss(pred_masks, gt_masks.unsqueeze(0))
                        
                        # Calcular IoU
                        with torch.no_grad():
                            iou = calculate_iou(
                                torch.sigmoid(pred_masks), 
                                gt_masks.unsqueeze(0),
                                threshold=0.5
                            )
                    
                    # Acumular solo si el loss es válido
                    if torch.isfinite(loss):
                        batch_loss += loss
                        batch_iou += iou
                        valid_samples += 1
                    
                except Exception as e:
                    print(f"Error en sample: {e}")
                    continue
            
            if valid_samples > 0:
                # Promedio del batch
                batch_loss = batch_loss / valid_samples
                batch_iou = batch_iou / valid_samples
                
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
                valid_samples = 0
                
                for sample in batch:
                    try:
                        # Mover a device
                        pixel_values = sample['pixel_values'].to(self.device)
                        
                        # Handle input_points and input_labels (same as training)
                        input_points = sample['input_points']
                        input_labels = sample['input_labels']
                        
                        # Convert nested lists to tensors and move to device
                        if isinstance(input_points, list):
                            input_points_tensors = []
                            for batch_points in input_points:
                                batch_tensors = []
                                for obj_points in batch_points:
                                    if isinstance(obj_points, list):
                                        tensor_points = torch.tensor(obj_points, dtype=torch.float32).to(self.device)
                                    else:
                                        tensor_points = obj_points.to(self.device)
                                    batch_tensors.append(tensor_points)
                                input_points_tensors.append(batch_tensors)
                            input_points = input_points_tensors
                        else:
                            input_points = input_points.to(self.device)
                        
                        if isinstance(input_labels, list):
                            input_labels_tensors = []
                            for batch_labels in input_labels:
                                batch_tensors = []
                                for obj_labels in batch_labels:
                                    if isinstance(obj_labels, list):
                                        tensor_labels = torch.tensor(obj_labels, dtype=torch.long).to(self.device)
                                    else:
                                        tensor_labels = obj_labels.to(self.device)
                                    batch_tensors.append(tensor_labels)
                                input_labels_tensors.append(batch_tensors)
                            input_labels = input_labels_tensors
                        else:
                            input_labels = input_labels.to(self.device)
                        
                        gt_masks = sample['ground_truth_masks'].to(self.device)
                        
                        # Verificar que gt_masks tenga la forma correcta
                        if len(gt_masks.shape) == 3:
                            gt_masks = gt_masks.max(dim=0)[0]
                        
                        if len(gt_masks.shape) == 1:
                            h, w = pixel_values.shape[-2:]
                            gt_masks = gt_masks.view(h, w)
                        
                        # Forward pass
                        outputs = self.model(
                            pixel_values=pixel_values,
                            input_points=input_points,
                            input_labels=input_labels
                        )
                        
                        # Calcular métricas
                        pred_masks = outputs.pred_masks.squeeze(1)
                        
                        # Asegurar que las dimensiones coincidan
                        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                            gt_masks = torch.nn.functional.interpolate(
                                gt_masks.unsqueeze(0).unsqueeze(0), 
                                size=pred_masks.shape[-2:], 
                                mode='nearest'
                            ).squeeze()
                        
                        loss, _, _ = self.compute_loss(pred_masks, gt_masks.unsqueeze(0))
                        iou = calculate_iou(
                            torch.sigmoid(pred_masks), 
                            gt_masks.unsqueeze(0),
                            threshold=0.5
                        )
                        
                        if torch.isfinite(loss):
                            batch_loss += loss
                            batch_iou += iou
                            valid_samples += 1
                        
                    except Exception as e:
                        print(f"Error en validación: {e}")
                        continue
                
                if valid_samples > 0:
                    total_loss += (batch_loss / valid_samples).item()
                    total_iou += batch_iou / valid_samples
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
