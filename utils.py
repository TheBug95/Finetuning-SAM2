"""
Utilidades comunes para el finetuning de SAM2
Soporta datasets en formato COCO para segmentación médica
"""

import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import mask as coco_mask
from transformers import Sam2Processor
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


class COCOSegmentationDataset(Dataset):
    """Dataset personalizado para datos COCO de segmentación médica"""
    
    def __init__(self, 
                 data_dir: str, 
                 split: str = "train",
                 processor: Optional[Sam2Processor] = None,
                 max_samples: Optional[int] = None):
        """
        Args:
            data_dir: Directorio raíz del dataset
            split: "train", "valid" o "test"
            processor: Procesador de SAM para preprocessing
            max_samples: Límite máximo de muestras (para debugging)
        """
        self.data_dir = data_dir
        self.split = split
        self.processor = processor
        
        # Cargar anotaciones COCO
        ann_file = os.path.join(data_dir, split, "_annotations.coco.json")
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Crear mapeos
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Agrupar anotaciones por imagen
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Lista de IDs de imágenes válidas (que tienen anotaciones)
        self.image_ids = list(self.image_annotations.keys())
        
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]
        
        print(f"Dataset {split}: {len(self.image_ids)} imágenes cargadas")
        print(f"Categorías: {[cat['name'] for cat in self.categories.values()]}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Cargar imagen
        image_info = self.images[image_id]
        image_path = os.path.join(self.data_dir, self.split, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Obtener anotaciones
        annotations = self.image_annotations[image_id]
        
        # Preparar máscaras y bboxes
        masks = []
        bboxes = []
        labels = []
        
        for ann in annotations:
            # Convertir segmentación a máscara
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Polígono
                    mask = self._polygon_to_mask(ann['segmentation'], 
                                                image_info['height'], 
                                                image_info['width'])
                else:
                    # RLE encoding
                    mask = coco_mask.decode(ann['segmentation'])
                
                masks.append(mask)
                bboxes.append(ann['bbox'])  # [x, y, width, height]
                labels.append(ann['category_id'])
        
        # Convertir formato de bbox de COCO [x, y, w, h] a [x1, y1, x2, y2]
        bboxes_xyxy = []
        for bbox in bboxes:
            x, y, w, h = bbox
            bboxes_xyxy.append([x, y, x + w, y + h])
        
        sample = {
            'image': image,
            'masks': np.array(masks),
            'bboxes': np.array(bboxes_xyxy),
            'labels': np.array(labels),
            'image_id': image_id
        }
        
        # Aplicar procesamiento si está disponible
        if self.processor:
            sample = self._process_sample(sample)
        
        return sample
    
    def _polygon_to_mask(self, segmentation, height, width):
        """Convierte polígonos COCO a máscara binaria"""
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in segmentation:
            poly = np.array(poly).reshape(-1, 2)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        return mask
    
    def _process_sample(self, sample):
        """Aplica el procesamiento específico de SAM2"""
        # Generar puntos de entrada automáticamente desde las máscaras
        input_points = []
        input_labels = []
        
        for mask in sample['masks']:
            # Obtener centroide de la máscara como punto positivo
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                centroid_x = int(np.mean(x_indices))
                centroid_y = int(np.mean(y_indices))
                input_points.append([centroid_x, centroid_y])
                input_labels.append(1)  # Punto positivo
        
        if len(input_points) == 0:
            # Si no hay máscaras válidas, usar punto central de la imagen
            h, w = sample['image'].size[1], sample['image'].size[0]
            input_points = [[w//2, h//2]]
            input_labels = [1]
        
        # Formato correcto para SAM2: [imagen, objeto, punto, coordenadas]
        # Para una imagen con múltiples objetos
        formatted_points = []
        formatted_labels = []
        
        for point, label in zip(input_points, input_labels):
            formatted_points.append([point])  # Cada objeto tiene una lista de puntos
            formatted_labels.append([label])  # Cada objeto tiene una lista de etiquetas
        
        # Procesar con SAM2 processor
        processed = self.processor(
            sample['image'],
            input_points=[formatted_points],  # [imagen[objeto[punto[x,y]]]]
            input_labels=[formatted_labels],  # [imagen[objeto[etiqueta]]]
            return_tensors="pt"
        )
        
        # Agregar máscaras ground truth - asegurar formato correcto
        masks = sample['masks']
        if isinstance(masks, np.ndarray):
            if len(masks.shape) == 3 and masks.shape[0] > 0:
                # Si hay múltiples máscaras, combinarlas
                combined_mask = np.any(masks, axis=0).astype(np.float32)
            elif len(masks.shape) == 2:
                combined_mask = masks.astype(np.float32)
            else:
                # Crear máscara vacía si no hay máscaras válidas
                h, w = sample['image'].size[1], sample['image'].size[0]
                combined_mask = np.zeros((h, w), dtype=np.float32)
        else:
            # Si masks es una lista, convertir a array
            masks = np.array(masks)
            if len(masks) > 0:
                combined_mask = np.any(masks, axis=0).astype(np.float32)
            else:
                h, w = sample['image'].size[1], sample['image'].size[0]
                combined_mask = np.zeros((h, w), dtype=np.float32)
        
        processed['ground_truth_masks'] = torch.tensor(combined_mask, dtype=torch.float32)
        processed['labels'] = torch.tensor(sample['labels'])
        processed['image_id'] = sample['image_id']
        
        return processed


def create_data_loaders(cataract_dir: str, 
                       retinopathy_dir: str,
                       processor: Sam2Processor,
                       batch_size: int = 4,
                       max_samples_per_dataset: Optional[int] = None):
    """
    Crea data loaders para ambos datasets médicos
    """
    from torch.utils.data import DataLoader, ConcatDataset
    
    # Datasets de cataratas
    cataract_train = COCOSegmentationDataset(
        cataract_dir, "train", processor, max_samples_per_dataset
    )
    cataract_val = COCOSegmentationDataset(
        cataract_dir, "valid", processor, max_samples_per_dataset
    )
    
    # Datasets de retinopatía diabética
    retinopathy_train = COCOSegmentationDataset(
        retinopathy_dir, "train", processor, max_samples_per_dataset
    )
    retinopathy_val = COCOSegmentationDataset(
        retinopathy_dir, "valid", processor, max_samples_per_dataset
    )
    
    # Combinar datasets
    train_dataset = ConcatDataset([cataract_train, retinopathy_train])
    val_dataset = ConcatDataset([cataract_val, retinopathy_val])
    
    # Crear data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    """Función de collate personalizada para el batch processing"""
    # SAM espera entradas específicas, así que mantenemos estructura simple
    return batch


def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """Calcula IoU entre máscara predicha y ground truth"""
    pred_binary = (pred_mask > threshold).float()
    gt_binary = gt_mask.float()
    
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def visualize_prediction(image, pred_masks, gt_masks, input_points=None, save_path=None):
    """Visualiza predicciones vs ground truth"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagen original
    axes[0].imshow(image)
    if input_points is not None:
        for point in input_points:
            axes[0].plot(point[0], point[1], 'ro', markersize=8)
    axes[0].set_title('Imagen Original + Puntos')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(image)
    if len(gt_masks.shape) > 2:
        combined_gt = np.max(gt_masks, axis=0)
    else:
        combined_gt = gt_masks
    axes[1].imshow(combined_gt, alpha=0.5, cmap='Reds')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Predicción
    axes[2].imshow(image)
    if len(pred_masks.shape) > 2:
        combined_pred = np.max(pred_masks, axis=0)
    else:
        combined_pred = pred_masks
    axes[2].imshow(combined_pred, alpha=0.5, cmap='Blues')
    axes[2].set_title('Predicción')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def print_training_progress(epoch, loss, val_loss=None, iou=None):
    """Imprime progreso del entrenamiento"""
    progress = f"Epoch {epoch:3d} | Loss: {loss:.4f}"
    if val_loss is not None:
        progress += f" | Val Loss: {val_loss:.4f}"
    if iou is not None:
        progress += f" | IoU: {iou:.4f}"
    print(progress)


class MetricsTracker:
    """Clase para trackear métricas durante el entrenamiento"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.val_losses = []
        self.ious = []
        self.val_ious = []
    
    def update(self, loss, val_loss=None, iou=None, val_iou=None):
        self.losses.append(loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if iou is not None:
            self.ious.append(iou)
        if val_iou is not None:
            self.val_ious.append(val_iou)
    
    def plot_metrics(self, save_path=None):
        """Grafica las métricas de entrenamiento"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.losses, label='Train Loss', alpha=0.7)
        if self.val_losses:
            axes[0].plot(self.val_losses, label='Val Loss', alpha=0.7)
        axes[0].set_title('Loss durante el entrenamiento')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # IoU
        if self.ious:
            axes[1].plot(self.ious, label='Train IoU', alpha=0.7)
        if self.val_ious:
            axes[1].plot(self.val_ious, label='Val IoU', alpha=0.7)
        axes[1].set_title('IoU durante el entrenamiento')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def save_model_checkpoint(model, optimizer, epoch, loss, save_path):
    """Guarda checkpoint del modelo"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint guardado: {save_path}")


def load_model_checkpoint(model, optimizer, checkpoint_path):
    """Carga checkpoint del modelo"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint cargado: {checkpoint_path}")
    return epoch, loss
