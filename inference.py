"""
Script de inferencia para modelos SAM2 fine-tuneados
Permite probar modelos entrenados con Classic, LoRA y QLoRA
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from transformers import SamModel, SamProcessor
from peft import PeftModel
import json

from utils import visualize_prediction, calculate_iou


class SAM2Inferencer:
    """Clase para realizar inferencia con modelos SAM2 fine-tuneados"""
    
    def __init__(self, model_type, model_path, base_model_name="facebook/sam-vit-base"):
        """
        Args:
            model_type: "classic", "lora", o "qlora"
            model_path: Ruta al modelo entrenado
            base_model_name: Modelo base de SAM2
        """
        self.model_type = model_type
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Cargando modelo {model_type} desde {model_path}")
        
        # Cargar procesador
        self.processor = SamProcessor.from_pretrained(base_model_name)
        
        # Cargar modelo según el tipo
        if model_type == "classic":
            self.model = SamModel.from_pretrained(base_model_name)
            # Cargar checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        elif model_type in ["lora", "qlora"]:
            # Cargar modelo base
            base_model = SamModel.from_pretrained(base_model_name)
            # Cargar adaptadores LoRA/QLoRA
            self.model = PeftModel.from_pretrained(base_model, model_path)
        
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Modelo cargado exitosamente en {self.device}")
    
    def predict_single_image(self, image_path, input_points=None, input_labels=None):
        """
        Realiza predicción en una sola imagen
        
        Args:
            image_path: Ruta a la imagen
            input_points: Lista de puntos [x, y] o None para usar centro
            input_labels: Lista de etiquetas (1=positivo, 0=negativo) o None
        
        Returns:
            dict con imagen, máscaras predichas y puntos usados
        """
        # Cargar imagen
        image = Image.open(image_path).convert('RGB')
        
        # Si no se proporcionan puntos, usar el centro de la imagen
        if input_points is None:
            w, h = image.size
            input_points = [[w//2, h//2]]
            input_labels = [1]
        
        # Procesar entrada
        inputs = self.processor(
            image, 
            input_points=[input_points], 
            input_labels=[input_labels], 
            return_tensors="pt"
        )
        
        # Mover a device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inferencia
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Procesar salida
        pred_masks = outputs.pred_masks.squeeze().cpu().numpy()
        pred_masks = torch.sigmoid(torch.tensor(pred_masks)).numpy()
        
        return {
            'image': np.array(image),
            'pred_masks': pred_masks,
            'input_points': input_points,
            'input_labels': input_labels
        }
    
    def predict_batch(self, image_paths, points_list=None):
        """
        Realiza predicción en un batch de imágenes
        
        Args:
            image_paths: Lista de rutas a imágenes
            points_list: Lista de listas de puntos para cada imagen
        
        Returns:
            Lista de resultados de predicción
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            points = points_list[i] if points_list else None
            result = self.predict_single_image(image_path, points)
            results.append(result)
        
        return results
    
    def evaluate_on_dataset(self, dataset_dir, split="test", max_samples=10):
        """
        Evalúa el modelo en un dataset COCO
        
        Args:
            dataset_dir: Directorio del dataset
            split: "train", "valid" o "test"
            max_samples: Número máximo de muestras a evaluar
        
        Returns:
            Métricas de evaluación
        """
        from utils import COCOSegmentationDataset
        
        # Crear dataset
        dataset = COCOSegmentationDataset(
            dataset_dir, 
            split, 
            processor=None,  # No procesamos aquí
            max_samples=max_samples
        )
        
        ious = []
        results = []
        
        print(f"Evaluando en {len(dataset)} muestras del split {split}")
        
        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample['image']
            gt_masks = sample['masks']
            
            # Convertir PIL a array numpy
            image_array = np.array(image)
            h, w = image_array.shape[:2]
            
            # Usar centroide de la primera máscara como punto de entrada
            if len(gt_masks) > 0:
                y_indices, x_indices = np.where(gt_masks[0] > 0)
                if len(y_indices) > 0:
                    centroid_x = int(np.mean(x_indices))
                    centroid_y = int(np.mean(y_indices))
                    input_points = [[centroid_x, centroid_y]]
                else:
                    input_points = [[w//2, h//2]]
            else:
                input_points = [[w//2, h//2]]
            
            # Guardar imagen temporalmente para predicción
            temp_path = f"temp_image_{i}.jpg"
            image.save(temp_path)
            
            try:
                # Predicción
                pred_result = self.predict_single_image(temp_path, input_points, [1])
                
                # Calcular IoU con la primera máscara ground truth
                if len(gt_masks) > 0:
                    iou = calculate_iou(
                        torch.tensor(pred_result['pred_masks']), 
                        torch.tensor(gt_masks[0]),
                        threshold=0.5
                    )
                    ious.append(iou)
                
                results.append({
                    'image_id': sample['image_id'],
                    'iou': iou if len(gt_masks) > 0 else 0,
                    'prediction': pred_result
                })
                
            except Exception as e:
                print(f"Error en muestra {i}: {e}")
            
            finally:
                # Limpiar archivo temporal
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Calcular métricas
        metrics = {
            'mean_iou': np.mean(ious) if ious else 0,
            'std_iou': np.std(ious) if ious else 0,
            'num_samples': len(ious)
        }
        
        print(f"Métricas de evaluación:")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"  Std IoU: {metrics['std_iou']:.4f}")
        print(f"  Muestras: {metrics['num_samples']}")
        
        return metrics, results


def compare_models(model_configs, test_image_path, input_points=None):
    """
    Compara múltiples modelos en la misma imagen
    
    Args:
        model_configs: Lista de diccionarios con configuración de modelos
        test_image_path: Ruta a imagen de prueba
        input_points: Puntos de entrada
    """
    results = {}
    
    # Realizar predicciones con cada modelo
    for config in model_configs:
        name = config['name']
        model_type = config['type']
        model_path = config['path']
        
        print(f"Probando modelo {name}...")
        
        try:
            inferencer = SAM2Inferencer(model_type, model_path)
            result = inferencer.predict_single_image(test_image_path, input_points)
            results[name] = result
        except Exception as e:
            print(f"Error con modelo {name}: {e}")
    
    # Visualizar comparación
    if results:
        num_models = len(results)
        fig, axes = plt.subplots(2, num_models, figsize=(5*num_models, 10))
        
        if num_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, (name, result) in enumerate(results.items()):
            # Imagen original con puntos
            axes[0, i].imshow(result['image'])
            for point in result['input_points']:
                axes[0, i].plot(point[0], point[1], 'ro', markersize=8)
            axes[0, i].set_title(f'{name} - Original')
            axes[0, i].axis('off')
            
            # Predicción
            axes[1, i].imshow(result['image'])
            if len(result['pred_masks'].shape) > 2:
                combined_mask = np.max(result['pred_masks'], axis=0)
            else:
                combined_mask = result['pred_masks']
            axes[1, i].imshow(combined_mask, alpha=0.5, cmap='Blues')
            axes[1, i].set_title(f'{name} - Predicción')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Comparación guardada como 'model_comparison.png'")


def main():
    parser = argparse.ArgumentParser(description='Inferencia con modelos SAM2 fine-tuneados')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['classic', 'lora', 'qlora'],
                       help='Tipo de modelo entrenado')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Ruta al modelo entrenado')
    parser.add_argument('--test_image', type=str,
                       help='Imagen de prueba individual')
    parser.add_argument('--test_dataset', type=str,
                       help='Directorio del dataset para evaluación')
    parser.add_argument('--dataset_split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Split del dataset a usar')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Máximo número de muestras para evaluación')
    parser.add_argument('--points', type=str,
                       help='Puntos de entrada como "x1,y1;x2,y2"')
    parser.add_argument('--base_model', type=str, default='facebook/sam-vit-base',
                       help='Modelo base de SAM2')
    parser.add_argument('--compare_models', type=str,
                       help='JSON con configuración de múltiples modelos para comparar')
    
    args = parser.parse_args()
    
    # Parsear puntos si se proporcionan
    input_points = None
    if args.points:
        point_pairs = args.points.split(';')
        input_points = []
        for pair in point_pairs:
            x, y = map(int, pair.split(','))
            input_points.append([x, y])
    
    # Modo comparación de modelos
    if args.compare_models:
        with open(args.compare_models, 'r') as f:
            model_configs = json.load(f)
        
        if not args.test_image:
            print("Error: Se requiere --test_image para comparación de modelos")
            return
        
        compare_models(model_configs, args.test_image, input_points)
        return
    
    # Crear inferencer
    inferencer = SAM2Inferencer(args.model_type, args.model_path, args.base_model)
    
    # Prueba en imagen individual
    if args.test_image:
        print(f"Realizando inferencia en: {args.test_image}")
        result = inferencer.predict_single_image(args.test_image, input_points)
        
        # Visualizar resultado
        visualize_prediction(
            result['image'],
            result['pred_masks'],
            np.zeros_like(result['pred_masks']),  # No hay GT
            result['input_points'],
            save_path=f'inference_result_{args.model_type}.png'
        )
        
        print(f"Resultado guardado como: inference_result_{args.model_type}.png")
    
    # Evaluación en dataset
    if args.test_dataset:
        print(f"Evaluando en dataset: {args.test_dataset}")
        metrics, results = inferencer.evaluate_on_dataset(
            args.test_dataset, 
            args.dataset_split, 
            args.max_samples
        )
        
        # Guardar resultados
        results_file = f'evaluation_results_{args.model_type}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'model_type': args.model_type,
                'model_path': args.model_path,
                'dataset': args.test_dataset,
                'split': args.dataset_split
            }, f, indent=2)
        
        print(f"Resultados de evaluación guardados en: {results_file}")


if __name__ == "__main__":
    main()
