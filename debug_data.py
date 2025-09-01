"""
Test de debugging para identificar el problema con los datos
"""

import os
import torch
from utils import COCOSegmentationDataset
from torch.utils.data import DataLoader

def debug_data_loading():
    """Prueba la carga de datos sin necesidad del modelo SAM2"""
    
    print("[DEBUG] Iniciando test de debugging de datos...")
    
    # Cargar dataset sin processor (para ver estructura original)
    print("\n1. Cargando dataset SIN processor...")
    dataset_no_proc = COCOSegmentationDataset(
        data_dir="data/Cataract COCO Segmentation",
        split="train",
        processor=None,  # Sin processor
        max_samples=2
    )
    
    if len(dataset_no_proc) > 0:
        sample = dataset_no_proc[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image type: {type(sample['image'])}, size: {sample['image'].size}")
        print(f"Masks type: {type(sample['masks'])}, shape: {sample['masks'].shape}")
        print(f"Labels type: {type(sample['labels'])}, shape: {sample['labels'].shape}")
        print(f"Bboxes type: {type(sample['bboxes'])}, shape: {sample['bboxes'].shape}")
    
    # Ahora intentar con un processor ficticio para ver qué pasa
    print("\n2. Intentando crear processor ficticio...")
    
    try:
        # Crear un processor mock para testing
        from transformers import AutoImageProcessor
        
        # Vamos a usar un processor genérico para testing
        mock_processor = None  # Por ahora dejarlo None para ver estructura
        
        dataset_with_proc = COCOSegmentationDataset(
            data_dir="data/Cataract COCO Segmentation",
            split="train", 
            processor=mock_processor,
            max_samples=2
        )
        
        print(f"Dataset con processor creado: {len(dataset_with_proc)} samples")
        
        # Crear dataloader
        dataloader = DataLoader(dataset_with_proc, batch_size=1, shuffle=False)
        
        print("\n3. Probando dataloader...")
        for i, batch in enumerate(dataloader):
            print(f"\nBatch {i}:")
            for j, sample in enumerate(batch):
                print(f"  Sample {j}:")
                if isinstance(sample, dict):
                    for key, value in sample.items():
                        print(f"    {key}: {type(value)}")
                        if hasattr(value, 'shape'):
                            print(f"      shape: {value.shape}")
                        elif isinstance(value, list):
                            print(f"      list length: {len(value)}")
                            if len(value) > 0:
                                print(f"      first element type: {type(value[0])}")
                else:
                    print(f"    Sample is {type(sample)}")
            
            if i >= 1:  # Solo probar 2 batches
                break
                
    except Exception as e:
        print(f"Error en test con processor: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[DEBUG] Test de debugging completado")

if __name__ == "__main__":
    debug_data_loading()
