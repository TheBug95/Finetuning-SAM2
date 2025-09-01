"""
Test de la estructura de datos del processor SAM2
"""

import torch
import numpy as np
from PIL import Image

def mock_sam2_processor(image, input_points, input_labels, return_tensors="pt"):
    """
    Mock del processor SAM2 basado en la documentación
    """
    # El processor SAM2 devuelve algo como esto:
    return {
        'pixel_values': torch.randn(1, 3, 1024, 1024),  # Imagen procesada
        'input_points': input_points,  # Los puntos tal como se enviaron
        'input_labels': input_labels,  # Las etiquetas tal como se enviaron
    }

def test_processor_format():
    """Test del formato de datos del processor"""
    
    print("[TEST] Probando formato del processor SAM2...")
    
    # Simular imagen
    image = Image.new('RGB', (640, 640), color='red')
    
    # Simular puntos en formato SAM2: [imagen[objeto[punto[x,y]]]]
    formatted_points = [[[320, 320]]]  # Una imagen, un objeto, un punto
    formatted_labels = [[[1]]]  # Una imagen, un objeto, una etiqueta
    
    print(f"Input points format: {formatted_points}")
    print(f"Input labels format: {formatted_labels}")
    
    # Simular processor
    processed = mock_sam2_processor(
        image,
        input_points=formatted_points,
        input_labels=formatted_labels,
        return_tensors="pt"
    )
    
    print(f"\\nProcessor output:")
    for key, value in processed.items():
        print(f"  {key}: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"    shape: {value.shape}")
        elif isinstance(value, list):
            print(f"    list structure: {value}")
    
    # Simular lo que pasa en el training loop
    print(f"\\n[TEST] Simulando training loop...")
    
    # Esto es lo que hacemos en el training:
    input_points = processed['input_points']
    input_labels = processed['input_labels']
    
    print(f"input_points type: {type(input_points)}")
    print(f"input_labels type: {type(input_labels)}")
    
    # Aquí es donde falla - intentamos hacer esto:
    try:
        if isinstance(input_points, list):
            print("input_points is a list")
            points_tensors = [torch.tensor(points) for points in input_points]
            print(f"Converted to tensors: {[type(pt) for pt in points_tensors]}")
        else:
            print("input_points is not a list")
    except Exception as e:
        print(f"Error converting input_points: {e}")
    
    try:
        if isinstance(input_labels, list):
            print("input_labels is a list")
            labels_tensors = [torch.tensor(labels) for labels in input_labels]
            print(f"Converted to tensors: {[type(lt) for lt in labels_tensors]}")
        else:
            print("input_labels is not a list")
    except Exception as e:
        print(f"Error converting input_labels: {e}")

if __name__ == "__main__":
    test_processor_format()
