"""
Test simple de la implementación sin descargar modelos grandes
"""

import os
import sys
from utils import COCOSegmentationDataset

def test_dataset_loading():
    """Prueba la carga del dataset sin necesidad de modelos"""
    
    print("[TEST] Probando carga de datasets...")
    
    # Verificar que existen los datasets
    cataract_dir = "data/Cataract COCO Segmentation"
    retinopathy_dir = "data/Diabetic-Retinopathy COCO Segmentation"
    
    if not os.path.exists(cataract_dir):
        print(f"[ERROR] No se encuentra {cataract_dir}")
        return False
    
    if not os.path.exists(retinopathy_dir):
        print(f"[ERROR] No se encuentra {retinopathy_dir}")
        return False
    
    print("[OK] Directorios de datasets encontrados")
    
    try:
        # Probar carga de dataset de cataratas
        print("[TEST] Cargando dataset de cataratas...")
        cataract_dataset = COCOSegmentationDataset(
            data_dir=cataract_dir,
            split="train",
            max_samples=3  # Solo 3 muestras para probar
        )
        print(f"[OK] Dataset cataratas: {len(cataract_dataset)} muestras")
        
        # Probar carga de dataset de retinopatía
        print("[TEST] Cargando dataset de retinopatía...")
        retinopathy_dataset = COCOSegmentationDataset(
            data_dir=retinopathy_dir,
            split="train",
            max_samples=3  # Solo 3 muestras para probar
        )
        print(f"[OK] Dataset retinopatía: {len(retinopathy_dataset)} muestras")
        
        # Probar obtener una muestra
        print("[TEST] Obteniendo muestra de ejemplo...")
        sample = cataract_dataset[0]
        print(f"[OK] Muestra obtenida: {sample['image'].size}")
        print(f"[OK] Máscaras encontradas: {len(sample['masks'])}")
        
        print("[SUCCESS] Test de datasets completado exitosamente!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error en test de datasets: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Verifica que todos los archivos necesarios existen"""
    
    print("[TEST] Verificando estructura de archivos...")
    
    required_files = [
        "utils.py",
        "finetune_classic.py", 
        "finetune_lora.py",
        "finetune_qlora.py",
        "main.py",
        "inference.py"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file}")
        else:
            print(f"[ERROR] {file} no encontrado")
            return False
    
    print("[SUCCESS] Todos los archivos necesarios están presentes!")
    return True

def main():
    print("=" * 60)
    print("[TEST] PRUEBA SIMPLE DE IMPLEMENTACIÓN (SIN MODELOS)")
    print("=" * 60)
    
    # Test 1: Estructura de archivos
    if not test_file_structure():
        print("[FAIL] Falló test de estructura de archivos")
        return
    
    # Test 2: Carga de datasets
    if not test_dataset_loading():
        print("[FAIL] Falló test de carga de datasets")
        return
    
    print("\n[SUCCESS] TODOS LOS TESTS BÁSICOS PASARON!")
    print("[INFO] La implementación está lista para usar")
    print("\n[NOTE] Para usar los modelos SAM2, necesitas:")
    print("1. Autenticarte con Hugging Face: huggingface-cli login")
    print("2. O usar un modelo que no requiera autenticación")
    print("\n[NEXT] Próximo paso: autenticarte y ejecutar:")
    print("python main.py --cataract_dir 'data/Cataract COCO Segmentation' \\")
    print("               --retinopathy_dir 'data/Diabetic-Retinopathy COCO Segmentation' \\")
    print("               --run_qlora --qlora_epochs 3 --max_samples 10")

if __name__ == "__main__":
    main()
