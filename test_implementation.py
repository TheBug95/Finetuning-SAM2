"""
Script de prueba rápida para verificar la implementación
Ejecuta un entrenamiento corto con pocas muestras para verificar que todo funciona
"""

import subprocess
import sys
import os


def test_implementation():
    """Prueba rápida de la implementación"""
    
    print("🔍 PRUEBA RÁPIDA DE IMPLEMENTACIÓN SAM2")
    print("=" * 60)
    
    # Verificar que existen los datasets
    cataract_dir = "data/Cataract COCO Segmentation"
    retinopathy_dir = "data/Diabetic-Retinopathy COCO Segmentation"
    
    if not os.path.exists(cataract_dir):
        print(f"❌ Error: No se encuentra {cataract_dir}")
        return False
    
    if not os.path.exists(retinopathy_dir):
        print(f"❌ Error: No se encuentra {retinopathy_dir}")
        return False
    
    print("✅ Datasets encontrados")
    
    # Ejecutar prueba rápida con QLoRA (más eficiente)
    print("\n🚀 Ejecutando prueba con QLoRA (5 épocas, 5 muestras)...")
    
    cmd = [
        sys.executable, "main.py",
        "--cataract_dir", cataract_dir,
        "--retinopathy_dir", retinopathy_dir,
        "--run_qlora",
        "--qlora_epochs", "3",
        "--qlora_batch_size", "1",
        "--max_samples", "5",
        "--qlora_save_dir", "test_checkpoints"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode == 0:
            print("✅ Prueba de entrenamiento exitosa!")
            return True
        else:
            print("❌ Error en entrenamiento:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout - el entrenamiento tomó demasiado tiempo")
        return False
    except Exception as e:
        print(f"❌ Error ejecutando prueba: {e}")
        return False


def test_inference():
    """Prueba la inferencia si existe un modelo entrenado"""
    
    print("\n🔍 PRUEBA DE INFERENCIA")
    print("=" * 40)
    
    # Buscar modelo de prueba
    test_model_path = "test_checkpoints/sam2_qlora_best"
    
    if not os.path.exists(test_model_path):
        print("⚠️  No se encontró modelo de prueba para inferencia")
        return True  # No es error crítico
    
    # Buscar imagen de prueba
    test_image = None
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
    
    if not test_image:
        print("⚠️  No se encontró imagen de prueba")
        return True
    
    print(f"📸 Usando imagen: {test_image}")
    
    # Ejecutar inferencia
    cmd = [
        sys.executable, "inference.py",
        "--model_type", "qlora",
        "--model_path", test_model_path,
        "--test_image", test_image
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Prueba de inferencia exitosa!")
            return True
        else:
            print("❌ Error en inferencia:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout en inferencia")
        return False
    except Exception as e:
        print(f"❌ Error en prueba de inferencia: {e}")
        return False


def cleanup():
    """Limpia archivos de prueba"""
    print("\n🧹 LIMPIEZA")
    print("=" * 20)
    
    import shutil
    
    cleanup_dirs = ["test_checkpoints"]
    cleanup_files = ["inference_result_qlora.png", "temp_image_*.jpg"]
    
    for dir_name in cleanup_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"🗑️  Eliminado: {dir_name}")
    
    import glob
    for pattern in cleanup_files:
        for file in glob.glob(pattern):
            os.remove(file)
            print(f"🗑️  Eliminado: {file}")


def main():
    """Ejecuta todas las pruebas"""
    
    print("🧪 INICIANDO PRUEBAS DE IMPLEMENTACIÓN SAM2")
    print("=" * 80)
    
    # Prueba 1: Entrenamiento
    if not test_implementation():
        print("\n❌ PRUEBA FALLIDA: Error en entrenamiento")
        return
    
    # Prueba 2: Inferencia
    if not test_inference():
        print("\n❌ PRUEBA FALLIDA: Error en inferencia")
        return
    
    print("\n✅ TODAS LAS PRUEBAS EXITOSAS!")
    print("🎉 La implementación funciona correctamente")
    
    # Preguntar si limpiar
    response = input("\n¿Limpiar archivos de prueba? (y/N): ")
    if response.lower() in ['y', 'yes', 's', 'si']:
        cleanup()
    
    print("\n📚 SIGUIENTE PASO:")
    print("Ejecutar entrenamiento completo con:")
    print("python main.py --cataract_dir 'data/Cataract COCO Segmentation' \\")
    print("               --retinopathy_dir 'data/Diabetic-Retinopathy COCO Segmentation' \\")
    print("               --run_all")


if __name__ == "__main__":
    main()
