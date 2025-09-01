"""
Script principal para ejecutar finetuning de SAM2 con diferentes métodos
Soporta: Classic, LoRA y QLoRA
"""

import argparse
import os
import sys
import subprocess
import time


def run_classic_finetuning(args):
    """Ejecuta finetuning clásico"""
    print("\n" + "="*80)
    print("INICIANDO FINETUNING CLÁSICO DE SAM2")
    print("="*80)
    
    cmd = [
        sys.executable, "finetune_classic.py",
        "--cataract_dir", args.cataract_dir,
        "--retinopathy_dir", args.retinopathy_dir,
        "--model_name", args.model_name,
        "--batch_size", str(args.classic_batch_size),
        "--num_epochs", str(args.classic_epochs),
        "--learning_rate", str(args.classic_lr),
        "--save_dir", args.classic_save_dir
    ]
    
    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"Finetuning clásico completado en {end_time - start_time:.2f} segundos")
    return result.returncode == 0


def run_lora_finetuning(args):
    """Ejecuta finetuning con LoRA"""
    print("\n" + "="*80)
    print("INICIANDO FINETUNING LoRA DE SAM2")
    print("="*80)
    
    cmd = [
        sys.executable, "finetune_lora.py",
        "--cataract_dir", args.cataract_dir,
        "--retinopathy_dir", args.retinopathy_dir,
        "--model_name", args.model_name,
        "--batch_size", str(args.lora_batch_size),
        "--num_epochs", str(args.lora_epochs),
        "--learning_rate", str(args.lora_lr),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--save_dir", args.lora_save_dir
    ]
    
    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])
    
    if args.merge_lora:
        cmd.append("--merge_final")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"Finetuning LoRA completado en {end_time - start_time:.2f} segundos")
    return result.returncode == 0


def run_qlora_finetuning(args):
    """Ejecuta finetuning con QLoRA"""
    print("\n" + "="*80)
    print("INICIANDO FINETUNING QLoRA DE SAM2")
    print("="*80)
    
    cmd = [
        sys.executable, "finetune_qlora.py",
        "--cataract_dir", args.cataract_dir,
        "--retinopathy_dir", args.retinopathy_dir,
        "--model_name", args.model_name,
        "--batch_size", str(args.qlora_batch_size),
        "--num_epochs", str(args.qlora_epochs),
        "--learning_rate", str(args.qlora_lr),
        "--lora_r", str(args.qlora_r),
        "--lora_alpha", str(args.qlora_alpha),
        "--lora_dropout", str(args.qlora_dropout),
        "--save_dir", args.qlora_save_dir
    ]
    
    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"Finetuning QLoRA completado en {end_time - start_time:.2f} segundos")
    return result.returncode == 0


def compare_results(args):
    """Compara resultados de diferentes métodos"""
    print("\n" + "="*80)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*80)
    
    methods = []
    if args.run_classic:
        methods.append(("Clásico", args.classic_save_dir))
    if args.run_lora:
        methods.append(("LoRA", args.lora_save_dir))
    if args.run_qlora:
        methods.append(("QLoRA", args.qlora_save_dir))
    
    print("Métodos ejecutados:")
    for method, save_dir in methods:
        metrics_file = os.path.join(save_dir, f"training_metrics_{method.lower()}.png")
        if os.path.exists(metrics_file):
            print(f"✓ {method}: Métricas guardadas en {metrics_file}")
        else:
            print(f"✗ {method}: No se encontraron métricas")
    
    print("\nRecomendaciones:")
    print("- Clásico: Mejor rendimiento, pero requiere más memoria y tiempo")
    print("- LoRA: Buen balance entre eficiencia y rendimiento")
    print("- QLoRA: Máxima eficiencia de memoria, ideal para GPUs limitadas")


def main():
    parser = argparse.ArgumentParser(
        description='Finetuning de SAM2 para segmentación médica',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Ejecutar solo finetuning clásico:
   python main.py --cataract_dir "data/Cataract COCO Segmentation" 
                  --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation"
                  --run_classic

2. Ejecutar todos los métodos:
   python main.py --cataract_dir "data/Cataract COCO Segmentation"
                  --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation"
                  --run_all

3. Solo LoRA y QLoRA para comparar eficiencia:
   python main.py --cataract_dir "data/Cataract COCO Segmentation"
                  --retinopathy_dir "data/Diabetic-Retinopathy COCO Segmentation"
                  --run_lora --run_qlora
        """)
    
    # Datasets
    parser.add_argument('--cataract_dir', type=str, required=True,
                       help='Directorio del dataset de cataratas')
    parser.add_argument('--retinopathy_dir', type=str, required=True,
                       help='Directorio del dataset de retinopatía diabética')
    
    # Métodos a ejecutar
    parser.add_argument('--run_classic', action='store_true',
                       help='Ejecutar finetuning clásico')
    parser.add_argument('--run_lora', action='store_true',
                       help='Ejecutar finetuning LoRA')
    parser.add_argument('--run_qlora', action='store_true',
                       help='Ejecutar finetuning QLoRA')
    parser.add_argument('--run_all', action='store_true',
                       help='Ejecutar todos los métodos')
    
    # Configuración general
    parser.add_argument('--model_name', type=str, default='facebook/sam2-hiera-base-plus',
                       help='Nombre del modelo SAM2')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Máximo número de muestras por dataset (para testing)')
    
    # Configuración finetuning clásico
    parser.add_argument('--classic_batch_size', type=int, default=2,
                       help='Batch size para finetuning clásico')
    parser.add_argument('--classic_epochs', type=int, default=10,
                       help='Épocas para finetuning clásico')
    parser.add_argument('--classic_lr', type=float, default=1e-5,
                       help='Learning rate para finetuning clásico')
    parser.add_argument('--classic_save_dir', type=str, default='checkpoints_classic',
                       help='Directorio para guardar checkpoints clásicos')
    
    # Configuración LoRA
    parser.add_argument('--lora_batch_size', type=int, default=4,
                       help='Batch size para LoRA')
    parser.add_argument('--lora_epochs', type=int, default=15,
                       help='Épocas para LoRA')
    parser.add_argument('--lora_lr', type=float, default=1e-4,
                       help='Learning rate para LoRA')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='Rango de LoRA')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='Alpha de LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='Dropout de LoRA')
    parser.add_argument('--lora_save_dir', type=str, default='checkpoints_lora',
                       help='Directorio para guardar checkpoints LoRA')
    parser.add_argument('--merge_lora', action='store_true',
                       help='Fusionar LoRA con modelo base al final')
    
    # Configuración QLoRA
    parser.add_argument('--qlora_batch_size', type=int, default=2,
                       help='Batch size para QLoRA')
    parser.add_argument('--qlora_epochs', type=int, default=20,
                       help='Épocas para QLoRA')
    parser.add_argument('--qlora_lr', type=float, default=2e-4,
                       help='Learning rate para QLoRA')
    parser.add_argument('--qlora_r', type=int, default=8,
                       help='Rango de LoRA para QLoRA')
    parser.add_argument('--qlora_alpha', type=int, default=16,
                       help='Alpha de LoRA para QLoRA')
    parser.add_argument('--qlora_dropout', type=float, default=0.05,
                       help='Dropout de LoRA para QLoRA')
    parser.add_argument('--qlora_save_dir', type=str, default='checkpoints_qlora',
                       help='Directorio para guardar checkpoints QLoRA')
    
    args = parser.parse_args()
    
    # Verificar que se especificó al menos un método
    if args.run_all:
        args.run_classic = True
        args.run_lora = True
        args.run_qlora = True
    
    if not (args.run_classic or args.run_lora or args.run_qlora):
        print("Error: Debe especificar al menos un método de finetuning")
        print("Use --run_classic, --run_lora, --run_qlora o --run_all")
        return
    
    # Verificar que existen los directorios de datos
    if not os.path.exists(args.cataract_dir):
        print(f"Error: No se encuentra el directorio de cataratas: {args.cataract_dir}")
        return
    
    if not os.path.exists(args.retinopathy_dir):
        print(f"Error: No se encuentra el directorio de retinopatía: {args.retinopathy_dir}")
        return
    
    # Mostrar configuración
    print("CONFIGURACIÓN DEL EXPERIMENTO")
    print("=" * 50)
    print(f"Modelo: {args.model_name}")
    print(f"Dataset cataratas: {args.cataract_dir}")
    print(f"Dataset retinopatía: {args.retinopathy_dir}")
    if args.max_samples:
        print(f"Límite de muestras: {args.max_samples}")
    
    print("\nMétodos a ejecutar:")
    if args.run_classic:
        print(f"✓ Clásico: {args.classic_epochs} épocas, batch={args.classic_batch_size}, lr={args.classic_lr}")
    if args.run_lora:
        print(f"✓ LoRA: {args.lora_epochs} épocas, batch={args.lora_batch_size}, lr={args.lora_lr}, r={args.lora_r}")
    if args.run_qlora:
        print(f"✓ QLoRA: {args.qlora_epochs} épocas, batch={args.qlora_batch_size}, lr={args.qlora_lr}, r={args.qlora_r}")
    
    # Ejecutar métodos
    total_start_time = time.time()
    results = {}
    
    if args.run_classic:
        success = run_classic_finetuning(args)
        results['classic'] = success
    
    if args.run_lora:
        success = run_lora_finetuning(args)
        results['lora'] = success
    
    if args.run_qlora:
        success = run_qlora_finetuning(args)
        results['qlora'] = success
    
    total_end_time = time.time()
    
    # Mostrar resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"Tiempo total: {total_end_time - total_start_time:.2f} segundos")
    
    for method, success in results.items():
        status = "✓ EXITOSO" if success else "✗ FALLÓ"
        print(f"{method.upper()}: {status}")
    
    # Comparar resultados si se ejecutó más de un método
    if len(results) > 1:
        compare_results(args)
    
    print("\n¡Experimento completado!")


if __name__ == "__main__":
    main()
