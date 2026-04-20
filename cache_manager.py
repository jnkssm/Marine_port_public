# cache_manager.py
import os
import shutil
import subprocess
from pathlib import Path

def get_ollama_cache_size():
    """Get total size of Ollama cache"""
    ollama_dir = Path.home() / '.ollama'
    if ollama_dir.exists():
        total_size = sum(f.stat().st_size for f in ollama_dir.rglob('*') if f.is_file())
        return total_size / (1024**3)  # GB
    return 0

def clear_ollama_cache():
    """Clear various Ollama caches"""
    
    # 1. Clear temporary files
    tmp_dir = Path.home() / '.ollama' / 'tmp'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        print("✓ Cleared temporary files")
    
    # 2. Clear logs
    log_dir = Path.home() / '.ollama' / 'logs'
    if log_dir.exists():
        for log_file in log_dir.glob('*.log'):
            log_file.unlink()
        print("✓ Cleared log files")
    
    # 3. Unload models from memory
    subprocess.run(['pkill', 'ollama'], capture_output=True)
    print("✓ Unloaded models from memory")
    
    # 4. Restart Ollama
    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✓ Restarted Ollama service")

def list_models_with_size():
    """List all models with their disk usage"""
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    print("\n📦 Installed Models:")
    print(result.stdout)
    
    # Show detailed disk usage
    blobs_dir = Path.home() / '.ollama' / 'models' / 'blobs'
    if blobs_dir.exists():
        print("\n💾 Model files disk usage:")
        for blob in blobs_dir.glob('sha256-*'):
            size = blob.stat().st_size / (1024**3)
            print(f"  {blob.name[:20]}... : {size:.2f} GB")

if __name__ == "__main__":
    print(f"Current cache size: {get_ollama_cache_size():.2f} GB")
    
    choice = input("\nOptions:\n1. Clear all caches\n2. List models\n3. Remove a model\nChoice: ")
    
    if choice == '1':
        clear_ollama_cache()
        print(f"New cache size: {get_ollama_cache_size():.2f} GB")
    
    elif choice == '2':
        list_models_with_size()
    
    elif choice == '3':
        model = input("Model name to remove: ")
        subprocess.run(['ollama', 'rm', model])
        print(f"Removed {model}")