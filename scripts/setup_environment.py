#!/usr/bin/env python3
"""
Setup script for Big Mood Detector PAT-L training environment on Windows WSL2
Handles Python 3.12+ installation, CUDA setup, and dependency management
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class EnvironmentSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.venv_name = ".venv-wsl"
        self.python_version = "3.12"
        
    def run_command(self, cmd, check=True, shell=True):
        """Run a command and handle errors gracefully"""
        print(f"🔧 Running: {cmd}")
        try:
            result = subprocess.run(cmd, shell=shell, check=check, 
                                  capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr and check:
                print(f"⚠️  Warning: {result.stderr}")
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            if check:
                raise
            return e
    
    def check_wsl(self):
        """Verify we're running in WSL"""
        if "microsoft" not in platform.release().lower():
            print("❌ This script must be run in WSL2!")
            print("Please run from Windows Terminal in WSL2 Ubuntu")
            sys.exit(1)
        print("✅ Running in WSL2")
    
    def install_python312(self):
        """Install Python 3.12 if not available"""
        print("\n📦 Checking Python 3.12...")
        result = self.run_command("python3.12 --version", check=False)
        
        if result.returncode != 0:
            print("🔄 Installing Python 3.12...")
            commands = [
                "sudo apt update",
                "sudo apt install -y software-properties-common",
                "sudo add-apt-repository -y ppa:deadsnakes/ppa",
                "sudo apt update",
                "sudo apt install -y python3.12 python3.12-venv python3.12-dev python3.12-distutils"
            ]
            for cmd in commands:
                self.run_command(cmd)
        else:
            print("✅ Python 3.12 already installed")
    
    def setup_venv(self):
        """Create a clean virtual environment"""
        print(f"\n🔄 Setting up virtual environment: {self.venv_name}")
        venv_path = self.project_root / self.venv_name
        
        # Remove existing venv if it exists
        if venv_path.exists():
            print(f"🗑️  Removing existing {self.venv_name}...")
            self.run_command(f"rm -rf {venv_path}")
        
        # Create new venv with Python 3.12
        print(f"🔨 Creating new virtual environment with Python 3.12...")
        self.run_command(f"python3.12 -m venv {venv_path}")
        
        # Upgrade pip and setuptools
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
        
        print("📦 Upgrading pip and setuptools...")
        self.run_command(f"{pip_path} install --upgrade pip setuptools wheel")
        
        return venv_path, pip_path, python_path
    
    def install_cuda_pytorch(self, pip_path):
        """Install PyTorch with CUDA 12.8 support"""
        print("\n🚀 Installing PyTorch with CUDA 12.8 support...")
        
        # First install numpy<2.0 to avoid conflicts
        self.run_command(f"{pip_path} install 'numpy<2.0'")
        
        # Install PyTorch with CUDA 12.1 (compatible with 12.8)
        pytorch_cmd = f"{pip_path} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        self.run_command(pytorch_cmd)
    
    def install_project_dependencies(self, pip_path):
        """Install project dependencies"""
        print("\n📦 Installing project dependencies...")
        
        # Install core ML dependencies with numpy constraint
        core_deps = [
            "'numpy<2.0'",
            "transformers",
            "h5py",
            "scikit-learn",
            "pandas",
            "matplotlib",
            "tqdm"
        ]
        
        for dep in core_deps:
            self.run_command(f"{pip_path} install {dep}")
        
        # Install the project in editable mode
        print("\n📦 Installing big-mood-detector in editable mode...")
        self.run_command(f"{pip_path} install -e '{self.project_root}[dev,ml,monitoring]'")
    
    def verify_cuda(self, python_path):
        """Verify CUDA is working"""
        print("\n🔍 Verifying CUDA setup...")
        test_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
"""
        result = self.run_command(f'{python_path} -c "{test_script}"')
        return result.returncode == 0
    
    def create_activation_script(self, venv_path):
        """Create a convenient activation script"""
        script_path = self.project_root / "activate_pat_training.sh"
        content = f"""#!/bin/bash
# Activation script for PAT-L training environment

# Activate virtual environment
source {venv_path}/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="{self.project_root}/src:$PYTHONPATH"
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging

# Verify setup
echo "✅ Environment activated!"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}')"

echo ""
echo "🚀 Ready to train! Run:"
echo "   python scripts/pat_training/train_pat_l_run_now.py"
"""
        
        with open(script_path, 'w') as f:
            f.write(content)
        
        self.run_command(f"chmod +x {script_path}")
        print(f"\n✅ Created activation script: {script_path}")
        return script_path
    
    def run(self):
        """Run the complete setup"""
        print("🏁 Starting Big Mood Detector PAT-L Training Setup")
        print("=" * 60)
        
        try:
            # Change to project directory
            os.chdir(self.project_root)
            
            # Check we're in WSL
            self.check_wsl()
            
            # Install Python 3.12
            self.install_python312()
            
            # Setup virtual environment
            venv_path, pip_path, python_path = self.setup_venv()
            
            # Install PyTorch with CUDA
            self.install_cuda_pytorch(pip_path)
            
            # Install project dependencies
            self.install_project_dependencies(pip_path)
            
            # Verify CUDA
            cuda_ok = self.verify_cuda(python_path)
            
            # Create activation script
            activation_script = self.create_activation_script(venv_path)
            
            print("\n" + "=" * 60)
            print("✅ Setup complete!")
            print("\n📋 Next steps:")
            print(f"1. Activate environment: source {activation_script.name}")
            print("2. Run training: python scripts/pat_training/train_pat_l_run_now.py")
            
            if not cuda_ok:
                print("\n⚠️  Warning: CUDA verification failed.")
                print("Make sure:")
                print("- WSL2 GPU support is enabled")
                print("- NVIDIA drivers are installed on Windows")
                print("- Run 'nvidia-smi' in WSL to verify")
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.run()