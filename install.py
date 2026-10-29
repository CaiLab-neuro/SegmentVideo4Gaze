#!/usr/bin/env python3
"""
SAM2 Video UI Setup Script
=========================

Installs dependencies and downloads models for SAM2 Video UI.

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import platform
import urllib.request
import shutil
from pathlib import Path

def print_header():
    print("=" * 50)
    print("SAM2 Video UI Setup")
    print("=" * 50)
    print()

def check_python():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info

    # Handle Python 2
    if version.major == 2:
        if version.minor < 7:
            print(f"ERROR: Python {version.major}.{version.minor} detected.")
            print("   Python 2.6 and below are not supported.")
            print("   Please upgrade to Python 3.12 or higher.")
            return False
        else:
            # Python 2.7
            print(f"WARNING: Python {version.major}.{version.minor} (legacy) detected.")
            print("   Python 2.7 is end-of-life and not recommended.")
            print("   SAM2/SAM3 require Python 3.10 or higher.")
            choice = input("\nContinue with Python 2.7? (y/n) [default: n]: ").strip().lower()
            if choice != 'y':
                print("Please install Python 3.12 or higher and re-run setup.")
                return False

    # Handle Python 3
    elif version.major == 3:
        if version.minor < 10:
            print(f"ERROR: Python {version.major}.{version.minor} detected.")
            print("   SAM2 requires Python 3.10 or higher.")
            print("   Recommended: Python 3.12 or higher for full feature support.")
            return False
        elif version.minor < 12:
            # Python 3.10 or 3.11
            print(f"WARNING: Python {version.major}.{version.minor} detected.")
            print("   SAM2 will work, but SAM3 (text-based prompting) requires Python 3.12+")
            choice = input("\nContinue with Python 3.{}? (y/n) [default: y]: ".format(version.minor)).strip().lower()
            if choice and choice != 'y':
                return False

    print(f"OK: Python {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 12:
        print("   Full feature support (SAM2 + SAM3)")

    return True

def install_packages():
    """Install required packages"""
    print("\nInstalling Python packages...")

    # Detect CUDA availability and version
    cuda_available = False
    cuda_version = "cpu"
    print("\nDetecting CUDA...")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("NVIDIA GPU detected")
            # Try to detect CUDA version from nvidia-smi output
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    cuda_str = line.split(':')[-1].strip().split()[0]
                    cuda_major, cuda_minor = cuda_str.split('.')[:2]
                    cuda_version = f"cu{cuda_major}{cuda_minor}"
                    cuda_available = True
                    print(f"Detected CUDA {cuda_str} -> {cuda_version}")
                    break
    except Exception as e:
        pass

    if not cuda_available:
        print("WARNING: CUDA not detected. Installing CPU-only PyTorch.")
        print("For GPU acceleration:")
        print("  1. Install CUDA from: https://developer.nvidia.com/cuda-downloads")
        print("  2. Re-run setup.py to install CUDA-compatible PyTorch")
    else:
        print(f"Installing CUDA-compatible PyTorch ({cuda_version})")

    # Base packages for Sam2UI (compatible with SAM2 requirements)
    packages = [
        "torchvision>=0.20.1",
        "opencv-python>=4.5.0",
        "numpy>=1.24.4",
        "pandas>=1.3.0",
        "Pillow>=9.4.0",
        "omegaconf>=2.1.0",
        "hydra-core>=1.3.2",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
        "tqdm>=4.66.1",
        "pycocotools>=2.0.2"
    ]

    # Install PyTorch first with appropriate index URL
    print("\nInstalling PyTorch and torchvision...")
    try:
        if cuda_available:
            # Install CUDA-compatible PyTorch
            index_url = f"https://download.pytorch.org/whl/{cuda_version}"
            print(f"Using PyTorch index: {index_url}")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch>=2.5.1", "torchvision>=0.20.1",
                "--index-url", index_url
            ], check=True, capture_output=True)
        else:
            # Install CPU-only PyTorch
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch>=2.5.1", "torchvision>=0.20.1"
            ], check=True, capture_output=True)
        print("OK: PyTorch installed")
    except subprocess.CalledProcessError as e:
        print("FAILED: PyTorch installation failed")
        return False

    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package],
                          check=True, capture_output=True)
            print(f"OK: {package}")
        except subprocess.CalledProcessError:
            print(f"FAILED: {package}")
            return False

    return True

def create_directories():
    """Create required directories"""
    print("\nCreating directories...")

    dirs = ["demo/data"]

    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"OK: {directory}")

def clone_sam2_repository():
    """Clone SAM2 repository into Sam2UI/sam_models/sam2/"""
    print("\nCloning SAM2 repository...")

    sam2_dir = Path("sam_models/sam2")

    # Check if already exists with valid structure
    if sam2_dir.exists():
        # Check if it looks like a valid SAM2 installation
        sam2_package_dir = sam2_dir / "sam2"
        if sam2_package_dir.exists() and (sam2_package_dir / "__init__.py").exists():
            print("OK: SAM2 already installed (skipping clone)")
            print("   To reinstall SAM2, delete the 'sam_models/sam2' directory and re-run setup.py")
            return True
        else:
            # Directory exists but doesn't look like valid SAM2
            print("WARNING: 'sam2' directory exists but appears incomplete")
            print("\nCurrent contents of 'sam2' directory:")
            try:
                contents = list(sam2_dir.iterdir())
                if contents:
                    for item in sorted(contents)[:10]:  # Show first 10 items
                        print(f"  - {item.name}")
                    if len(contents) > 10:
                        print(f"  ... and {len(contents) - 10} more items")
                else:
                    print("  (empty directory)")
            except Exception as e:
                print(f"  (could not list contents: {e})")

            user_input = input("\nRemove and re-clone SAM2? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Keeping existing directory (setup may fail)")
                return True
            print("Removing directory...")
            shutil.rmtree(sam2_dir)

    # Clone repository
    try:
        # Create sam_models directory if it doesn't exist
        Path("sam_models").mkdir(parents=True, exist_ok=True)

        subprocess.run([
            "git", "clone",
            "https://github.com/facebookresearch/sam2.git",
            "sam_models/sam2"
        ], check=True, capture_output=True, text=True)
        print("OK: SAM2 repository cloned")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: Could not clone SAM2 repository")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ERROR: git not found. Please install git first.")
        print("  Download from: https://git-scm.com/downloads")
        return False

def install_sam2_package():
    """Install SAM2 as editable package and compile CUDA extension"""
    print("\nInstalling SAM2 package...")

    sam2_dir = Path("sam_models/sam2")
    if not sam2_dir.exists():
        print("ERROR: SAM2 directory not found")
        return False

    # Check if already importable — skip pip install if so
    already_installed = subprocess.run(
        [sys.executable, "-c", "from sam2.build_sam import build_sam2_video_predictor"],
        capture_output=True, text=True
    ).returncode == 0

    if already_installed:
        print("OK: SAM2 package already installed (skipping pip install)")
    else:
        try:
            # Install SAM2 in editable mode (verbose so CUDA build errors are visible)
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-v", "-e", "./sam_models/sam2"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FAILED: Could not install SAM2 package")
                print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
                return False
            print("OK: SAM2 package installed")
        except subprocess.CalledProcessError:
            print(f"FAILED: Could not install SAM2 package")
            return False

    # Verify that the CUDA extension (_C) was compiled
    check = subprocess.run([
        sys.executable, "-c",
        "from sam2 import _C; print('OK')"
    ], capture_output=True, text=True, cwd=str(sam2_dir))

    if check.returncode == 0:
        print("OK: SAM2 CUDA extension (_C) compiled successfully")
    else:
        print("WARNING: SAM2 CUDA extension (_C) was not compiled.")
        print("  This is non-fatal — segmentation still works, but some post-processing")
        print("  refinements will be skipped.")
        print("  To compile manually later:")
        print(f"    cd {sam2_dir.resolve()} && SAM2_BUILD_CUDA=1 python setup.py build_ext --inplace")
        print()
        # Attempt explicit build
        print("Attempting explicit build of CUDA extension...")
        build_result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            capture_output=True, text=True, cwd=str(sam2_dir.resolve()),
            env={**os.environ, "SAM2_BUILD_CUDA": "1"}
        )
        if build_result.returncode == 0:
            # Verify again
            recheck = subprocess.run([
                sys.executable, "-c", "from sam2 import _C; print('OK')"
            ], capture_output=True, text=True, cwd=str(sam2_dir))
            if recheck.returncode == 0:
                print("OK: CUDA extension compiled on retry")
            else:
                print("WARNING: Extension still unavailable after build attempt.")
                print(build_result.stderr[-1000:] if build_result.stderr else "")
        else:
            print("WARNING: Explicit build failed. Check CUDA toolkit installation.")
            if build_result.stderr:
                print(build_result.stderr[-1000:])

    return True

def select_models_to_download():
    """Interactive model selection for download"""
    print("\n" + "=" * 50)
    print("SAM2 Model Selection")
    print("=" * 50)

    # Define all available models with sizes
    models_sam21 = [
        ("sam2.1_hiera_tiny.pt", 156, "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"),
        ("sam2.1_hiera_small.pt", 184, "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"),
        ("sam2.1_hiera_base_plus.pt", 324, "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"),
        ("sam2.1_hiera_large.pt", 898, "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"),
    ]

    models_sam2 = [
        ("sam2_hiera_tiny.pt", 156, "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"),
        ("sam2_hiera_small.pt", 184, "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"),
        ("sam2_hiera_base_plus.pt", 324, "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"),
        ("sam2_hiera_large.pt", 898, "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"),
    ]

    print("\nAvailable models:")
    print("\nSAM2.1 Models (Recommended - Released Sept 2024):")
    print("  1. Tiny   (156 MB) - Fastest, lowest quality")
    print("  2. Small  (184 MB) - Good balance")
    print("  3. Base+  (324 MB) - High quality")
    print("  4. Large  (898 MB) - Best quality, slowest")
    print("\nSAM2 Models (Legacy - Released July 2024):")
    print("  5. Tiny   (156 MB)")
    print("  6. Small  (184 MB)")
    print("  7. Base+  (324 MB)")
    print("  8. Large  (898 MB)")
    print("\nOptions:")
    print("  A. Download all SAM2.1 models (~1.5 GB)")
    print("  B. Download recommended (SAM2.1 Small + Base+) (~500 MB)")
    print("  C. Custom selection")
    print("\nYou can also specify:")
    print("  - Single model: 2")
    print("  - Multiple models: 2,3,4")
    print("  - Range: 2-4")

    choice = input("\nEnter your choice (1-8, A, B, C, or range) [default: B]: ").strip().upper()
    if not choice:
        choice = 'B'

    selected = []
    all_models = models_sam21 + models_sam2

    if choice == 'A':
        selected = models_sam21
        print(f"Selected: All SAM2.1 models")
    elif choice == 'B':
        selected = [models_sam21[1], models_sam21[2]]  # Small and Base+
        print(f"Selected: SAM2.1 Small and Base+")
    elif choice == 'C':
        print("\nEnter model numbers to download (comma-separated, e.g., 2,3):")
        selections = input("Models: ").strip().split(',')
        for sel in selections:
            try:
                idx = int(sel.strip()) - 1
                if 0 <= idx < len(all_models):
                    selected.append(all_models[idx])
            except ValueError:
                continue
    elif choice.isdigit() and 1 <= int(choice) <= 8:
        idx = int(choice) - 1
        selected = [all_models[idx]]
        model_name = all_models[idx][0]
        print(f"Selected: {model_name}")
    elif '-' in choice:
        # Handle range syntax (e.g., "2-4")
        try:
            parts = choice.split('-')
            if len(parts) == 2:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                if 1 <= start <= end <= 8:
                    for i in range(start, end + 1):
                        selected.append(all_models[i - 1])
                    model_names = [m[0] for m in selected]
                    print(f"Selected: {', '.join(model_names)}")
                else:
                    print("Invalid range. Downloading recommended models (Small + Base+)")
                    selected = [models_sam21[1], models_sam21[2]]
            else:
                print("Invalid range format. Downloading recommended models (Small + Base+)")
                selected = [models_sam21[1], models_sam21[2]]
        except ValueError:
            print("Invalid range. Downloading recommended models (Small + Base+)")
            selected = [models_sam21[1], models_sam21[2]]
    elif ',' in choice:
        # Handle comma-separated list (e.g., "2,3,4")
        try:
            selections = choice.split(',')
            for sel in selections:
                idx = int(sel.strip()) - 1
                if 0 <= idx < len(all_models):
                    selected.append(all_models[idx])
            if selected:
                model_names = [m[0] for m in selected]
                print(f"Selected: {', '.join(model_names)}")
            else:
                print("No valid models selected. Downloading recommended models (Small + Base+)")
                selected = [models_sam21[1], models_sam21[2]]
        except ValueError:
            print("Invalid input. Downloading recommended models (Small + Base+)")
            selected = [models_sam21[1], models_sam21[2]]
    else:
        print("Invalid choice. Downloading recommended models (Small + Base+)")
        selected = [models_sam21[1], models_sam21[2]]

    return selected

def download_checkpoints(selected_models):
    """Download selected model checkpoints with progress tracking"""
    print("\nDownloading model checkpoints...")

    checkpoint_dir = Path("sam_models/sam2/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_size = sum(size for _, size, _ in selected_models)
    print(f"\nTotal download size: {total_size} MB")

    for filename, size, url in selected_models:
        filepath = checkpoint_dir / filename

        if filepath.exists():
            print(f"OK: {filename} (already exists)")
            continue

        print(f"\nDownloading {filename} ({size} MB)...")
        try:
            # Download with progress tracking
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)

            urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
            print(f"\nOK: {filename}")
        except Exception as e:
            print(f"\nFAILED: {filename}: {e}")
            return False

    return True

def detect_environment_info():
    """Detect current Python environment type and activation method"""
    env_info = {
        'type': 'system',  # 'conda', 'venv', 'uv', or 'system'
        'python_path': sys.executable,
        'conda_path': None,
        'conda_env_name': None,
        'venv_path': None,
    }

    # Check for conda
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_default_env = os.environ.get('CONDA_DEFAULT_ENV')

    if conda_prefix and conda_default_env:
        env_info['type'] = 'conda'
        env_info['conda_env_name'] = conda_default_env
        # Find conda.sh or conda installation root
        # CONDA_PREFIX points to the environment, go up to find conda installation
        conda_exe = shutil.which('conda')
        if conda_exe:
            # Get conda base from conda info
            try:
                result = subprocess.run(
                    ['conda', 'info', '--base'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    env_info['conda_path'] = result.stdout.strip()
            except:
                # Fallback: try to infer from CONDA_PREFIX
                # Typically envs are in <conda_base>/envs/<env_name>
                if '/envs/' in conda_prefix:
                    env_info['conda_path'] = conda_prefix.split('/envs/')[0]
                elif '\\envs\\' in conda_prefix:
                    env_info['conda_path'] = conda_prefix.split('\\envs\\')[0]

    # Check for venv (including uv-managed venvs)
    elif hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # Check if it's uv-managed
        if os.path.exists(os.path.join(sys.prefix, 'pyvenv.cfg')):
            with open(os.path.join(sys.prefix, 'pyvenv.cfg'), 'r') as f:
                content = f.read()
                if 'uv' in content.lower():
                    env_info['type'] = 'uv'
                else:
                    env_info['type'] = 'venv'
        else:
            env_info['type'] = 'venv'
        env_info['venv_path'] = sys.prefix

    return env_info

def create_launcher(script_name="sam2_ui.py", output_basename="run", ui_label="SAM2 Video UI"):
    """Create launcher script that activates the detected environment"""
    print(f"\nCreating launcher for {script_name}...")

    # Detect current environment
    env_info = detect_environment_info()
    print(f"Detected environment type: {env_info['type']}")
    if env_info['type'] == 'conda':
        print(f"  Conda base: {env_info['conda_path']}")
        print(f"  Environment: {env_info['conda_env_name']}")
    elif env_info['type'] in ['venv', 'uv']:
        print(f"  Virtual environment: {env_info['venv_path']}")

    if platform.system() == "Windows":
        # Generate Windows batch file
        if env_info['type'] == 'conda':
            launcher_content = f"""@echo off
echo Starting {ui_label}...
echo.

:: Activate conda environment (detected during setup)
set CONDA_BASE={env_info['conda_path'] or 'NOT_DETECTED'}
set CONDA_ENV={env_info['conda_env_name'] or 'sam'}

if exist "%CONDA_BASE%\\Scripts\\activate.bat" (
    echo Activating conda environment '%CONDA_ENV%' from %CONDA_BASE%...
    call "%CONDA_BASE%\\Scripts\\activate.bat" %CONDA_ENV%
    python {script_name}
    pause
    exit /b
)

:: Fallback: try conda run
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using 'conda run' to launch with environment '%CONDA_ENV%'...
    conda run -n %CONDA_ENV% python {script_name}
    pause
    exit /b
)

:: Final fallback
echo.
echo WARNING: Could not activate conda environment '%CONDA_ENV%'
echo Please activate manually with: conda activate %CONDA_ENV%
echo.
pause
python {script_name}
pause
"""
        elif env_info['type'] in ['venv', 'uv']:
            venv_activate = f"{env_info['venv_path']}\\Scripts\\activate.bat"
            launcher_content = f"""@echo off
echo Starting {ui_label}...
echo.

:: Activate virtual environment (detected during setup)
set VENV_PATH={env_info['venv_path']}

if exist "%VENV_PATH%\\Scripts\\activate.bat" (
    echo Activating virtual environment from %VENV_PATH%...
    call "%VENV_PATH%\\Scripts\\activate.bat"
    python {script_name}
    pause
    exit /b
)

:: Fallback: use absolute python path
echo Using detected Python: {env_info['python_path']}
"{env_info['python_path']}" {script_name}
pause
"""
        else:
            # System Python
            launcher_content = f"""@echo off
echo Starting {ui_label}...
echo.
echo Using system Python: {env_info['python_path']}
"{env_info['python_path']}" {script_name}
pause
"""

        bat_path = f"{output_basename}.bat"
        with open(bat_path, "w") as f:
            f.write(launcher_content)
        print(f"OK: {bat_path} created")

    else:  # Linux/Mac
        if env_info['type'] == 'conda':
            conda_base = env_info['conda_path'] or '/opt/miniconda3'
            conda_env = env_info['conda_env_name'] or 'sam'
            launcher_content = f"""#!/bin/bash
echo "Starting {ui_label}..."
echo ""

# Activate conda environment (detected during setup)
CONDA_BASE="{conda_base}"
CONDA_ENV="{conda_env}"

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Activating conda environment '$CONDA_ENV' from $CONDA_BASE..."
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    python3 {script_name}
    exit $?
fi

# Fallback: try conda run if conda is in PATH
if command -v conda &> /dev/null; then
    echo "Using 'conda run' to launch with environment '$CONDA_ENV'..."
    conda run -n "$CONDA_ENV" python3 {script_name}
    exit $?
fi

# Final fallback
echo ""
echo "WARNING: Could not activate conda environment '$CONDA_ENV'"
echo "Please activate manually with: conda activate $CONDA_ENV"
echo ""
read -p "Press Enter to try anyway..."
python3 {script_name}
"""
        elif env_info['type'] in ['venv', 'uv']:
            venv_activate = f"{env_info['venv_path']}/bin/activate"
            launcher_content = f"""#!/bin/bash
echo "Starting {ui_label}..."
echo ""

# Activate virtual environment (detected during setup)
VENV_PATH="{env_info['venv_path']}"

if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment from $VENV_PATH..."
    source "$VENV_PATH/bin/activate"
    python3 {script_name}
    exit $?
fi

# Fallback: use absolute python path
echo "Using detected Python: {env_info['python_path']}"
"{env_info['python_path']}" {script_name}
"""
        else:
            # System Python
            launcher_content = f"""#!/bin/bash
echo "Starting {ui_label}..."
echo ""
echo "Using system Python: {env_info['python_path']}"
"{env_info['python_path']}" {script_name}
"""

        sh_path = f"{output_basename}.sh"
        with open(sh_path, "w") as f:
            f.write(launcher_content)
        os.chmod(sh_path, 0o755)
        print(f"OK: {sh_path} created")

def prompt_sam3_installation():
    """Ask user if they want to install SAM3 (optional)"""
    print("\n" + "=" * 50)
    print("SAM3 Installation (Optional)")
    print("=" * 50)
    print("\nSAM3 provides text-based prompting capabilities.")
    print("Requirements:")
    print("  - Python 3.12+")
    print("  - PyTorch 2.7+")
    print("  - HuggingFace account with model access")
    print("  - ~848M parameter model")
    print("\nNote: SAM3 and SAM3.1 share the same codebase — only the checkpoint differs.")
    print("  SAM3   checkpoint: facebook/sam3 on HuggingFace")
    print("  SAM3.1 checkpoint: facebook/sam3.1 on HuggingFace")

    choice = input("\nInstall SAM3? (y/n) [default: n]: ").strip().lower()
    if choice != 'y':
        print("Skipping SAM3 installation")
        return True

    print("\nWhich checkpoint(s) do you want instructions for?")
    print("  1. SAM3   - Original release")
    print("  2. SAM3.1 - Latest release")
    print("  3. Both")
    ckpt_choice = input("Choice (1/2/3) [default: 2]: ").strip()
    if ckpt_choice == '1':
        versions = ["3"]
    elif ckpt_choice == '3':
        versions = ["3", "3.1"]
    else:
        versions = ["3.1"]

    return install_sam3(versions=versions)

def _display_sam3_checkpoint_instructions(version="3"):
    """Display SAM3/SAM3.1 checkpoint download instructions"""
    hf_repo = f"facebook/sam3.1" if version == "3.1" else "facebook/sam3"
    checkpoint_file = "sam3.1_multiplex.pt" if version == "3.1" else "sam3.pt"
    label = f"SAM{version}"

    print("\n" + "=" * 50)
    print(f"{label} Checkpoint Download Instructions")
    print("=" * 50)
    print(f"\n1. REQUEST ACCESS TO CHECKPOINTS:")
    print(f"   Visit: https://huggingface.co/{hf_repo}")
    print(f"   Click 'Request Access' and wait for approval")
    print(f"\n2. AUTHENTICATE WITH HUGGINGFACE:")
    print(f"   a. Generate token at: https://huggingface.co/settings/tokens")
    print(f"   b. Run: huggingface-cli login")
    print(f"   c. Paste your token when prompted")
    print(f"\n3. DOWNLOAD CHECKPOINT (choose one method):")
    print(f"   ")
    print(f"   Method 1 - huggingface-cli (recommended):")
    print(f"   huggingface-cli download {hf_repo} --local-dir sam_models/sam3/checkpoints")
    print(f"   ")
    print(f"   Method 2 - Python snapshot_download:")
    print(f"   python -c \"from huggingface_hub import snapshot_download; \\")
    print(f"       snapshot_download(repo_id='{hf_repo}', local_dir='sam_models/sam3/checkpoints')\"")
    print(f"   ")
    print(f"   Method 3 - Single file:")
    print(f"   huggingface-cli download {hf_repo} {checkpoint_file} \\")
    print(f"       --local-dir sam_models/sam3/checkpoints")
    print(f"\n4. VERIFY INSTALLATION:")
    print(f"   Run: python -c 'from sam3.model_builder import build_sam3_video_predictor; print(\"{label} OK\")'")
    print(f"\nNote: {label} will not work until checkpoint is downloaded")
    print(f"      Checkpoint will be saved to: sam_models/sam3/checkpoints/")

def install_sam3(versions=None):
    """Clone and install SAM3 repository, then show checkpoint instructions for requested versions."""
    if versions is None:
        versions = ["3.1"]

    sam3_dir = Path("sam_models/sam3")
    package_installed = False

    if sam3_dir.exists():
        print("\nSAM3 repository already present — skipping clone/install")
        package_installed = True
    else:
        print("\nCloning SAM3 repository...")
        try:
            # Check Python version
            if sys.version_info < (3, 12):
                print("ERROR: SAM3 requires Python 3.12+")
                print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
                print("Please create a Python 3.12+ environment for SAM3")
                return False

            # Check PyTorch version
            try:
                import torch
                torch_version = torch.__version__.split('+')[0]
                major, minor = map(int, torch_version.split('.')[:2])
                if major < 2 or (major == 2 and minor < 7):
                    print("ERROR: SAM3 requires PyTorch 2.7+")
                    print(f"Current version: {torch_version}")
                    print("Please upgrade PyTorch:")
                    print("  pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126")
                    return False

                if torch.cuda.is_available():
                    cuda_version = torch.version.cuda
                    if cuda_version:
                        cuda_major = int(cuda_version.split('.')[0])
                        cuda_minor = int(cuda_version.split('.')[1])
                        if cuda_major < 12 or (cuda_major == 12 and cuda_minor < 6):
                            print(f"WARNING: SAM3 requires CUDA 12.6+ (current: {cuda_version})")
                            print("SAM3 may not work correctly with older CUDA versions")
                else:
                    print("WARNING: CUDA not available. SAM3 requires CUDA 12.6+ for optimal performance")
            except ImportError:
                print("WARNING: PyTorch not installed. Please install PyTorch 2.7+ first")
                return False

            Path("sam_models").mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "git", "clone",
                "https://github.com/facebookresearch/sam3.git",
                "sam_models/sam3"
            ], check=True, capture_output=True)
            print("OK: SAM3 repository cloned")

            subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", "./sam_models/sam3"
            ], check=True, capture_output=True)
            print("OK: SAM3 package installed")
            package_installed = True

        except subprocess.CalledProcessError:
            print("FAILED: Could not clone/install SAM3")
            return False

    # Always ensure additional dependencies are installed
    print("\nInstalling/verifying SAM3 dependencies...")
    additional_deps = [
        "einops",
        "huggingface-hub",
        "decord",
        "scikit-learn",
        "ftfy==6.1.1",
        "regex",
        "iopath>=0.1.10",
        "timm>=1.0.17",
        # Imported by SAM3 at import time (sam3_video_predictor, edt)
        "psutil",
        "triton; sys_platform == 'linux'",
    ]
    for dep in additional_deps:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], check=True, capture_output=True)
            print(f"OK: {dep}")
        except subprocess.CalledProcessError:
            print(f"WARNING: Failed to install {dep} — install manually: pip install {dep}")

    # Create checkpoint directory
    checkpoint_dir = Path("sam_models/sam3/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"OK: Checkpoint directory: {checkpoint_dir}")

    # Show instructions for each requested checkpoint version
    for version in versions:
        checkpoint_file = "sam3.1_multiplex.pt" if version == "3.1" else "sam3.pt"
        sam3_checkpoint = checkpoint_dir / checkpoint_file
        if sam3_checkpoint.exists():
            print(f"\nOK: SAM{version} checkpoint already present ({checkpoint_file})")
        else:
            _display_sam3_checkpoint_instructions(version=version)

    return True

def verify_setup():
    """Verify installation"""
    print("\nVerifying setup...")

    try:
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        import tkinter as tk
        print("OK: All packages imported successfully")

        # Check SAM2 installation (must use subprocess: editable installs via
        # 'pip install -e' add .pth entries only visible to new interpreters)
        sam2_check = subprocess.run([
            sys.executable, "-c",
            "from sam2.build_sam import build_sam2_video_predictor; print('OK')"
        ], capture_output=True, text=True, timeout=30)
        if sam2_check.returncode == 0:
            print("OK: SAM2 package imported successfully")
        else:
            print(f"ERROR: Could not import SAM2: {sam2_check.stderr.strip()}")
            return False

        # Check CUDA extension
        c_ext_check = subprocess.run([
            sys.executable, "-c", "from sam2 import _C; print('OK')"
        ], capture_output=True, text=True, timeout=10)
        if c_ext_check.returncode == 0:
            print("OK: SAM2 CUDA extension (_C) compiled and available")
        else:
            print("WARNING: SAM2 CUDA extension (_C) not compiled (non-fatal)")
            print("         Recompile if needed: cd sam_models/sam2 && python setup.py build_ext --inplace")

        if torch.cuda.is_available():
            print(f"OK: CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA not available - will use CPU")

        # Check files and directories
        required_paths = [
            "sam2_ui.py",
            "sam_models/sam2/",
            "sam_models/sam2/sam2/",
            "sam_models/sam2/checkpoints/"
        ]

        for path in required_paths:
            if Path(path).exists():
                print(f"OK: {path}")
            else:
                print(f"MISSING: {path}")
                return False

        # Check at least one checkpoint exists
        checkpoint_dir = Path("sam_models/sam2/checkpoints")
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            print(f"OK: Found {len(checkpoints)} checkpoint(s)")
        else:
            print("WARNING: No checkpoints found")
            return False

        # Optional: Check SAM3
        if Path("sam_models/sam3").exists():
            print("OK: SAM3 installed (optional)")
            # Check SAM3 dependencies (subprocess for same reason as SAM2 above)
            sam3_dep_checks = [
                ("einops", "import einops"),
                ("scikit-learn", "import sklearn"),
                ("timm", "import timm"),
                ("ftfy", "import ftfy"),
                ("iopath", "import iopath"),
                ("decord", "import decord"),
            ]
            for dep_name, import_stmt in sam3_dep_checks:
                dep_check = subprocess.run([
                    sys.executable, "-c", f"{import_stmt}; print('OK')"
                ], capture_output=True, text=True, timeout=10)
                if dep_check.returncode == 0:
                    print(f"OK: SAM3 dependency ({dep_name}) available")
                else:
                    print(f"WARNING: {dep_name} not found - SAM3 may not work")
                    print(f"  Install with: pip install {dep_name}")

        return True

    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False

def main():
    """Main setup function"""
    print_header()

    # Step 1: Check Python version
    if not check_python():
        return False

    # Step 2: Install base packages
    if not install_packages():
        print("\nERROR: Package installation failed")
        return False

    # Step 3: Create directories
    create_directories()

    # Step 4: Clone SAM2 repository
    if not clone_sam2_repository():
        print("\nERROR: SAM2 repository cloning failed")
        return False

    # Step 5: Install SAM2 package
    if not install_sam2_package():
        print("\nERROR: SAM2 package installation failed")
        return False

    # Step 6: Select and download model checkpoints
    selected_models = select_models_to_download()
    if not download_checkpoints(selected_models):
        print("\nERROR: Model download failed")
        return False

    # Step 7: Optional SAM3 installation
    if not prompt_sam3_installation():
        print("\nWARNING: SAM3 installation failed (continuing anyway)")

    # Step 8: Create launcher(s)
    create_launcher(script_name="sam2_ui.py", output_basename="run", ui_label="SAM2 Video UI")
    if Path("sam_models/sam3").exists():
        create_launcher(script_name="sam3_ui.py", output_basename="run_sam3", ui_label="SAM3 Video UI")

    # Step 9: Verify setup
    if not verify_setup():
        print("\nERROR: Setup verification failed")
        return False

    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print("\nTo run the SAM2 Video UI:")
    if platform.system() == "Windows":
        print("  1. Double-click run.bat (environment auto-activates)")
        print("  2. OR manually: conda activate sam && python sam2_ui.py")
    else:
        print("  1. Double-click or run: ./run.sh (environment auto-activates)")
        print("  2. OR manually: conda activate sam && python3 sam2_ui.py")

    if Path("sam_models/sam3").exists():
        print("\nTo run the SAM3 Video UI:")
        if platform.system() == "Windows":
            print("  1. Double-click run_sam3.bat (environment auto-activates)")
            print("  2. OR manually: conda activate sam && python sam3_ui.py")
        else:
            print("  1. Double-click or run: ./run_sam3.sh (environment auto-activates)")
            print("  2. OR manually: conda activate sam && python3 sam3_ui.py")

    print("\nThe launcher script(s) will automatically activate your Python environment.")
    print()

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nSetup interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)