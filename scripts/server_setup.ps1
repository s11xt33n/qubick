# Установка окружения на сервере (Windows 11, Tesla V100 = GPU 1, Quadro P600 = GPU 0).
# Запуск из корня проекта:  powershell -ExecutionPolicy Bypass -File scripts\server_setup.ps1
$ErrorActionPreference = "Stop"

if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
# Сборка PyTorch под CUDA 12.6 поддерживает архитектуру Volta (sm_70) у V100
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
& $py -m pip install -e ".[pennylane,vision,dev]"

# Выбираем именно V100 (в порядке шины PCI она вторая)
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "1"
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('arch', torch.cuda.get_arch_list())"
& $py -m pytest -q
