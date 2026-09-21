"""qhnn — библиотека для построения, обучения и исследования гибридных
квантово-классических нейронных сетей на PyTorch."""
from .circuits import Circuit, Gate, build_circuit, to_qasm
from .layers import QuantumLayer
from .models import build_model, n_params, MODEL_NAMES

__version__ = "0.2.0"
__all__ = ["Circuit", "Gate", "build_circuit", "to_qasm", "QuantumLayer",
           "build_model", "n_params", "MODEL_NAMES"]
