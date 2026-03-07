#!/usr/bin/env python3
"""Verify PyTorch and common deep learning libraries are installed and usable."""

def main():
    print("Checking development environment...\n")

    # Core PyTorch
    try:
        import torch
        print(f"  torch:        {torch.__version__}")
    except ImportError as e:
        print(f"  torch:        FAILED - {e}")
        return 1

    try:
        import torchvision
        print(f"  torchvision:  {torchvision.__version__}")
    except ImportError as e:
        print(f"  torchvision: FAILED - {e}")

    try:
        import torchaudio
        print(f"  torchaudio:  {torchaudio.__version__}")
    except ImportError as e:
        print(f"  torchaudio:  FAILED - {e}")

    # Device availability
    print("\nDevice availability:")
    print(f"  CUDA:         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device:  {torch.cuda.get_device_name(0)}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"  MPS (Metal):  available (Apple Silicon)")
    else:
        print(f"  MPS (Metal):  not available")
    print(f"  CPU:          always available")

    # Other common libs
    libs = ("numpy", "scipy", "matplotlib", "PIL", "cv2", "sklearn", "tqdm", "tensorboard")
    print("\nOther libraries:")
    for name in libs:
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "ok")
            print(f"  {name:12} {ver}")
        except ImportError:
            print(f"  {name:12} not installed")

    # Quick tensor test
    print("\nQuick tensor test:")
    x = torch.rand(2, 3)
    print(f"  torch.rand(2, 3) =\n{x}")
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    exit(main())
