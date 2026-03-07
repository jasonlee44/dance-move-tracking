# dance-move-tracking

## Development setup (PyTorch)

**Requirements:** Python 3.10+ (3.10–3.12 recommended).

1. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   ```

2. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Verify the environment:**

   ```bash
   python check_env.py
   ```

On **Apple Silicon**, PyTorch will use MPS (Metal) when available. For **CUDA** (Linux/Windows), install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) and then install the rest: `pip install -r requirements.txt`.