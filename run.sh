#!/usr/bin/env bash
set -e

# 1. Create & activate venv
python3 -m venv venv
# macOS / Linux
source venv/bin/activate
# (on Windows PowerShell use:  venv\Scripts\Activate.ps1 )

# 2. Install dependencies
pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Verify no annotation errors
python -c "from miniSam import MiniSam; print('OK')"

# 4. Run the U-Matrix plot
python plot_som_umatrix.py

# 5. Deactivate when done
deactivate
