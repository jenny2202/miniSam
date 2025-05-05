# MiniSam SOM Visualization

This project provides a simple Self-Organizing Map (SOM) implementation (`miniSam.py`), a script to visualize the U-Matrix heatmap of a trained SOM on synthetic data and top 5 recommendation explained in `Improvement_Analysis.ipynb`.

## Prerequisites

* Python 3 (>= 3.7)
* `bash` shell (macOS/Linux) or PowerShell (Windows) for the `run.sh` script

## Setup & Installation

1. Ensure the repository contains the following files:

   * `miniSam.py` (SOM implementation)
   * `plot_som_umatrix.py` (visualization script)
   * `run.sh` (setup and execution script)
   * `requirements.txt` (dependencies list)

2. Make the helper script executable (once):

   ```bash
   chmod +x run.sh
   ```

## Usage

From the project root, simply run:

```bash
./run.sh
```

This script will:

1. Create and activate a virtual environment (`venv/`).
2. Install all required Python packages from `requirements.txt`.
3. Execute the SOM training and U-Matrix plotting (`plot_som_umatrix.py`).
4. Deactivate the virtual environment when done.

After running, a window will pop up displaying the SOM U-Matrix heatmap. Close the window to end the script.

---

Enjoy exploring Self-Organizing Maps!
