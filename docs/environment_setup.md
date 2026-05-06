# Environment Setup

This guide sets up the Python environment for the authorship attribution and topic modelling project on Windows using PowerShell. Run the commands from the project root directory, where `requirements.txt` is located.

## Step 1 - Create a Virtual Environment

```powershell
python -m venv .venv
```

This creates an isolated Python environment in `.venv/` so project dependencies do not interfere with other Python projects on the same machine.

## Step 2 - Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

This activates the virtual environment for the current PowerShell session. After activation, Python and pip commands will use `.venv/`.

If PowerShell blocks the activation script, run this command once for the current session and then activate again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

## Step 3 - Upgrade pip

```powershell
python -m pip install --upgrade pip
```

This upgrades pip inside the virtual environment before installing larger NLP and ML packages.

## Step 4 - Install Project Dependencies

```powershell
pip install -r requirements.txt
```

This installs the project dependencies for transformer models, datasets, evaluation, PEFT/TRL experiments, classical ML, topic modelling, visualization, configuration, and notebooks.

## Step 5 - Verify the Environment

```powershell
python --version
pip --version
python -c "import torch, transformers, datasets, pandas, sklearn; print('Environment OK')"
```

These commands confirm that the active Python and pip come from the virtual environment and that core packages import successfully.

## Step 6 - Launch Jupyter When Needed

```powershell
jupyter notebook
```

Use notebooks for exploration only. Reproducible experiment logic should later be placed in `src/`, configured through `configs/`, and run through scripts in `scripts/`.

## Step 7 - Deactivate the Environment

```powershell
deactivate
```

This exits the virtual environment when the session is finished.
