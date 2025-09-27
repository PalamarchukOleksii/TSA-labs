# TSA Labs Project

This repository contains several lab projects, each with its own Jupyter notebook.

---

## Getting Started

Follow the instructions below to set up the project.

### Prerequisites

* **Python 3.10 or higher**
* `git` installed on your system

---

### 1. Clone the Repository

```bash
git clone https://github.com/PalamarchukOleksii/TSA-labs.git
```

### 2. Navigate to the Project Directory

```bash
cd TSA-labs
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

* **Linux/macOS:**

```bash
source .venv/bin/activate
```

* **Windows (cmd):**

```bat
.venv\Scripts\activate
```

* **Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies from `pyproject.toml`

Since this project uses `pyproject.toml` for dependency management:

```bash
pip install .
```

**Main dependencies include:**

* `numpy`
* `pandas`
* `matplotlib`
* `jupyterlab`

> **Note:** If you need to add a new dependency, make sure to add it to the `pyproject.toml` under `[project]` and then run:
>
> ```bash
> pip install .
> ```
>
> again to install it in your virtual environment.

---

### 6. Run Jupyter Notebooks

1. Start Jupyter Lab:

```bash
jupyter lab
```

2. Open the notebook you want to run from the `lab*/` folder.

---

### 7. Deactivate the Virtual Environment

Once you're done:

```bash
deactivate
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
