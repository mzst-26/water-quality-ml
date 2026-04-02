# water-quality-ml
Predicting nitrite levels in water samples using Linear Regression, Random Forest, and Neural Networks. Includes evolutionary optimisation benchmarking with Hill Climber and EA on the McCormick function.


## Run locally

1. Create and activate a virtual environment.

Run this in terminal within the project directory.

<details>
<summary><strong>Unix/macOS</strong></summary>

~~~bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
~~~

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

~~~powershell
py -3 -m venv venv
venv\Scripts\Activate.ps1
py -3 -m pip install --upgrade pip
py -3 -m pip install -r requirements.txt
~~~

</details>

## Debug issues with installation of dependencies

To confirm the virtual environment is activated, check the location of your Python interpreter:

<details>
<summary><strong>Unix/macOS</strong></summary>

~~~bash
which python

# Example output:
# /Users/user/Github/water-quality-ml/venv/bin/python
~~~

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

~~~powershell
Get-Command python

# Example output:
# CommandType     Name      Version    Source
# -----------     ----      -------    ------
# Application     python.exe 3.x.x.x  C:\Users\user\Github\water-quality-ml\venv\Scripts\python.exe
~~~

</details>

Check if pip is installed:

<details>
<summary><strong>Unix/macOS</strong></summary>

~~~bash
python3 -m pip --version

# Example output:
# pip 26.0.1 from /Users/user/Github/water-quality-ml/venv/lib/python3.14/site-packages/pip (python 3.14)
~~~

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

~~~powershell
py -3 -m pip --version

# Example output:
# pip 26.0.1 from C:\Users\user\Github\water-quality-ml\venv\Lib\site-packages\pip (python 3.14)
~~~

</details>


## Badges
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
