# IT1244 Group Project

## Setup

```
git clone <repo-url> .
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then open `main.ipynb` and select the `.venv` kernel.

## Structure

```
main.ipynb             pipeline
src/helpers.py         device (cuda/cpu)
src/data.py            data
src/train.py           training
src/models/linear.py   model
src/models/mlp.py      model
```

One `.py` file per model in `src/models/`, each with a `build(device)` function.
To add a model, make a new file there and import it in `main.ipynb`.
