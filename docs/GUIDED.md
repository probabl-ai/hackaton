# Guided lab

Work **in Bob** (IDE or CLI) with the lab skills. Every Kaggle file must be produced there, then uploaded with a Skore Hub **EstimatorReport URL** in the Submission Description. See the [competition rules](https://www.kaggle.com/competitions/ibm-probabl-hackaton/rules).

Snippets match **installed** libraries: `skore` 0.25.0, `scikit-learn` 1.9.0, `skrub` 0.10.0. Do not invent other symbol names. Put a **new** hub report key for each Kaggle file (`01_dummy`, `02_ridge`, …). Do **not** use the reserved key `eda` for a model.

Join Kaggle and form your team **before** you install Python on your machine.

## 1. Join Kaggle and form a team

**Kaggle** is a website for data-science competitions: you download the data, upload a prediction file, and get a public score on a **leaderboard**. This lab’s private competition lives there. You need a free Kaggle account.

![Example Kaggle leaderboard: teams ranked by score.](figures/kaggle_leaderboard.png)

*Example leaderboard — teams ranked by score (lower is better here). Yours will look like this after the first Submissions.*

Open the competition and accept the rules: [https://www.kaggle.com/t/1d3caaf98f12426cb621b47f2063ab28](https://www.kaggle.com/t/1d3caaf98f12426cb621b47f2063ab28).

Then form **one Kaggle team** (maximum four people). Remember the **exact team name**. You will reuse it as the Skore Hub workspace name in step 3.

## 2. Python

You need a working **Python 3.10+** interpreter on your `PATH`. Check:

```bash
python --version
```

If that fails, try `python3 --version`. Use that same command (`python` or `python3`) for the rest of this lab. Install Python from [python.org](https://www.python.org/downloads/) if neither works.

## 3. Hub account and team workspace

Create (or sign in to) a Skore Hub account on the **custom lab hub**: [https://ibm.skore.probabl.ai](https://ibm.skore.probabl.ai).

**One Hub workspace per Kaggle team — not one per person.**

1. **One** teammate creates the workspace.
2. The workspace **name must match the Kaggle team name** (no `/` in the name; if Hub rejects spaces or punctuation, use the same words with hyphens).
3. That person **invites the other teammates** into that workspace.
4. Everyone else **joins the invite**. Do not create a second workspace.

`python scripts/install_skore.py` (next step) expects you to be a member of **exactly one** Hub workspace. Extra personal workspaces break that install.

You will use **Bob IDE or Bob CLI** plus **skore** for every Submission - not Cursor, Claude Code, Copilot, ChatGPT, or Kaggle Notebooks.

## 4. Install skore

From the repo root, after you are in the team Hub workspace:

```bash
python scripts/install_skore.py
```

Confirm `.skore` and `.bob/skills/` exist.

## 5. Data into `data/`

Competition tables are **not** in git (`data/*.csv` is ignored). Download the files from the [competition **Data** tab](https://www.kaggle.com/competitions/ibm-probabl-hackaton/data) and unzip them into a `data/` folder at the repo root.

You should see `X_train.csv`, `y_train.csv`, `X_test.csv`, and `sample_submission.csv`. Every teammate still needs these CSVs locally (Hub share does not replace the download).

## 6. Explore the data (EDA)

**Exploratory data analysis (EDA)** is looking at the tables *before* you pick a model: shape, missingness, `patient_id` groups, what looks like a leak. In Bob this is the **G-EDA** step (`explore-ml-data`). It writes `data/eda.py`, a short narrative `data/eda.md`, and interactive `data/eda_<table>.html` pages.

**Only one teammate computes the EDA.** That person chooses **run** in Bob. When it finishes, Bob uploads those files to the team Hub workspace under the reserved key `eda`.

Everyone else **must not run a second EDA**. When Bob looks up Hub:

- if key `eda` is already there, it **fetches** the files onto their machine;
- if a teammate is still computing it, choose **wait** — do not start modelling, do not skip. When they have uploaded, say so and Bob fetches.

Same dataset, same findings. One run is enough; Hub is the shared copy.

## 7. Dummy mean: a floor

If you cannot beat predicting the training-set mean, nothing else is working (data load, metric, upload). Start by loading the visits.

Each **row** is a **visit**. The target is a **true / unbiased OFF** MDS-UPDRS motor score (`y_train.target`). Clinic ON/OFF scores are biased (subjectivity, missing values, levodopa timing).

- `patient_id`: several visits per patient. The Kaggle holdout is **by patient**.
- **Missingness**: `on`, `off`, `ledd`, `gene`, intake delays are often missing.
- **ON / OFF / LEDD / timing**: `on`, `off`, `ledd`, `time_since_intake_on`, `time_since_intake_off`. On train, years since diagnosis is `age - age_at_diagnosis` (test may already have `time_since_diagnosis`).

```python
import pandas as pd

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")
y = visits["target"]
```

Use `sample_submission.csv` as the shape template: columns `Index,target`.

- **What it is**: Dummy always predicts the same number: the average true OFF in train.
- **How it works**: it ignores every column. You still pass an `X` so `skore.evaluate` can line up the same rows as `y`.
- **Why here**: if you cannot beat this floor, the data load, the metric, or the upload is broken - not the model.

```python
from sklearn.dummy import DummyRegressor
from skore import evaluate

feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
X = visits[feature_cols]
y = visits["target"]

dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
report.metrics.rmse()
```

`skore.evaluate` is the entry point. With no `splitter=`, it uses default `splitter=0.2` (a random row holdout). That is enough to check the floor; grouped splits come in step 10.

**Project** stores reports. **Hub** is required for a valid Submission (the URL printed after `put`). `name=` is your project inside the hub workspace from `.skore`. Always `load_skore_credentials()` then `login(mode="hub")`:

```python
from parkinson.hub import load_skore_credentials
from skore import Project, login

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="ibm-hackaton", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
# The console prints: Consult your report at https://ibm.skore.probabl.ai/…
```

`Project.get` is by **id** from `project.summarize()`, not by the string key you passed to `put`.

## 8. A simple linear model

Numeric clinical variables (`age`, `on`, `off`, `ledd`, delays) may carry a simple additive signal.

- **What it is**: Ridge draws a straight line: predicted true OFF ≈ intercept + (a weight × each column).
- **How it works**: it chooses weights so predictions match `y`, then `alpha` pulls those weights toward zero so one noisy column cannot dominate.
- **Why here**: a first check that the numbers matter at all. Ridge only eats numbers with no holes, so median-impute first and leave strings like `gene` for later.

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skore import evaluate

ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)
report = evaluate(ridge, X, y)
```

`make_pipeline(*steps)` stitches transformers then the regressor. Compare dummy vs ridge in one report:

```python
report = evaluate(
    {"dummy": dummy, "ridge": ridge},
    X,
    y,
)
```

`put` with a new key. The default row holdout can still leak the same patient into both sides - grouped CV is step 10.

## 9. Tweak Ridge and submit on Kaggle

`Ridge(alpha=…)` is the knob: larger `alpha` shrinks coefficients more. Change it, re-run `evaluate`, keep the one that beats dummy by the most.

```python
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=10.0),  # try 0.1, 1.0, 10.0, …
)
report = evaluate(ridge, X, y)
report.metrics.rmse()
project.put("02_ridge", report)
```

`evaluate` only scores on train. The file you upload is that pipeline **fit on all training visits**, then `predict` on `X_test` (same columns as `X`):

```python
from sklearn.base import clone

final = clone(ridge).fit(X, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission.to_csv("submission.csv", index=False)
```

Upload `submission.csv` on Kaggle. A Submission is valid only if:

1. Code was written and run in **Bob**, evaluation through **skore**.
2. You `put` this Ridge report; the console shows `https://ibm.skore.probabl.ai/…`.
3. CSV header is `Index,target`, one row per test visit.
4. The Kaggle **Submission Description** contains that report URL (otherwise the row is invalid even if Kaggle scored the file).

Later models use the same fit → CSV → URL path.

## 10. Evaluate with patient-grouped CV

The default `splitter=0.2` shuffles **rows**. That is too kind here.

- **What it is**: GroupKFold is a dress rehearsal of the Kaggle split: a whole patient goes to train or to the held-out fold, never both.
- **How it works**: it cuts patients into 5 groups. Five times, it trains on 4 groups and scores on the 5th, then you read the average RMSE.
- **Why here**: the same person has many visits. A random row split is **leakage**: visits from one patient sit on both sides, so the model memorizes that person instead of generalizing, and metric looks better than the Kaggle competition (which holds out entire patients). Grouped CV blocks that leak.

skore’s sklearn path calls `splitter.split(X, y)` **without** `groups=`, so precompute the index pairs:

```python
from sklearn.model_selection import GroupKFold
from skore import evaluate

groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
report = evaluate(
    {"dummy": dummy, "ridge": ridge},
    X,
    y,
    splitter=cv_splits,
)
report.metrics.summarize()          # table of metrics
report.metrics.get("rmse")          # or report.metrics.rmse()
```

- `splitter=` a list of index pairs (or `5`, or a CV splitter) → `CrossValidationReport`
- several estimators (list or dict) → `ComparisonReport`

Use `cv_splits` from here on. `put` this report with a new key.

skrub DataOps can attach `groups` on the graph - see step 13.

## 11. HistGradientBoosting: missingness as signal

ON/OFF/LEDD/genetics are often missing; missingness is part of the generative process (see [CONTEXT.md](CONTEXT.md)). Ridge had to *fill* those holes with a median. Here we keep the holes.

- **What it is**: a **decision tree** is a flowchart of yes/no questions (`age > 62?`, `on` missing?). Each visit falls down the flowchart into a leaf, and that leaf predicts a number (a typical true OFF for visits that landed there). **Boosting** means we do not stop at one tree: we grow many small trees in a sequence, and each new tree is trained on the *mistakes* of the ones before it. The final prediction is the sum of all those small corrections. **Hist** (histogram) is an implementation trick: each numeric column is first cut into a few buckets (like a histogram), so the model looks at bucket ids instead of every distinct age. That keeps it fast on tens of thousands of visits.
- **How it works** -
  - Tree 1 fits a rough sketch of true OFF.
  - Tree 2 looks at the residuals (true OFF minus what tree 1 predicted) and tries to explain what is still wrong.
  - Trees 3, 4, … keep nipping at the remaining error. That is gradient boosting, in plain language: keep adding a small expert on the leftover mistakes.
  - At a split, the algorithm may send **missing** values left or right on purpose. It learns that route from the training data - so “OFF was not measured” can be a signal, not a defect.
  - `random_state=0` only makes the run repeatable.
- **Why here**: in this table, a missing OFF often means the visit was ON-only (uncomfortable OFF exams are skipped). That is information about the patient and the protocol, not noise to impute away. Do **not** median-fill NaNs unless you are testing that ablation (does the model get *worse* when you hide the holes?).

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from skore import evaluate

hgbr = HistGradientBoostingRegressor(random_state=0)
report = evaluate(hgbr, X, y, splitter=cv_splits)
```

Strings are still a problem: by default the model wants numbers or pandas `category` dtypes. `categorical_features="from_dtype"` (the default) treats a `category` column as “pick among a few labels” instead of as a fake number. Convert `cohort` / `gene` with `.astype("category")` if you add them to `X`.

## 12. skrub `tabular_pipeline`: mixed types without hand-encoding

Ridge and the numeric HGBR above never saw `gene` or `cohort`: those are **strings**. A sklearn regressor cannot multiply `"GBA"` by a weight. Something has to turn text into numbers first. Doing that by hand (a custom encoder per column) is where pipelines usually rot.

- **What it is**: `tabular_pipeline("regressor")` is a ready-made two-step recipe from skrub: (1) `TableVectorizer` turns a messy table into a numeric matrix, (2) `HistGradientBoostingRegressor` predicts. You drop ids, pass the rest, and the vectorizer chooses an encoder **per column**.
- **How it works**: “cardinality” means how many distinct values a column has.
  - **Low cardinality** (a handful of labels, e.g. `sexM`, maybe `cohort`) → **one-hot**: one new column per label, `1` if that row has it, `0` otherwise. The model sees a switch, not a made-up ranking of labels.
  - **High cardinality** (many distinct strings, e.g. `gene` if it is messy) → `StringEncoder`: compress the text into a few numeric dimensions instead of hundreds of one-hot columns. You keep signal without exploding the width of `X`.
  - **Numbers** (`age`, `on`, `ledd`, …) pass through. HGBR can still use their NaNs, as in step 11.
  - `Index` and `patient_id` are identifiers, not clinical features. If you leave them in, the model can memorize ids - another form of leakage. Drop them (and `target`) before fitting.
- **Why here**: the interesting PD columns are mixed: numbers with holes *and* categoricals. This step is how you stop throwing `gene` / `cohort` away just because Ridge could not read them.

```python
from skrub import tabular_pipeline
from skore import evaluate

# include categoricals; keep Index / patient_id out of features
X_full = visits.drop(columns=["Index", "patient_id", "target"])
model = tabular_pipeline("regressor")
report = evaluate(model, X_full, y, splitter=cv_splits)
```

Same vectorizer, your own estimator (if you want to tweak HGBR):

```python
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor

model = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=0),
)
```



## 13. skrub DataOps: groups baked into the graph

Until now you have juggled separate objects: a DataFrame `visits`, a list `cv_splits`, a pipeline `model`. Easy to fit on the wrong columns or forget `groups` after a copy-paste. **DataOps** is skrub’s way to write the *recipe* once, as a graph, so features, target, and the grouped split live on the same object.

- **What it is**: a DataOp is not the prediction yet. It is a delayed plan: “when you give me a table named `visits`, drop `target`, vectorize, then apply HGBR.” `skore.evaluate` walks that plan and runs the CV that you attached to it.
- **How it works** -
  - `skrub.var("visits", visits)` names an **input**. Later, at predict time, you can pass a different table under the same name (`X_test`).
  - `.skb.mark_as_X(...)` tells skore “this node is the feature table.” `.skb.mark_as_y()` marks the target. Those two marks are how `evaluate` knows what to split.
  - Passing `cv=GroupKFold(...)` and `split_kwargs={"groups": groups}` **on** `mark_as_X` bakes patient-grouped CV into the graph. You no longer keep a side list `cv_splits` that can go stale.
  - `.skb.apply(transformer)` / `.skb.apply(estimator, y=...)` appends a step, like `make_pipeline`, but on the graph.
  - `evaluate(pred)` with **no** `splitter=` reads `cv` / `groups` from the DataOp. That is the point: the split cannot drift away from the data.
  - `skrub.X(value)` is shorthand for `skrub.var("X", value).skb.mark_as_X()`: no extra `cv` unless you call `mark_as_X` again.
  - `.skb.make_learner()` **freezes** the graph into a `SkrubLearner`. Then `fit` / `predict` take an **environment dict** (`{"visits": ...}`), because the input was a named `var`, not a naked array. You need that for the Kaggle file: test has no `target`.
- **Why here**: GroupKFold only works if `groups` is the `patient_id` of the *same* rows as `X`. Putting that on the graph is how you stop the leakage from step 10 from creeping back in through a forgotten argument.

```python
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import evaluate

data = skrub.var("visits", visits)
groups = data["patient_id"]
X_op = data.drop("target", axis=1).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y_op = data["target"].skb.mark_as_y()
pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(random_state=0),
    y=y_op,
)

# evaluate reads cv / groups from the DataOp when splitter is omitted
report = evaluate(pred)
```



## 14. Fit the chosen model and submit again

Grouped CV (steps 10–13) tells you which idea is better. Submit it the same way as Ridge (step 9): `put` a **new** hub report, fit on **all** training visits, `predict` on `X_test`, upload the CSV with that URL in the Description.

```python
from sklearn.base import clone

final = clone(model).fit(X_full, y)
submission = X_test[["Index"]].copy()
submission["target"] = final.predict(X_test.drop(columns=["Index", "patient_id"], errors="ignore"))
submission.to_csv("submission.csv", index=False)
```

Align test columns with whatever you trained on (same drops, same dtypes). For a DataOp / `SkrubLearner`:

```python
learner = pred.skb.make_learner()
learner.fit({"visits": visits})
pred_test = learner.predict({"visits": X_test})
```

Next: [CONTEXT.md](CONTEXT.md) for timing / missingness / progression, the Kaggle Description page for files and metric.