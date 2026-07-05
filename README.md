# HDPE Reactor MFI Prediction - Streamlit App

A web app that predicts HDPE reactor Melt Flow Index (MFI at 1.2 kg) from live process
parameters, so an operator can react before a batch goes off-spec.

Model: **CatBoost Regressor** (auto-selected out of 10 models compared in
`HDPE_Model_Comparison_Final.ipynb` - lowest test error, best generalization).
Test MAE ≈ 4.3 MFI units, Test R² ≈ 0.64.

This README assumes you have never used Streamlit before - every step is spelled out.

---

## 1. Folder structure

Your project folder should look exactly like this:

```
mfi_streamlit_app/
├── app.py                     <- the Streamlit application
├── requirements.txt           <- list of Python packages needed
├── sample_input.csv           <- example file for the CSV-upload feature
├── README.md                  <- this file
└── model/
    ├── mfi_model.pkl          <- the trained CatBoost model
    ├── mfi_scaler.pkl         <- the fitted MinMaxScaler
    └── mfi_feature_list.pkl   <- the exact list/order of features the model expects
```

All of these files are provided for you. Just keep this folder structure - the app looks
for the model files at `model/mfi_model.pkl` etc, relative to `app.py`.

---

## 2. Install Python (skip if you already have it)

Download Python 3.10 or newer from [python.org](https://www.python.org/downloads/) and
install it. During installation on Windows, tick **"Add Python to PATH"**.

Check it worked by opening a terminal (Command Prompt / PowerShell / Terminal) and running:

```
python --version
```

---

## 3. Create the folder and add the files

1. Create a folder anywhere on your computer, e.g. `mfi_streamlit_app`.
2. Put `app.py`, `requirements.txt`, `sample_input.csv`, `README.md` directly inside it.
3. Create a subfolder called `model` inside it, and put the three `.pkl` files inside `model/`.

(This matches the structure in Section 1 above.)

---

## 4. Install Streamlit and the other packages

Open a terminal, navigate into your project folder, and run:

```
cd path/to/mfi_streamlit_app
pip install -r requirements.txt
```

This installs Streamlit, pandas, scikit-learn, CatBoost, and everything else the app needs.

(Optional but recommended: create a virtual environment first, so these packages don't
mix with other Python projects on your computer:

```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```
)

---

## 5. About the trained model and scaler files

You don't need to do anything here - `mfi_model.pkl`, `mfi_scaler.pkl` and
`mfi_feature_list.pkl` are already trained and saved for you (produced at the end of
`HDPE_Model_Comparison_Final.ipynb`, in the "Save the final model and scaler" cell).

If you retrain the model later (e.g. with new data), just re-run that notebook cell and
replace the three files in the `model/` folder with the new ones - the app will
automatically use the new model next time it starts.

No separate encoder file is included: the final model does not use GRADE as an input
(this was tested directly in the modelling notebook - including GRADE did not improve
predictions once the process variables and the "last known MFI" feature were included),
so there's no categorical encoding step to save.

---

## 6. Run the app

From inside the project folder, with your virtual environment active (if you made one):

```
streamlit run app.py
```

You'll see output like:

```
Local URL: http://localhost:8501
```

---

## 7. Open it in your browser

Streamlit usually opens your browser automatically. If not, copy the **Local URL** from
the terminal (usually `http://localhost:8501`) and paste it into your browser.

You'll see two tabs:
- **Manual Entry** - type in current reactor readings and the last known lab MFI, get one prediction.
- **CSV Upload** - upload a file like `sample_input.csv` to get predictions for many rows at once.

To stop the app, go back to the terminal and press `Ctrl + C`.

---

## 8. How to edit the UI

Everything the app shows is in `app.py`, organised in clearly labelled sections:

- **Sidebar** - project description, model info, instructions (edit the `with st.sidebar:` block).
- **Manual entry tab** - the input form (edit the `with tab_manual:` block). To change a
  slider's range or default, edit the numbers in the `FEATURE_RANGES` dictionary near the top.
- **CSV upload tab** - the batch prediction logic (edit the `with tab_csv:` block).

After editing `app.py`, just save the file - if the app is still running, Streamlit shows
a "Rerun" prompt in the browser automatically. Otherwise, run `streamlit run app.py` again.

---

## 9. Deploy it for free with Streamlit Community Cloud

### Step 1 - Put your project on GitHub

1. Create a free account at [github.com](https://github.com) if you don't have one.
2. Create a new repository (e.g. `hdpe-mfi-app`) - keep it **public** (required for the free tier).
3. Upload all the files from Section 1 into that repository. You can do this either:
   - Through the GitHub website: open the repo, click **"Add file" -> "Upload files"**, drag in everything (keeping the `model/` subfolder), then commit.
   - Or with git from your terminal:
     ```
     cd path/to/mfi_streamlit_app
     git init
     git add .
     git commit -m "Initial commit - HDPE MFI prediction app"
     git branch -M main
     git remote add origin https://github.com/YOUR_USERNAME/hdpe-mfi-app.git
     git push -u origin main
     ```

### Step 2 - Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
2. Click **"New app"**.
3. Choose your repository (`hdpe-mfi-app`), the branch (`main`), and the main file path (`app.py`).
4. Click **"Deploy"**.

Streamlit Cloud will install everything from `requirements.txt` and launch the app. You'll
get a public URL like `https://your-app-name.streamlit.app` that anyone can open - no
installation needed on their end.

### Step 3 - Update the app later

Whenever you want to change something:
1. Edit the files locally (or directly on GitHub).
2. Commit and push the changes:
   ```
   git add .
   git commit -m "Describe what you changed"
   git push
   ```
3. Streamlit Community Cloud automatically detects the update and redeploys within a
   minute or two - no manual redeploy step needed.

---

## 10. Troubleshooting

- **"Could not load the model files"** - check the `model/` folder is next to `app.py`
  and contains all three `.pkl` files with the exact names shown in Section 1.
- **Package install errors** - make sure you're using Python 3.10+, and that you're
  running `pip install -r requirements.txt` from inside the project folder.
- **CSV upload says "Missing required column(s)"** - check your file has `Timestamp`,
  `GRADE`, and all 10 process variable columns, matching the names in `sample_input.csv`.
- **Predictions look off** - the model was trained on 4 specific grades with more than
  100 lab samples (see the model comparison notebook); predictions for other grades, or
  inputs far outside the ranges in `sample_input.csv`, are extrapolating and less reliable.

---

## 11. Known limitations (be upfront about these with anyone using the app)

- Manual entry has no history, so it can't compute real rolling averages - it assumes
  the current reading has been steady for the last 2-4 hours. The CSV upload path
  computes real rolling averages from the uploaded time series, which is more accurate.
- The single strongest feature is "last known MFI for this grade" - the model is least
  reliable right after a grade change or long shutdown, when there's no recent reading
  to build from.
- Trained on 4 grades only. Don't use it for grades it hasn't seen (`002DP48`, `012DB54`,
  `010DP45`, `004DP44`).
