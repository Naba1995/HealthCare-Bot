# MedBot — Healthcare Chatbot

A neural-network-based medical FAQ chatbot built with **Python, NLTK, and TensorFlow/Keras**, with a **modern Tkinter chat-bubble GUI**. Ask MedBot about symptoms, medicines, prevention tips, or how to find a doctor for **12 common health conditions**.

The repo also includes a separate **Heart Disease classification notebook** that compares 7 classical ML algorithms on the UCI Cleveland dataset.

---

## Features

- 🩺 **Symptom checker** — describe what you feel, MedBot guesses the condition
- 💊 **Medicine & prevention tips** — what to take, what to do, what to eat
- 👨‍⚕️ **Doctor consultation links** — get pointed to online doctor services
- 🤷 **Honest fallback** — says *"I don't know that one"* instead of guessing
- 🖥️ **Two interfaces** — CLI (`chatbot_py.py`) and modern Tkinter GUI (`gui.py`)
- ✅ **Automated test suite** — 57 test cases covering all intents
- 📊 **Bonus** — heart-disease ML notebook (`Data_Analysis.ipynb`)

## Conditions covered

| Symptoms | Prevention/Medicine |
|---|---|
| Common cold | ✓ |
| Fever | ✓ |
| Diabetes | ✓ |
| Depression | ✓ |
| Asthma | ✓ |
| Headache / Migraine | ✓ |
| Allergies | ✓ |
| Hypertension | ✓ |
| COVID-19 | ✓ |
| Flu / Influenza | ✓ |
| Anxiety | ✓ |
| Stomach / Indigestion | ✓ |

Plus general intents: greetings, goodbye, thanks, name, age, doctor consultation, and a smart fallback for unknown questions.

---

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| NLP | NLTK (tokenizer + WordNet lemmatizer) |
| ML | TensorFlow / Keras (Sequential dense network) |
| Architecture | 3-layer feed-forward NN (128 → 64 → softmax) trained for 200 epochs with SGD |
| GUI | Tkinter (Canvas-based chat bubbles) |
| Dataset (notebook) | UCI Cleveland Heart Disease |

---

## Project structure

```
Health-Care-Chatbot/
├── chatbot_py.py          # CLI version of the chatbot
├── gui.py                 # Modern Tkinter chat GUI
├── training_py.py         # Trains the neural network
├── test_chatbot.py        # 57 automated test cases
├── intents.json           # Training data: tags, patterns, responses
├── chatbotmodel.h5        # Trained model weights
├── words.pkl              # Vocabulary built during training
├── classes.pkl            # Intent labels built during training
├── Data_Analysis.ipynb    # Heart disease ML notebook
├── heart-disease.csv      # UCI Cleveland dataset (303 rows)
├── chat.txt               # Sample conversation transcript
├── README.md
├── LICENSE
└── .gitignore
```

---

## Setup

### Requirements
- **Python 3.11** (TensorFlow doesn't yet support 3.12+)
- **Visual C++ Redistributable** (Windows only — TensorFlow needs it)
- ~1 GB free disk for the venv

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Naba1995/HealthCare-Bot.git
cd HealthCare-Bot

# 2. Create a virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Mac/Linux

# 3. Install dependencies
pip install --upgrade pip
pip install nltk "numpy<2" tensorflow

# 4. Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

> **Windows DLL error?** Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) — TensorFlow needs it. Or via winget: `winget install Microsoft.VCRedist.2015+.x64`.

---

## Usage

Make sure your venv is active first: `venv\Scripts\activate`

### Run the GUI (recommended)

```bash
python gui.py
```

A modern dark-themed chat window will open with WhatsApp-style bubbles. Type in the box, press **Enter** or click the green ➜ button.

### Run the CLI

```bash
python chatbot_py.py
```

Type messages, press Enter. **Ctrl+C** to exit.

### Run the test suite

```bash
python test_chatbot.py
```

Runs all 57 tests and shows pass/fail per case. Exit code is 0 on full pass.

### Retrain the model (only if you edit `intents.json`)

```bash
python training_py.py
```

Regenerates `words.pkl`, `classes.pkl`, and `chatbotmodel.h5`.

### Run the Heart Disease notebook

```bash
pip install jupyter pandas matplotlib seaborn scikit-learn xgboost
jupyter notebook Data_Analysis.ipynb
```

---

## Sample conversation

```
You:    hi
MedBot: hey!

You:    what are the symptoms of diabetes?
MedBot: It seems that you are suffering from Diabetes

You:    how to prevent diabetes
MedBot: Medicines: Insulin, Amylinomimetic drug, DPP-4 inhibitors.
        Preventions: Cut Sugar and Refined Carbs, Work Out Regularly,
        Drink Water, Lose Weight, Quit Smoking, Watch Portion Sizes.
        Suggested foods: Leafy greens, Avocados, Eggs.

You:    is there any doctor available?
MedBot: You can contact various doctors here for any kind of consultation:
        1. https://www.1mg.com/online-doctor-consultation
        2. https://www.tatahealth.com/online-doctor-consultation/general-physician
        3. https://www.doconline.com/
        ...

You:    do i have cancer
MedBot: Sorry, I don't have information on that yet. I currently know about:
        common cold, fever, diabetes, depression, asthma, headache/migraine,
        allergies, hypertension, COVID-19, flu, anxiety, and stomach problems.
        You can also ask me to find a doctor.

You:    bye
MedBot: have a nice day
```

---

## How it works

1. **Tokenize** the user's message with NLTK
2. **Lemmatize** each word ("running" → "run") so `walk` and `walking` map to the same feature
3. **Bag-of-words encode** against the vocabulary built during training
4. **Predict** the intent via a 3-layer dense network
5. **Threshold check** — if confidence < 0.5, return the fallback message; otherwise pick a random response from the matched intent

```
"i have fever" → [tokenize+lemmatize] → bag-of-words → [Dense 128 → Dense 64 → Softmax]
              → top class = "fever symptoms" (confidence 1.00)
              → random response from intents.json
```

---

## Limitations

This is a **closed-domain** chatbot — it only recognizes the 12 conditions in `intents.json`. It is **not a substitute for medical advice**. For anything serious, see a real doctor.

The training data is small (~250 patterns across 28 intents), so very different phrasings of known queries can still misclassify. Adding more patterns and retraining improves coverage.

---

## How to add a new condition

1. Open `intents.json`
2. Add a new entry with `tag`, `patterns` (sample user inputs), and `responses`. **Important:** include the disease name in patterns (e.g., `"do i have X"`, `"X symptoms"`) — otherwise the model can't learn it.
3. Update the `fallback` intent's response so users know about the new coverage
4. Retrain: `python training_py.py`
5. Add test cases in `test_chatbot.py`
6. Run tests: `python test_chatbot.py`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: No matching distribution found for tensorflow` | You're on Python 3.12+. TF needs 3.11. Install Python 3.11 and rebuild venv. |
| `DLL load failed while importing _pywrap_tensorflow_internal` | Install Visual C++ Redistributable (Windows only). |
| `LookupError: Resource punkt_tab not found` | Run the NLTK download command from step 4 above. |
| `ModuleNotFoundError: No module named 'pandas'` (when running `training_py.py`) | Already removed — make sure you have the latest `training_py.py`. |
| GUI shows "Sorry, I don't get that" for everything | You haven't retrained after editing `intents.json`. Run `python training_py.py`. |
| Push to GitHub rejected (file too large) | You committed `venv/`. The `.gitignore` should prevent this — make sure it exists before staging. |

---

## License

MIT — see [LICENSE](LICENSE).

---

## Credits