"""Automated tests for the Health-Care-Chatbot.

Run from the project root with the venv active:

    python test_chatbot.py
"""

import json
import os
import pickle
import sys

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

os.chdir(os.path.dirname(os.path.abspath(__file__)))

lemmatizer = WordNetLemmatizer()
with open("intents.json") as f:
    intents = json.load(f)
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbotmodel.h5")

CONFIDENCE_THRESHOLD = 0.5


def predict_intent(sentence: str) -> str | None:
    tokens = [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(sentence)]
    bag = np.array([1 if w in tokens else 0 for w in words])
    probs = model.predict(np.array([bag]), verbose=0)[0]
    top = int(np.argmax(probs))
    return classes[top] if probs[top] >= CONFIDENCE_THRESHOLD else None


CASES = [
    ("hi",                                                              "greetings"),
    ("hello there",                                                     "greetings"),
    ("good morning",                                                    "greetings"),
    ("who are you?",                                                    "name"),
    ("what is your name",                                               "name"),
    ("how old are you?",                                                "age"),
    ("thanks",                                                          "thanks"),
    ("thank you so much",                                               "thanks"),

    ("Runny or stuffy nose",                                            "common cold symptoms"),
    ("what are the symptoms of common cold",                            "common cold symptoms"),
    ("i have a cold",                                                   "common cold symptoms"),

    ("Chills and shivering",                                            "fever symptoms"),
    ("what are the symptoms of fever",                                  "fever symptoms"),
    ("i have fever",                                                    "fever symptoms"),

    ("frequent urination",                                              "Diabetes symptoms"),
    ("what are the symptoms of diabetes",                               "Diabetes symptoms"),
    ("i think i have diabetes",                                         "Diabetes symptoms"),

    ("Hopeless outlook",                                                "Depression symptoms"),
    ("what are the symptoms of depression",                             "Depression symptoms"),
    ("i feel depressed",                                                "Depression symptoms"),

    ("shortness of breath",                                             "Asthma symptoms"),
    ("what are the symptoms of asthma",                                 "Asthma symptoms"),

    ("my head hurts",                                                   "headache symptoms"),
    ("what are the symptoms of headache",                               "headache symptoms"),
    ("i have a migraine",                                               "headache symptoms"),

    ("itchy skin",                                                      "allergy symptoms"),
    ("what are the symptoms of allergy",                                "allergy symptoms"),

    ("what are the symptoms of hypertension",                           "hypertension symptoms"),
    ("i have high bp",                                                  "hypertension symptoms"),

    ("what are the symptoms of covid",                                  "covid symptoms"),
    ("loss of smell",                                                   "covid symptoms"),

    ("what are the symptoms of flu",                                    "flu symptoms"),
    ("i have the flu",                                                  "flu symptoms"),

    ("what are the symptoms of anxiety",                                "anxiety symptoms"),
    ("i feel anxious",                                                  "anxiety symptoms"),

    ("my stomach hurts",                                                "stomach symptoms"),
    ("what are the symptoms of stomach problems",                       "stomach symptoms"),

    ("What medicines can I buy to help me with my common cold?",        "common cold prevention"),
    ("How can I keep from getting a fever?",                            "fever prevention"),
    ("how to prevent diabetes",                                         "diabetes prevention"),
    ("treatment for depression",                                        "depression prevention"),
    ("medicine for asthma",                                             "asthma prevention"),
    ("how to prevent migraine",                                         "headache prevention"),
    ("what medicines for allergy",                                      "allergy prevention"),
    ("how to lower blood pressure",                                     "hypertension prevention"),
    ("how to prevent covid",                                            "covid prevention"),
    ("how to prevent flu",                                              "flu prevention"),
    ("how to manage anxiety",                                           "anxiety prevention"),
    ("medicine for bloating",                                           "stomach prevention"),

    ("is there any doctor available?",                                  "Consultation"),
    ("i want to talk to a doctor",                                      "Consultation"),
    ("bye",                                                             "goodbye"),
    ("see you later",                                                   "goodbye"),
]

FALLBACK_CASES = [
    "do i have cancer",
    "what are the symptoms of tuberculosis",
    "tell me about kidney stones",
    "asdfghjkl",
]


def main() -> int:
    passed = failed = 0
    print(f"Running {len(CASES)} intent tests + {len(FALLBACK_CASES)} fallback tests...\n")

    for i, (text, expected) in enumerate(CASES, 1):
        actual = predict_intent(text)
        ok = actual == expected
        marker = "  " if ok else ">>"
        status = "PASS" if ok else "FAIL"
        print(f"{marker}[{status}] {i:>2}. {text!r:<60} -> {actual}  (expected {expected})")
        if ok: passed += 1
        else:  failed += 1

    print("\n--- Fallback tests (out-of-scope queries) ---")
    for i, text in enumerate(FALLBACK_CASES, 1):
        actual = predict_intent(text)
        ok = actual is None or actual == "fallback"
        marker = "  " if ok else ">>"
        status = "PASS" if ok else "FAIL"
        print(f"{marker}[{status}] {i}. {text!r:<60} -> {actual}  (expected: confidence < {CONFIDENCE_THRESHOLD} or fallback)")
        if ok: passed += 1
        else:  failed += 1

    total = len(CASES) + len(FALLBACK_CASES)
    print(f"\n{passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
