"""MedBot GUI — modern chat interface."""

import json
import os
import pickle
import random
import time
import tkinter.messagebox
from tkinter import (BOTH, BOTTOM, END, FLAT, LEFT, RIGHT, TOP, VERTICAL, W, E, X, Y,
                     Button, Canvas, Entry, Frame, Label, Menu, Scrollbar, Tk)

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

os.chdir(os.path.dirname(os.path.abspath(__file__)))

_lemmatizer = WordNetLemmatizer()
with open("intents.json") as _f:
    _intents = json.load(_f)
_words = pickle.load(open("words.pkl", "rb"))
_classes = pickle.load(open("classes.pkl", "rb"))
_model = load_model("chatbotmodel.h5")


def _fallback_response() -> str:
    for intent in _intents["intents"]:
        if intent["tag"] == "fallback":
            return random.choice(intent["responses"])
    return "Sorry, I didn't get that."


def chat(message: str) -> str:
    tokens = [_lemmatizer.lemmatize(w) for w in nltk.word_tokenize(message)]
    bag = np.array([1 if w in tokens else 0 for w in _words])
    probs = _model.predict(np.array([bag]), verbose=0)[0]
    top = int(np.argmax(probs))
    if probs[top] < 0.5:
        return _fallback_response()
    tag = _classes[top]
    for intent in _intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return _fallback_response()


COLORS = {
    "bg":          "#0B141A",
    "header":      "#202C33",
    "header_fg":   "#E9EDEF",
    "header_sub":  "#8696A0",
    "user_bubble": "#005C4B",
    "user_fg":     "#E9EDEF",
    "bot_bubble":  "#202C33",
    "bot_fg":      "#E9EDEF",
    "time_fg":     "#8696A0",
    "input_bg":    "#2A3942",
    "input_fg":    "#E9EDEF",
    "send_bg":     "#00A884",
    "send_hover":  "#06CF9C",
    "send_fg":     "#FFFFFF",
}


class MedBotGUI:
    def __init__(self, root: Tk):
        self.root = root
        root.title("MedBot")
        root.geometry("540x700")
        root.minsize(420, 540)
        root.configure(bg=COLORS["bg"])

        self._build_menu()
        self._build_header()
        self._build_chat_area()
        self._build_input_area()

        self._add_bot_message(
            "Hi! I'm MedBot — your medical assistant.\n"
            "Tell me a symptom, ask about a medicine, or how to find a doctor."
        )

    def _build_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        file_m = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_m)
        file_m.add_command(label="Clear Chat", command=self.clear_chat)
        file_m.add_separator()
        file_m.add_command(label="Exit", command=self.root.destroy)

        help_m = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_m)
        help_m.add_command(label="About MedBot", command=self._about)
        help_m.add_command(label="Developers", command=self._developers)

    def _build_header(self):
        header = Frame(self.root, bg=COLORS["header"], height=72)
        header.pack(side=TOP, fill=X)
        header.pack_propagate(False)

        avatar = Label(
            header, text="\U0001FA7A", bg=COLORS["header"], fg="white",
            font=("Segoe UI Emoji", 24),
        )
        avatar.pack(side=LEFT, padx=(16, 12), pady=12)

        text_frame = Frame(header, bg=COLORS["header"])
        text_frame.pack(side=LEFT, fill=Y, pady=14)
        Label(
            text_frame, text="MedBot",
            bg=COLORS["header"], fg=COLORS["header_fg"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor=W)
        Label(
            text_frame, text="online · medical assistant",
            bg=COLORS["header"], fg=COLORS["header_sub"],
            font=("Segoe UI", 9),
        ).pack(anchor=W)

    def _build_chat_area(self):
        wrapper = Frame(self.root, bg=COLORS["bg"])
        wrapper.pack(fill=BOTH, expand=True)

        self.canvas = Canvas(
            wrapper, bg=COLORS["bg"], highlightthickness=0, bd=0,
        )
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)

        sb = Scrollbar(wrapper, orient=VERTICAL, command=self.canvas.yview)
        sb.pack(side=RIGHT, fill=Y)
        self.canvas.configure(yscrollcommand=sb.set)

        self.chat_frame = Frame(self.canvas, bg=COLORS["bg"])
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.chat_frame, anchor="nw",
        )

        self.chat_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.window_id, width=e.width),
        )
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

    def _build_input_area(self):
        bar = Frame(self.root, bg=COLORS["header"], height=66)
        bar.pack(side=BOTTOM, fill=X)
        bar.pack_propagate(False)

        self.entry = Entry(
            bar, bg=COLORS["input_bg"], fg=COLORS["input_fg"],
            insertbackground=COLORS["input_fg"], relief=FLAT,
            font=("Segoe UI", 11), bd=0,
        )
        self.entry.pack(
            side=LEFT, fill=BOTH, expand=True,
            padx=(14, 8), pady=14, ipady=9, ipadx=12,
        )
        self.entry.bind("<Return>", lambda e: self._on_send())
        self.entry.focus_set()

        self.send_btn = Button(
            bar, text="➜", bg=COLORS["send_bg"], fg=COLORS["send_fg"],
            font=("Segoe UI", 15, "bold"), relief=FLAT, bd=0,
            activebackground=COLORS["send_hover"], activeforeground="white",
            cursor="hand2", command=self._on_send, padx=20,
        )
        self.send_btn.pack(side=RIGHT, padx=(0, 14), pady=14, ipady=4)
        self.send_btn.bind("<Enter>", lambda e: self.send_btn.config(bg=COLORS["send_hover"]))
        self.send_btn.bind("<Leave>", lambda e: self.send_btn.config(bg=COLORS["send_bg"]))

    def _add_bubble(self, text: str, sender: str):
        is_user = sender == "user"
        bubble_bg = COLORS["user_bubble"] if is_user else COLORS["bot_bubble"]
        bubble_fg = COLORS["user_fg"] if is_user else COLORS["bot_fg"]

        row = Frame(self.chat_frame, bg=COLORS["bg"])
        row.pack(fill=X, padx=12, pady=4)

        bubble = Frame(row, bg=bubble_bg)
        if is_user:
            bubble.pack(side=RIGHT, padx=(70, 0))
        else:
            bubble.pack(side=LEFT, padx=(0, 70))

        msg = Label(
            bubble, text=text, bg=bubble_bg, fg=bubble_fg,
            font=("Segoe UI", 10), wraplength=340, justify=LEFT,
            padx=14, pady=8,
        )
        msg.pack(anchor=W)

        ts = Label(
            bubble, text=time.strftime("%I:%M %p"),
            bg=bubble_bg, fg=COLORS["time_fg"],
            font=("Segoe UI", 7), padx=12,
        )
        ts.pack(anchor=E, pady=(0, 4))

        self.root.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def _add_user_message(self, text: str):
        self._add_bubble(text, "user")

    def _add_bot_message(self, text: str):
        self._add_bubble(text, "bot")

    def _on_send(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, END)
        self._add_user_message(text)
        try:
            reply = chat(text)
        except Exception as e:
            reply = f"Sorry, something went wrong: {e}"
        self.root.after(180, lambda: self._add_bot_message(reply))

    def clear_chat(self):
        for w in self.chat_frame.winfo_children():
            w.destroy()
        self._add_bot_message("Chat cleared. How can I help?")

    def _about(self):
        tkinter.messagebox.showinfo(
            "About MedBot",
            "MedBot v2.0\n\n"
            "A neural-network chatbot for medical FAQs.\n"
            "Built with NLTK + TensorFlow Keras.\n"
            "GUI: Tkinter.",
        )

    def _developers(self):
        tkinter.messagebox.showinfo(
            "MedBot Developers",
            "1. Samarth Kumar Pal\n"
            "2. Rakesh Kumar\n"
            "3. Amber Kakkar\n"
            "4. Akash Upadhyay",
        )


if __name__ == "__main__":
    root = Tk()
    MedBotGUI(root)
    root.mainloop()
