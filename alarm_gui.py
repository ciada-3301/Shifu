#!/usr/bin/env python3
"""
alarm_gui.py — Shifu Alarm Popup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standalone Tkinter window that pops up when a Shifu alarm automation fires.
Launched as a subprocess by  tools/alarm.py:TriggerAlarmTool._run().

Usage (called by the daemon — you normally don't run this directly):
    python alarm_gui.py --label "Team standup" --message "Daily 10am sync" --sound default

CLI args
────────
  --label   SHORT alarm title  (window title + large header text)
  --message BODY text shown beneath the title
  --sound   default | urgent | silent
              default  → single system bell on open
              urgent   → repeated bell for 3 seconds
              silent   → no bell

Window behaviour
────────────────
  • Always-on-top, centred on screen.
  • Animated ring icon that pulses on open.
  • Snooze button: 5 / 10 / 15 min — re-launches the process after a delay.
  • Dismiss button: closes the window.
  • Auto-closes after 5 minutes if not interacted with.

Snooze mechanism
────────────────
Clicking "Snooze Xm" calls  alarm_gui.py  again via subprocess after X minutes
using  sched / threading  (not APScheduler — this file has no external deps
beyond stdlib + tkinter).
"""

import argparse
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from datetime import datetime


# ── CLI args ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shifu Alarm Popup")
    p.add_argument("--label",   default="Alarm",   help="Short alarm title")
    p.add_argument("--message", default="",        help="Body message")
    p.add_argument("--sound",   default="default", choices=["default", "urgent", "silent"])
    return p.parse_args()


# ── Snooze ─────────────────────────────────────────────────────────────────────

def _snooze(minutes: int, label: str, message: str, sound: str):
    """Re-launch this script after `minutes` minutes in a background thread."""
    def _delayed():
        time.sleep(minutes * 60)
        cmd = [
            sys.executable, __file__,
            "--label",   label,
            "--message", message,
            "--sound",   sound,
        ]
        kwargs: dict = {"start_new_session": True}
        if os.name == "nt":
            import subprocess as sp
            kwargs = {"creationflags": sp.CREATE_NEW_PROCESS_GROUP}
        subprocess.Popen(cmd, **kwargs)

    t = threading.Thread(target=_delayed, daemon=True)
    t.start()


# ── GUI ────────────────────────────────────────────────────────────────────────

# Colour palette — dark orange matches Shifu's terminal theme
_BG        = "#1a1a1a"
_ACCENT    = "#d4780a"
_ACCENT_LT = "#f0a030"
_TEXT      = "#f0ece4"
_DIM       = "#888880"
_BTN_BG    = "#2a2a2a"
_BTN_HV    = "#3a3a3a"
_DISMISS   = "#c0392b"
_DISMISS_H = "#e74c3c"

_AUTO_CLOSE_SECS = 300   # 5 minutes


class AlarmWindow:
    def __init__(self, label: str, message: str, sound: str):
        self.label   = label
        self.message = message
        self.sound   = sound

        self.root = tk.Tk()
        self._build_window()
        self._build_ui()
        self._play_sound()
        self._start_auto_close()
        self._start_pulse()

    # ── Window setup ──────────────────────────────────────────────────────────

    def _build_window(self):
        r = self.root
        r.title(f"⏰  {self.label}")
        r.configure(bg=_BG)
        r.resizable(False, False)
        r.attributes("-topmost", True)
        r.lift()
        r.focus_force()

        # Centre on screen
        r.update_idletasks()
        w, h = 460, 340
        sw   = r.winfo_screenwidth()
        sh   = r.winfo_screenheight()
        x    = (sw - w) // 2
        y    = (sh - h) // 2
        r.geometry(f"{w}x{h}+{x}+{y}")
        r.minsize(460, 340)

        # Window close → dismiss
        r.protocol("WM_DELETE_WINDOW", self._dismiss)

    # ── UI components ─────────────────────────────────────────────────────────

    def _build_ui(self):
        r = self.root

        # ── Top accent bar ─────────────────────────────────────────────────
        bar = tk.Frame(r, bg=_ACCENT, height=4)
        bar.pack(fill="x", side="top")

        # ── Main content frame ────────────────────────────────────────────
        body = tk.Frame(r, bg=_BG, padx=30, pady=20)
        body.pack(fill="both", expand=True)

        # ── Animated ring canvas ──────────────────────────────────────────
        self._canvas = tk.Canvas(body, width=80, height=80,
                                 bg=_BG, highlightthickness=0)
        self._canvas.pack(pady=(0, 8))
        self._ring = self._canvas.create_oval(8, 8, 72, 72,
                                              outline=_ACCENT, width=4)
        self._bell = self._canvas.create_text(40, 40, text="⏰",
                                              font=("Segoe UI Emoji", 28),
                                              fill=_TEXT)
        self._pulse_dir = 1
        self._pulse_val = 0

        # ── Title ─────────────────────────────────────────────────────────
        title_font = tkfont.Font(family="Segoe UI", size=18, weight="bold")
        self._title_lbl = tk.Label(body, text=self.label,
                                   font=title_font, fg=_ACCENT_LT, bg=_BG,
                                   wraplength=380)
        self._title_lbl.pack(pady=(0, 6))

        # ── Message ───────────────────────────────────────────────────────
        if self.message and self.message != self.label:
            msg_font = tkfont.Font(family="Segoe UI", size=11)
            self._msg_lbl = tk.Label(body, text=self.message,
                                     font=msg_font, fg=_TEXT, bg=_BG,
                                     wraplength=380, justify="center")
            self._msg_lbl.pack(pady=(0, 4))

        # ── Timestamp ─────────────────────────────────────────────────────
        ts_font = tkfont.Font(family="Segoe UI", size=9)
        ts_lbl  = tk.Label(body,
                           text=datetime.now().strftime("Fired at %H:%M, %a %d %b %Y"),
                           font=ts_font, fg=_DIM, bg=_BG)
        ts_lbl.pack(pady=(0, 16))

        # ── Auto-close countdown ──────────────────────────────────────────
        self._countdown_var = tk.StringVar(value="")
        cd_lbl = tk.Label(body, textvariable=self._countdown_var,
                          font=ts_font, fg=_DIM, bg=_BG)
        cd_lbl.pack()

        # ── Buttons frame ─────────────────────────────────────────────────
        btn_frame = tk.Frame(r, bg=_BG, pady=16, padx=30)
        btn_frame.pack(fill="x", side="bottom")

        # Snooze buttons
        snooze_frame = tk.Frame(btn_frame, bg=_BG)
        snooze_frame.pack(side="left")

        snooze_lbl = tk.Label(snooze_frame, text="Snooze:", font=("Segoe UI", 9),
                              fg=_DIM, bg=_BG)
        snooze_lbl.pack(side="left", padx=(0, 6))

        for mins in (5, 10, 15):
            btn = self._make_btn(snooze_frame, f"{mins}m",
                                 lambda m=mins: self._snooze_and_close(m),
                                 width=4)
            btn.pack(side="left", padx=3)

        # Dismiss button
        dismiss_btn = tk.Button(
            btn_frame,
            text="  Dismiss  ",
            font=("Segoe UI", 10, "bold"),
            fg=_TEXT, bg=_DISMISS,
            activebackground=_DISMISS_H, activeforeground=_TEXT,
            relief="flat", cursor="hand2", bd=0, padx=12, pady=6,
            command=self._dismiss,
        )
        dismiss_btn.pack(side="right")
        self._hover(dismiss_btn, _DISMISS, _DISMISS_H)

        # Bottom accent bar
        tk.Frame(r, bg=_ACCENT, height=2).pack(fill="x", side="bottom")

    def _make_btn(self, parent, text: str, command, width: int = 6) -> tk.Button:
        btn = tk.Button(
            parent,
            text=text, width=width,
            font=("Segoe UI", 9),
            fg=_TEXT, bg=_BTN_BG,
            activebackground=_BTN_HV, activeforeground=_TEXT,
            relief="flat", cursor="hand2", bd=0, padx=6, pady=5,
            command=command,
        )
        self._hover(btn, _BTN_BG, _BTN_HV)
        return btn

    @staticmethod
    def _hover(widget: tk.Button, normal: str, hover: str):
        widget.bind("<Enter>", lambda _: widget.config(bg=hover))
        widget.bind("<Leave>", lambda _: widget.config(bg=normal))

    # ── Sound ─────────────────────────────────────────────────────────────────

    def _play_sound(self):
        if self.sound == "silent":
            return
        if self.sound == "urgent":
            for _ in range(6):
                self.root.bell()
                self.root.after(500)
        else:
            self.root.bell()

    # ── Pulse animation ───────────────────────────────────────────────────────

    def _start_pulse(self):
        self._pulse_step()

    def _pulse_step(self):
        try:
            self._pulse_val += self._pulse_dir * 2
            if self._pulse_val >= 60:
                self._pulse_dir = -1
            elif self._pulse_val <= 0:
                self._pulse_dir = 1

            r   = int(0xd4 + (0xf0 - 0xd4) * self._pulse_val / 60)
            g   = int(0x78 + (0xa0 - 0x78) * self._pulse_val / 60)
            b   = int(0x0a + (0x30 - 0x0a) * self._pulse_val / 60)
            col = f"#{r:02x}{g:02x}{b:02x}"

            self._canvas.itemconfig(self._ring, outline=col)
            self.root.after(40, self._pulse_step)
        except tk.TclError:
            pass  # window was destroyed

    # ── Auto-close countdown ──────────────────────────────────────────────────

    def _start_auto_close(self):
        self._remaining = _AUTO_CLOSE_SECS
        self._tick()

    def _tick(self):
        try:
            if self._remaining <= 0:
                self._dismiss()
                return
            if self._remaining <= 30:
                self._countdown_var.set(f"Auto-dismissing in {self._remaining}s")
            self._remaining -= 1
            self.root.after(1000, self._tick)
        except tk.TclError:
            pass

    # ── Actions ───────────────────────────────────────────────────────────────

    def _snooze_and_close(self, minutes: int):
        _snooze(minutes, self.label, self.message, self.sound)
        self._dismiss()

    def _dismiss(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    AlarmWindow(
        label   = args.label,
        message = args.message or args.label,
        sound   = args.sound,
    ).run()


if __name__ == "__main__":
    main()