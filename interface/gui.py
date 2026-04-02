from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import datetime, timedelta
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from typing import Callable, Literal

import customtkinter as ctk  # type: ignore[reportMissingImports]
from PIL import Image

from core.config import AppConfig, load_config
from first_run import build_settings_yaml, check_tesseract
from services.llm_provider import LlmProvider
from services.ocr_service import OcrResult, OcrService
from services.research_service import ResearchService
from services.voice_service import VoiceService


_bootstrap_pid = os.environ.get("SAGE_V5_GUI_BOOTSTRAP_PID")
if _bootstrap_pid and _bootstrap_pid != str(os.getpid()):
    print("[INFO] Sage v5 GUI child launch suppressed.")
    raise SystemExit(0)
os.environ["SAGE_V5_GUI_BOOTSTRAP_PID"] = str(os.getpid())


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def _acquire_single_instance_lock() -> socket.socket | None:
    """Bind a localhost lock port so only one GUI process can run."""
    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        try:
            lock_socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        except OSError:
            pass
    try:
        lock_socket.bind(("127.0.0.1", 50973))
        lock_socket.listen(1)
        return lock_socket
    except OSError:
        try:
            lock_socket.close()
        except Exception:
            pass
        return None


class SageDesktopGUI(ctk.CTk):
    MAX_OCR_PROMPT_CHARS = 1500  # 2B model; larger prompts cause OOM/server crash
    MODEL_STAMMER_TOKEN = "[MODEL_STAMMER]"

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.llm = LlmProvider(cfg)
        self.ocr = OcrService(cfg)
        self.research = ResearchService(cfg)
        self.voice = VoiceService(cfg)

        self.project_root = Path(__file__).resolve().parents[1]
        self.inbox_dir = self.project_root / self.cfg.paths.ocr_inbox_dir
        self.notes_dir = self.project_root / "data" / "notes"
        self.memory_file = self.project_root / self.cfg.paths.memory_dir / "session_history.jsonl"
        self.rolling_memory_file = Path("C:/Sage_v5/convo-memory.txt")
        self.research_status_file = self.project_root / "data" / "research" / "research_status.json"
        self.research_report_file = self.project_root / "data" / "research" / "RESEARCH_REPORT.md"
        self.ollama_log_file = self.project_root / "data" / "logs" / "engine_watchdog.log"
        self.ollama_log_file.parent.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.rolling_memory_file.touch(exist_ok=True)

        self.pending_images: list[Path] = []
        self.notes_files: list[Path] = []
        self.last_summary_text: str = ""
        self.chat_lane: str = "gemma"
        self._ollama_restart_lock = threading.Lock()
        self._last_ollama_restart_ts = 0.0
        self._ollama_proc: subprocess.Popen | None = None
        self._watchdog_consecutive_misses: int = 0  # only act after 3 in a row
        self._watchdog_suppressed_until: float = 0.0
        self._gui_busy: bool = False                 # tab switch / active request guard
        self._chat_cancel_event = threading.Event()
        self._research_cancel_requested = False
        self._research_complex_mode = False
        self._research_gemma_conclusion = ""
        self._research_gemma_inflight = False
        self._translate_after_id: str | None = None
        self._active_chat_future: concurrent.futures.Future | None = None
        self._active_chat_future_lock = threading.Lock()
        self._voice_auto_listen: bool = False
        self._voice_auto_lane: Literal["sage_local", "gemma"] | None = None
        self._cancel_requested: bool = False
        self._active_stt_proc: subprocess.Popen | None = None
        self._active_stt_proc_lock = threading.Lock()
        self._active_view: str = "dashboard"
        self._text_only_mode = tk.BooleanVar(value=True)
        self._last_mode_notice: str = ""
        self._chat_open_hint_shown: bool = False
        self._is_shutting_down: bool = False
        self._reminder_poll_ms: int = 30000
        self._recent_reminder_hits: dict[str, float] = {}

        self.title("Sage v5 [C-DRIVE SYNC: March 31]")
        default_w = int(round(float(self.winfo_fpixels("120m"))))
        default_h = int(round(float(self.winfo_fpixels("90m"))))
        self.geometry(f"{default_w}x{default_h}")
        # Allow shrinking/expanding substantially from the original 1100x700 footprint.
        self.minsize(1, max(260, default_h - 60))
        self.maxsize(1600, 1200)
        self.resizable(True, True)

        self.grid_columnconfigure(0, weight=0, minsize=210)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content_host()
        self._build_status_indicator()
        self._enable_clipboard_shortcuts()

        self._show_view("dashboard")
        self._refresh_dashboard()
        self._refresh_ocr_inbox()
        self._refresh_notes()
        self._refresh_memory_roll()
        self._update_ollama_status()
        self._poll_voice_busy()
        self._ollama_watchdog_tick()
        self.after(1200, self._memory_live_refresh_tick)
        self.after(3000, self._reminder_tick)
        self.after(500, self._start_model_warmup)
        self.protocol("WM_DELETE_WINDOW", self._on_app_close)

    def _on_app_close(self) -> None:
        if self._is_shutting_down:
            return
        self._is_shutting_down = True

        self._voice_auto_listen = False
        self._voice_auto_lane = None
        self._cancel_requested = True
        self._chat_cancel_event.set()

        with self._active_chat_future_lock:
            future = self._active_chat_future
            self._active_chat_future = None
        if future is not None and not future.done():
            try:
                future.cancel()
            except Exception:
                pass

        with self._active_stt_proc_lock:
            stt_proc = self._active_stt_proc
            self._active_stt_proc = None
        if stt_proc is not None:
            try:
                stt_proc.terminate()
            except Exception:
                pass

        try:
            self.voice.stop()
        except Exception:
            pass

        try:
            self.llm.run_coroutine(self.llm.aclose())
        except Exception:
            pass

        proc = self._ollama_proc
        self._ollama_proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        self.destroy()

    def _enable_clipboard_shortcuts(self) -> None:
        # Global clipboard shortcuts so CTk input dialogs and in-app fields
        # reliably support copy/paste/select-all on Windows.
        self.bind_all("<Control-c>", self._clipboard_copy, add="+")
        self.bind_all("<Control-C>", self._clipboard_copy, add="+")
        self.bind_all("<Control-x>", self._clipboard_cut, add="+")
        self.bind_all("<Control-X>", self._clipboard_cut, add="+")
        self.bind_all("<Control-v>", self._clipboard_paste, add="+")
        self.bind_all("<Control-V>", self._clipboard_paste, add="+")
        self.bind_all("<Control-a>", self._clipboard_select_all, add="+")
        self.bind_all("<Control-A>", self._clipboard_select_all, add="+")

        self._edit_menu = tk.Menu(self, tearoff=0)
        self._edit_menu.add_command(label="Cut", command=self._menu_cut)
        self._edit_menu.add_command(label="Copy", command=self._menu_copy)
        self._edit_menu.add_command(label="Paste", command=self._menu_paste)
        self._edit_menu.add_separator()
        self._edit_menu.add_command(label="Select All", command=self._menu_select_all)
        self.bind_all("<Button-3>", self._show_edit_menu, add="+")

    def _focused_edit_widget(self):
        widget = self.focus_get()
        if widget is None:
            return None
        if isinstance(widget, (tk.Entry, tk.Text, tk.Listbox)):
            return widget
        return widget if hasattr(widget, "event_generate") else None

    def _clipboard_copy(self, _event=None):
        widget = self._focused_edit_widget()
        if widget is not None:
            widget.event_generate("<<Copy>>")
            return "break"
        return None

    def _clipboard_cut(self, _event=None):
        widget = self._focused_edit_widget()
        if widget is not None:
            widget.event_generate("<<Cut>>")
            return "break"
        return None

    def _clipboard_paste(self, _event=None):
        widget = self._focused_edit_widget()
        if widget is not None:
            widget.event_generate("<<Paste>>")
            return "break"
        return None

    def _clipboard_select_all(self, _event=None):
        widget = self._focused_edit_widget()
        if widget is None:
            return None

        try:
            if isinstance(widget, tk.Entry):
                widget.select_range(0, tk.END)
                widget.icursor(tk.END)
                return "break"
            if isinstance(widget, tk.Text):
                widget.tag_add("sel", "1.0", "end-1c")
                return "break"
            widget.event_generate("<<SelectAll>>")
            return "break"
        except Exception:
            return None

    def _menu_cut(self) -> None:
        self._clipboard_cut()

    def _menu_copy(self) -> None:
        self._clipboard_copy()

    def _menu_paste(self) -> None:
        self._clipboard_paste()

    def _menu_select_all(self) -> None:
        self._clipboard_select_all()

    def _show_edit_menu(self, event) -> None:
        widget = event.widget
        if not isinstance(widget, (tk.Entry, tk.Text, tk.Listbox)):
            return
        try:
            self._edit_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._edit_menu.grab_release()

    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkFrame(self, corner_radius=0, width=210)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(9, weight=1)

        title = ctk.CTkLabel(self.sidebar, text="Sage v5", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=16, pady=(14, 8), sticky="w")

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self.nav_images: dict[str, ctk.CTkImage] = {}
        icon_path = self.project_root / "data" / "assets" / "icons"
        nav_items = [
            ("dashboard", "dashboard.png", "Dashboard"),
            ("ocr", "ocr.png", "OCR Inbox"),
            ("sage_local", "chat.png", "Chat / Voice"),
            ("translate", "translate.png", "Translate"),
            ("research", "research.png", "Research"),
            ("memory", "memory.png", "Chat Memory"),
            ("notes", "notes.png", "Notes"),
        ]
        for idx, (key, icon_file, name) in enumerate(nav_items, start=1):
            pil_img = Image.open(icon_path / icon_file).convert("RGBA")
            img_obj = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(20, 20))
            self.nav_images[key] = img_obj

            btn = ctk.CTkButton(
                self.sidebar,
                text=name,
                image=img_obj,
                compound="left",
                anchor="w",
                command=lambda k=key: self._show_view(k),
                width=176,
                fg_color="transparent",
                hover_color=("#1F1F1F", "#2B2B2B"),
                border_width=0,
                corner_radius=8,
                border_spacing=10,
            )
            btn.grid(row=idx, column=0, padx=(12, 10), pady=2, sticky="ew")
            self.nav_buttons[key] = btn

    def _build_content_host(self) -> None:
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self.content, fg_color="transparent")
        self.header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        self.header.grid_columnconfigure(0, weight=1)

        self.page_title = ctk.CTkLabel(self.header, text="Dashboard", font=ctk.CTkFont(size=19, weight="bold"))
        self.page_title.grid(row=0, column=0, sticky="w")

        self.view_host = ctk.CTkFrame(self.content)
        self.view_host.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.view_host.grid_columnconfigure(0, weight=1)
        self.view_host.grid_rowconfigure(0, weight=1)

        self.views: dict[str, ctk.CTkFrame] = {}
        self._build_dashboard_view()
        self._build_ocr_view()
        self._build_chat_view()
        self._build_translate_view()
        self._build_research_view()
        self._build_memory_view()
        self._build_notes_view()

    def _build_status_indicator(self) -> None:
        self.status_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        self.status_frame.grid(row=0, column=1, sticky="e")

        self.led_canvas = tk.Canvas(self.status_frame, width=16, height=16, bg=self._apply_appearance_mode("#1F1F1F"), highlightthickness=0)
        self.led_canvas.grid(row=0, column=0, padx=(0, 8))
        self.led_dot = self.led_canvas.create_oval(2, 2, 14, 14, fill="#b71c1c", outline="#000000")

        self.status_label = ctk.CTkLabel(self.status_frame, text="Ollama: offline")
        self.status_label.grid(row=0, column=1, sticky="e")

    def _build_dashboard_view(self) -> None:
        frame = ctk.CTkFrame(self.view_host)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        top.grid_columnconfigure(0, weight=1)

        intro = ctk.CTkLabel(top, text="Recent", font=ctk.CTkFont(size=15, weight="bold"))
        intro.grid(row=0, column=0, sticky="w")

        setup = ctk.CTkButton(
            top,
            text="First\nSetup",
            width=54,
            font=ctk.CTkFont(size=11),
            command=self._run_first_setup,
        )
        setup.grid(row=0, column=1, sticky="e", padx=(10, 6))

        health = ctk.CTkButton(top, text="Health", width=48, command=self._run_health_check)
        health.grid(row=0, column=2, sticky="e", padx=(0, 6))

        refresh = ctk.CTkButton(top, text="Refresh", width=46, command=self._refresh_dashboard)
        refresh.grid(row=0, column=3, sticky="e")

        self.dashboard_text = ctk.CTkTextbox(frame, wrap="word")
        self.dashboard_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self.views["dashboard"] = frame

    def _build_ocr_view(self) -> None:
        frame = ctk.CTkFrame(self.view_host)
        frame.grid_columnconfigure(0, weight=0, minsize=160)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(3, weight=1)

        left_top = ctk.CTkFrame(frame, fg_color="transparent")
        left_top.grid(row=0, column=0, sticky="ew", padx=(8, 6), pady=(8, 6))
        left_top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left_top, text="Select file", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            left_top,
            text="Refresh",
            width=60,
            font=ctk.CTkFont(size=12),
            command=self._refresh_ocr_inbox,
        ).grid(row=0, column=1, sticky="e")

        self.ocr_listbox = tk.Listbox(frame, exportselection=False, width=18)
        self.ocr_listbox.grid(row=1, column=0, sticky="nsew", padx=(8, 6), pady=(0, 8))

        right = ctk.CTkFrame(frame)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(6, 8), pady=(8, 4))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        actions = ctk.CTkFrame(right, fg_color="transparent")
        actions.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 4))
        actions.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(actions, text="Process\nOCR", width=92, command=self._process_selected_image).grid(row=0, column=0, sticky="ew")

        self.ocr_output = ctk.CTkTextbox(right, wrap="word")
        self.ocr_output.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        self.ocr_log_label = ctk.CTkLabel(frame, text="Scanned Text", font=ctk.CTkFont(size=14, weight="bold"))
        self.ocr_log_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 3))

        self.ocr_log = ctk.CTkTextbox(frame, wrap="word")
        self.ocr_log.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 8))

        self.views["ocr"] = frame

    def _build_chat_view(self) -> None:
        frame = ctk.CTkFrame(self.view_host)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        lane_bar = ctk.CTkFrame(frame, fg_color="transparent")
        lane_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        lane_bar.grid_columnconfigure(0, weight=0)
        lane_bar.grid_columnconfigure(1, weight=0)
        lane_bar.grid_columnconfigure(2, weight=1)
        lane_bar.grid_columnconfigure(3, weight=0)

        lane_btn_font = ctk.CTkFont(size=12)
        ctk.CTkButton(lane_bar, text="S", width=40, font=lane_btn_font, command=lambda: self._set_chat_lane("sage_local", "S")).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.chat_cs_btn = ctk.CTkButton(lane_bar, text="CS", width=44, font=lane_btn_font, command=lambda: self._start_lane_voice_capture("sage_local", "CS"))
        self.chat_cs_btn.grid(row=0, column=1, padx=4, sticky="ew")

        self.chat_lane_label = ctk.CTkLabel(
            lane_bar,
            text="Gemma",
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self.chat_lane_label.grid(row=0, column=2, sticky="ew", padx=(8, 0))

        self.chat_text_mode_btn = ctk.CTkButton(
            lane_bar,
            text="Text\nOnly",
            width=56,
            font=ctk.CTkFont(size=10),
            command=self._toggle_text_only_mode,
        )
        self.chat_text_mode_btn.grid(row=0, column=3, sticky="e", padx=(6, 0))
        self._sync_text_only_button()

        self.chat_transcript = ctk.CTkTextbox(frame, wrap="word")
        self.chat_transcript.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 6))

        bottom = ctk.CTkFrame(frame, fg_color="transparent")
        bottom.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        bottom.grid_columnconfigure(0, weight=1)

        self.chat_command_ribbon = ctk.CTkLabel(
            bottom,
            text="Gemma | /help | /command",
            anchor="w",
            height=20,
            fg_color=("#1A1A1A", "#111111"),
            corner_radius=6,
        )
        self.chat_command_ribbon.grid(row=0, column=0, columnspan=5, sticky="ew", pady=(0, 6))

        self.chat_input = ctk.CTkEntry(bottom, placeholder_text="Ask Gemma...")
        self.chat_input.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.chat_input.bind("<Return>", lambda _e: self._send_chat())

        self.chat_mic_btn = ctk.CTkButton(bottom, text="🎤", width=40, command=self._send_chat_voice)
        self.chat_mic_btn.grid(row=1, column=1, sticky="e", padx=(0, 8))

        self.stop_voice_btn = ctk.CTkButton(bottom, text="⏹", width=40, command=self._stop_voice)
        self.stop_voice_btn.grid(row=1, column=2, sticky="e", padx=(0, 8))
        self.stop_voice_btn.grid_remove()

        send = ctk.CTkButton(bottom, text="Send", width=86, command=self._send_chat)
        send.grid(row=1, column=3, sticky="e")

        self.stop_chat_btn = ctk.CTkButton(bottom, text="Stop", width=72, command=self._cancel_active_chat)
        self.stop_chat_btn.grid(row=1, column=4, sticky="e", padx=(8, 0))
        self.stop_chat_btn.configure(state="disabled")
        self.stop_btn = self.stop_chat_btn

        self._apply_chat_mode_controls()

        self.views["sage_local"] = frame

    def _is_text_only_mode(self) -> bool:
        try:
            return bool(self._text_only_mode.get())
        except Exception:
            return False

    def _apply_chat_mode_controls(self) -> None:
        disabled = self._is_text_only_mode()
        state = "disabled" if disabled else "normal"
        for widget_name in ("chat_mic_btn",):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except Exception:
                    pass

    def _toggle_text_only_mode(self) -> None:
        self._text_only_mode.set(not self._is_text_only_mode())
        self._on_chat_mode_toggle()

    def _sync_text_only_button(self) -> None:
        btn = getattr(self, "chat_text_mode_btn", None)
        if btn is None:
            return
        if self._is_text_only_mode():
            btn.configure(
                text="Text\nOnly ON",
                fg_color=("#2e7d32", "#1b5e20"),
                hover_color=("#388e3c", "#2e7d32"),
            )
        else:
            btn.configure(
                text="Text\nOnly",
                fg_color=("#3B8ED0", "#1F6AA5"),
                hover_color=("#36719F", "#144870"),
            )

    def _on_chat_mode_toggle(self) -> None:
        self._sync_text_only_button()
        self._apply_chat_mode_controls()
        if self._is_text_only_mode():
            self._set_chat_lane("gemma", "TEXT")
            self._voice_auto_listen = False
            self._voice_auto_lane = None
            self.voice.stop()
            self._append_mode_notice("[MODE] Text-only mode enabled. Gemma text prompting is active.")
        else:
            self._append_mode_notice("[MODE] Voice+Text mode enabled.")

    def _append_mode_notice(self, message: str) -> None:
        text = str(message or "").strip()
        if not text or text == self._last_mode_notice:
            return
        try:
            self.chat_transcript.insert("end", text + "\n")
            self.chat_transcript.see("end")
            self._last_mode_notice = text
        except Exception:
            pass

    def _build_translate_view(self) -> None:
        frame = ctk.CTkFrame(self.view_host)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(5, weight=1)

        source_top = ctk.CTkFrame(frame, fg_color="transparent")
        source_top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        source_top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(source_top, text="Text for Translation", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.translate_clear_btn = ctk.CTkButton(
            source_top,
            text="Clear Text for Translation",
            width=120,
            command=self._clear_translate_fields,
        )
        self.translate_clear_btn.grid(row=0, column=1, sticky="e")
        self.translate_clear_btn.grid_remove()

        ctk.CTkLabel(frame, text="Target Language", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=2, column=0, sticky="w", padx=10, pady=(8, 4)
        )

        self.translate_source_text = ctk.CTkTextbox(frame, wrap="word")
        self.translate_source_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        self.translate_source_text.bind("<KeyRelease>", self._schedule_translate_from_source)

        self.translate_target_lang_entry = ctk.CTkEntry(frame, placeholder_text="name the language to translate to")
        self.translate_target_lang_entry.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.translate_target_lang_entry.bind("<KeyRelease>", self._schedule_translate_from_source)

        ctk.CTkLabel(frame, text="Translated Text", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=4, column=0, sticky="w", padx=10, pady=(2, 4)
        )

        self.translate_output_text = ctk.CTkTextbox(frame, wrap="word")
        self.translate_output_text.grid(row=5, column=0, sticky="nsew", padx=10, pady=(0, 8))

        action_bar = ctk.CTkFrame(frame, fg_color="transparent")
        action_bar.grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 10))
        action_bar.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(action_bar, text="Translate", width=120, command=self._run_translate_from_source).grid(
            row=0, column=1, sticky="e"
        )
        self.views["translate"] = frame

    def _clear_translate_fields(self) -> None:
        if hasattr(self, "translate_source_text"):
            self.translate_source_text.delete("1.0", "end")
        if hasattr(self, "translate_output_text"):
            self.translate_output_text.delete("1.0", "end")
        if hasattr(self, "translate_clear_btn"):
            self.translate_clear_btn.grid_remove()

    def _build_research_view(self) -> None:
        frame = ctk.CTkFrame(self.view_host)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        self.research_output = ctk.CTkTextbox(frame, wrap="word")
        self.research_output.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))

        bottom = ctk.CTkFrame(frame, fg_color="transparent")
        bottom.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=1, uniform="research_btns")
        bottom.grid_columnconfigure(2, weight=1, uniform="research_btns")
        bottom.grid_columnconfigure(3, weight=1, uniform="research_btns")

        self.research_topic_entry = ctk.CTkEntry(bottom, placeholder_text="Research Topic")
        self.research_topic_entry.grid(row=0, column=0, columnspan=4, sticky="ew", padx=(0, 0), pady=(0, 8))
        self.research_topic_entry.bind("<Return>", lambda _e: self._start_research())

        self.research_send_btn = ctk.CTkButton(bottom, text="Send", width=96, command=self._start_research)
        self.research_send_btn.grid(row=1, column=1, sticky="ew", padx=(0, 8))

        self.research_stop_btn = ctk.CTkButton(bottom, text="Stop", width=96, command=self._stop_research)
        self.research_stop_btn.grid(row=1, column=2, sticky="ew", padx=(0, 8))

        self.research_clear_btn = ctk.CTkButton(bottom, text="Clear", width=96, command=self._clear_research_output)
        self.research_clear_btn.grid(row=1, column=3, sticky="ew")

        self._render_research_idle_state()
        self.views["research"] = frame

    def _render_research_idle_state(self) -> None:
        if not hasattr(self, "research_output"):
            return
        self.research_output.delete("1.0", "end")
        self.research_output.insert("end", "[Model] n/a | [Start Time] n/a | [Status] idle\n[Finished] n/a\n\n[Conclusion]\n")

    def _clear_research_output(self) -> None:
        self._research_cancel_requested = False
        self._research_complex_mode = False
        self._research_gemma_conclusion = ""
        self._research_gemma_inflight = False

        if hasattr(self, "research_topic_entry"):
            try:
                self.research_topic_entry.delete(0, "end")
            except Exception:
                pass

        for path in (self.research_status_file, self.research_report_file):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

        self._render_research_idle_state()

    def _build_memory_view(self) -> None:
        frame = ctk.CTkFrame(self.view_host)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(top, text="Summarize\nMemory", width=59, command=self._summarize_memory_roll).grid(row=0, column=1, sticky="e", padx=(0, 8))
        ctk.CTkButton(top, text="Clear\nMemory", width=59, command=self._clear_memory_roll).grid(row=0, column=2, sticky="e")

        self.memory_roll = ctk.CTkTextbox(frame, wrap="word")
        self.memory_roll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.views["memory"] = frame

    def _build_notes_view(self) -> None:
        frame = ctk.CTkFrame(self.view_host)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=2)
        frame.grid_rowconfigure(2, weight=1)

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 6))
        top.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(top, text="Search:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.notes_search = ctk.CTkEntry(top, placeholder_text="filename or text")
        self.notes_search.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.notes_search.bind("<Return>", lambda _e: self._refresh_notes())
        ctk.CTkButton(top, text="Find", width=78, command=self._refresh_notes).grid(row=0, column=2, sticky="e")

        self.notes_listbox = tk.Listbox(frame, exportselection=False)
        self.notes_listbox.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(8, 6), pady=(0, 8))
        self.notes_listbox.bind("<<ListboxSelect>>", lambda _e: self._load_selected_note())

        self.notes_viewer = ctk.CTkTextbox(frame, wrap="word")
        self.notes_viewer.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=(6, 8), pady=(0, 8))

        self.views["notes"] = frame

    def _show_view(self, key: str) -> None:
        self._gui_busy = True
        self._active_view = key
        labels = {
            "dashboard": "Dashboard",
            "ocr": "OCR",
            "sage_local": "Chat / Voice",
            "translate": "Translate",
            "research": "Research",
            "memory": "Chat Memory",
            "notes": "Notes",
        }
        for name, frame in self.views.items():
            if name == key:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

        self.page_title.configure(text=labels.get(key, "Sage"))
        for name, btn in self.nav_buttons.items():
            btn.configure(fg_color="transparent")
        if key == "memory":
            self._refresh_memory_roll()
        elif key == "research":
            self._refresh_research_status_once()
        elif key == "sage_local" and not self._chat_open_hint_shown:
            self._append_mode_notice("[MODE] Text mode: Gemma active. Click S or CS for Sage voice.")
            self._chat_open_hint_shown = True
        self._gui_busy = False

    def _memory_live_refresh_tick(self) -> None:
        if self._active_view == "memory":
            self._refresh_memory_roll()
        self.after(1200, self._memory_live_refresh_tick)

    def _set_chat_command_ribbon(self, lane: Literal["sage_local", "gemma"], status: str) -> None:
        model = "Sage" if lane == "sage_local" else "Gemma"
        self.chat_command_ribbon.configure(text=f"{model} | /help | /command")

    def _apply_chat_command_interceptor(self, raw_prompt: str) -> tuple[str, Literal["sage_local", "gemma"]]:
        prompt = str(raw_prompt or "").strip()
        lane: Literal["sage_local", "gemma"] = "sage_local" if self.chat_lane == "sage_local" else "gemma"
        lower = prompt.lower()

        if lower.startswith("chat-sage"):
            prompt = prompt[len("chat-sage") :].lstrip(" :,-")
            lane = "sage_local"
            self._set_chat_lane("sage_local", "CS")
            self._set_chat_command_ribbon(lane, "chat-sage")
        elif lower.startswith("chat-"):
            cmd = lower.split()[0]
            if cmd in {"chat-local", "chat-legacy", "chat-bitney", "chat-bitnet"}:
                prompt = prompt[len(cmd) :].lstrip(" :,-")
                lane = "sage_local"
                self._set_chat_lane("sage_local", "CS")
                self._set_chat_command_ribbon(lane, "chat-legacy-alias")
        elif lower.startswith("chat-gemma"):
            prompt = prompt[len("chat-gemma") :].lstrip(" :,-")
            lane = "gemma"
            self._set_chat_lane("gemma", "CG")
            self._set_chat_command_ribbon(lane, "chat-gemma")
        else:
            self._set_chat_command_ribbon(lane, "direct")

        return prompt, lane

    def _extract_translate_to_intent(self, prompt: str) -> tuple[str, str] | None:
        text = str(prompt or "").strip()
        if not text:
            return None

        patterns = [
            # translate to French: hello world
            r"(?is)^translate\s+(?:this\s+|the\s+following\s+|following\s+)?(?:to|into)\s+([^:\n]+?)\s*[:\-]\s*(.+)$",
            # translate this into French "hello world"
            r"(?is)^translate\s+(?:this\s+|the\s+following\s+|following\s+)?(?:to|into)\s+([^\n]+?)\s+[\"'\u201c](.+)[\"'\u201d]\s*$",
            # translate this into French hello world
            r"(?is)^translate\s+(?:this\s+|the\s+following\s+|following\s+)?(?:to|into)\s+([a-z][a-z\- ]{1,30})\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, text)
            if not match:
                continue
            target_lang = match.group(1).strip(" \t\"'`.,;:!?")
            target_lang = re.sub(r"^[<\[\(\{\s]+", "", target_lang)
            target_lang = re.sub(r"[>\]\)\}\s]+$", "", target_lang)
            source_text = match.group(2).strip().strip("\"'\u201c\u201d")
            if target_lang and source_text:
                return target_lang, source_text
        return None

    def _looks_like_math_or_finance_prompt(self, prompt: str) -> bool:
        text = str(prompt or "")
        lower = text.lower()
        has_number = bool(re.search(r"\d", text))
        has_signal = any(
            key in lower
            for key in (
                "calculate",
                "what is",
                "how much",
                "yield",
                "revenue",
                "growth",
                "percent",
                "%",
                "roi",
                "interest",
                "rate",
                "fund",
                "profit",
                "loss",
                "square root",
                "sqrt",
            )
        )
        return has_number and has_signal

    def _apply_sage_prompt_guardrails(self, prompt: str, lane: Literal["sage_local", "gemma"]) -> str:
        """Add targeted task guardrails for chat prompts, with lane-specific defaults."""
        text = str(prompt or "").strip()
        if not text:
            return text
        model_name = "Sage" if lane == "sage_local" else "Gemma"

        translate_intent = self._extract_translate_to_intent(text)
        if translate_intent is not None:
            target_lang, source_text = translate_intent
            wants_analysis = bool(re.search(r"\bwith\s+analysis\b", text, flags=re.IGNORECASE))
            analysis_clause = (
                "After the translation, provide a brief analysis because the user explicitly asked for Analysis."
                if wants_analysis
                else "Return only the translated string. No intro text, no bullet points, and no explanations."
            )
            return (
                f"You are {model_name}. This is a translation task. "
                "Prioritize culinary accuracy for food terms. "
                "A 'rack of lamb' is 'carre d'agnello' in Italian and 'carre d'agneau' in French. "
                f"Translate the source text into {target_lang}. "
                f"{analysis_clause} Do not include markdown.\n\n"
                f"Source:\n{source_text}"
            )

        if self._looks_like_math_or_finance_prompt(text):
            return (
                f"You are {model_name}. Solve the user's numeric/finance question with strict arithmetic. "
                "Do not invent missing assumptions or starting values. "
                "If key inputs are missing, reply exactly with 'Insufficient data.' and ask one short follow-up question. "
                "Otherwise provide bulleted output (not numbered) with three bullets: Formula, Substituted numbers, Final result.\n\n"
                f"Question:\n{text}"
            )

        if lane == "gemma":
            return (
                "You are Gemma. Keep answers concise and direct by default. "
                "Use plain text and avoid markdown headings/bullets unless the user asks for detailed formatting.\n\n"
                f"User request:\n{text}"
            )

        return text

    def _run_worker(
        self,
        fn: Callable[[], object],
        on_done: Callable[[object, Exception | None], None] | None = None,
    ) -> None:
        def _task() -> None:
            self._gui_busy = True
            try:
                result = fn()
                error = None
            except Exception as exc:
                result = None
                error = exc
            finally:
                self._gui_busy = False

            callback = on_done
            if callback is not None:
                self.after(0, lambda cb=callback, r=result, e=error: cb(r, e))

        threading.Thread(target=_task, daemon=True).start()

    def _log_ollama_event(self, message: str, include_chat: bool = False) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = f"[{stamp}] {message}"
        try:
            self.ollama_log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.ollama_log_file.open("a", encoding="utf-8") as fh:
                fh.write(row + "\n")
        except Exception:
            pass

        # Surface event in OCR log for active workflow transparency.
        try:
            self._append_ocr_log(f"OLLAMA: {message}")
        except Exception:
            pass

        if include_chat:
            try:
                self.chat_transcript.insert("end", f"[OLLAMA] {message}\n")
                self.chat_transcript.see("end")
            except Exception:
                pass

    def _start_ollama_background(self) -> bool:
        """External GGUF server launch is disabled; Sage v5 uses Ollama directly."""
        return False

    def _ensure_ollama_online(self, timeout_sec: float = 18.0) -> bool:
        """Best-effort guard to keep local engine available for local lane requests."""
        if self._is_ollama_alive():
            return True

        deadline = time.time() + max(1.0, float(timeout_sec))
        while time.time() < deadline:
            if self._is_ollama_alive():
                self._log_ollama_event("Engine is back online", include_chat=True)
                return True
            time.sleep(0.3)
        ok = self._is_ollama_alive()
        if not ok:
            self._log_ollama_event("Engine did not recover within timeout", include_chat=True)
        return ok

    def _ollama_watchdog_tick(self) -> None:
        # Skip this tick entirely if the GUI is busy (tab switch or active request).
        if self._gui_busy:
            self.after(10000, self._ollama_watchdog_tick)
            return

        if time.time() < self._watchdog_suppressed_until:
            # Hold watchdog actions while voice-once flow is in progress to avoid
            # engine restart churn during local model warmup.
            self.after(10000, self._ollama_watchdog_tick)
            return

        if self._is_ollama_alive():
            # Engine is healthy — reset miss counter silently.
            self._watchdog_consecutive_misses = 0
        else:
            self._watchdog_consecutive_misses += 1
            # Only log after 3 consecutive misses to avoid
            # false positives from brief momentary disconnects during tab switches.
            if self._watchdog_consecutive_misses >= 3:
                if not self._is_ollama_alive():
                    self._log_ollama_event(
                        f"Watchdog: engine down for {self._watchdog_consecutive_misses} checks.",
                        include_chat=True,
                    )
                self._watchdog_consecutive_misses = 0

        self.after(10000, self._ollama_watchdog_tick)

    def _suppress_watchdog(self, seconds: float = 30.0) -> None:
        hold_until = time.time() + max(1.0, float(seconds))
        if hold_until > self._watchdog_suppressed_until:
            self._watchdog_suppressed_until = hold_until

    def _split_sentences(self, text: str) -> tuple[list[str], str]:
        out: list[str] = []
        current = ""
        value = str(text or "")
        for i, ch in enumerate(value):
            current += ch
            if ch not in ".!?。！？":
                continue
            # Do not split decimals such as 2.5 or 68.4
            if ch == ".":
                prev_ch = value[i - 1] if i > 0 else ""
                next_ch = value[i + 1] if i + 1 < len(value) else ""
                if prev_ch.isdigit() and next_ch.isdigit():
                    continue
                # In streaming output, decimals can arrive as "45." then "5" later.
                # Defer split so the next token can complete the decimal value.
                if prev_ch.isdigit():
                    j = i + 1
                    while j < len(value) and value[j].isspace():
                        j += 1
                    if j >= len(value) or value[j].isdigit():
                        continue
            if ch in ".!?。！？":
                out.append(current.strip())
                current = ""
        return out, current

    def _prepare_spoken_text(self, text: str) -> str:
        """Filter model/meta diagnostics from TTS while keeping them in transcript."""
        value = self._normalize_tts_text(text).strip()
        if not value:
            return ""

        if re.search(r"!{4,}|\.{4,}", value):
            return self.MODEL_STAMMER_TOKEN

        # Ignore punctuation-only chunks (for example streamed "..."),
        # which otherwise cause rapid stop/start TTS artifacts.
        if not any(ch.isalnum() for ch in value):
            return ""

        lower = value.lower()
        if "[model notice]" in lower:
            return ""
        if "sage_local_retries_exhausted" in lower or "bitnet_retries_exhausted" in lower:
            return ""
        if "switching to gemma" in lower:
            return ""
        if lower.startswith("reason:"):
            return ""

        return value

    def _normalize_tts_text(self, text: str) -> str:
        """Convert common math/markdown syntax into TTS-friendly speech text."""
        raw = str(text or "")
        if not raw:
            return ""

        value = raw.replace("\r", " ").replace("\n", " ")

        # Read numeric ranges naturally (for example 15-30 mph -> 15 to 30 mph)
        # while keeping arithmetic phrases like "12 - 5" as subtraction.
        value = re.sub(
            r"\b(\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*(\d+(?:\.\d+)?)(?=\s*(?:mph|mi/h|miles?\s+per\s+hour|km/h|kph|kilometers?\s+per\s+hour|%|percent|years?|months?|weeks?|days?|hours?|minutes?|seconds?))",
            r"\1 to \2",
            value,
            flags=re.IGNORECASE,
        )

        # Common LaTeX math forms.
        value = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1 over \2", value)
        value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"square root of \1", value)

        replacements = {
            r"\\times": " times ",
            r"\\cdot": " times ",
            r"\\div": " divided by ",
            r"\\pm": " plus or minus ",
            r"\\approx": " approximately ",
            r"\\neq": " not equal to ",
            r"\\le": " less than or equal to ",
            r"\\ge": " greater than or equal to ",
            r"\\%": " percent ",
            r"\\$": " dollar ",
            r"\\[": " ",
            r"\\]": " ",
            r"\\(": " ",
            r"\\)": " ",
            "$$": " ",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)

        # Drop remaining command-like tokens (for example \text, \mathrm).
        value = re.sub(r"\\[A-Za-z]+", " ", value)

        # Remove markup/control characters that read badly in TTS.
        value = value.replace("{", " ").replace("}", " ")
        value = re.sub(r"[`*_#]", " ", value)

        # If a chunk is mostly symbols and has no clear words/numbers, skip speaking it.
        compact = re.sub(r"\s+", "", value)
        if compact:
            symbol_count = len(re.findall(r"[^A-Za-z0-9]", compact))
            if symbol_count / max(1, len(compact)) > 0.6 and not re.search(r"[A-Za-z0-9]", compact):
                return ""

        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _set_chat_lane(self, lane: str, trigger: str) -> None:
        if lane == "sage_local" and self._is_text_only_mode():
            # Sage lane implies voice-capable mode for Chat/Voice interactions.
            self._text_only_mode.set(False)
            self._sync_text_only_button()
            self._apply_chat_mode_controls()
        self.chat_lane = lane
        if lane == "sage_local":
            self.chat_lane_label.configure(text="Sage")
            self._set_chat_command_ribbon("sage_local", f"lane:{trigger}")
        else:
            self.chat_lane_label.configure(text="Gemma")
            self._set_chat_command_ribbon("gemma", f"lane:{trigger}")

    def _start_lane_voice_capture(self, lane: str, trigger: str) -> None:
        self._set_chat_lane(lane, trigger)
        self._voice_auto_listen = True
        self._voice_auto_lane = "gemma" if lane == "gemma" else "sage_local"
        self._chat_cancel_event.clear()
        self.chat_input.delete(0, "end")
        self._send_chat_voice()

    def _resume_voice_loop_when_idle(self, delay_ms: int = 300) -> None:
        if not self._voice_auto_listen:
            return
        if self._chat_cancel_event.is_set():
            return
        if self.voice.is_busy():
            self.after(300, self._resume_voice_loop_when_idle)
            return
        self.after(max(0, int(delay_ms)), self._send_chat_voice)

    def _schedule_translate_from_source(self, _event=None) -> None:
        if self._translate_after_id is not None:
            try:
                self.after_cancel(self._translate_after_id)
            except Exception:
                pass
        self._translate_after_id = self.after(600, self._run_translate_from_source)

    def _run_translate_from_source(self) -> None:
        if self._translate_after_id is not None:
            self._translate_after_id = None

        source = self.translate_source_text.get("1.0", "end").strip() if hasattr(self, "translate_source_text") else ""
        if not source:
            return

        target_language = "English"
        if hasattr(self, "translate_target_lang_entry"):
            target_language = (self.translate_target_lang_entry.get() or "").strip() or "English"

        self.translate_output_text.delete("1.0", "end")
        self.translate_output_text.insert("end", "Translating with Gemma...\n")
        if hasattr(self, "translate_clear_btn"):
            self.translate_clear_btn.grid_remove()

        prompt = (
            f"Translate the source text into clear {target_language}. "
            "Prioritize culinary accuracy for food terms. "
            "A 'rack of lamb' is 'carre d'agnello' in Italian and 'carre d'agneau' in French. "
            "Return only the translated text with no explanation, no quotes, and no markdown.\n\n"
            f"Source:\n{source}"
        )

        def _work() -> str:
            return self.llm.run_coroutine(self.llm.chat_text(prompt, target="gemma", stream=False))

        def _done(result, error) -> None:
            self.translate_output_text.delete("1.0", "end")
            if error is not None:
                self.translate_output_text.insert("end", f"[ERROR] {error}")
                if hasattr(self, "translate_clear_btn"):
                    self.translate_clear_btn.grid()
                return
            self.translate_output_text.insert("end", str(result or ""))
            if hasattr(self, "translate_clear_btn"):
                self.translate_clear_btn.grid()

        self._run_worker(_work, _done)

    def _refresh_memory_roll(self) -> None:
        if not hasattr(self, "memory_roll"):
            return

        self.memory_roll.delete("1.0", "end")
        if not self.rolling_memory_file.exists():
            self.memory_roll.insert("end", "No memory history yet.")
            return

        text = self.rolling_memory_file.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            self.memory_roll.insert("end", "No memory entries found.")
            return

        self.memory_roll.insert("end", text)
        self.memory_roll.see("end")

    def _summarize_memory_roll(self) -> None:
        packed = ""
        if self.rolling_memory_file.exists():
            packed = self.rolling_memory_file.read_text(encoding="utf-8", errors="replace").strip()
        if not packed:
            self.memory_roll.delete("1.0", "end")
            self.memory_roll.insert("end", "No memory entries to summarize.")
            return

        prompt = (
            "You are a Financial Manager. Summarize the key points of this conversation roll for a permanent record.\n\n"
            f"Conversation Roll:\n{packed}"
        )

        self.memory_roll.delete("1.0", "end")
        self.memory_roll.insert("end", "Sage (Qwen2.5) summarizing memory roll...\n")

        def _work() -> str:
            return self.llm.run_coroutine(self.llm.chat_text(prompt, target="sage_local", stream=False))

        def _done(result, error) -> None:
            self.memory_roll.delete("1.0", "end")
            if error is not None:
                self.memory_roll.insert("end", f"[ERROR] {error}")
                return
            self.memory_roll.insert("end", str(result or ""))

        self._run_worker(_work, _done)

    def _clear_memory_roll(self) -> None:
        try:
            self.rolling_memory_file.parent.mkdir(parents=True, exist_ok=True)
            self.rolling_memory_file.write_text("", encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Clear Memory", f"Unable to clear memory: {exc}", parent=self)
            return
        self._refresh_memory_roll()
        self._refresh_dashboard()

    def _append_rolling_memory_turn(self, user_text: str, assistant_text: str) -> None:
        user_line = str(user_text or "").strip().replace("\n", " ")
        assistant_line = str(assistant_text or "").strip().replace("\n", " ")
        if not user_line:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        block = (
            f"[{stamp}]\n"
            f"You: {user_line}\n"
            f"Assistant: {assistant_line}\n\n"
        )
        try:
            self.rolling_memory_file.parent.mkdir(parents=True, exist_ok=True)
            with self.rolling_memory_file.open("a", encoding="utf-8") as fh:
                fh.write(block)
        except Exception:
            pass

    def _read_last_rolling_turns(self, limit: int = 10) -> list[tuple[str, str]]:
        if not self.rolling_memory_file.exists():
            return []
        lines = self.rolling_memory_file.read_text(encoding="utf-8", errors="replace").splitlines()
        turns: list[tuple[str, str]] = []
        current_user = ""
        for line in lines:
            row = line.strip()
            if row.startswith("You:"):
                current_user = row[len("You:") :].strip()
                continue
            if row.startswith("Assistant:") and current_user:
                assistant = row[len("Assistant:") :].strip()
                turns.append((current_user, assistant))
                current_user = ""
        return turns[-max(1, int(limit)) :]

    def _build_historical_context(self, limit: int = 10) -> str:
        turns = self._read_last_rolling_turns(limit=limit)
        if not turns:
            return ""
        out: list[str] = []
        for idx, (user_line, assistant_line) in enumerate(turns, start=1):
            out.append(f"Turn {idx} User: {user_line}")
            out.append(f"Turn {idx} Assistant: {assistant_line}")
        return "\n".join(out)

    def _extract_research_conclusion(self, report_text: str) -> str:
        text = str(report_text or "")
        if not text.strip():
            return "No report yet."

        # Prefer an explicit final conclusion section from the sidecar report.
        match = re.search(r"(?ims)^##\s+Final\s+Conclusion\s*\n(.*?)(?=^##\s+|\Z)", text)
        if match:
            result = match.group(1).strip()
            if result:
                return result

        # Fallback to agent synthesis when available.
        match = re.search(r"(?ims)^##\s+Agent\s+Synthesis\s*\n(.*?)(?=^##\s+|\Z)", text)
        if match:
            result = match.group(1).strip()
            if result:
                return result

        # Last fallback: executive summary block.
        match = re.search(r"(?ims)^##\s+Executive\s+Summary\s*\n(.*?)(?=^##\s+|\Z)", text)
        if match:
            result = match.group(1).strip()
            if result:
                return result

        return text[-2000:].strip() or "No report yet."

    def _start_research(self) -> None:
        topic = ""
        if hasattr(self, "research_topic_entry"):
            topic = (self.research_topic_entry.get() or "").strip()
        if not topic:
            return

        lower_topic = topic.lower()
        self._research_complex_mode = ("[complex]" in lower_topic) or lower_topic.startswith("complex:")
        clean_topic = topic.replace("[complex]", "").strip()
        if clean_topic.lower().startswith("complex:"):
            clean_topic = clean_topic.split(":", 1)[1].strip()
        if not clean_topic:
            clean_topic = topic
        self._research_gemma_conclusion = ""
        self._research_gemma_inflight = False

        self._research_cancel_requested = False
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.research_output.delete("1.0", "end")
        model_banner = str(self.cfg.models.names.sage_local or "Sage Local")
        if self._research_complex_mode:
            model_banner = f"{self.cfg.models.names.sage_local} -> gemma fallback"
        self.research_output.insert(
            "end",
            f"[Model] {model_banner} | [Start Time] {stamp} | [Status: Processing...]\n"
            "[Finished] pending\n\n[Conclusion]\nResearch is running...\n",
        )
        started = self.research.launch_detached(clean_topic)
        if not started:
            self.research_output.insert("end", "\nFailed to launch research sidecar.\n")
            return
        self.after(1200, self._refresh_research_status_once)

    def _stop_research(self) -> None:
        self._research_cancel_requested = True
        try:
            if self.research_status_file.exists():
                status = json.loads(self.research_status_file.read_text(encoding="utf-8", errors="replace"))
                pid = int(status.get("pid") or 0)
                if pid > 0:
                    os.kill(pid, 9)
        except Exception:
            pass
        if hasattr(self, "research_output"):
            self.research_output.insert("end", "\n[Status] stop requested\n")

    def _refresh_research_status_once(self) -> None:
        if not hasattr(self, "research_output"):
            return

        status_obj: dict[str, object] = {}
        if self.research_status_file.exists():
            try:
                status_obj = json.loads(self.research_status_file.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                status_obj = {}

        state = str(status_obj.get("state") or "idle")
        started_at = str(status_obj.get("started_at") or "n/a")
        finished_at = str(status_obj.get("completed_at") or status_obj.get("failed_at") or "pending")
        model_name = str(self.cfg.models.names.sage_local or "Sage Local")
        if self._research_complex_mode:
            model_name = f"{self.cfg.models.names.sage_local} -> gemma fallback"

        conclusion = "No report yet."
        if self.research_report_file.exists():
            report_text = self.research_report_file.read_text(encoding="utf-8", errors="replace").strip()
            if report_text:
                conclusion = self._extract_research_conclusion(report_text)

        if self._research_complex_mode and state == "completed" and (not self._research_gemma_inflight) and (not self._research_gemma_conclusion) and conclusion.strip():
            self._research_gemma_inflight = True

            def _work() -> str:
                prompt = (
                    "You are Gemma. Produce a final strategic conclusion from this research report. "
                    "Return concise sections: Final Thesis, Key Risks, Recommended Next Steps.\n\n"
                    f"Report:\n{conclusion}"
                )
                return self.llm.run_coroutine(self.llm.chat_text(prompt, target="gemma", stream=False))

            def _done(result, error) -> None:
                self._research_gemma_inflight = False
                if error is None:
                    self._research_gemma_conclusion = str(result or "").strip()
                self._refresh_research_status_once()

            self._run_worker(_work, _done)

        status_line = "Processing..." if state == "running" else state.capitalize()
        self.research_output.delete("1.0", "end")
        self.research_output.insert(
            "end",
            f"[Model] {model_name} | [Start Time] {started_at} | [Status: {status_line}]\n"
            f"[Finished: {finished_at}]\n\n[Conclusion]\n{conclusion}\n"
            f"\nWhen completed Research reports are saved in\n{self.research_report_file.as_posix()}\n",
        )
        if self._research_gemma_conclusion:
            self.research_output.insert("end", "\n[Gemma Final Conclusion]\n" + self._research_gemma_conclusion + "\n")

        if state == "running" and not self._research_cancel_requested:
            self.after(1200, self._refresh_research_status_once)

    def _stop_voice(self) -> None:
        self._voice_auto_listen = False
        self._voice_auto_lane = None
        self.voice.stop()
        self._append_ocr_log("Voice playback stop requested.")

    def _set_chat_stop_enabled(self, enabled: bool) -> None:
        if self._voice_auto_listen:
            enabled = True
        state = "normal" if enabled else "disabled"
        try:
            self.stop_chat_btn.configure(state=state)
        except Exception:
            pass

    def _cancel_active_chat(self) -> None:
        self._voice_auto_listen = False
        self._voice_auto_lane = None
        self._cancel_requested = True
        self._chat_cancel_event.set()

        with self._active_stt_proc_lock:
            proc = self._active_stt_proc
            self._active_stt_proc = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

        self.voice.stop()
        try:
            lane: Literal["sage_local", "gemma"] = "gemma" if self.chat_lane == "gemma" else "sage_local"
            if not self._is_text_only_mode():
                self.voice.speak_text_nonblocking("OK, stopping now.", target=lane)
        except Exception:
            pass
        with self._active_chat_future_lock:
            future = self._active_chat_future
        if future is not None and not future.done():
            future.cancel()
        self._set_chat_stop_enabled(False)

    def _poll_voice_busy(self) -> None:
        if hasattr(self, "stop_voice_btn"):
            if self.voice.is_busy():
                self.stop_voice_btn.grid()
            else:
                self.stop_voice_btn.grid_remove()
        self.after(250, self._poll_voice_busy)

    def _run_health_check(self) -> None:
        self.dashboard_text.delete("1.0", "end")
        self.dashboard_text.insert("end", "Running system health check...\n")

        def _work() -> list[str]:
            lines: list[str] = []

            # 1) Ollama connectivity status probe.
            connected = self._is_ollama_alive()
            endpoint = str(self.cfg.models.endpoints.sage_local or "http://127.0.0.1:11434").strip()
            lines.append(f"[OLLAMA] {'PASS' if connected else 'WARN'} - {endpoint} {'reachable' if connected else 'disconnected'}")

            # 2) OCR inbox empty-state availability.
            pending = [
                p
                for p in sorted(self.inbox_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if p.is_file() and p.suffix.lower() in OcrService.IMAGE_EXTENSIONS
            ]
            if pending:
                lines.append(f"[OCR INBOX] PASS - {len(pending)} file(s) detected")
            else:
                lines.append("[OCR INBOX] PASS - empty state ready (No files found.)")

            # 3) Notes empty-state availability.
            notes = sorted(self.notes_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if notes:
                lines.append(f"[NOTES] PASS - {len(notes)} note file(s) detected")
            else:
                lines.append("[NOTES] PASS - empty state ready (No .txt notes found.)")

            # 4) Voice plumbing readiness.
            voice_ok = hasattr(self.voice, "_tts") and self.voice is not None
            lines.append(f"[VOICE] {'PASS' if voice_ok else 'FAIL'} - VoiceService initialized")

            lines.append("Health check complete.")
            return lines

        def _done(result, error) -> None:
            self.dashboard_text.delete("1.0", "end")
            if error is not None:
                self.dashboard_text.insert("end", f"[HEALTH] FAIL - {error}\n")
                return
            self.dashboard_text.insert("end", "\n".join(result or []))

        self._run_worker(_work, _done)

    def _run_first_setup(self) -> None:
        values = self._prompt_first_setup_values()
        if values is None:
            return
        local_url, notes_dir = values

        settings_path = self.project_root / "config" / "settings.yaml"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(build_settings_yaml(llama_url=local_url, notes_dir=notes_dir), encoding="utf-8")

        ok, detail = check_tesseract()
        state = "SUCCESS" if ok else "WARNING"

        self.dashboard_text.delete("1.0", "end")
        self.dashboard_text.insert("end", "First setup complete.\n")
        self.dashboard_text.insert("end", f"settings: {settings_path.as_posix()}\n")
        self.dashboard_text.insert("end", f"local_model.url: {local_url}\n")
        self.dashboard_text.insert("end", f"paths.notes_dir: {notes_dir}\n")
        self.dashboard_text.insert("end", f"tesseract: [{state}] {detail}\n")

    def _prompt_first_setup_values(self) -> tuple[str, str] | None:
        dialog = ctk.CTkToplevel(self)
        dialog.title("First Setup")
        dialog.geometry("560x220")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.lift()
        dialog.focus_force()
        dialog.grab_set()

        frame = ctk.CTkFrame(dialog)
        frame.pack(fill="both", expand=True, padx=14, pady=14)

        ctk.CTkLabel(frame, text="Enter local model endpoint URL").pack(anchor="w", pady=(4, 4))
        url_entry = ctk.CTkEntry(frame)
        url_entry.pack(fill="x", pady=(0, 10))
        url_entry.insert(0, "http://127.0.0.1:11434")

        ctk.CTkLabel(frame, text="Enter notes directory (relative or absolute)").pack(anchor="w", pady=(2, 4))
        notes_entry = ctk.CTkEntry(frame)
        notes_entry.pack(fill="x", pady=(0, 12))
        notes_entry.insert(0, "data/notes")

        result: dict[str, str] = {}

        def _safe_close() -> None:
            try:
                dialog.grab_release()
            except Exception:
                pass
            dialog.destroy()

        def _on_ok() -> None:
            result["url"] = (url_entry.get() or "").strip() or "http://127.0.0.1:11434"
            result["notes"] = (notes_entry.get() or "").strip() or "data/notes"
            _safe_close()

        def _on_cancel() -> None:
            _safe_close()

        dialog.protocol("WM_DELETE_WINDOW", _on_cancel)

        button_row = ctk.CTkFrame(frame, fg_color="transparent")
        button_row.pack(fill="x")
        button_row.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(button_row, text="Cancel", command=_on_cancel).grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkButton(button_row, text="Save", command=_on_ok).grid(row=0, column=1, padx=(6, 0), sticky="ew")

        url_entry.focus_set()
        dialog.wait_window()

        if not result:
            return None
        return result["url"], result["notes"]

    # ------------------------------------------------------------------
    # Model warmup
    # ------------------------------------------------------------------

    def _start_model_warmup(self) -> None:
        """Send a lightweight ping to both models so they are loaded and ready.

        Both pings run sequentially inside a single asyncio.run() / single thread so
        that the shared httpx.AsyncClient is never accessed from two event loops at
        once.  Sage local result is posted to the GUI as soon as it arrives; Gemma result
        follows whenever Ollama responds.

        Gemma uses ping_gemma_warmup() which hits /api/generate with stream=False.
        That makes the httpx timeout apply to the complete response, not per SSE
        chunk — so a cold Ollama model load can't silently reset the clock and hang.
        """
        self._append_dashboard("[WARMUP] Starting model warmup...")

        def _ping_both_sequential() -> None:

            async def _run() -> None:
                # --- Sage Local ---
                try:
                    reply = await self.llm.chat_text("hi", target="sage_local", stream=False)
                    _ = self._clean_warmup_preview(reply)
                    snippet = "Hi there! 👋"
                    self.after(0, lambda s=snippet: self._append_dashboard(
                        f"[WARMUP] Sage Local [OK] ready - \"{s}\""
                    ))
                except Exception as exc:
                    self.after(0, lambda e=exc: self._append_dashboard(
                        f"[WARMUP] Sage Local [FAIL] {e}"
                    ))

                # --- Gemma Local ---
                # ping_gemma_warmup uses POST /api/generate stream=False so the
                # 60 s timeout is a true wall-clock limit, not a per-chunk idle timer.
                try:
                    gemma_reply = await self.llm.ping_gemma_warmup(timeout=60.0)
                    snippet = self._clean_warmup_preview(gemma_reply)
                    self.after(0, lambda s=snippet: self._append_dashboard(
                        f"[WARMUP] Gemma Local [OK] ready - \"{s}\""
                    ))
                except Exception as exc:
                    self.after(0, lambda e=exc: self._append_dashboard(
                        f"[WARMUP] Gemma Local [FAIL] {e}"
                    ))

            self.llm.run_coroutine(_run())

        self._run_worker(_ping_both_sequential, None)

    def _clean_warmup_preview(self, raw_text: str) -> str:
        text = str(raw_text or "").replace("\r", "").strip()
        if not text:
            return ""
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        preview = first_line or text
        if len(preview) > 64:
            preview = preview[:64].rstrip()
            if " " in preview:
                preview = preview.rsplit(" ", 1)[0]
        if preview.endswith(("I'", "I\u2019")):
            preview = preview[:-2].rstrip()
        return preview

    def _append_dashboard(self, message: str) -> None:
        """Append a timestamped line to the dashboard textbox."""
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            self.dashboard_text.insert("end", f"[{stamp}] {message}\n")
            self.dashboard_text.see("end")
        except Exception:
            pass

    def _refresh_dashboard(self) -> None:
        self.dashboard_text.delete("1.0", "end")
        if not self.memory_file.exists():
            self.dashboard_text.insert(
                "end",
                "Welcome to Sage v5! Your activity feed is empty for now. Try Chat, Voice, OCR, or Research to get started.",
            )
            return

        lines = self.memory_file.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        if not lines:
            self.dashboard_text.insert(
                "end",
                "Welcome back! No recent activity yet. Ask Sage or Gemma something to start your session.",
            )
            return

        rendered: list[str] = []
        for row in lines:
            try:
                obj = json.loads(row)
                ts = obj.get("ts", "")
                role = obj.get("role", "")
                lane = obj.get("lane", "")
                text = str(obj.get("text", "")).replace("\n", " ")
                rendered.append(f"[{ts}] {role}/{lane}: {text[:180]}")
            except Exception:
                rendered.append(row)

        self.dashboard_text.insert("end", "\n".join(rendered))

    def _refresh_ocr_inbox(self) -> None:
        self.pending_images = [
            p for p in sorted(self.inbox_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if p.is_file() and p.suffix.lower() in OcrService.IMAGE_EXTENSIONS
        ]

        self.ocr_listbox.delete(0, tk.END)
        if not self.pending_images:
            self.ocr_listbox.insert(tk.END, "No files found.")
            return

        for p in self.pending_images:
            self.ocr_listbox.insert(tk.END, p.name)

    def _get_selected_image(self) -> Path | None:
        sel = self.ocr_listbox.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.pending_images):
            return None
        return self.pending_images[idx]

    def _build_ocr_prompt(self, ocr_result: OcrResult) -> str:
        raw = str(ocr_result.text or "")
        text = raw[: self.MAX_OCR_PROMPT_CHARS].strip()
        if len(raw) > self.MAX_OCR_PROMPT_CHARS:
            self._append_ocr_log(
                f"OCR text truncated from {len(raw)} to {self.MAX_OCR_PROMPT_CHARS} chars to protect local model stability."
            )
        return (
            "You are Sage. Summarize this OCR extraction clearly for a general user. "
            "Highlight key findings and obvious OCR uncertainty.\n\n"
            f"Source image: {ocr_result.image_path}\n"
            f"Confidence: {ocr_result.confidence:.2f}\n"
            "Text:\n"
            f"{text}"
        )

    def _append_ocr_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.ocr_output.insert("end", f"\n[{stamp}] {message}\n")
        self.ocr_output.see("end")

    def _process_newest_image(self) -> None:
        if not self.pending_images:
            self._refresh_ocr_inbox()
        if not self.pending_images:
            messagebox.showinfo("OCR Inbox", "No pending images found.", parent=self)
            return
        self._process_image(self.pending_images[0])

    def _process_selected_image(self) -> None:
        image = self._get_selected_image()
        if image is None:
            messagebox.showinfo("OCR Inbox", "Select an image first.", parent=self)
            return
        self._process_image(image)

    def _process_image(self, image_path: Path) -> None:
        self.ocr_output.delete("1.0", "end")
        self.ocr_output.insert("end", f"Processing {image_path.name}...\n")
        self._append_ocr_log(f"Queued image: {image_path.name}")

        def _work() -> OcrResult:
            self.after(0, lambda: self._append_ocr_log("OCR extraction started."))
            ocr_result = self.ocr.extract_text(str(image_path))
            self.after(0, lambda: self._append_ocr_log(f"OCR complete. confidence={ocr_result.confidence:.2f}"))

            def _write_header() -> None:
                self.ocr_output.delete("1.0", "end")
                self.ocr_output.insert("end", f"[OCR] file: {ocr_result.image_path}\n")
                self.ocr_output.insert("end", f"[OCR] confidence: {ocr_result.confidence:.2f}\n\n")
                self.ocr_output.insert("end", "[SAGE SUMMARY]\n")

                self.ocr_log.delete("1.0", "end")
                self.ocr_log.insert("end", (ocr_result.text or "(empty)").strip() or "(empty)")
                self.ocr_log.see("1.0")

            self.after(0, _write_header)
            self.after(0, lambda: self._append_ocr_log("Sage summary generation started."))
            if not self._ensure_ollama_online():
                self.after(0, lambda: self._append_ocr_log("Ollama check failed; local lane unavailable."))

            async def _stream_summary() -> str:
                parts: list[str] = []
                async for token in self.llm.chat(user_text=self._build_ocr_prompt(ocr_result), target="sage_local", stream=True):
                    if not token:
                        continue
                    parts.append(token)
                    self.after(0, lambda t=token: (self.ocr_output.insert("end", t), self.ocr_output.see("end")))
                return "".join(parts)

            summary = self.llm.run_coroutine(_stream_summary())
            self.last_summary_text = summary
            self.after(0, lambda: self._append_ocr_log("Sage summary generation complete."))
            return ocr_result

        def _done(result: object, error: Exception | None) -> None:
            if error is not None:
                self._append_ocr_log(f"Processing failed: {error}")
                self.ocr_output.insert("end", f"\n[ERROR] {error}\n")
                return
            self._append_ocr_log("Processing pipeline finished successfully.")
            self._refresh_dashboard()

        self._run_worker(_work, _done)

    def _capture_voice_prompt_blocking(self, timeout_sec: int = 8) -> str:
        """Capture one utterance from the default microphone on Windows.

        Returns recognized text or an empty string when no speech is captured.
        """
        whisper_text = self._capture_voice_prompt_whisper(timeout_sec=timeout_sec)
        if whisper_text:
            return whisper_text

        if os.name != "nt":
            return ""

        seconds = max(2, int(timeout_sec))
        # Small lead-in so the recognizer is fully ready before users start speaking.
        lead_ms = 450
        ps_script = (
            "Add-Type -AssemblyName System.Speech;"
            "$ci=[System.Globalization.CultureInfo]::GetCultureInfo('en-US');"
            "$rec=New-Object System.Speech.Recognition.SpeechRecognitionEngine($ci);"
            "$rec.SetInputToDefaultAudioDevice();"
            "$rec.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar));"
            "$rec.InitialSilenceTimeout=[TimeSpan]::FromSeconds(3);"
            "$rec.EndSilenceTimeout=[TimeSpan]::FromMilliseconds(700);"
            "$rec.BabbleTimeout=[TimeSpan]::FromSeconds(2);"
            f"Start-Sleep -Milliseconds {lead_ms};"
            f"$res=$rec.Recognize([TimeSpan]::FromSeconds({seconds}));"
            "if($res -and $res.Text){Write-Output $res.Text}"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=seconds + 6,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.returncode != 0:
                return ""
            return (proc.stdout or "").strip()
        except Exception:
            return ""

    def _capture_voice_prompt_whisper(self, timeout_sec: int = 8) -> str:
        """Capture one utterance via local faster-whisper sidecar when available."""
        stt_script = self.project_root / "models" / "Voice" / "faster_whisper" / "run_stt.py"
        if not stt_script.exists():
            return ""
        if self._cancel_requested:
            return ""

        seconds = max(2, int(timeout_sec))
        cmd = [
            sys.executable,
            str(stt_script),
            "--seconds",
            str(seconds),
            "--samplerate",
            "16000",
            "--model-size",
            "base",
            "--language",
            "en",
            "--chunk-size",
            "1024",
            "--silence-reset-seconds",
            "2.4",
            "--min-stop-seconds",
            "2.4",
            "--stop-on-silence",
        ]

        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            with self._active_stt_proc_lock:
                self._active_stt_proc = proc

            # Keep capture bounded so voice-once cannot linger excessively.
            deadline = time.time() + max(10, seconds + 6)
            while proc.poll() is None:
                if self._cancel_requested or self._chat_cancel_event.is_set() or not self._voice_auto_listen and self._voice_auto_lane is not None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break
                if time.time() >= deadline:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    break
                time.sleep(0.1)

            stdout_text, _stderr_text = proc.communicate(timeout=2)
            if proc.returncode != 0:
                return ""

            payload = None
            for line in reversed((stdout_text or "").splitlines()):
                row = line.strip()
                if not row:
                    continue
                try:
                    payload = json.loads(row)
                    break
                except Exception:
                    continue

            if not isinstance(payload, dict):
                return ""
            if not bool(payload.get("ok")):
                return ""

            return str(payload.get("text") or "").strip()
        except Exception:
            return ""
        finally:
            with self._active_stt_proc_lock:
                if proc is not None and self._active_stt_proc is proc:
                    self._active_stt_proc = None

    def _send_chat(self) -> None:
        prompt = self.chat_input.get().strip()
        if not prompt:
            return

        prompt, target_lane = self._apply_chat_command_interceptor(prompt)
        if not prompt:
            return
        historical_context = self._build_historical_context(limit=10)
        guarded_prompt = self._apply_sage_prompt_guardrails(prompt, target_lane)
        effective_prompt = self.llm.with_historical_context(guarded_prompt, historical_context)
        assistant_name = "Sage" if target_lane == "sage_local" else "Gemma"

        self.chat_input.delete(0, "end")
        self.chat_transcript.insert("end", f"You: {prompt}\n\n")
        self.chat_transcript.insert("end", f"{assistant_name}: ")
        self.chat_transcript.see("end")
        self._cancel_requested = False
        self._chat_cancel_event.clear()
        self._set_chat_stop_enabled(True)

        def _work() -> str:
            async def _stream() -> str:
                full: list[str] = []
                speak_buf = ""
                stammer_noted = False
                if target_lane == "sage_local":
                    _ = self._ensure_ollama_online(timeout_sec=8.0)
                    if self._chat_cancel_event.is_set():
                        return ""
                async for token in self.llm.chat(user_text=effective_prompt, target=target_lane, stream=True):
                    if self._chat_cancel_event.is_set():
                        break
                    if not token:
                        continue
                    full.append(token)
                    self.after(0, lambda t=token: (self.chat_transcript.insert("end", t), self.chat_transcript.see("end")))
                    speak_buf += token
                    done_sentences, rem = self._split_sentences(speak_buf)
                    speak_buf = rem
                    for sentence in done_sentences:
                        spoken = self._prepare_spoken_text(sentence)
                        if spoken == self.MODEL_STAMMER_TOKEN:
                            if not stammer_noted:
                                stammer_noted = True
                                self.after(0, lambda: (self.chat_transcript.insert("end", "\n[MODEL STAMMER] Output suppressed for TTS.\n"), self.chat_transcript.see("end")))
                            continue
                        if spoken and not self._is_text_only_mode() and target_lane == "sage_local":
                            self.voice.speak_text_nonblocking(spoken, target=target_lane)
                if not self._chat_cancel_event.is_set():
                    spoken_tail = self._prepare_spoken_text(speak_buf)
                    if spoken_tail == self.MODEL_STAMMER_TOKEN:
                        if not stammer_noted:
                            stammer_noted = True
                            self.after(0, lambda: (self.chat_transcript.insert("end", "\n[MODEL STAMMER] Output suppressed for TTS.\n"), self.chat_transcript.see("end")))
                        spoken_tail = ""
                    if spoken_tail and not self._is_text_only_mode() and target_lane == "sage_local":
                        self.voice.speak_text_nonblocking(spoken_tail, target=target_lane)
                return "".join(full)

            fut = self.llm.submit_coroutine(_stream())
            with self._active_chat_future_lock:
                self._active_chat_future = fut
            try:
                while True:
                    if self._chat_cancel_event.is_set():
                        fut.cancel()
                        return ""
                    try:
                        return fut.result(timeout=0.1)
                    except concurrent.futures.TimeoutError:
                        continue
            finally:
                with self._active_chat_future_lock:
                    if self._active_chat_future is fut:
                        self._active_chat_future = None

        def _done(result, error) -> None:
            self._set_chat_stop_enabled(False)
            if self._chat_cancel_event.is_set() or isinstance(error, (concurrent.futures.CancelledError, asyncio.CancelledError)):
                self.chat_transcript.insert("end", "[STOPPED] Convo stopped by user.")
            elif error is not None:
                self.chat_transcript.insert("end", f"[ERROR] {error}")
            else:
                self._append_rolling_memory_turn(prompt, str(result or ""))
            self.chat_transcript.insert("end", "\n\n")
            self.chat_transcript.see("end")
            self._refresh_dashboard()
            self._refresh_memory_roll()

        self._run_worker(_work, _done)

    def _send_chat_voice(self) -> None:
        if self._is_text_only_mode():
            prompt = self.chat_input.get().strip()
            if prompt:
                self._send_chat()
            else:
                self.chat_transcript.insert("end", "[MODE] Text-only mode is active. Type your prompt and press Send.\n")
                self.chat_transcript.see("end")
            return

        prompt = self.chat_input.get().strip()
        if not prompt:
            if self._voice_auto_listen and self._voice_auto_lane is not None:
                self._set_chat_lane(self._voice_auto_lane, "AUTO")
            self._cancel_requested = False
            self._suppress_watchdog(30.0)
            if self._voice_auto_listen:
                self._set_chat_stop_enabled(True)
            listen_state = {"done": False}

            def _show_listening_banner() -> None:
                if listen_state["done"]:
                    return
                self.chat_transcript.insert("end", "[Voice] Listening...\n")
                self.chat_transcript.see("end")

            # Show listening feedback slightly later so recognizer warm-up has already begun.
            self.after(500, _show_listening_banner)

            def _listen_work() -> str:
                return self._capture_voice_prompt_blocking(timeout_sec=10)

            def _listen_done(result, error) -> None:
                listen_state["done"] = True
                if self._chat_cancel_event.is_set() or not self._voice_auto_listen and self._voice_auto_lane is not None:
                    return
                if error is not None:
                    self.chat_transcript.insert("end", "[Voice] Capture failed.\n\n")
                    self.chat_transcript.see("end")
                    if self._voice_auto_listen:
                        self.after(1200, self._send_chat_voice)
                    else:
                        messagebox.showerror("Voice Prompt", f"Microphone capture failed: {error}", parent=self)
                    return

                heard = str(result or "").strip()
                if not heard:
                    if self._voice_auto_listen:
                        self.chat_transcript.insert("end", "[Voice] No speech detected. Still listening...\n")
                    else:
                        messagebox.showinfo("Voice Prompt", "No speech detected. Try again.", parent=self)
                        self.chat_transcript.insert("end", "[Voice] No speech detected.\n\n")
                    self.chat_transcript.see("end")
                    if self._voice_auto_listen:
                        self.after(500, self._send_chat_voice)
                    return

                self.chat_input.delete(0, "end")
                self.chat_input.insert(0, heard)
                self.chat_transcript.insert("end", f"[Voice] Heard: {heard}\n")
                self.chat_transcript.see("end")

                confirm_lane: Literal["sage_local", "gemma"] = "gemma" if self.chat_lane == "gemma" else "sage_local"
                confirm_text = "OK."
                self.chat_transcript.insert("end", f"[Voice] Confirmation speaking: {confirm_text}\n")
                self.chat_transcript.see("end")

                def _confirm_work() -> None:
                    self.voice.speak_text_blocking(confirm_text, target=confirm_lane)

                def _confirm_done(_result, confirm_error) -> None:
                    if confirm_error is not None:
                        self.chat_transcript.insert("end", f"[Voice] Confirmation failed: {confirm_error}\n")
                    else:
                        self.chat_transcript.insert("end", "[Voice] Confirmation done.\n")
                    self.chat_transcript.see("end")
                    self._send_chat_voice()

                self._run_worker(_confirm_work, _confirm_done)

            self._run_worker(_listen_work, _listen_done)
            return

        prompt, target_lane = self._apply_chat_command_interceptor(prompt)
        if not prompt:
            return
        if target_lane == "gemma":
            # Gemma is text-only in Chat/Voice; route voice prompts to Sage.
            self._append_mode_notice("[MODE] Gemma voice is disabled. Routing this voice request to Sage.")
            target_lane = "sage_local"
            self._set_chat_lane("sage_local", "VOICE")
        historical_context = self._build_historical_context(limit=10)
        guarded_prompt = self._apply_sage_prompt_guardrails(prompt, target_lane)
        effective_prompt = self.llm.with_historical_context(guarded_prompt, historical_context)
        self._suppress_watchdog(30.0)
        assistant_name = "Sage" if target_lane == "sage_local" else "Gemma"

        self.chat_input.delete(0, "end")
        self.chat_transcript.insert("end", f"You (voice): {prompt}\n\n")
        self.chat_transcript.insert("end", f"{assistant_name}: ")
        self.chat_transcript.see("end")
        self._cancel_requested = False
        self._chat_cancel_event.clear()
        self._set_chat_stop_enabled(True)

        def _work() -> str:
            async def _stream_and_queue() -> str:
                full: list[str] = []
                speak_buf = ""
                stammer_noted = False
                if target_lane == "sage_local":
                    _ = self._ensure_ollama_online(timeout_sec=8.0)
                    if self._chat_cancel_event.is_set():
                        return ""
                async for token in self.llm.chat(user_text=effective_prompt, target=target_lane, stream=True):
                    if self._chat_cancel_event.is_set():
                        break
                    if not token:
                        continue
                    full.append(token)
                    self.after(0, lambda t=token: (self.chat_transcript.insert("end", t), self.chat_transcript.see("end")))
                    speak_buf += token
                    done_sentences, rem = self._split_sentences(speak_buf)
                    speak_buf = rem
                    for sentence in done_sentences:
                        spoken = self._prepare_spoken_text(sentence)
                        if spoken == self.MODEL_STAMMER_TOKEN:
                            if not stammer_noted:
                                stammer_noted = True
                                self.after(0, lambda: (self.chat_transcript.insert("end", "\n[MODEL STAMMER] Output suppressed for TTS.\n"), self.chat_transcript.see("end")))
                            continue
                        if spoken and not self._is_text_only_mode() and target_lane == "sage_local":
                            self.voice.speak_text_nonblocking(spoken, target=target_lane)

                if not self._chat_cancel_event.is_set():
                    spoken_tail = self._prepare_spoken_text(speak_buf)
                    if spoken_tail == self.MODEL_STAMMER_TOKEN:
                        if not stammer_noted:
                            stammer_noted = True
                            self.after(0, lambda: (self.chat_transcript.insert("end", "\n[MODEL STAMMER] Output suppressed for TTS.\n"), self.chat_transcript.see("end")))
                        spoken_tail = ""
                    if spoken_tail and not self._is_text_only_mode() and target_lane == "sage_local":
                        self.voice.speak_text_nonblocking(spoken_tail, target=target_lane)

                return "".join(full).strip()

            fut = self.llm.submit_coroutine(_stream_and_queue())
            with self._active_chat_future_lock:
                self._active_chat_future = fut
            try:
                while True:
                    if self._chat_cancel_event.is_set():
                        fut.cancel()
                        return ""
                    try:
                        return fut.result(timeout=0.1)
                    except concurrent.futures.TimeoutError:
                        continue
            finally:
                with self._active_chat_future_lock:
                    if self._active_chat_future is fut:
                        self._active_chat_future = None

        def _done(result, error) -> None:
            self._set_chat_stop_enabled(False)
            if self._chat_cancel_event.is_set() or isinstance(error, (concurrent.futures.CancelledError, asyncio.CancelledError)):
                self._voice_auto_listen = False
                self._voice_auto_lane = None
                self.chat_transcript.insert("end", "[STOPPED] Convo stopped by user.\n\n")
            elif error is not None:
                self._voice_auto_listen = False
                self._voice_auto_lane = None
                self.chat_transcript.insert("end", f"[ERROR] {error}\n\n")
            else:
                self.chat_transcript.insert("end", "\n\n")
                self._append_rolling_memory_turn(prompt, str(result or ""))
            self.chat_transcript.see("end")
            self._refresh_dashboard()
            self._refresh_memory_roll()
            if self._voice_auto_listen and not self._chat_cancel_event.is_set():
                self._resume_voice_loop_when_idle(delay_ms=350)

        self._run_worker(_work, _done)

    def _refresh_notes(self) -> None:
        query = self.notes_search.get().strip().lower()
        all_files = sorted(self.notes_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)

        if query:
            filtered = [p for p in all_files if query in p.name.lower()]
        else:
            filtered = all_files

        self.notes_files = filtered
        self.notes_listbox.delete(0, tk.END)
        for p in self.notes_files:
            self.notes_listbox.insert(tk.END, p.name)

        self.notes_viewer.delete("1.0", "end")
        if not self.notes_files:
            self.notes_viewer.insert("end", "No .txt notes found.")

    def _load_selected_note(self) -> None:
        sel = self.notes_listbox.curselection()
        if not sel:
            return

        idx = int(sel[0])
        if idx < 0 or idx >= len(self.notes_files):
            return

        path = self.notes_files[idx]
        text = path.read_text(encoding="utf-8", errors="replace")
        self.notes_viewer.delete("1.0", "end")
        self.notes_viewer.insert("end", text)

    def _parse_note_frontmatter(self, text: str) -> tuple[dict[str, str], str]:
        lines = str(text or "").splitlines()
        if len(lines) < 3 or lines[0].strip() != "---":
            return {}, str(text or "")

        end_idx = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx <= 1:
            return {}, str(text or "")

        meta: dict[str, str] = {}
        for row in lines[1:end_idx]:
            if ":" not in row:
                continue
            key, value = row.split(":", 1)
            key = key.strip().lower()
            if not key:
                continue
            meta[key] = value.strip()
        body = "\n".join(lines[end_idx + 1 :])
        return meta, body

    def _serialize_note_frontmatter(self, meta: dict[str, str], body: str) -> str:
        if not meta:
            return body

        preferred_keys = [
            "type",
            "title",
            "status",
            "due_at",
            "repeat",
            "last_triggered_at",
            "completed_at",
        ]
        seen: set[str] = set()
        ordered: list[tuple[str, str]] = []
        for key in preferred_keys:
            if key in meta and str(meta.get(key) or "").strip():
                ordered.append((key, str(meta[key]).strip()))
                seen.add(key)
        for key in sorted(meta.keys()):
            if key in seen:
                continue
            value = str(meta.get(key) or "").strip()
            if value:
                ordered.append((key, value))

        if not ordered:
            return body

        rows = ["---"]
        rows.extend(f"{k}: {v}" for k, v in ordered)
        rows.append("---")
        rows.append("")
        rows.append(str(body or ""))
        return "\n".join(rows).rstrip() + "\n"

    def _parse_reminder_datetime(self, raw_value: str) -> datetime | None:
        text = str(raw_value or "").strip()
        if not text:
            return None

        normalized = text.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except Exception:
            pass

        formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d",
        )
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
        return None

    def _write_note_with_meta(self, path: Path, meta: dict[str, str], body: str) -> None:
        updated = self._serialize_note_frontmatter(meta, body)
        path.write_text(updated, encoding="utf-8")

    def _advance_reminder_schedule(self, due_at: datetime, repeat: str, now: datetime) -> datetime | None:
        cadence = str(repeat or "").strip().lower()
        if cadence in {"", "none", "once", "one-time", "onetime"}:
            return None

        step = {
            "daily": timedelta(days=1),
            "weekly": timedelta(days=7),
            "monthly": timedelta(days=30),
        }.get(cadence)
        if step is None:
            return None

        next_due = due_at
        guard = 0
        while next_due <= now and guard < 500:
            next_due += step
            guard += 1
        return next_due

    def _reminder_tick(self) -> None:
        if self._is_shutting_down:
            return

        now = datetime.now()
        try:
            stale_keys = [k for k, ts in self._recent_reminder_hits.items() if now.timestamp() - ts > 6 * 3600]
            for key in stale_keys:
                self._recent_reminder_hits.pop(key, None)

            due_items: list[tuple[Path, dict[str, str], str, datetime]] = []
            for path in sorted(self.notes_dir.glob("*.txt")):
                try:
                    raw_text = path.read_text(encoding="utf-8", errors="replace")
                    meta, body = self._parse_note_frontmatter(raw_text)
                    if str(meta.get("type") or "").strip().lower() != "reminder":
                        continue

                    status = str(meta.get("status") or "pending").strip().lower()
                    if status in {"done", "completed", "cancelled"}:
                        continue

                    due_raw = str(meta.get("due_at") or meta.get("remind_at") or "")
                    due_at = self._parse_reminder_datetime(due_raw)
                    if due_at is None or due_at > now:
                        continue

                    dedupe_key = f"{path.as_posix()}|{due_at.isoformat()}"
                    if dedupe_key in self._recent_reminder_hits:
                        continue
                    self._recent_reminder_hits[dedupe_key] = now.timestamp()
                    due_items.append((path, meta, body, due_at))
                except Exception:
                    continue

            for path, meta, body, due_at in due_items:
                title = str(meta.get("title") or path.stem).strip() or "Reminder"
                preview = ""
                for row in body.splitlines():
                    row = row.strip()
                    if row:
                        preview = row
                        break
                message = f"{title}\n\n{preview}" if preview else title
                try:
                    messagebox.showinfo("Sage Reminder", message, parent=self)
                except Exception:
                    pass

                repeat = str(meta.get("repeat") or "none").strip().lower()
                next_due = self._advance_reminder_schedule(due_at=due_at, repeat=repeat, now=now)
                meta["last_triggered_at"] = now.isoformat(timespec="seconds")
                if next_due is None:
                    meta["status"] = "done"
                    meta["completed_at"] = now.isoformat(timespec="seconds")
                else:
                    meta["status"] = "pending"
                    meta["due_at"] = next_due.strftime("%Y-%m-%d %H:%M")
                    meta.pop("completed_at", None)

                try:
                    self._write_note_with_meta(path, meta, body)
                except Exception:
                    continue
        finally:
            self.after(self._reminder_poll_ms, self._reminder_tick)

    def _is_ollama_alive(self) -> bool:
        endpoint = str(self.cfg.models.endpoints.sage_local or "").strip() or "http://127.0.0.1:11434"
        base = endpoint.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        parsed = urllib.parse.urlparse(base)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or (443 if parsed.scheme == "https" else 80))

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.8)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False
        finally:
            try:
                s.close()
            except Exception:
                pass

    def _get_ollama_state(self) -> Literal["online", "offline", "loading"]:
        if not self._is_ollama_alive():
            return "offline"

        if not self.llm._prefer_openai_for_local(self.cfg.models.endpoints.sage_local):
            # Ollama: tags endpoint is enough to report service readiness.
            base = self.llm._normalize_ollama_base(self.cfg.models.endpoints.sage_local)
            tags_url = f"{base}/api/tags"
            try:
                req = urllib.request.Request(tags_url, method="GET")
                with urllib.request.urlopen(req, timeout=0.8):
                    return "online"
            except Exception:
                return "offline"

        # Probe a lightweight endpoint for temporary model warmup/loading states.
        base = str(self.cfg.models.endpoints.sage_local or "").strip().rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        health_url = f"{base}/health"
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                body = resp.read(2048).decode("utf-8", errors="ignore").lower()
                if any(k in body for k in ("loading", "initializing", "warming", "not ready", "starting")):
                    return "loading"
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read(2048).decode("utf-8", errors="ignore").lower()
            except Exception:
                body = ""
            if exc.code in (425, 429, 503):
                return "loading"
            if any(k in body for k in ("loading", "initializing", "warming", "not ready", "starting")):
                return "loading"
        except Exception:
            pass

        return "online"

    def _update_ollama_status(self) -> None:
        state = self._get_ollama_state()
        if state == "online":
            dot_color = "#2e7d32"
            label = "Ollama: online"
        elif state == "loading":
            dot_color = "#FFBF00"
            label = "Ollama: loading"
            self._suppress_watchdog(60.0)
        else:
            dot_color = "#b71c1c"
            label = "Ollama: offline"

        self.led_canvas.itemconfig(self.led_dot, fill=dot_color)
        self.status_label.configure(text=label)
        self.after(2000, self._update_ollama_status)


def run_gui() -> None:
    instance_lock = _acquire_single_instance_lock()
    if instance_lock is None:
        print("[INFO] Sage v5 GUI is already running; skipping second instance.")
        return

    cfg = load_config()
    app = SageDesktopGUI(cfg)
    try:
        app.mainloop()
    finally:
        try:
            instance_lock.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_gui()
