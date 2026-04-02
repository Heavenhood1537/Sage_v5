from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from queue import Empty, Queue
import threading
from typing import Literal

from core.config import AppConfig
from services.tts_kokoro import TtsService


def _num_to_words(n: int) -> str:
    """Convert a non-negative integer to English words."""
    if n < 0:
        return "negative " + _num_to_words(-n)
    _ones = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    ]
    _tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    if n < 20:
        return _ones[n]
    if n < 100:
        return _tens[n // 10] + ("" if n % 10 == 0 else "-" + _ones[n % 10])
    if n < 1_000:
        tail = n % 100
        return _ones[n // 100] + " hundred" + ("" if tail == 0 else " " + _num_to_words(tail))
    if n < 1_000_000:
        tail = n % 1_000
        return _num_to_words(n // 1_000) + " thousand" + ("" if tail == 0 else " " + _num_to_words(tail))
    if n < 1_000_000_000:
        tail = n % 1_000_000
        return _num_to_words(n // 1_000_000) + " million" + ("" if tail == 0 else " " + _num_to_words(tail))
    if n < 1_000_000_000_000:
        tail = n % 1_000_000_000
        return _num_to_words(n // 1_000_000_000) + " billion" + ("" if tail == 0 else " " + _num_to_words(tail))
    if n < 1_000_000_000_000_000:
        tail = n % 1_000_000_000_000
        return _num_to_words(n // 1_000_000_000_000) + " trillion" + ("" if tail == 0 else " " + _num_to_words(tail))
    if n < 1_000_000_000_000_000_000:
        tail = n % 1_000_000_000_000_000
        return _num_to_words(n // 1_000_000_000_000_000) + " quadrillion" + ("" if tail == 0 else " " + _num_to_words(tail))
    tail = n % 1_000_000_000_000_000_000
    return _num_to_words(n // 1_000_000_000_000_000_000) + " quintillion" + ("" if tail == 0 else " " + _num_to_words(tail))


def _expand_numbers(text: str) -> str:
    """Replace digit strings (including comma-formatted) with spoken words.

    Examples: '1,000' -> 'one thousand', '42' -> 'forty-two'.
    Decimals are expanded as spoken decimals: '3.14' -> 'three point one four'.
    4-digit years like '2026' are kept as digits so TTS handles them naturally.
    """
    def _replace(m: re.Match) -> str:
        raw = m.group(0).replace(",", "")
        try:
            value = int(raw)
            # Leave 4-digit years alone (1000–2999) — they read fine as digits.
            if 1000 <= value <= 2999 and "," not in m.group(0) and len(raw) == 4:
                return m.group(0)
            return _num_to_words(value)
        except (ValueError, OverflowError):
            return m.group(0)

    # Match comma-grouped integers like 1,000,000 or plain integers.
    # --- Step 0: normalize decimal separators and spacing around decimals ---
    # 68,4 -> 68.4 (decimal comma), while preserving thousands groups like 12,345.
    text = re.sub(r"\b(\d{1,3}),(\d{1,2})(?![\d.,])", r"\1.\2", text)
    # 68 . 4 -> 68.4 (spaced decimal point)
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)

    # --- Step 1: expand decimals first, e.g. "10.6" -> "ten point six" ---
    _digit_names = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    def _expand_dec(m: re.Match) -> str:
        int_part_str = m.group(1).replace(",", "")
        frac_digits = m.group(2)
        try:
            int_words = _num_to_words(int(int_part_str))
        except (ValueError, OverflowError):
            return m.group(0)
        frac_words = " ".join(_digit_names[int(d)] for d in frac_digits)
        # Use spaces instead of hyphenated tens to avoid odd prosody around
        # punctuation in some TTS voices.
        int_words = int_words.replace("-", " ")
        return int_words + " point " + frac_words
    text = re.sub(r"\b(\d{1,3}(?:,\d{3})*|\d+)\.(\d+)\b", _expand_dec, text)
    # --- Step 2: expand remaining integers ---
    out = re.sub(r"(?<!\d\.)\b\d{1,3}(?:,\d{3})+\b|\b(?<!\.)(\d+)(?!\.)\b", _replace, text)
    # Keep spoken numbers smooth for TTS engines that over-pause hyphens.
    return out.replace("-", " ")


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting symbols so Kokoro doesn't read them aloud.

    Strips:  ** * __ _ (bold/italic), # (headings), ` (code), ~~ (strikethrough),
             > (blockquote), --- / *** / ___ (horizontal rules), | (table pipes).
    Keeps the visible text content intact.
    """
    # Horizontal rules on their own line → silence
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Headings: strip leading # symbols
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Bold+italic: ***text*** or ___text___
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)
    # Bold: **text** or __text__
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
    # Italic: *text* or _text_  (word-boundary guard avoids touching contractions)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Inline code: `text`
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Blockquote markers
    text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)
    # Table pipes
    text = re.sub(r"\|", " ", text)
    # Leftover bare asterisks / underscores not part of words
    text = re.sub(r"(?<!\w)[*_]+(?!\w)", " ", text)
    return text


def _expand_math_symbols(text: str) -> str:
    """Convert common math/programming symbols into speakable words."""
    def _root_index_word(value: str) -> str:
        idx = int(value)
        names = {
            2: "square",
            3: "cube",
            4: "fourth",
            5: "fifth",
            6: "sixth",
            7: "seventh",
            8: "eighth",
            9: "ninth",
            10: "tenth",
        }
        return names.get(idx, f"{_num_to_words(idx)}th")

    # Function-style math first.
    text = re.sub(r"\bsqrt\s*\(\s*([^)]+)\s*\)", r"square root of \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcbrt\s*\(\s*([^)]+)\s*\)", r"cube root of \1", text, flags=re.IGNORECASE)

    # Indexed radical first so 4√16 does not get consumed by the plain √ rule.
    def _replace_indexed_root(m: re.Match) -> str:
        idx_word = _root_index_word(m.group(1))
        radicand = m.group(2)
        return f"{idx_word} root of {radicand}"

    # n√x -> nth root of x
    text = re.sub(r"\b(\d{1,2})\s*√\s*\(?\s*([^\)\s]+(?:\s+[^\)\s]+)*)\s*\)?", _replace_indexed_root, text)

    # Unicode radical forms.
    text = re.sub(r"∛\s*\(?\s*([^\)\s]+(?:\s+[^\)\s]+)*)\s*\)?", r"cube root of \1", text)
    text = re.sub(r"∜\s*\(?\s*([^\)\s]+(?:\s+[^\)\s]+)*)\s*\)?", r"fourth root of \1", text)
    text = re.sub(r"√\s*\(?\s*([^\)\s]+(?:\s+[^\)\s]+)*)\s*\)?", r"square root of \1", text)

    # Multi-char operators first so they are not broken by single-char replacements.
    text = re.sub(r"<=|≤", " less than or equal to ", text)
    text = re.sub(r">=|≥", " greater than or equal to ", text)
    text = re.sub(r"!=|≠", " not equal to ", text)
    text = re.sub(r"==", " equals ", text)
    text = re.sub(r"\+/-|±", " plus or minus ", text)
    text = re.sub(r"~=|≈", " approximately ", text)

    # Single-char operators in arithmetic contexts.
    text = re.sub(r"(?<=\w)\s*\^\s*(?=\w)", " to the power of ", text)
    text = re.sub(r"(?<=\w)\s*[×*]\s*(?=\w)", " times ", text)
    text = re.sub(r"(?<=\w)\s*/\s*(?=\w)", " divided by ", text)
    text = re.sub(r"(?<=\w)\s*\+\s*(?=\w)", " plus ", text)
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " minus ", text)
    text = re.sub(r"(?<=\w)\s*=\s*(?=\w)", " equals ", text)

    # Percent after numbers.
    text = re.sub(r"(?<=\d)\s*%", " percent", text)

    return text


def _expand_financial_amounts(text: str) -> str:
    """Convert common currency amounts into speakable English phrases."""
    currency_from_symbol = {
        "$": ("dollar", "cent"),
        "€": ("euro", "cent"),
        "£": ("pound", "pence"),
        "¥": ("yen", "sen"),
        "₹": ("rupee", "paise"),
    }
    currency_from_code = {
        "USD": ("dollar", "cent"),
        "EUR": ("euro", "cent"),
        "GBP": ("pound", "pence"),
        "JPY": ("yen", "sen"),
        "INR": ("rupee", "paise"),
        "CAD": ("canadian dollar", "cent"),
        "AUD": ("australian dollar", "cent"),
        "CHF": ("swiss franc", "centime"),
        "CNY": ("yuan", "jiao"),
    }

    def _plural(word: str, value: int) -> str:
        if value == 1:
            return word
        if word in {"pence", "yen", "paise"}:
            return word
        if word.endswith("s"):
            return word
        return word + "s"

    def _number_phrase(raw_amount: str) -> tuple[str, float]:
        raw = str(raw_amount or "").replace(",", "").strip()
        if not raw:
            return "zero", 0.0
        try:
            as_float = float(raw)
        except Exception:
            as_float = 0.0
        if "." in raw:
            whole_str, frac_str = raw.split(".", 1)
            whole_val = int(whole_str or "0")
            frac_digits = "".join(ch for ch in frac_str if ch.isdigit())
            if frac_digits:
                digit_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
                return f"{_num_to_words(whole_val)} point {' '.join(digit_names[int(d)] for d in frac_digits)}", as_float
            return _num_to_words(whole_val), as_float
        return _num_to_words(int(raw)), as_float

    def _amount_to_words(amount_str: str, major: str, minor: str) -> str:
        # Accept amounts like 1,234.56 or 2.5 (kept as point form unless 2-digit minor unit).
        raw = str(amount_str or "").replace(",", "").strip()
        if not raw:
            return ""
        if "." in raw:
            whole_str, frac_str = raw.split(".", 1)
            whole_val = int(whole_str or "0")
            frac_digits = "".join(ch for ch in frac_str if ch.isdigit())
            whole_words = _num_to_words(whole_val)
            if len(frac_digits) == 2:
                frac_val = int(frac_digits)
                if frac_val > 0:
                    return (
                        f"{whole_words} {_plural(major, whole_val)} and "
                        f"{_num_to_words(frac_val)} {_plural(minor, frac_val)}"
                    )
                return f"{whole_words} {_plural(major, whole_val)}"
            # Non-2-digit fractional parts are better spoken as decimal point.
            digit_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            frac_words = " ".join(digit_names[int(d)] for d in frac_digits) if frac_digits else "zero"
            return f"{whole_words} point {frac_words} {_plural(major, whole_val)}"

        whole_val = int(raw)
        return f"{_num_to_words(whole_val)} {_plural(major, whole_val)}"

    # Handle symbol amounts first: $1,200.50, €2.5 million, etc.
    def _replace_symbol_amount(m: re.Match) -> str:
        sym = m.group("sym")
        amount = m.group("amount")
        scale = (m.group("scale") or "").strip().lower()
        major, minor = currency_from_symbol.get(sym, ("unit", "subunit"))
        if scale:
            num_words, numeric_value = _number_phrase(amount)
            return f"{num_words} {scale} {_plural(major, 1 if abs(numeric_value - 1.0) < 1e-9 else 2)}"
        spoken = _amount_to_words(amount, major, minor)
        return spoken

    text = re.sub(
        r"(?P<sym>[\$€£¥₹])\s*(?P<amount>[\d,]+(?:\.\d+)?)\s*(?P<scale>million|billion|trillion)?",
        _replace_symbol_amount,
        text,
    )

    # Handle code-suffixed amounts: 1,200 USD, 2.5 million EUR
    def _replace_code_amount(m: re.Match) -> str:
        amount = m.group("amount")
        scale = (m.group("scale") or "").strip().lower()
        code = m.group("code").upper()
        major, minor = currency_from_code.get(code, (code.lower(), "subunit"))
        if scale:
            num_words, numeric_value = _number_phrase(amount)
            return f"{num_words} {scale} {_plural(major, 1 if abs(numeric_value - 1.0) < 1e-9 else 2)}"
        spoken = _amount_to_words(amount, major, minor)
        return spoken

    text = re.sub(
        r"(?P<amount>[\d,]+(?:\.\d+)?)\s*(?P<scale>million|billion|trillion)?\s*(?P<code>USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|CNY)\b",
        _replace_code_amount,
        text,
        flags=re.IGNORECASE,
    )

    # Handle code-prefix amounts: EUR 1,200.50, JPY 1200000, USD 2.5 million
    text = re.sub(
        r"\b(?P<code>USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|CNY)\s*(?P<amount>[\d,]+(?:\.\d+)?)\s*(?P<scale>million|billion|trillion)?",
        _replace_code_amount,
        text,
        flags=re.IGNORECASE,
    )

    return text


def _sanitise_for_tts(text: str) -> str:
    """
    Strip characters that Kokoro TTS cannot pronounce cleanly.

    Keeps:
      - ASCII printable characters (Kokoro handles English, numbers, punctuation)
      - Latin-extended characters used by Spanish/German/French (U+00C0–U+024F)
      - Whitespace

    Replaces runs of stripped characters with a single space so sentence rhythm
    is mostly preserved and the English parts still read naturally.
    """
    cleaned = str(text or "")
    # Preserve symbols long enough to map them into speakable words first.
    cleaned = _strip_markdown(cleaned)
    cleaned = _expand_math_symbols(cleaned)
    cleaned = _expand_financial_amounts(cleaned)
    cleaned = _expand_numbers(cleaned)
    # Strip any remaining unsupported symbols after conversion.
    cleaned = re.sub(r"[^\x20-\x7E\u00C0-\u024F\s]+", " ", cleaned)
    # Collapse multiple spaces / blank lines introduced by stripping
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned


@dataclass
class VoiceService:
    """
    Local voice adapter for desktop/UI callers.
    Uses the existing Kokoro local sidecar service under the hood.
    """

    cfg: AppConfig

    def __post_init__(self) -> None:
        self._tts = TtsService(self.cfg)
        self._queue: Queue[tuple[str, Literal["sage_local", "bitnet", "gemma"]]] = Queue()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._is_speaking = False

    async def speak_text(self, text: str, target: Literal["sage_local", "bitnet", "gemma"] = "sage_local") -> None:
        value = _sanitise_for_tts(text)
        if not value:
            return
        await self._tts.speak(value, target=target)

    def speak_text_blocking(self, text: str, target: Literal["sage_local", "bitnet", "gemma"] = "sage_local") -> None:
        value = _sanitise_for_tts(text)
        if not value:
            return
        asyncio.run(self.speak_text(value, target=target))

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._voice_worker, daemon=True)
            self._worker.start()

    def speak_text_nonblocking(self, text: str, target: Literal["sage_local", "bitnet", "gemma"] = "sage_local") -> None:
        value = _sanitise_for_tts(text)
        if not value:
            return
        self._queue.put((value, target))
        self._ensure_worker()

    def _voice_worker(self) -> None:
        while True:
            try:
                text, target = self._queue.get(timeout=0.8)
            except Empty:
                break

            self._is_speaking = True
            try:
                self.speak_text_blocking(text, target=target)
            except Exception:
                # Keep worker alive for queued items; GUI logs surface failures.
                pass
            finally:
                self._is_speaking = False
                self._queue.task_done()

    def stop(self) -> None:
        self._tts.request_stop()
        try:
            while True:
                _ = self._queue.get_nowait()
                self._queue.task_done()
        except Empty:
            pass

    def is_busy(self) -> bool:
        return self._is_speaking or (not self._queue.empty())


__all__ = ["VoiceService"]
