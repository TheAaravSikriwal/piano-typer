"""Types text into whatever Windows app has focus (SendInput, Unicode)."""

import ctypes
import sys
from ctypes import wintypes

if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    VK_BACK = 0x08
    VK_RETURN = 0x0D

    ULONG_PTR = ctypes.c_size_t

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    def _send(inputs):
        arr = (INPUT * len(inputs))(*inputs)
        user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))

    def _vk(vk):
        return [INPUT(INPUT_KEYBOARD, _U(ki=KEYBDINPUT(vk, 0, 0, 0, 0))),
                INPUT(INPUT_KEYBOARD, _U(ki=KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP, 0, 0)))]

    def _uni(ch):
        code = ord(ch)
        return [INPUT(INPUT_KEYBOARD, _U(ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE, 0, 0))),
                INPUT(INPUT_KEYBOARD, _U(ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0)))]

    def backspace(n):
        if n > 0:
            _send(sum((_vk(VK_BACK) for _ in range(n)), []))

    def type_text(s):
        seq = []
        for ch in s:
            seq += _vk(VK_RETURN) if ch == "\n" else _uni(ch)
        if seq:
            _send(seq)
else:
    def backspace(n):
        pass

    def type_text(s):
        pass


def apply_diff(old, new):
    """Make the focused app's text go from `old` to `new` with backspaces + typing."""
    common = 0
    for a, b in zip(old, new):
        if a != b:
            break
        common += 1
    backspace(len(old) - common)
    type_text(new[common:])
