"""jesus.py - the Zhizho interpreter.

    python3 jesus.py program.jesus      intonate an existing Kontakion
    python3 jesus.py '"Some text!"'     generate a Kontakion for the string,
                                        write it to a .jesus file, and intonate it
    python3 jesus.py performance.wav    decode a performance back to text

A .jesus file is a Kontakion: one registry row per line, in the
Alexandrion format  index:HEPT<terscii address tail>terscii word.
"""
import math
import re
import struct
import sys
import wave

M = 3 ** 9  # 19683
HEPT = "0ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_ROMAN = [
    ["ES", "SP", "0", "9", "I", "R", "_", "i", "r"],
    ["EL", "-", "1", "A", "J", "S", "a", "j", "s"],
    ["ET", "'", "2", "B", "K", "T", "b", "k", "t"],
    ["LR", ",", "3", "C", "L", "U", "c", "l", "u"],
    ["OP", ";", "4", "D", "M", "V", "d", "m", "v"],
    ["RL", ":", "5", "E", "N", "W", "e", "n", "w"],
    ["SU", ".", "6", "F", "O", "X", "f", "o", "x"],
    ["HT", "!", "7", "G", "P", "Y", "g", "p", "y"],
    ["SD", "?", "8", "H", "Q", "Z", "h", "q", "z"],
]
_C2T, _T2C = {}, {}
for _r, _row in enumerate(_ROMAN):
    for _c, _ch in enumerate(_row):
        code = f"{_r}{_c}"
        if _ch == "SP":
            _C2T[" "] = code
            _T2C[code] = " "
        elif len(_ch) == 1:
            _C2T[_ch] = code
            _T2C[code] = _ch


def terscii(s):
    return "".join(_C2T.get(ch, "00") for ch in s)


def unterscii(t):
    return "".join(_T2C.get(t[i:i + 2], "\0") for i in range(0, len(t), 2))


def hept_to_int(h):
    v = 0
    for ch in h:
        v = v * 27 + HEPT.index(ch)
    return v


def to_hept(n):
    n %= M
    if n == 0:
        return "0"
    d = []
    while n:
        n, r = divmod(n, 27)
        d.append(HEPT[r])
    return "".join(reversed(d))


def dlog2(v):
    n, k = 1, 0
    while n != v % M:
        n, k = n * 2 % M, k + 1
    return k


# --------------------------------------------------------------------------
# Kontakion generation: assemble a 9 trits, 19683 trytes program that
# prints the given text, and registry it in the Alexandrion format.
# --------------------------------------------------------------------------

def _bal9(v):
    v %= M
    if v > (M - 1) // 2:
        v -= M
    out = []
    for _ in range(9):
        r = v % 3
        if r == 0:
            out.append('0'); v //= 3
        elif r == 1:
            out.append('1'); v = (v - 1) // 3
        else:
            out.append('t'); v = (v + 1) // 3
    return "".join(reversed(out))


def assemble(text):
    """Trytes of: while p != endptr do putch [p] ; p *= 2 end halt, data."""
    addr = lambda t: pow(2, t, M)
    data = [ord(c) % M for c in text]
    T_P, T_DATA = 7, 15
    T_CONST2 = T_DATA + len(data)
    T_ENDPTR = T_CONST2 + 1

    s = ""

    def pad():
        nonlocal s
        while len(s) % 9:
            s += '0'

    s += ("11t1" + "110t10" + "111t00"
          + "0" + _bal9(addr(T_P)) + "0" + _bal9(addr(T_ENDPTR)) + "10")
    pad()
    s += "1t"; pad()
    s += "111t1t" + "1t" + "0" + _bal9(addr(T_DATA)) + "1100"; pad()
    s += ("t" + _bal9(addr(T_P)) + "111001"
          + "0" + _bal9(addr(T_P)) + "0" + _bal9(addr(T_CONST2)))
    pad()
    s += "1100"; pad()
    s += "110t00"; pad()

    trytes = []
    for i in range(0, len(s), 9):
        v = 0
        for c in s[i:i + 9]:
            v = v * 3 + {'0': 0, '1': 1, 't': -1}[c]
        trytes.append(v % M)
    return trytes + data + [2, addr(T_CONST2)]


def generate_kontakion(text):
    trytes = assemble(text)
    word = terscii(text)
    rows = []
    for k, v in enumerate(trytes):
        radix = 29996 - k
        a = f"{format(radix & 0xFFFFFF, '06X')}{format(k, '03X')}"
        rows.append((to_hept(v), a, k))
    rows.sort(key=lambda x: (hept_to_int(x[0]), x[2]))
    return [f"{i}:{h}<{terscii(a)}>{word}" for i, (h, a, k) in enumerate(rows)]


# --------------------------------------------------------------------------
# Kontakion loading and text extraction
# --------------------------------------------------------------------------

def read_jesus(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ":" in ln and "<" in ln]


def load_kontakion(lines):
    cells = {}
    for line in lines:
        hept = line.split(":", 1)[1].split("<", 1)[0]
        t9 = line.split("<", 1)[1].split(">", 1)[0]
        k = int(unterscii(t9)[-3:], 16)
        cells[k] = hept_to_int(hept)
    return cells


def extract_text(cells):
    k_start = dlog2(cells[7])
    k_end = dlog2(cells[max(cells)])
    return "".join(chr(cells[k]) for k in range(k_start, k_end))


# --------------------------------------------------------------------------
# DMT intonation: each tryte's three heptavintimal digits are a chord.
# Dominant / Mediant / Tonic occupy their own octave bands in 27-EDO.
# --------------------------------------------------------------------------

RATE = 44100
DUR = 0.25
BANDS = {"D": 1760.0, "M": 880.0, "T": 440.0}


def hept3(v):
    d = []
    for _ in range(3):
        v, r = divmod(v, 27)
        d.append(r)
    return d[::-1]


def digit_freq(band, d):
    return BANDS[band] * 2.0 ** (d / 27.0)


def dmt_triad(value):
    D, Mm, T = hept3(value)
    return [digit_freq("D", D), digit_freq("M", Mm), digit_freq("T", T)]


def encode_to_wav(text, path):
    samples = []
    for ch in text:
        freqs = dmt_triad(ord(ch))
        for n in range(int(RATE * DUR)):
            t = n / RATE
            samples.append(sum(math.sin(2.0 * math.pi * f * t) for f in freqs))
    mx = max(abs(s) for s in samples) or 1.0
    pcm = b"".join(struct.pack("<h", int(s / mx * 32767)) for s in samples)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    print("Wrote", path)


def goertzel(seg, freq):
    sp = sp2 = 0.0
    n = len(seg)
    k = int(0.5 + n * freq / RATE)
    coeff = 2.0 * math.cos(2.0 * math.pi * k / n)
    for x in seg:
        s = x + coeff * sp - sp2
        sp2, sp = sp, s
    return sp2 * sp2 + sp * sp - coeff * sp * sp2


def decode_from_wav(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    samples = [struct.unpack("<h", raw[i:i + 2])[0] / 32768.0
               for i in range(0, len(raw), 2)]
    seg_len = int(RATE * DUR)
    n_chars = len(samples) // seg_len
    chars = []
    for i in range(n_chars):
        seg = samples[i * seg_len:(i + 1) * seg_len]
        digits = [max(range(27), key=lambda d: goertzel(seg, digit_freq(b, d)))
                  for b in ("D", "M", "T")]
        chars.append(chr(digits[0] * 729 + digits[1] * 27 + digits[2]))
    return "".join(chars)


# --------------------------------------------------------------------------

def slug(text):
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (s[:24] or "kontakion") + ".jesus"


def main():
    if len(sys.argv) < 2:
        with open("helloworld.jesus", "w") as f:
            f.write("\n".join(generate_kontakion("Hello, World!")) + "\n")
        print("Generated helloworld.jesus")
        print()
        print("usage:")
        print('  python3 jesus.py file.jesus        intonate a Kontakion')
        print('  python3 jesus.py \'"Some text!"\'    generate a Kontakion for the string')
        return

    arg = sys.argv[1]

    if '"' in arg or "'" in arg:         # quoted string: generate, then intonate
        text = arg.strip().strip('"\'')
        path = slug(text)
        with open(path, "w") as f:
            f.write("\n".join(generate_kontakion(text)) + "\n")
        print(f"Generated Kontakion for {text!r} -> {path}")
    elif arg.endswith(".wav"):           # a performance: decode it back to text
        print("Decoded text:", repr(decode_from_wav(arg)))
        return
    else:                                # a .jesus file: intonate
        path = arg

    cells = load_kontakion(read_jesus(path))
    text = extract_text(cells)
    print("Text from Kontakion:", repr(text))
    encode_to_wav(text, path.rsplit(".", 1)[0] + ".wav")


if __name__ == "__main__":
    main()