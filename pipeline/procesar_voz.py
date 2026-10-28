#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Procesa la grabación de voz de un episodio.

  python procesar_voz.py --ficha episodios/002-.../ficha.json --entrada audio.m4a \
         --salida-audio voz.wav --salida-tiempos tiempos.json

Pasos:
  1. Detecta los tramos con voz (el resto se considera pausa/respiración/ruido).
  2. Atenúa fuerte las pausas (quita respiraciones y ruido) y nivela cada tramo,
     de modo que el final de frase sin aire no se hunda.
  3. Cadena "radio": EQ, compresión en dos etapas, ajuste a -18,4 LUFS y limitador.
  4. Reparte los tramos en los bloques de la ficha (por nº de palabras y pausas)
     y escribe los tiempos reales de cada bloque.
Los cortes entre bloques son una ESTIMACIÓN (no se transcribe el audio).
"""
import argparse
import json
import re
import subprocess
import tempfile
import os
import wave

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

SR = 48000
LUFS_OBJ = -18.4
CADENA = ("highpass=f=80,afftdn=nr=8:nf=-55,"
          "equalizer=f=100:t=q:w=0.8:g=5,equalizer=f=220:t=q:w=1:g=2,"
          "equalizer=f=380:t=q:w=1.2:g=-3,equalizer=f=3500:t=q:w=1:g=3.5,"
          "equalizer=f=10000:t=h:w=4000:g=-4,"
          "acompressor=threshold=0.04:ratio=4.5:attack=5:release=100:makeup=3,"
          "acompressor=threshold=0.1:ratio=2:attack=10:release=150")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def leer_wav(ruta):
    w = wave.open(ruta)
    x = np.frombuffer(w.readframes(w.getnframes()), "int16").astype("float64") / 32768.0
    return x, w.getframerate()


def escribir_wav(ruta, y):
    y = np.clip(y, -1, 1)
    o = wave.open(ruta, "wb")
    o.setnchannels(1)
    o.setsampwidth(2)
    o.setframerate(SR)
    o.writeframes((y * 32767).astype("int16").tobytes())
    o.close()


def detectar_tramos(x, sr, on=-44.0, off=-48.0, unir=0.20, minimo=0.10):
    xb = sosfilt(butter(4, [180, 4500], "band", fs=sr, output="sos"), x)
    hop = sr // 100
    n = len(x) // hop
    env = np.array([20 * np.log10(np.sqrt(np.mean(xb[i * hop:(i + 1) * hop] ** 2)) + 1e-9)
                    for i in range(n)])
    activo, st = np.zeros(n, bool), False
    for i in range(n):
        if not st and env[i] > on:
            st = True
        elif st and env[i] < off:
            st = False
        activo[i] = st
    tramos, i = [], 0
    while i < n:
        if activo[i]:
            j = i
            while j < n and activo[j]:
                j += 1
            tramos.append([i * 0.01, j * 0.01])
            i = j
        else:
            i += 1
    fus = []
    for t in tramos:
        if fus and t[0] - fus[-1][1] < unir:
            fus[-1][1] = t[1]
        else:
            fus.append(t)
    return [t for t in fus if t[1] - t[0] >= minimo]


def limpiar_y_nivelar(x, sr, tramos, silencio_db=-42, pre=0.05, post=0.18, objetivo=0.045):
    g = np.full(len(x), 10 ** (silencio_db / 20))
    for a, b in tramos:
        a, b = max(0, a - pre), min(len(x) / sr, b + post)
        seg = x[int(a * sr):int(b * sr)]
        rms = np.sqrt(np.mean(seg ** 2)) + 1e-9
        g[int(a * sr):int(b * sr)] = np.clip(objetivo / rms, 10 ** (-3 / 20), 10 ** (8 / 20))
    g = uniform_filter1d(g, int(sr * 0.04))
    g = uniform_filter1d(g, int(sr * 0.03))
    return x * g


def lufs(ruta):
    r = run(["ffmpeg", "-hide_banner", "-i", ruta, "-af", "ebur128=peak=true", "-f", "null", "-"])
    m = re.findall(r"I:\s+(-?\d+\.\d+) LUFS", r.stderr)
    return float(m[-1])


def repartir_bloques(ficha, tramos, t0, lam=1.5):
    """Reparte los tramos de voz entre los bloques (programación dinámica).

    Cada bloque debería hablarse a un ritmo (seg/palabra) parecido al del
    conjunto, y se prefiere cortar donde la pausa es más larga."""
    pal = [len(b["texto"].split()) for b in ficha["bloques"]]
    K, N = len(pal), len(tramos)
    dur = np.array([b - a for a, b in tramos])
    pref = np.concatenate([[0], np.cumsum(dur)])
    r = pref[-1] / sum(pal)
    gap = [tramos[i + 1][0] - tramos[i][1] for i in range(N - 1)] + [0.0]

    def coste(i, j, k):  # bloque k usa tramos i..j-1
        d = pref[j] - pref[i]
        return pal[k] * (d / pal[k] - r) ** 2 * 10 - (lam * gap[j - 1] if k < K - 1 else 0)

    INF = 1e18
    dp = [[INF] * (N + 1) for _ in range(K + 1)]
    prev = [[0] * (N + 1) for _ in range(K + 1)]
    dp[0][0] = 0
    for k in range(K):
        for j in range(k + 1, N + 1):
            for i in range(k, j):
                if dp[k][i] < INF:
                    c = dp[k][i] + coste(i, j, k)
                    if c < dp[k + 1][j]:
                        dp[k + 1][j], prev[k + 1][j] = c, i
    cortes, j = [], N
    for k in range(K, 0, -1):
        i = prev[k][j]
        cortes.append((i, j))
        j = i
    cortes.reverse()
    return [{"id": b["id"], "inicio": round(tramos[i][0] - t0, 2), "fin": round(tramos[j - 1][1] - t0, 2)}
            for b, (i, j) in zip(ficha["bloques"], cortes)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ficha", required=True)
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--salida-audio", required=True)
    ap.add_argument("--salida-tiempos", required=True)
    ap.add_argument("--lead", type=float, default=0.5, help="silencio antes de la 1ª palabra")
    a = ap.parse_args()

    ficha = json.load(open(a.ficha, encoding="utf-8"))
    tmp = tempfile.mkdtemp()
    base = os.path.join(tmp, "base.wav")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.entrada, "-ar", str(SR), "-ac", "1", base])
    x, sr = leer_wav(base)
    tramos = detectar_tramos(x, sr)
    print(len(tramos), "tramos de voz")
    y = limpiar_y_nivelar(x, sr, tramos)
    t0 = max(0.0, tramos[0][0] - a.lead)
    y = y[int(t0 * sr):int((tramos[-1][1] + 0.8) * sr)]
    gate = os.path.join(tmp, "gate.wav")
    escribir_wav(gate, y)
    proc = os.path.join(tmp, "proc.wav")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", gate, "-af", CADENA, proc])
    ganancia = LUFS_OBJ - lufs(proc)
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", proc, "-af",
         f"volume={ganancia:.2f}dB,alimiter=limit=0.89:level=disabled", a.salida_audio])
    print("LUFS final: %.1f (objetivo %.1f)" % (lufs(a.salida_audio), LUFS_OBJ))
    tiempos = repartir_bloques(ficha, tramos, t0)
    json.dump(tiempos, open(a.salida_tiempos, "w", encoding="utf-8"), indent=1)
    for t in tiempos:
        print(t)


if __name__ == "__main__":
    main()
