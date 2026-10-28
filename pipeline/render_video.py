#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renderizador de Shorts verticales (1080x1920) de "Código de Bolsillo".

Uso:
  python render_video.py --ficha episodios/001-variables/ficha.json --salida previa.mp4
  python render_video.py --ficha ... --frame 20.5 --png fotograma.png        (un solo fotograma)
  python render_video.py --ficha ... --tiempos tiempos.json --audio voz.wav --salida video.mp4

Sin --tiempos se usan tiempos ESTIMADOS (por número de palabras). Con --tiempos se usan
los tiempos reales de cada bloque de la grabación:
  [{"id": 1, "inicio": 1.0, "fin": 6.2}, ...]
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

W, H, FPS = 1080, 1920, 30

# ---- Paleta de marca ------------------------------------------------------
BG = (15, 23, 42)
CARD = (30, 41, 70)
CARD2 = (22, 31, 56)
EDGE = (52, 68, 112)
AMBER = (255, 176, 0)
WHITE = (245, 247, 250)
GRAY = (148, 163, 184)
GREEN = (134, 239, 172)
PINK = (249, 168, 212)
BLUE = (125, 211, 252)
YELLOW = (253, 224, 71)
DARK = (15, 23, 42)
COLORES = {"verde": GREEN, "ambar": AMBER, "rosa": PINK, "azul": BLUE}

# ---- Fuentes (licencias libres) -------------------------------------------
POPPINS_B = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
POPPINS_M = "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# ---- Geometría (zona segura: x 70..950, evita la franja inferior de YouTube)
X0, X1 = 70, 950
CODE_Y0, CODE_Y1 = 720, 1180
LH = 92            # alto de línea de código
LINE_TOP = 800     # y de la primera línea
CODE_X = 185       # x del código
OUT_Y0, OUT_Y1 = 1200, 1390
SUB_Y = 1475
MAX_LINEAS = 4
CPS = 14.0         # caracteres por segundo al "escribir" código

RE_ASIG = re.compile(r"^\s*(\w+)\s*=\s*(-?\d+(?:\.\d+)?)\s*$")


# ---- Utilidades -----------------------------------------------------------
_fc = {}


def font(path, size):
    k = (path, max(8, int(round(size))))
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, k[1])
    return _fc[k]


def clamp01(x):
    return max(0.0, min(1.0, x))


def lerp(a, b, x):
    return a + (b - a) * x


def ease_out_back(x):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def mix(c1, c2, a):
    """a=1 -> c1, a=0 -> c2."""
    a = clamp01(a)
    return tuple(int(c1[i] * a + c2[i] * (1 - a)) for i in range(3))


def color_token(tt):
    if tt in Token.Name.Builtin:
        return BLUE
    if tt in Token.Name:
        return GREEN
    if tt in Token.Number:
        return AMBER
    if tt in Token.Operator or tt in Token.Keyword:
        return PINK
    if tt in Token.Literal.String:
        return YELLOW
    if tt in Token.Punctuation:
        return (203, 213, 225)
    return WHITE


# ---- Tiempos y eventos ----------------------------------------------------
def tiempos_estimados(ficha, inicio=0.8, pausa=1.0, seg_palabra=0.36):
    t = inicio
    res = {}
    for b in ficha["bloques"]:
        d = len(b["texto"].split()) * seg_palabra + 0.3
        res[b["id"]] = (t, t + d)
        t += d + pausa
    return res, (t - pausa) + 1.2


def construir_eventos(ficha, T):
    ev = {"lineas": {}, "activas": [], "valores": [], "salidas": [], "anot": []}
    prev_code, prev_out = [], []
    for b in ficha["bloques"]:
        t0, t1 = T[b["id"]]
        D = t1 - t0
        code = b.get("codigo_en_pantalla", [])
        out = b["salida"].split("\n") if b.get("salida") else []
        t = t0 + b.get("momento", 0.5) * D
        fin_escritura = None
        for j, linea in enumerate(code):
            if j >= len(prev_code) or prev_code[j] != linea:
                dur = max(0.6, len(linea) / CPS)
                ev["lineas"].setdefault(j, []).append((t, t + dur, linea))
                ev["activas"].append((t, j))
                m = RE_ASIG.match(linea)
                if m:
                    ev["valores"].append((t + dur + 0.1, m.group(1), m.group(2)))
                fin_escritura = t + dur
                t += dur + 0.15
        nuevas = out[len(prev_out):]
        if nuevas:
            ts = (fin_escritura + 0.3) if fin_escritura else t0 + b.get("momento_salida", 0.5) * D
            for k, s in enumerate(nuevas):
                ev["salidas"].append((ts + 0.45 * k, len(prev_out) + k, s))
        for a in b.get("anotaciones", []):
            hasta = a.get("hasta", b["id"])
            ev["anot"].append(dict(a, t_in=t0 + a.get("en", 0.5) * D, t_out=T[hasta][1] + 0.4))
        prev_code, prev_out = code, out
    return ev


def preparar_subs(ficha, T):
    subs = []
    for b in ficha["bloques"]:
        t0, t1 = T[b["id"]]
        palabras = b["texto"].split()
        pesos = [len(w) + (4 if w[-1] in ".:;?!" else 2 if w[-1] == "," else 0) for w in palabras]
        tot = float(sum(pesos))
        acc = 0.0
        items = []
        for w, p in zip(palabras, pesos):
            ws = t0 + (t1 - t0) * acc / tot
            acc += p
            items.append((w, ws, t0 + (t1 - t0) * acc / tot))
        chunks, cur = [], []
        for it in items:
            if cur and (len(" ".join(x[0] for x in cur + [it])) > 22 or len(cur) >= 4):
                chunks.append(cur)
                cur = []
            cur.append(it)
            w = it[0]
            if w[-1] in ".:;?!" or (w[-1] == "," and len(cur) >= 2):
                chunks.append(cur)
                cur = []
        if cur:
            chunks.append(cur)
        for i, c in enumerate(chunks):
            ce = chunks[i + 1][0][1] if i + 1 < len(chunks) else c[-1][2] + 0.25
            subs.append((c[0][1], ce, c))
    return subs


# ---- Capa estática (fondo, tarjetas, cabecera) ------------------------------
def capa_estatica(ficha, ruta_logo):
    ys = np.linspace(0, 1, H)[:, None]
    top, bot = np.array([18, 27, 50]), np.array([10, 15, 30])
    grad = top[None, :] * (1 - ys) + bot[None, :] * ys
    arr = np.repeat(grad[:, None, :], W, axis=1).astype("uint8")
    SS = 2
    big = Image.fromarray(arr, "RGB").resize((W * SS, H * SS), Image.BILINEAR)
    d = ImageDraw.Draw(big)
    for yy in range(30, H, 60):
        for xx in range(30, W, 60):
            r = 2 * SS
            d.ellipse((xx * SS - r, yy * SS - r, xx * SS + r, yy * SS + r), fill=(25, 37, 66))
    # tarjeta de código
    d.rounded_rectangle((X0 * SS, CODE_Y0 * SS, X1 * SS, CODE_Y1 * SS), radius=30 * SS,
                        fill=CARD, outline=EDGE, width=3 * SS)
    d.rounded_rectangle((X0 * SS, CODE_Y0 * SS, X1 * SS, (CODE_Y0 + 70) * SS), radius=30 * SS,
                        fill=CARD2, corners=(True, True, False, False))
    for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx, cy, r = (110 + 40 * k) * SS, (CODE_Y0 + 35) * SS, 11 * SS
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=c)
    # tarjeta de salida
    d.rounded_rectangle((X0 * SS, OUT_Y0 * SS, X1 * SS, OUT_Y1 * SS), radius=30 * SS,
                        fill=CARD2, outline=EDGE, width=3 * SS)
    img = big.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    # cabecera
    logo = Image.open(ruta_logo).convert("RGB").resize((96, 96), Image.LANCZOS)
    mask = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, 95, 95), fill=255)
    img.paste(logo, (X0, 118), mask)
    d.text((X0 + 118, 150), "Código de Bolsillo", font=font(POPPINS_B, 44), fill=WHITE, anchor="lm")
    d.text((X0 + 118, 194), "Python desde cero", font=font(POPPINS_M, 28), fill=GRAY, anchor="lm")
    d.text((X1, 150), "#" + ficha["id"], font=font(POPPINS_B, 44), fill=AMBER, anchor="rm")
    d.text((X1, 194), ficha["concepto"], font=font(POPPINS_M, 28), fill=GRAY, anchor="rm")
    # rótulos
    d.text((510, CODE_Y0 + 35), "ejemplo.py", font=font(POPPINS_M, 30), fill=GRAY, anchor="mm")
    d.text((X0 + 36, OUT_Y0 + 34), "Salida", font=font(POPPINS_M, 30), fill=GRAY, anchor="lm")
    return img


# ---- Elementos dinámicos -----------------------------------------------------
def progreso(d, t, ctx):
    d.rounded_rectangle((X0, 262, X1, 270), radius=4, fill=CARD)
    w = (X1 - X0) * clamp01(t / ctx.total)
    if w > 8:
        d.rounded_rectangle((X0, 262, X0 + w, 270), radius=4, fill=AMBER)


def titular(d, t, ctx):
    vis = ctx.visual
    t_fin = t_primera_entrada(vis, ctx)
    if t > t_fin:
        return
    e = ease_out_back(clamp01(t / 0.55))
    a = min(clamp01(t / 0.25), clamp01((t_fin - t) / 0.35))
    escala = 0.8 + 0.2 * e
    for k, txt in enumerate(vis["titular"]):
        col = WHITE if k == 0 else AMBER
        d.text((510, 440 + k * 125), txt, font=font(POPPINS_B, 104 * escala),
               fill=mix(col, BG, a), anchor="mm")


def t_primera_entrada(vis, ctx):
    if vis.get("tipo") == "tarjetas":
        return min(t_tarjeta(c, ctx) for c in vis["tarjetas"])
    return ctx.T[vis["entra_bloque"]][0]


def t_tarjeta(c, ctx):
    t0, t1 = ctx.T[c["entra_bloque"]]
    return t0 + c.get("en", 0.5) * (t1 - t0)


def tarjetas(d, t, ctx):
    vis = ctx.visual
    t_sale = ctx.T[vis["sale_bloque"]][0] + 0.05
    if t > t_sale + 0.35:
        return
    fuera = 1 - clamp01((t - t_sale) / 0.3)
    n = len(vis["tarjetas"])
    xs = [510] if n == 1 else [300 + i * (420 // max(1, n - 1)) for i in range(n)]
    for c, cx in zip(vis["tarjetas"], xs):
        ti = t_tarjeta(c, ctx)
        if t < ti:
            continue
        e = ease_out_back(clamp01((t - ti) / 0.5))
        a = clamp01((t - ti) / 0.25) * fuera
        col = COLORES[c["color"]]
        w, h = vis.get("ancho_tarjeta", 380) * (0.7 + 0.3 * e), 235 * (0.7 + 0.3 * e)
        cy = 470
        d.rounded_rectangle((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), radius=28,
                            fill=mix(CARD, BG, a), outline=mix(col, BG, a), width=5)
        d.text((cx, cy - 25), c["valor"], font=font(MONO_B, 78 * (0.7 + 0.3 * e)),
               fill=mix(WHITE, CARD, a), anchor="mm")
        tt = ti + 0.45
        if t >= tt:
            p = clamp01((t - tt) / 0.3)
            pe = ease_out_back(p)
            f = font(MONO_B, 46 * (0.8 + 0.2 * pe))
            pw = f.getlength(c["tipo"]) + 70
            py = cy + h / 2 - 8
            d.rounded_rectangle((cx - pw / 2, py - 35, cx + pw / 2, py + 35), radius=35,
                                fill=mix(col, BG, clamp01(p * 2) * fuera))
            d.text((cx, py), c["tipo"], font=f, fill=mix(DARK, col, clamp01(p * 2) * fuera), anchor="mm")
            if c.get("descripcion"):
                d.text((cx, py + 62), c["descripcion"], font=font(POPPINS_M, 30),
                       fill=mix(GRAY, BG, clamp01(p * 2) * fuera), anchor="mm")

    for op in vis.get("operadores", []):
        t0, t1 = ctx.T[op["entra_bloque"]]
        ti = t0 + op.get("en", 0.2) * (t1 - t0)
        ts = ctx.T[op["sale_bloque"]][0] + op.get("sale_en", 0.0) if op.get("sale_bloque") else 1e9
        if t < ti or t > ts + 0.3:
            continue
        e = ease_out_back(clamp01((t - ti) / 0.4))
        a = clamp01((t - ti) / 0.2) * (1 - clamp01((t - ts) / 0.25)) * fuera
        r = 52 * (0.5 + 0.5 * e)
        d.ellipse((510 - r, 470 - r, 510 + r, 470 + r), fill=mix(AMBER, BG, a))
        d.text((510, 470), op["texto"], font=font(POPPINS_B, 70 * (0.6 + 0.4 * e)),
               fill=mix(DARK, AMBER, a), anchor="mm")


def pulso(t, ctx):
    p = 0.0
    for ts, _, _ in ctx.ev["salidas"]:
        if ts <= t <= ts + 0.5:
            p = max(p, 1 - (t - ts) / 0.5)
    return p


def valor_actual(t, ctx):
    vals = [(tv, n, v) for tv, n, v in ctx.ev["valores"] if tv <= t]
    if not vals:
        return None, None, 1.0
    tv, _, v = vals[-1]
    prev = vals[-2][2] if len(vals) > 1 else None
    return v, prev, clamp01((t - tv) / 0.55)


def caja(d, t, ctx):
    vis = ctx.visual
    t_in = ctx.T[vis["entra_bloque"]][0]
    t_sale = ctx.T[vis["sale_bloque"]][0] + 0.05
    if t < t_in or t > t_sale + 0.45:
        return
    s = lerp(0.5, 1.0, ease_out_back(clamp01((t - t_in) / 0.6)))
    a = clamp01((t - t_in) / 0.25) * (1 - clamp01((t - t_sale) / 0.3))
    cx, yb = 510, 690

    def X(x):
        return cx + (x - cx) * s

    def Y(y):
        return yb + (y - yb) * s

    pl = pulso(t, ctx)
    borde = mix(WHITE, AMBER, pl)
    d.rounded_rectangle((X(260), Y(420), X(760), Y(478)), radius=14 * s,
                        fill=mix((45, 60, 100), BG, a))
    d.rounded_rectangle((X(285), Y(470), X(735), Y(690)), radius=24 * s, fill=mix(CARD, BG, a),
                        outline=mix(borde, BG, a), width=max(2, int((5 + 4 * pl) * s)))
    # valor
    v, prev, p = valor_actual(t, ctx)
    vf = lambda sz: font(POPPINS_B, sz * s)
    if v is None:
        d.text((cx, Y(585)), "?", font=vf(150), fill=mix(GRAY, CARD, a), anchor="mm")
    else:
        if prev is None:
            e = ease_out_back(p)
            d.text((cx, Y(585)), v, font=vf(140 * (0.3 + 0.7 * e)),
                   fill=mix(AMBER, CARD, clamp01(p * 2) * a), anchor="mm")
        else:
            d.text((cx, Y(585 - 110 * p)), prev, font=vf(140),
                   fill=mix(AMBER, CARD, clamp01(1 - p * 1.6) * a), anchor="mm")
            yv = 585 - (1 - ease_out_back(p)) * 150
            d.text((cx, Y(yv)), v, font=vf(140), fill=mix(AMBER, CARD, clamp01(p * 2) * a), anchor="mm")
    # etiqueta con el nombre
    if ctx.ev["valores"]:
        nombre = ctx.ev["valores"][0][1]
        t_tag = t_in + vis.get("etiqueta_en", 0.35) * (ctx.T[vis["entra_bloque"]][1] - t_in)
        if t >= t_tag:
            p = clamp01((t - t_tag) / 0.35)
            e = ease_out_back(p)
            at = clamp01(p * 2) * a
            f = font(MONO_B, 54)
            w = max(280, f.getlength(nombre) + 90)
            cy = 360 - (1 - e) * 30
            d.line((cx, cy + 40, cx, Y(420)), fill=mix(GREEN, BG, at), width=5)
            d.rounded_rectangle((cx - w / 2, cy - 40, cx + w / 2, cy + 40), radius=40,
                                fill=mix((20, 46, 40), BG, at), outline=mix(GREEN, BG, at), width=5)
            d.text((cx, cy), nombre, font=f, fill=mix(GREEN, BG, at), anchor="mm")


def cierre(d, t, ctx):
    vis = ctx.visual
    c = vis["cierre"]
    t0 = ctx.T[vis["sale_bloque"]][0] + 0.55
    if t < t0:
        return
    a = clamp01((t - t0) / 0.4)
    e = ease_out_back(clamp01((t - t0) / 0.6))
    cy = 430
    for txt, cxp, col in ((c["izquierda"], 300, COLORES[c.get("color_izq", "verde")]),
                          (c["derecha"], 720, COLORES[c.get("color_der", "ambar")])):
        f = font(MONO_B, 56 * (0.85 + 0.15 * e))
        w, h = f.getlength(txt) + 80, 110
        d.rounded_rectangle((cxp - w / 2, cy - h / 2, cxp + w / 2, cy + h / 2), radius=h / 2,
                            fill=mix(col, BG, a))
        d.text((cxp, cy), txt, font=f, fill=mix(DARK, col, a), anchor="mm")
    d.text((510, cy), c.get("signo", "+"), font=font(POPPINS_B, 84), fill=mix(WHITE, BG, a), anchor="mm")
    d.text((510, 560), c["texto"], font=font(POPPINS_B, 80 * (0.9 + 0.1 * e)),
           fill=mix(WHITE, BG, a), anchor="mm")
    d.text((510, 650), c["siguiente"], font=font(POPPINS_M, 42), fill=mix(GRAY, BG, a), anchor="mm")


def pill(d, cx, cy, texto, col, a, fondo=CARD):
    f = font(POPPINS_B, 28)
    w, h = f.getlength(texto) + 44, 46
    d.polygon([(cx - 9, cy - h / 2), (cx + 9, cy - h / 2), (cx, cy - h / 2 - 12)], fill=mix(col, fondo, a))
    d.rounded_rectangle((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), radius=h / 2, fill=mix(col, fondo, a))
    d.text((cx, cy + 1), texto, font=f, fill=mix(DARK, col, a), anchor="mm")


def texto_linea(j, t, ctx):
    """Devuelve (texto_visible, longitud_total, escribiendo) de la línea j en t."""
    vers = [v for v in ctx.ev["lineas"].get(j, []) if v[0] <= t]
    if not vers:
        return "", 0, False
    ta, tb, txt = vers[-1]
    n = int(round(len(txt) * clamp01((t - ta) / (tb - ta))))
    return txt[:n], len(txt), t < tb


def codigo(d, t, ctx):
    fm = font(MONO_B, ctx.code_size)
    cw = fm.getlength("M")
    act, ta_act = 0, -1.0
    for ta, j in ctx.ev["activas"]:
        if ta <= t:
            act, ta_act = j, ta
    # resaltado de la línea activa
    yt = LINE_TOP + act * LH
    al = clamp01((t - ta_act) / 0.2) if ta_act >= 0 else 1.0
    d.rectangle((X0 + 12, yt, X1 - 12, yt + LH), fill=mix((38, 52, 90), CARD, al))
    d.rectangle((X0 + 12, yt, X0 + 20, yt + LH), fill=mix(AMBER, CARD, al))
    # brillo detrás de tokens
    for a_ in ctx.ev["anot"]:
        if a_.get("brillo") and a_["t_in"] <= t <= a_["t_out"]:
            txt = ctx.ev["lineas"][a_["linea"]][-1][2]
            i = txt.index(a_["token"])
            x = CODE_X + i * cw
            yc = LINE_TOP + a_["linea"] * LH + LH / 2
            ab = min(clamp01((t - a_["t_in"]) / 0.3), clamp01((a_["t_out"] - t) / 0.3))
            k = 0.45 + 0.25 * math.sin(t * 7)
            d.rounded_rectangle((x - 8, yc - 38, x + cw * len(a_["token"]) + 8, yc + 38), radius=12,
                                fill=mix(mix(AMBER, (38, 52, 90), k), (38, 52, 90), ab))
    # líneas
    for j in range(MAX_LINEAS):
        visible, total, escribiendo = texto_linea(j, t, ctx)
        yc = LINE_TOP + j * LH + LH / 2
        if visible or j == act:
            d.text((150, yc), str(j + 1), font=font(POPPINS_M, 34), fill=GRAY, anchor="rm")
        if total:
            completo = ctx.ev["lineas"][j]
            txt = [v for v in completo if v[0] <= t][-1][2]
            off = 0
            for tt, val in lex(txt, PythonLexer()):
                val = val.rstrip("\n")
                if not val:
                    continue
                n = len(visible) - off
                if n > 0:
                    d.text((CODE_X + off * cw, yc), val[:n], font=fm, fill=color_token(tt), anchor="lm")
                off += len(val)
        if j == act and (escribiendo or int(t * 2.2) % 2 == 0):
            x = CODE_X + len(visible) * cw
            d.rectangle((x + 2, yc - 34, x + cw * 0.5, yc + 34), fill=AMBER)
    # anotaciones
    for a_ in ctx.ev["anot"]:
        if a_["t_in"] <= t <= a_["t_out"]:
            txt = ctx.ev["lineas"][a_["linea"]][-1][2]
            i = txt.index(a_["token"])
            cx = CODE_X + (i + len(a_["token"]) / 2) * cw
            ab = min(clamp01((t - a_["t_in"]) / 0.3), clamp01((a_["t_out"] - t) / 0.3))
            cy = LINE_TOP + a_["linea"] * LH + LH + 16 + (1 - clamp01((t - a_["t_in"]) / 0.3)) * 10
            pill(d, cx, cy, a_["texto"], COLORES[a_["color"]], ab)


def salida(d, t, ctx):
    fm = font(MONO_B, 64)
    cw = fm.getlength("M")
    for ts, k, txt in ctx.ev["salidas"]:
        if t < ts:
            continue
        n = int(round(len(txt) * clamp01((t - ts) / 0.3)))
        yc = 1285 + k * 64
        d.text((X0 + 40, yc), ">", font=fm, fill=GRAY, anchor="lm")
        d.text((X0 + 40 + cw * 1.6, yc), txt[:n], font=fm, fill=GREEN, anchor="lm")


def subtitulos(d, t, ctx):
    for cs, ce, c in ctx.subs:
        if cs <= t < ce:
            e = ease_out_back(clamp01((t - cs) / 0.18))
            f = font(POPPINS_B, 62 * (0.9 + 0.1 * e))
            esp = f.getlength(" ")
            anchos = [f.getlength(w) for w, _, _ in c]
            x = 510 - (sum(anchos) + esp * (len(c) - 1)) / 2
            for (w, ws, we), aw in zip(c, anchos):
                d.text((x, SUB_Y), w, font=f, fill=AMBER if ws <= t < we else WHITE, anchor="lm",
                       stroke_width=7, stroke_fill=(8, 12, 24))
                x += aw + esp
            break


def fotograma(t, ctx):
    img = ctx.estatica.copy()
    d = ImageDraw.Draw(img)
    progreso(d, t, ctx)
    if ctx.visual:
        titular(d, t, ctx)
        if ctx.visual.get("tipo") == "tarjetas":
            tarjetas(d, t, ctx)
        else:
            caja(d, t, ctx)
        cierre(d, t, ctx)
    codigo(d, t, ctx)
    salida(d, t, ctx)
    subtitulos(d, t, ctx)
    return img


# ---- Principal --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ficha", required=True)
    ap.add_argument("--tiempos")
    ap.add_argument("--audio")
    ap.add_argument("--salida", default="video.mp4")
    ap.add_argument("--frame", type=float)
    ap.add_argument("--png", default="fotograma.png")
    ap.add_argument("--logo", default=os.path.join(os.path.dirname(__file__), "..", "marca", "logo.png"))
    a = ap.parse_args()

    ficha = json.load(open(a.ficha, encoding="utf-8"))
    if a.tiempos:
        lst = json.load(open(a.tiempos, encoding="utf-8"))
        T = {x["id"]: (x["inicio"], x["fin"]) for x in lst}
        total = max(x["fin"] for x in lst) + 1.2
    else:
        T, total = tiempos_estimados(ficha)
    mx = max([len(l) for b in ficha["bloques"] for l in b.get("codigo_en_pantalla", [])] + [1])
    tam = min(80, (X1 - CODE_X - 25) / (mx * 0.602))
    ctx = SimpleNamespace(code_size=tam, T=T, total=total, ev=construir_eventos(ficha, T), subs=preparar_subs(ficha, T),
                          visual=ficha.get("visual"), estatica=capa_estatica(ficha, a.logo))

    if a.frame is not None:
        fotograma(a.frame, ctx).save(a.png)
        print("fotograma guardado:", a.png)
        return

    n = int(math.ceil(total * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if a.audio:
        cmd += ["-i", a.audio]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"]
    if a.audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [a.salida]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            p.stdin.write(fotograma(i / FPS, ctx).tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass  # ffmpeg cerró al terminar el audio (-shortest)
    p.wait()
    print("vídeo guardado:", a.salida, "| %.1f s | %d fotogramas" % (total, n))


if __name__ == "__main__":
    main()
