"""L'Artisan KANOYA メニューPR動画（Instagram リール 1080x1920 / 30秒）

ある日のディナーより「フルーツマリネ、倭紅茶、ココナッツ」

使い方:
    python3 video/make_video.py --fonts <フォントのディレクトリ> --out video/kanoya_fruit_marinade_reel.mp4
"""
import argparse
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
FPS = 30
DUR = 30.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images")

GOLD = (212, 180, 122)
IVORY = (246, 240, 230)


def img(n):
    # macOS からのアップロードでファイル名が NFD になっているため番号で探す
    for f in os.listdir(IMG_DIR):
        if f.endswith(f"{n:04d}.jpg"):
            return os.path.join(IMG_DIR, f)
    raise FileNotFoundError(n)


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_io(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out_cubic(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


# ---------------------------------------------------------------- shots
# (画像番号, 開始, 終了, ズーム開始, ズーム終了, パン(dx,dy) 開始→終了)
SHOTS = [
    (197, 0.0, 6.0, 1.16, 1.02, (0.00, 0.04), (0.00, 0.00)),
    (202, 5.2, 9.2, 1.02, 1.12, (0.00, -0.03), (0.00, 0.02)),
    (207, 8.4, 11.6, 1.10, 1.02, (0.02, 0.00), (-0.02, 0.00)),
    (210, 10.8, 14.4, 1.04, 1.14, (0.00, 0.00), (0.00, 0.03)),
    (214, 13.6, 17.6, 1.12, 1.03, (0.00, 0.03), (0.00, -0.02)),
    (223, 16.8, 20.6, 1.03, 1.12, (-0.02, 0.00), (0.02, 0.00)),
    (225, 19.8, 23.6, 1.12, 1.04, (0.00, -0.03), (0.00, 0.02)),
    (231, 22.8, 30.0, 1.00, 1.10, (0.00, 0.00), (0.00, 0.00)),
]
# shot i → i+1 の切り替え方
TRANSITIONS = ["wipe_down", "light_leak", "zoom_through", "parallax_up",
               "iris", "blur_dissolve", "dip_black"]

PRELOAD_H = 2300


def load_shots():
    out = []
    for n, *_ in SHOTS:
        im = Image.open(img(n)).convert("RGB")
        s = PRELOAD_H / im.height
        out.append(im.resize((round(im.width * s), PRELOAD_H), Image.LANCZOS))
    return out


def render_shot(i, t, pics, extra_zoom=1.0, dy_px=0.0):
    n, s, e, z0, z1, p0, p1 = SHOTS[i]
    p = ease_io((t - s) / (e - s))
    z = (z0 + (z1 - z0) * p) * extra_zoom
    im = pics[i]
    bh = im.height
    bw = bh * W / H
    if bw > im.width:
        bw = im.width
        bh = bw * H / W
    ww, wh = bw / z, bh / z
    cx = im.width / 2 + (p0[0] + (p1[0] - p0[0]) * p) * im.width
    cy = im.height / 2 + (p0[1] + (p1[1] - p0[1]) * p) * im.height
    cx = clamp(cx, ww / 2, im.width - ww / 2)
    cy = clamp(cy, wh / 2, im.height - wh / 2)
    box = (cx - ww / 2, cy - wh / 2 + dy_px, cx + ww / 2, cy + wh / 2 + dy_px)
    fr = im.resize((W, H), Image.BILINEAR, box=box)
    return np.asarray(fr, dtype=np.float32) / 255.0


def blur(a, amount):
    """amount 0..1 の簡易ぼかし（縮小→拡大）"""
    if amount <= 0.01:
        return a
    k = 1 + amount * 23
    im = Image.fromarray((a * 255).astype(np.uint8))
    sm = im.resize((max(1, int(W / k)), max(1, int(H / k))), Image.BILINEAR)
    sm = sm.filter(ImageFilter.GaussianBlur(1.2))
    return np.asarray(sm.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0


YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
RR = np.sqrt((XX - W / 2) ** 2 + (YY - H / 2) ** 2)
WARM = np.array([1.0, 0.86, 0.66], dtype=np.float32)


def transition(kind, a_fn, b_fn, p):
    """a_fn/b_fn は (extra_zoom, dy) → フレーム"""
    if kind == "wipe_down":
        # 紅茶を注ぐように、上から下へ柔らかく塗り替える
        q = ease_in_out_cubic(p)
        edge = -350 + q * (H + 700)
        m = np.clip((edge - YY) / 350 + 0.5, 0, 1)[..., None]
        a, b = a_fn(1, 0), b_fn(1, 0)
        out = a * (1 - m) + b * m
        line = np.exp(-((YY - edge) / 6) ** 2)[..., None] * math.sin(math.pi * p) * 0.35
        return np.clip(out + line * np.array(GOLD, np.float32) / 255, 0, 1)
    if kind == "light_leak":
        q = ease_io(p)
        a, b = a_fn(1 + 0.04 * q, 0), b_fn(1.04 - 0.04 * q, 0)
        out = a * (1 - q) + b * q
        glow = math.sin(math.pi * p) ** 2
        cx = W * (0.1 + 0.8 * p)
        leak = np.exp(-(((XX - cx) / 520) ** 2 + ((YY - H * 0.35) / 900) ** 2))[..., None]
        out = out + leak * WARM * glow * 0.85 + glow * 0.12
        return np.clip(out, 0, 1)
    if kind == "zoom_through":
        q = ease_in_out_cubic(p)
        a = blur(a_fn(1 + 0.35 * q, 0), q)
        b = blur(b_fn(1.25 - 0.25 * q, 0), 1 - q)
        return a * (1 - q) + b * q
    if kind == "parallax_up":
        q = ease_in_out_cubic(p)
        off = (1 - q) * H
        a = a_fn(1, 0)
        b = b_fn(1, 0)
        # 下から新しい画像がせり上がり、前の画像はゆっくり押し上げられる
        sa = int(round(q * H * 0.35))
        a = np.roll(a, -sa, axis=0) * (1 - 0.55 * q)
        out = a.copy()
        o = int(round(off))
        if o < H:
            out[o:] = b[: H - o]
            shade = np.exp(-((YY[o:] - o) / 60)) [..., None] * 0.0
            out[o:] -= shade
            if o > 0:
                out[max(0, o - 2):o] = np.array(GOLD, np.float32) / 255 * 0.8
        return out
    if kind == "iris":
        q = ease_in_out_cubic(p)
        r = q * math.hypot(W, H) * 0.6
        m = np.clip((r - RR) / 140 + 0.5, 0, 1)[..., None]
        a, b = a_fn(1 + 0.06 * q, 0), b_fn(1.06 - 0.06 * q, 0)
        return a * (1 - m) + b * m
    if kind == "blur_dissolve":
        q = ease_io(p)
        bl = math.sin(math.pi * p) * 0.6
        return blur(a_fn(1, 0), bl) * (1 - q) + blur(b_fn(1, 0), bl) * q
    if kind == "dip_black":
        if p < 0.5:
            return a_fn(1, 0) * (1 - ease_io(p * 2))
        return b_fn(1, 0) * ease_io(p * 2 - 1)
    raise ValueError(kind)


def base_frame(t, pics):
    active = [i for i, sh in enumerate(SHOTS) if sh[1] <= t < sh[2]]
    if not active:
        active = [len(SHOTS) - 1]
    if len(active) == 1:
        i = active[0]
        return render_shot(i, t, pics)
    i, j = active[0], active[1]
    s, e = SHOTS[j][1], SHOTS[i][2]
    p = (t - s) / (e - s)
    return transition(
        TRANSITIONS[i],
        lambda z, dy: render_shot(i, t, pics, z, dy),
        lambda z, dy: render_shot(j, t, pics, z, dy),
        p,
    )


# ---------------------------------------------------------------- grade
VIGNETTE = (1 - 0.55 * np.clip((RR / math.hypot(W / 2, H / 2)) ** 2.2, 0, 1))[..., None]
rng = np.random.default_rng(7)
GRAIN = [rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
GRAIN = [np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None] for g in GRAIN]
SHADOW_TINT = np.array([0.02, 0.03, 0.06], np.float32)


def grade(a, fi):
    a = a ** 1.04
    lum = a.mean(axis=2, keepdims=True)
    a = a * np.array([1.03, 1.0, 0.95], np.float32)  # 暖色寄り
    a = a + SHADOW_TINT * (1 - lum)  # 影を深い藍色に
    a = a * VIGNETTE
    a = a + GRAIN[fi % len(GRAIN)] * 0.018
    return np.clip(a, 0, 1)


# ---------------------------------------------------------------- telops
class Fonts:
    def __init__(self, d):
        self.d = d

    def jp(self, size, medium=False):
        f = "ShipporiMincho-Medium.ttf" if medium else "ShipporiMincho-Regular.ttf"
        return ImageFont.truetype(os.path.join(self.d, f), size)

    def la(self, size, italic=False, weight=400):
        f = "CormorantItalic.ttf" if italic else "Cormorant.ttf"
        font = ImageFont.truetype(os.path.join(self.d, f), size)
        try:
            font.set_variation_by_axes([weight])
        except Exception:
            pass
        return font


def text_w(font, s, track):
    return sum(font.getlength(c) for c in s) + track * max(0, len(s) - 1)


def draw_tracked(d, x, y, s, font, fill, track):
    for c in s:
        d.text((x, y), c, font=font, fill=fill)
        x += font.getlength(c) + track


class Layer:
    """テロップ1枚（RGBA）。表示位置・出入りのタイミング・出方を持つ"""

    def __init__(self, im, cx, cy, t0, t1, reveal="lr", rise=14, fade_out=0.45, reveal_dur=1.1,
                 scrim=0.45):
        pad = 130
        sh = Image.new("RGBA", (im.width + pad * 2, im.height + pad * 2), (0, 0, 0, 0))
        sh.paste(im, (pad, pad), im)
        alpha = sh.split()[3]
        shadow = Image.new("RGBA", sh.size, (8, 10, 18, 0))
        shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(12)).point(lambda v: int(v * 0.8)))
        comp = Image.alpha_composite(shadow, sh)
        arr = np.asarray(comp, dtype=np.float32) / 255.0
        self.rgb = arr[..., :3]
        self.a = arr[..., 3:4]
        # 文字の背後にうっすら敷く影（写真の上でも読めるように）
        sc = Image.new("L", sh.size, 0)
        ImageDraw.Draw(sc).rounded_rectangle(
            (pad - 30, pad - 20, pad + im.width + 30, pad + im.height + 20), radius=80, fill=255)
        sc = sc.filter(ImageFilter.GaussianBlur(55))
        self.scrim = np.asarray(sc, dtype=np.float32)[..., None] / 255.0 * scrim
        self.h, self.w = self.a.shape[:2]
        self.x = int(cx - self.w / 2)
        self.y = int(cy - self.h / 2)
        self.t0, self.t1 = t0, t1
        self.reveal, self.rise, self.fade_out, self.rd = reveal, rise, fade_out, reveal_dur
        gy, gx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.coord = gx / self.w if reveal == "lr" else gy / self.h

    def apply(self, frame, t):
        if t < self.t0 or t > self.t1:
            return
        p = (t - self.t0) / self.rd
        soft = 0.35
        m = np.clip((ease_out(p) * (1 + soft) - self.coord) / soft, 0, 1)[..., None]
        out_a = 1 - ease_io((t - (self.t1 - self.fade_out)) / self.fade_out)
        dy = int(round((1 - ease_out(p)) * self.rise))
        a = self.a * m * out_a
        sa = self.scrim * (ease_io(p * 1.4) * out_a)
        y0 = self.y + dy
        y1, x1 = y0 + self.h, self.x + self.w
        fy0, fx0 = max(0, y0), max(0, self.x)
        fy1, fx1 = min(H, y1), min(W, x1)
        ly0, lx0 = fy0 - y0, fx0 - self.x
        reg = frame[fy0:fy1, fx0:fx1]
        la = a[ly0:ly0 + (fy1 - fy0), lx0:lx0 + (fx1 - fx0)]
        lr = self.rgb[ly0:ly0 + (fy1 - fy0), lx0:lx0 + (fx1 - fx0)]
        reg = reg * (1 - sa[ly0:ly0 + (fy1 - fy0), lx0:lx0 + (fx1 - fx0)])
        frame[fy0:fy1, fx0:fx1] = reg * (1 - la) + lr * la


def block(lines, gap=18):
    """lines: [(種類, ...)] を中央揃えで縦に積んだ RGBA を作る"""
    items = []
    for ln in lines:
        if ln[0] == "text":
            _, s, font, fill, track = ln
            w = text_w(font, s, track)
            asc, desc = font.getmetrics()
            items.append(("text", s, font, fill, track, w, asc + desc))
        elif ln[0] == "rule":
            _, w, fill = ln
            items.append(("rule", None, None, fill, 0, w, 3))
        elif ln[0] == "space":
            items.append(("space", None, None, None, 0, 0, ln[1]))
    bw = int(max(i[5] for i in items)) + 20
    bh = int(sum(i[6] for i in items) + gap * (len(items) - 1)) + 10
    im = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    y = 5
    for kind, s, font, fill, track, w, h in items:
        x = (bw - w) / 2
        if kind == "text":
            draw_tracked(d, x, y, s, font, fill + (255,), track)
        elif kind == "rule":
            d.line([(x, y + 1), (x + w, y + 1)], fill=fill + (230,), width=2)
            # 中央に小さな菱形
            cx = bw / 2
            d.polygon([(cx, y - 5), (cx + 5, y + 1), (cx, y + 7), (cx - 5, y + 1)], fill=fill + (255,))
        y += h + gap
    return im


def vertical(s, font, fill, step=1.18, latin=None):
    """縦書きのテロップ。latin があれば右側に小さく添える"""
    size = font.size
    cell = int(size * step)
    lw = 0
    if latin:
        lf, ltxt, lfill = latin
        lw = int(lf.size * 1.6)
    im = Image.new("RGBA", (size + 30 + lw, cell * len(s) + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for k, c in enumerate(s):
        x, y = 10, 10 + k * cell
        if c in "、。":
            x += int(size * 0.62)
            y -= int(size * 0.55)
        elif c in "ー〜":
            g = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
            ImageDraw.Draw(g).text((size // 2, size // 2), c, font=font, fill=fill + (255,))
            g = g.rotate(-90, resample=Image.BICUBIC)
            im.alpha_composite(g, (x - size // 2, y - size // 2))
            continue
        d.text((x, y), c, font=font, fill=fill + (255,))
    if latin:
        lf, ltxt, lfill = latin
        tw = int(text_w(lf, ltxt, 4)) + 10
        g = Image.new("RGBA", (tw, lf.size * 2), (0, 0, 0, 0))
        draw_tracked(ImageDraw.Draw(g), 5, 0, ltxt, lf, lfill + (255,), 4)
        g = g.rotate(-90, expand=True, resample=Image.BICUBIC)
        im.alpha_composite(g, (size + 26, 14))
    return im


def build_layers(F):
    L = []
    # オープニング
    L.append(Layer(block([
        ("text", "ある日のディナーより", F.jp(40), IVORY, 10),
        ("space", 6),
        ("rule", 260, GOLD),
        ("space", 4),
        ("text", "L'Artisan", F.la(132, italic=True, weight=400), IVORY, 2),
        ("text", "K A N O Y A", F.la(64, weight=500), GOLD, 14),
    ], gap=14), W / 2, H * 0.19, 0.4, 3.5, reveal="tb", rise=24, reveal_dur=1.6, fade_out=0.6))

    # 1. フルーツマリネ
    L.append(Layer(block([
        ("text", "Fruits marinés", F.la(64, italic=True), GOLD, 3),
        ("text", "フルーツマリネ", F.jp(78, medium=True), IVORY, 14),
        ("rule", 220, GOLD),
        ("text", "瑞々しい果実を、ひと匙ずつ", F.jp(38), IVORY, 6),
    ]), W / 2, H * 0.17, 3.7, 5.6))

    # 2. 注ぐ（縦書き）
    L.append(Layer(vertical("倭紅茶の香りを、そっと注いで", F.jp(50), IVORY,
                            latin=(F.la(40, italic=True), "Thé de Yamato", GOLD)),
                   110, H * 0.62, 6.1, 8.7, reveal="tb", rise=0, reveal_dur=1.6))

    # 3. 倭紅茶
    L.append(Layer(block([
        ("text", "Thé japonais", F.la(60, italic=True), GOLD, 3),
        ("text", "倭紅茶", F.jp(84, medium=True), IVORY, 22),
        ("rule", 200, GOLD),
        ("text", "大和の地で育まれた", F.jp(36), IVORY, 4),
        ("text", "やわらかな香り", F.jp(36), IVORY, 4),
    ], gap=14), W * 0.24, H * 0.19, 11.8, 14.1))

    # 4. ココナッツ
    L.append(Layer(block([
        ("text", "Noix de coco", F.la(60, italic=True), GOLD, 3),
        ("text", "ココナッツ", F.jp(80, medium=True), IVORY, 16),
        ("rule", 200, GOLD),
        ("text", "まろやかなコクが、すべてを包む", F.jp(36), IVORY, 4),
    ]), W / 2, H * 0.72, 14.7, 17.3))

    # 5. 縦書き（左）
    L.append(Layer(vertical("果実の瑞々しさと、紅茶の余韻", F.jp(50), IVORY), 110, H * 0.30,
                   17.8, 20.3, reveal="tb", rise=0, reveal_dur=1.6))

    # 6. 想い
    L.append(Layer(block([
        ("text", "奈良の四季を、ひと皿に。", F.jp(58), IVORY, 8),
        ("space", 2),
        ("text", "La saison de Nara, dans une assiette", F.la(40, italic=True), GOLD, 2),
    ]), W / 2, H * 0.73, 20.9, 23.1, reveal_dur=1.4))

    # エンディング
    L.append(Layer(block([
        ("text", "ある日のディナーより", F.jp(36), GOLD, 10),
        ("space", 4),
        ("text", "フルーツマリネ", F.jp(62, medium=True), IVORY, 10),
        ("text", "倭紅茶 ・ ココナッツ", F.jp(62, medium=True), IVORY, 10),
    ], gap=16), W / 2, H * 0.17, 24.0, 30.2, reveal="tb", rise=20, reveal_dur=1.5))
    L.append(Layer(block([
        ("text", "その日限りのフレンチを", F.jp(42), IVORY, 12),
        ("space", 2),
        ("rule", 240, GOLD),
        ("space", 2),
        ("text", "L'Artisan KANOYA", F.la(84, italic=True), IVORY, 2),
        ("text", "奈良春日 鹿のや 内　|　ご予約 0742-22-6666", F.jp(28), GOLD, 3),
    ], gap=14), W / 2, H * 0.58, 25.9, 30.2, reveal="tb", rise=20, reveal_dur=1.6, scrim=0.0))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "video", "kanoya_fruit_marinade_reel.mp4"))
    ap.add_argument("--still", type=float, nargs="*", help="指定秒のフレームだけPNGで書き出す")
    args = ap.parse_args()

    F = Fonts(args.fonts)
    pics = load_shots()
    layers = build_layers(F)

    def frame(fi):
        t = fi / FPS
        a = base_frame(t, pics)
        # エンディング：写真をぼかして沈め、ロゴを浮かび上がらせる
        if t > 25.2:
            k = ease_io((t - 25.2) / 1.2)
            a = blur(a, 0.5 * k) * (1 - 0.5 * k)
        a = grade(a, fi)
        # 冒頭はタイトルが映えるよう暗めから
        if t < 3.6:
            a = a * (0.45 + 0.55 * ease_io((t - 2.2) / 1.4))
        for L in layers:
            L.apply(a, t)
        if t < 0.5:
            a = a * ease_io(t / 0.5)
        if t > DUR - 0.6:
            a = a * (1 - ease_io((t - (DUR - 0.6)) / 0.6))
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)

    if args.still:
        for s in args.still:
            Image.fromarray(frame(int(s * FPS))).save(f"{os.path.splitext(args.out)[0]}_{s:05.2f}.png")
        return

    n = int(DUR * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
           "-shortest", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-maxrate", "12M", "-bufsize", "24M",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "128k", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(n):
        proc.stdin.write(frame(fi).tobytes())
        if fi % 90 == 0:
            print(f"{fi}/{n}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("done", args.out)


if __name__ == "__main__":
    main()
