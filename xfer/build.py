#!/usr/bin/env python3
"""Assemble a nationals meme from per-shot clips (Veo route).

  build.py plan.json out.mp4

plan.json: {"font": "FreeSansBold.ttf", "logo": "logo-white.png", "url": "www.compprep.app",
            "shots": [{"video": "c1.mp4", "vs": 0.0, "audio": "c1.mp4", "as": 0.0, "dur": 7.3,
                       "caps": [["DECA kids will wear|a three-piece suit", 0.0], ...]}, ...]}
vs/as = start time inside the video/audio file; caps = [text, start time inside the shot], "|" = line break.
Each shot: centre 4:3 crop -> 1440x1080, light low-bitrate softening (santeluca capture look), 30 fps.
Then a silent black end card with the logo and URL (2.2 s), captions on top, loudness -14 LUFS.
"""
import json, subprocess, sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1440, 1080, 30


def sh(c):
    subprocess.run(c, shell=True, check=True)


def cap_png(txt, path, F):
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); shd = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(im), ImageDraw.Draw(shd)
    lines = txt.split('|'); lh = 74; y0 = H * 0.90 - lh * len(lines)
    for i, l in enumerate(lines):
        w = d.textlength(l, font=F); assert w < 1340, f'caption too wide: {l!r}'
        x, y = (W - w) / 2, y0 + i * lh
        ds.text((x + 2, y + 3), l, font=F, fill=(0, 0, 0, 230), stroke_width=4, stroke_fill=(0, 0, 0, 230))
        d.text((x, y), l, font=F, fill=(255, 255, 255, 255))
    Image.alpha_composite(shd.filter(ImageFilter.GaussianBlur(4)), im).save(path)


def end_card(P, path):
    im = Image.new('RGB', (W, H), (0, 0, 0)); d = ImageDraw.Draw(im)
    logo = Image.open(P['logo']).convert('RGBA'); lw = 620; logo = logo.resize((lw, round(logo.height * lw / logo.width)), Image.LANCZOS)
    im.paste(logo, ((W - lw) // 2, 380), logo)
    F = ImageFont.truetype(P['font'], 64); w = d.textlength(P['url'], font=F)
    d.text(((W - w) / 2, 380 + logo.height + 70), P['url'], font=F, fill=(255, 255, 255)); im.save(path)


def main(planf, out):
    P = json.load(open(planf)); F = ImageFont.truetype(P['font'], 62)
    S = P['shots']; caps = []; t0 = 0.0
    for i, s in enumerate(S):
        d = round(s['dur'] * FPS) / FPS
        sh(f"ffmpeg -nostdin -loglevel error -y -ss {s['vs']} -i {s['video']} -t {d:.3f} -vf \"tpad=stop_mode=clone:stop_duration=2,trim=0:{d:.3f},setpts=PTS-STARTPTS,"
           f"crop=ih*4/3:ih,scale=720:540:flags=bilinear,scale={W}:{H}:flags=bicubic,gblur=sigma=0.5,eq=saturation=0.95,fps={FPS},format=yuv420p\" "
           f"-an -c:v libx264 -crf 24 -preset fast sh_{i}.mp4")
        sh(f"ffmpeg -nostdin -loglevel error -y -ss {s['as']} -i {s['audio']} -t {d:.3f} -vn -af \"afade=t=in:d=0.03,apad,atrim=0:{d:.3f},"
           f"afade=t=out:st={max(0, d - 0.06):.3f}:d=0.06\" -ar 44100 -ac 2 au_{i}.wav")
        for txt, t in s['caps']:
            caps.append([txt, t0 + t]);
        t0 += d
    n = len(S); end = 2.2
    end_card(P, 'end.png')
    sh(f"ffmpeg -nostdin -loglevel error -y -loop 1 -i end.png -t {end} -vf fps={FPS},format=yuv420p -c:v libx264 -crf 16 sh_{n}.mp4")
    sh(f"ffmpeg -nostdin -loglevel error -y -f lavfi -i anullsrc=r=44100:cl=stereo -t {end} au_{n}.wav")
    ins = ' '.join(f'-i sh_{i}.mp4' for i in range(n + 1))
    sh(f"ffmpeg -nostdin -loglevel error -y {ins} -filter_complex \"{''.join(f'[{i}:v]' for i in range(n + 1))}concat=n={n + 1}:v=1:a=0\" -c:v libx264 -crf 16 -preset fast vcat.mp4")
    ends = [c[1] for c in caps[1:]] + [t0]
    for k, (txt, _) in enumerate(caps): cap_png(txt, f'cap_{k}.png', F)
    ins = ' '.join(f'-i cap_{k}.png' for k in range(len(caps))); fc = ''; last = '0:v'
    for k, ((_, a), b) in enumerate(zip(caps, ends)):
        fc += f"[{last}][{k + 1}:v]overlay=0:0:enable='between(t,{a:.3f},{b - 0.001:.3f})'[c{k}];"; last = f'c{k}'
    sh(f"ffmpeg -nostdin -loglevel error -y -i vcat.mp4 {ins} -filter_complex \"{fc[:-1]}\" -map [{last}] -c:v libx264 -crf 17 -preset medium -pix_fmt yuv420p vcap.mp4")
    ins = ' '.join(f'-i au_{i}.wav' for i in range(n + 1))
    sh(f"ffmpeg -nostdin -loglevel error -y {ins} -filter_complex \"concat=n={n + 1}:v=0:a=1,highpass=f=70,"
       f"acompressor=threshold=-20dB:ratio=3:attack=5:release=80,loudnorm=I=-14:TP=-1.5:LRA=11\" -ar 44100 mix.wav")
    sh(f"ffmpeg -nostdin -loglevel error -y -i vcap.mp4 -i mix.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart {out}")
    print(json.dumps({'out': out, 'dur': round(t0 + end, 2), 'captions': [[c[0], round(c[1], 2)] for c in caps]}))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
