#!/usr/bin/env python3
"""Nationals meme pipeline (santeluca-style cutscene memes). Runs in the Higgsfield sandbox.

  meme.py prep     vo.mp3                 -> vo_f.wav, votrack.mp4 (black video for voice_change)
  meme.py align    vc.wav script.json     -> plan.json, seg_1..5.mp3, a6.wav, cap_*.png
  meme.py assemble plan.json              -> final.mp4 (needs v1..v5.mp4 lip-sync clips)

script.json: {"shots": [[chunk, ...] x5], "say_end": "compprep dot app", "url": "www.compprep.app"}
chunk = caption text, "|" = line break. Shots: 1 medium hook, 2 wide roasts, 3 close-up twist,
4 low-angle catchphrase, 5 wide "Get yours at". The end card is black with the URL.
"""
import json, os, re, subprocess, sys
from difflib import SequenceMatcher

import numpy as np

FPS = 30
PADS = [0.3, 0.35, 0.45, 0.6]          # pause after shots 1-4 (seconds)
FONT = 'MRound.ttf'                      # M PLUS Rounded 1c ExtraBold
FONT_URL = 'https://github.com/google/fonts/raw/main/ofl/mplusrounded1c/MPLUSRounded1c-ExtraBold.ttf'


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def dur(f):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', f],
                                capture_output=True, text=True).stdout)


def toks(s):
    return [t for t in re.sub(r"[^a-z0-9' ]", ' ', s.lower().replace('-', ' ')).replace("'", '').split() if t]


def prep(vo):
    # squeeze long pauses, speed to ~29 s of speech (the reference pace), black video for voice_change
    sh(f'ffmpeg -nostdin -loglevel error -y -i {vo} -af "silenceremove=stop_periods=-1:stop_duration=0.22:'
       f'stop_threshold=-40dB,apad=pad_dur=0.1" -ar 44100 vo_ns.wav')
    tempo = min(1.2, max(1.0, dur('vo_ns.wav') / 29.0))
    sh(f'ffmpeg -nostdin -loglevel error -y -i vo_ns.wav -af atempo={tempo:.3f} -ar 44100 vo_f.wav')
    sh('ffmpeg -nostdin -loglevel error -y -f lavfi -i color=c=black:s=640x480:r=30 -i vo_f.wav '
       '-map 0:v -map 1:a -c:v libx264 -c:a aac -b:a 192k -shortest votrack.mp4')
    print(json.dumps({'tempo': round(tempo, 3), 'dur': round(dur('vo_f.wav'), 2)}))


def words(wav):
    from faster_whisper import WhisperModel
    m = WhisperModel('small', compute_type='int8')
    segs, _ = m.transcribe(wav, word_timestamps=True)
    return [(w.start, w.end, w.word) for s in segs for w in s.words]


def rms_curve(wav):
    a = np.frombuffer(subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', wav, '-f', 's16le', '-ac', '1',
                                      '-ar', '16000', '-'], capture_output=True).stdout, dtype=np.int16).astype(float)
    return np.array([np.sqrt(np.mean(a[i:i + 320] ** 2)) for i in range(0, len(a) - 320, 160)])  # 10 ms hop


def quietest(r, a, b):
    i0, i1 = int(a * 100), max(int(a * 100) + 1, int(b * 100))
    return (i0 + int(np.argmin(r[i0:i1]))) / 100


def render_cap(txt, path, F):
    from PIL import Image, ImageDraw, ImageFilter
    W, H = 1440, 1080
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); sh_ = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(im), ImageDraw.Draw(sh_)
    lines = txt.split('|'); lh = 82; y0 = H * 0.885 - lh * len(lines)
    for i, l in enumerate(lines):
        w = d.textlength(l, font=F)
        assert w < 1300, f'caption line too wide: {l!r}'
        x, y = (W - w) / 2, y0 + i * lh
        for dx, dy in [(0, 0), (3, 4)]:
            ds.text((x + dx, y + dy), l, font=F, fill=(0, 0, 0, 235), stroke_width=5, stroke_fill=(0, 0, 0, 235))
        d.text((x, y), l, font=F, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(255, 255, 255, 255))
    Image.alpha_composite(sh_.filter(ImageFilter.GaussianBlur(7)), im).save(path)


def align(wav, script):
    from PIL import ImageFont
    S = json.load(open(script))
    shots = S['shots']; assert len(shots) == 5
    W = words(wav)
    wt = [toks(w[2]) for w in W]
    flat, owner = [], []                       # whisper tokens -> word index
    for i, t in enumerate(wt):
        for x in t: flat.append(x); owner.append(i)
    chunks = [(k, c) for k, sc in enumerate(shots) for c in sc]
    stoks, sown = [], []                       # script tokens -> chunk index
    for ci, (_, c) in enumerate(chunks):
        for x in toks(c.replace('|', ' ')): stoks.append(x); sown.append(ci)
    for x in toks(S.get('say_end', 'compprep dot app')): stoks.append(x); sown.append(len(chunks))
    mp = {}
    for a, b, n in SequenceMatcher(None, stoks, flat, autojunk=False).get_matching_blocks():
        for j in range(n): mp[a + j] = owner[b + j]
    # first/last whisper word of each chunk (interpolate unmatched)
    def span(ci):
        idx = [mp[i] for i in range(len(stoks)) if sown[i] == ci and i in mp]
        return (min(idx), max(idx)) if idx else None
    spans = [span(ci) for ci in range(len(chunks) + 1)]
    for ci, s in enumerate(spans):
        assert s is not None, f'could not align chunk {ci}: {chunks[ci][1] if ci < len(chunks) else "end"}'
    r = rms_curve(wav); total = dur(wav)
    # shot boundaries: quietest point between last word of shot k and first word of shot k+1
    first_chunk = [next(ci for ci, (k, _) in enumerate(chunks) if k == s) for s in range(5)] + [len(chunks)]
    cuts = [0.0]
    for s in range(1, 6):
        prev_end = W[spans[first_chunk[s] - 1][1]][1]; nxt = W[spans[first_chunk[s]][0]][0]
        a, b = (prev_end - 0.12, nxt + 0.12) if nxt - prev_end < 0.15 else (prev_end, nxt)
        cuts.append(quietest(r, max(a, cuts[-1] + 0.5), b))
    cuts.append(total)
    # audio segments: shots 1-4 + pause, shot 5 padded to 2.0 s for the lip-sync model, end card audio
    segs = []
    for s in range(5):
        a, b = cuts[s], cuts[s + 1]
        pad = f'apad=pad_dur={PADS[s]}' if s < 4 else 'apad=whole_dur=2.0'
        sh(f'ffmpeg -nostdin -loglevel error -y -ss {a} -to {b} -i {wav} -af "afade=t=in:d=0.02,areverse,'
           f'afade=t=in:d=0.04,areverse,{pad}" -ar 44100 -ac 1 -b:a 192k seg_{s + 1}.mp3')
        segs.append(dur(f'seg_{s + 1}.mp3'))
    sh(f'ffmpeg -nostdin -loglevel error -y -ss {cuts[5]} -i {wav} -ar 44100 -ac 1 a6.wav')
    # frame-exact shot lengths on the final timeline
    shot_len = [round(x * FPS) / FPS for x in segs[:4]] + [round((cuts[5] - cuts[4] + 0.2) * FPS) / FPS,
                                                           round((total - cuts[5] + 1.3) * FPS) / FPS]
    off = np.concatenate([[0], np.cumsum(shot_len)]).tolist()
    def fmap(t):
        for s in range(6):
            if cuts[s] <= t < cuts[s + 1]: return t - cuts[s] + off[s]
        return t
    if not os.path.exists(FONT): sh(f'curl -sSfL -o {FONT} {FONT_URL}')
    F = ImageFont.truetype(FONT, 66)
    tl = []
    for ci, (k, c) in enumerate(chunks):
        st = fmap(W[spans[ci][0]][0])
        if ci == first_chunk[k]: st = off[k]           # caption changes on the cut
        tl.append([st, None]); render_cap(c, f'cap_{ci}.png', F)
    for ci in range(len(tl)):
        tl[ci][1] = tl[ci + 1][0] if ci + 1 < len(tl) else off[5]
    plan = {'cuts': cuts, 'seg_dur': segs, 'shot_len': shot_len, 'offsets': off, 'captions': tl,
            'url': S.get('url', 'www.compprep.app'), 'chunks': [c for _, c in chunks]}
    json.dump(plan, open('plan.json', 'w'), indent=1)
    print(json.dumps({'seg_dur': [round(x, 2) for x in segs], 'wan_durations': [int(np.ceil(x)) for x in segs],
                      'total': round(off[-1], 2)}))


def assemble(planf):
    from PIL import Image, ImageDraw, ImageFont
    P = json.load(open(planf)); L = P['shot_len']
    for s in range(5):
        sh(f'ffmpeg -nostdin -loglevel error -y -i v{s + 1}.mp4 -t {L[s]:.3f} -vf "scale=1440:1080:flags=lanczos,'
           f'setsar=1,fps={FPS},format=yuv420p" -an -c:v libx264 -crf 15 -preset fast sh_{s + 1}.mp4')
        sh(f'ffmpeg -nostdin -loglevel error -y -i seg_{s + 1}.mp3 -af "apad,atrim=0:{L[s]:.3f}" -ar 44100 -ac 2 au_{s + 1}.wav')
    im = Image.new('RGB', (1440, 1080), (0, 0, 0)); d = ImageDraw.Draw(im); F = ImageFont.truetype(FONT, 92)
    w = d.textlength(P['url'], font=F); d.text(((1440 - w) / 2, 480), P['url'], font=F, fill=(255, 255, 255)); im.save('black.png')
    sh(f'ffmpeg -nostdin -loglevel error -y -loop 1 -i black.png -t {L[5]:.3f} -vf "fps={FPS},format=yuv420p" -c:v libx264 -crf 14 sh_6.mp4')
    sh(f'ffmpeg -nostdin -loglevel error -y -i a6.wav -af "apad,atrim=0:{L[5]:.3f}" -ar 44100 -ac 2 au_6.wav')
    ins = ' '.join(f'-i sh_{i}.mp4' for i in range(1, 7))
    sh(f'ffmpeg -nostdin -loglevel error -y {ins} -filter_complex "' + ''.join(f'[{i}:v]' for i in range(6)) +
       f'concat=n=6:v=1:a=0,fps={FPS},format=yuv420p" -c:v libx264 -crf 14 -preset fast vcat.mp4')
    caps = P['captions']; ins = ' '.join(f'-i cap_{k}.png' for k in range(len(caps))); fc = ''; last = '0:v'
    for k, (a, b) in enumerate(caps):
        fc += f"[{last}][{k + 1}:v]overlay=0:0:enable='between(t,{a:.3f},{b - 0.001:.3f})'[c{k}];"; last = f'c{k}'
    sh(f'ffmpeg -nostdin -loglevel error -y -i vcat.mp4 {ins} -filter_complex "{fc[:-1]}" -map [{last}] '
       f'-c:v libx264 -crf 16 -preset medium -pix_fmt yuv420p vcap.mp4')
    ins = ' '.join(f'-i au_{i}.wav' for i in range(1, 7))
    sh(f'ffmpeg -nostdin -loglevel error -y {ins} -filter_complex "concat=n=6:v=0:a=1,highpass=f=70,'
       f'acompressor=threshold=-20dB:ratio=3:attack=5:release=80,loudnorm=I=-14:TP=-1.5:LRA=11" -ar 44100 mix.wav')
    sh('ffmpeg -nostdin -loglevel error -y -i vcap.mp4 -i mix.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k '
       '-shortest -movflags +faststart final.mp4')
    print(json.dumps({'final': 'final.mp4', 'dur': round(dur('final.mp4'), 2)}))


if __name__ == '__main__':
    cmd = sys.argv[1]
    {'prep': lambda: prep(sys.argv[2]), 'align': lambda: align(sys.argv[2], sys.argv[3]),
     'assemble': lambda: assemble(sys.argv[2])}[cmd]()
