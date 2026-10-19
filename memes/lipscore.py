# Lip-sync check: correlation between mouth opening and speech loudness, per lip-sync clip.
# Needs face.xml next to this file:
#   curl -sSfL -o face.xml https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml
# Run with PYTHONPATH=/pkg/ (the sandbox's OpenCV). Usage: lipscore.py v1.mp4 v3.mp4 ...
import sys, subprocess, numpy as np, cv2


def score(video, audio=None):
    fc = cv2.CascadeClassifier(__file__.rsplit('/', 1)[0] + '/face.xml')
    cap = cv2.VideoCapture(video); fps = cap.get(5) or 30; op = []; box = None
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if box is None or len(op) % 15 == 0:
            d = fc.detectMultiScale(g, 1.1, 5, minSize=(40, 40))
            if len(d): box = max(d, key=lambda b: b[2] * b[3])
        if box is None: op.append(np.nan); continue
        x, y, w, h = box
        roi = g[y + int(h * 0.68): y + int(h * 0.95), x + int(w * 0.3): x + int(w * 0.7)].astype(float)
        op.append(float((roi < np.percentile(g[y:y + h, x:x + w], 12)).mean()))   # dark mouth-cavity share
    op = np.array(op); seen = np.mean(~np.isnan(op))
    a = np.frombuffer(subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', audio or video, '-f', 's16le',
                                      '-ac', '1', '-ar', '16000', '-'], capture_output=True).stdout, dtype=np.int16).astype(float)
    hop = int(16000 / fps); env = np.array([np.sqrt(np.mean(a[i:i + hop] ** 2)) for i in range(0, len(a) - hop, hop)])
    k = min(len(env), len(op)); env, op = env[:k], op[:k]
    m = ~np.isnan(op)
    if m.sum() < 10: return dict(face=round(float(seen), 2), corr=None)
    env = np.convolve(env, np.ones(3) / 3, 'same')
    best = max((np.corrcoef(np.roll(op[m], L), env[m])[0, 1], L) for L in range(-12, 13))
    sp = env > env.max() * 0.12
    pause = float(np.nanmean(op[~sp & m])) / (float(np.nanmean(op[sp & m])) + 1e-6) if (~sp & m).any() else None
    return dict(face=round(float(seen), 2), corr=round(float(best[0]), 3), lag_frames=int(best[1]),
                pause_vs_speech_open=None if pause is None else round(pause, 2))


if __name__ == '__main__':
    for v in sys.argv[1:]:
        print(v, score(v), flush=True)
