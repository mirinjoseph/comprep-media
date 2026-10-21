# Free lip-sync fallback (no credits): Wav2Lip on a still shot image -> lip-synced video of that shot.
# Setup in the Higgsfield sandbox (once):
#   git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git code
#   sed -i 's/librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,/' code/audio.py
#   curl -sSL -o code/checkpoints/wav2lip_gan.pth https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth
#   curl -sSfL -o face.xml https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml
# Usage: python3 w2l_still.py shot.png seg_16k.wav out.mp4 [x1,y1,x2,y2]
# Always check the printed face box on a frame: the detector can grab the neck or a floor reflection;
# then pass the box by hand (coordinates on the 1440x1080 image, forehead to chin).
import sys, os, subprocess, numpy as np, cv2, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'code'))
import audio
from models import Wav2Lip
img_path, wav_path, out_path = sys.argv[1:4]
BOX = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 else None
FPS = 25
img = cv2.resize(cv2.imread(img_path), (1440, 1080))
if BOX:
    x1, y1, x2, y2 = BOX
else:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fc = cv2.CascadeClassifier(os.path.join(HERE, 'face.xml'))
    d = fc.detectMultiScale(g, 1.05, 4, minSize=(30, 30))
    if not len(d): d = fc.detectMultiScale(g, 1.03, 2, minSize=(16, 16))
    assert len(d), 'no face found'
    x, y, w, h = max(d, key=lambda b: b[2] * b[3])
    y1, y2 = max(0, y), min(1080, y + int(h * 1.12)); x1, x2 = max(0, x), min(1440, x + w)   # whole face incl. chin
print('face box', x1, y1, x2, y2, flush=True)
wav = audio.load_wav(wav_path, 16000)
mel = audio.melspectrogram(wav)
n_frames = int(len(wav) / 16000 * FPS)
mult = 80. / FPS
chunks = [mel[:, -16:] if int(i * mult) + 16 > mel.shape[1] else mel[:, int(i * mult):int(i * mult) + 16] for i in range(n_frames)]
model = Wav2Lip()
ck = torch.load(os.path.join(HERE, 'code/checkpoints/wav2lip_gan.pth'), map_location='cpu', weights_only=False)
model.load_state_dict({k.replace('module.', ''): v for k, v in ck['state_dict'].items()}); model.eval()
face = cv2.resize(img[y1:y2, x1:x2], (96, 96))
fh, fw = y2 - y1, x2 - x1
m = np.zeros((fh, fw), np.float32)                     # feathered mask over the lower face only
m[int(fh * 0.45):int(fh * 0.98), int(fw * 0.12):int(fw * 0.88)] = 1
m = cv2.GaussianBlur(m, (0, 0), max(3, fw * 0.05))[..., None]
tmp = out_path + '.noaudio.mp4'
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (1440, 1080))
with torch.no_grad():
    for b in range(0, n_frames, 32):
        mb = np.array(chunks[b:b + 32])[:, None]
        fb = np.repeat(face[None], len(mb), 0).astype(np.float32)
        masked = fb.copy(); masked[:, 48:] = 0
        inp = np.concatenate([masked, fb], 3) / 255.
        pred = model(torch.FloatTensor(mb), torch.FloatTensor(np.transpose(inp, (0, 3, 1, 2)))).numpy()
        for p in (np.transpose(pred, (0, 2, 3, 1)) * 255.).clip(0, 255):
            pf = cv2.resize(p.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            fr = img.copy().astype(np.float32)
            fr[y1:y2, x1:x2] = fr[y1:y2, x1:x2] * (1 - m) + pf * m
            vw.write(fr.astype(np.uint8))
vw.release()
subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', tmp, '-i', wav_path, '-map', '0:v', '-map', '1:a',
                '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', out_path], check=True)
os.remove(tmp); print('done', out_path)
