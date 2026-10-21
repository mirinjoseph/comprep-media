# Slow push-in + slight camera sway on each still lip-sync shot (w1..w5.mp4 -> v1..v5.mp4), so the frame is never frozen
for i in 1 2 3 4 5; do
  case $i in 3) Z=0.0010;; 4) Z=0.0012;; *) Z=0.0007;; esac
  ffmpeg -nostdin -loglevel error -y -i w$i.mp4 -vf "scale=1728:1296,zoompan=z='1+$Z*on':x='iw/2-(iw/zoom/2)+10*sin(on/23)':y='ih/2-(ih/zoom/2)+6*sin(on/31)':d=1:s=1440x1080:fps=25,format=yuv420p" -c:v libx264 -crf 16 -preset fast -c:a copy v$i.mp4
done
