import os

INPUT_VIDEO_FILE = 'orig.mp4'
INPUT_FRAMES_DIR = '# nosync/input/frames'
INPUT_AUDIO_FILE = '# nosync/input/audio.wav'

OUTPUT_VIDEO_FILE = '# nosync/output/out.mp4'
OUTPUT_FRAMES_DIR = '# nosync/output/frames'
OUTPUT_AUDIO_FILE = '# nosync/output/audio.wav'

CACHE_DIR = '# nosync/cache'

LKH_EXEC = 'LKH-3'


os.makedirs(INPUT_FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

