import os
from concurrent.futures import ProcessPoolExecutor
import subprocess

import numpy as np
from tqdm import tqdm

import config
from process import mp_init, mp_func
from render import harmonic


EXTRACT = False
CACHING = True
TEST = False
EXPORT_VIDEO = True

def main():
	if EXTRACT:
		subprocess.run([
			'ffmpeg',
			'-y',
			'-i', config.INPUT_VIDEO_FILE,
			os.path.join(config.INPUT_FRAMES_DIR, '%04d.png'),
			'-map', '0:a:0',
			config.INPUT_AUDIO_FILE,
		], capture_output=True)


	files = os.listdir(config.INPUT_FRAMES_DIR)

	# Compute square wave and its ifft matrix
	print('Computing ifftmat')
	npoints = 2**12
	sqw = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / npoints  # divide by `npoints` (fft formula quirk)
	sqw = sqw[np.linspace(0, 4, npoints, endpoint=False).astype(int)]
	ifftmat = np.linalg.inv(np.stack([harmonic(sqw, k) for k in range(npoints)]))
	print('Done ifftmat')

	if TEST is True:
		mp_init(sqw, ifftmat, files[0], CACHING, show=True)
		mp_func(48, False, files[5000-1])

	elif TEST is False:
		filenames = files
		frames = len(filenames)

		nfreqs = np.full(frames, 96)
		nfreqs[0:100] = npoints - 1
		nfreqs[100:400] = np.round((npoints - 1) ** (np.linspace(1, 0, 300) ** 3))
		nfreqs[400:430] = 1
		nfreqs[430:750] = np.round((np.geomspace(1, 96, 320)))

		show_freqs = np.full(frames, False)
		show_freqs[40:850] = True

		with ProcessPoolExecutor(max_workers=os.cpu_count() - 1,
		                         initializer=mp_init,
		                         initargs=(sqw, ifftmat, files[0], CACHING, False, 1080)) as pool:
			for _ in tqdm(pool.map(mp_func, nfreqs, show_freqs, filenames), total=frames):
				pass

		if EXPORT_VIDEO:
			subprocess.run([
				'ffmpeg',
				'-y',
				'-framerate', '30',
				'-i', os.path.join(config.OUTPUT_FRAMES_DIR, '%04d.png'),
				'-i', config.OUTPUT_AUDIO_FILE,
				config.OUTPUT_VIDEO_FILE
			], capture_output=False)


if __name__ == '__main__':
	main()