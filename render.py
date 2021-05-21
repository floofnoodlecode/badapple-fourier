import os
import pickle
import subprocess
import tempfile

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from skimage.io import imread
from skimage.measure import find_contours

import config


class Renderer:
	def __init__(self):
		self.artists = []
		self.fig, self.ax = plt.subplots()
		self.caching = False
		self.show = False
		self.shape = None, None

	def init(self, filename, caching, show):
		self.caching = caching
		self.show = show

		frame = imread(os.path.join(config.INPUT_FRAMES_DIR, filename), as_gray=True)
		self.shape = height, width = frame.shape

		self.fig.set_tight_layout({'pad': 0})
		self.fig.set_size_inches(width * 1080 / height / 100, 1080 / 100)
		self.fig.set_facecolor('black')
		self.ax.axis('off')
		self.ax.set_xlim(0, width)
		self.ax.set_ylim(height, 0)

	def render(self, npoints, nfreqs, show_freqs, filename):
		npoints = int(npoints)
		nfreqs = int(nfreqs)

		# Read input file
		rootname = os.path.splitext(filename)[0]
		frame = imread(os.path.join(config.INPUT_FRAMES_DIR, filename), as_gray=True)

		# Get contours and convert to complex numbers
		zs = [c[:, 1] + 1j * c[:, 0] for c in find_contours(frame, 0.5)]

		if zs:
			# Compute timestamps for each point on the contours
			ts = [np.cumsum(np.abs(np.diff(z, prepend=z[0]))) for z in zs]

			# Compute canonical merged contour
			JUMP_TIME = 10 ** -5
			offsets = np.cumsum([0] + [t[-1] for t in ts[:-1]]) + np.arange(len(ts)) * JUMP_TIME
			t_canon = np.concatenate([t + o for t, o in zip(ts, offsets)])
			z_canon = np.concatenate(zs)

			# Convert to TSP nodes and edges
			REDUCTION = 5
			MINPOINTS = 3  # >= 2
			node_duals = []
			node_t_canon = []
			node_z_canon = []
			for cnt, (t, z) in enumerate(zip(ts, zs)):
				tn = np.linspace(0, t[-1], max(round(t[-1] / REDUCTION), MINPOINTS))
				zn = np.interp(tn, t, z)

				duals = np.arange((len(tn) - 1) * 2) + len(node_duals)
				duals[1:-1:2] += 1
				duals[2::2] -= 1
				if np.isclose(zn[0], zn[-1]):
					duals[0], duals[-1] = duals[-1], duals[0]
				else:
					duals[0] = duals[-1] = -1
				node_duals.extend(duals)

				tn += offsets[cnt]
				tn = np.stack([tn[:-1], tn[1:]], axis=1).reshape(-1)
				node_t_canon.extend(tn)

				zn = np.stack([zn[:-1], zn[1:]], axis=1).reshape(-1)
				node_z_canon.extend(zn)

			# Get TSP solution (from cache or recompute)
			fixed_edges = np.arange(len(node_t_canon)).reshape(-1, 2)
			cachefile = os.path.join(config.CACHE_DIR, f'{rootname}_lkh.pkl')
			if self.caching and os.path.exists(cachefile):
				with open(cachefile, 'rb') as f:
					nodes_tour = pickle.load(f)
			else:
				nodes_tour = self._run_LKH(node_z_canon, fixed_edges)
				if self.caching is not None:
					with open(cachefile, 'wb') as f:
						pickle.dump(nodes_tour, f)
			assert len(nodes_tour) % 2 == 0

			# Construct interpolation from new contour time to old contour time
			jumps = []
			t_t_tsp = [0] * 2
			t_z_tsp = [node_t_canon[nodes_tour[0]]] * 2
			for i in range(0, len(nodes_tour), 2):
				n12, n21 = nodes_tour[i : i + 2]
				t1, t2 = node_t_canon[n12], node_t_canon[n21]
				if t1 == t_z_tsp[-1]:
					t_t_tsp[-1] += abs(t2 - t1)
					t_z_tsp[-1] = t2
				else:
					newt0 = t_t_tsp[-1] + JUMP_TIME
					newt1 = newt0 + abs(t2 - t1)
					jumps.append((t_t_tsp[-1], newt0))
					t_t_tsp.extend([newt0, newt1])
					t_z_tsp.extend([t1, t2])
			jumps = np.array(jumps)

			# Create merged tour contour
			start_z, end_z = np.interp(np.interp([0, t_t_tsp[-1]], t_t_tsp, t_z_tsp), t_canon, z_canon)
			endpoint = not np.isclose(start_z, end_z)  # Exclude end point if it equals start point

			t_t_tour = np.linspace(0, t_t_tsp[-1], npoints, endpoint=endpoint)
			if len(jumps):
				jump_idxs = t_t_tour.searchsorted(jumps)
				jump_idxs[:,0] -= 1
				jump_idxs = jump_idxs.ravel()
				cond = jump_idxs < len(t_t_tour)
				jump_idxs = jump_idxs[cond]
				jumps = jumps.ravel()[cond]
				t_t_tour[jump_idxs] = jumps  # Make sure that segments end and start on a jump
			t_tour = np.interp(t_t_tour, t_t_tsp, t_z_tsp)
			z_tour = np.interp(t_tour, t_canon, z_canon)

			# Cut frequencies
			coefs = np.fft.fft(z_tour)
			arg_coefs = np.argsort(np.abs(coefs))
			coefs[arg_coefs[:len(arg_coefs) - 1 - nfreqs]] = 0
			z_fft = np.fft.ifft(coefs)

			# Make closed contour for plotting
			z_fft = np.append(z_fft, z_fft[0])

			# Plot
			# art, = self.ax.plot(z_fft.real, z_fft.imag, color='w', marker='.')
			# self.artists.append(art)

			dists = np.abs(np.diff(z_fft))
			colors = np.ones((len(dists), 4))
			colors[:, 3] = np.minimum(np.exp(-dists + (t_t_tour[-1] / (npoints - 1) * 2)), 1)

			line = np.stack([z_fft.real, z_fft.imag], axis=-1)
			lc = LineCollection(np.stack([line[:-1], line[1:]], axis=1), colors=colors)
			self.ax.add_collection(lc)
			self.artists.append(lc)

		if show_freqs:
			art = self.ax.text(.5, 0.02, f'Frequencies: {nfreqs :>4}', color='w', fontsize=40, ha='center', transform=self.ax.transAxes)
			self.artists.append(art)

		if self.show:
			self.ax.set_aspect('equal', 'datalim')
			plt.show()
		else:
			self.fig.savefig(os.path.join(config.OUTPUT_FRAMES_DIR, filename))

		[x.remove() for x in self.artists]
		self.artists = []

	def _run_LKH(self, node_coords, fixed_edges):
		if len(node_coords) < 3:
			return list(range(len(node_coords)))

		problem_file = tempfile.NamedTemporaryFile('w', delete=False)
		solution_file = tempfile.NamedTemporaryFile('w', delete=False)
		parameter_file = tempfile.NamedTemporaryFile('w', delete=False)

		try:
			# Write input files
			problem_file.write('\n'.join([
				'TYPE:TSP',
				f'DIMENSION:{len(node_coords)}',
				'EDGE_WEIGHT_TYPE:EUC_2D',
				'NODE_COORD_SECTION:',
				*[f'{i + 1} {round(z.real * 1000)} {round(z.imag * 1000)}' for i, z in enumerate(node_coords)],
				'FIXED_EDGES_SECTION:',
				*[f'{n1 + 1} {n2 + 1}' for n1, n2 in fixed_edges],
				'-1',
			]))
			parameter_file.write('\n'.join([
				f'PROBLEM_FILE={problem_file.name}',
				f'TOUR_FILE={solution_file.name}',
				'TRACE_LEVEL=0',
				'CANDIDATE_SET_TYPE=DELAUNAY',
			]))

			# Close files
			problem_file.close()
			solution_file.close()
			parameter_file.close()

			# Run LKH process
			subprocess.run([config.LKH_EXEC, parameter_file.name], input='\n', text=True, capture_output=True)

			# Get output
			with open(solution_file.name) as f:
				text = f.read()
			text = text.split('TOUR_SECTION\n', maxsplit=1)[1]
			text = text.split('\n-1', maxsplit=1)[0]

			nodes_tour = [int(n) - 1 for n in text.splitlines()]
		finally:
			os.remove(problem_file.name)
			os.remove(solution_file.name)
			os.remove(parameter_file.name)

		return nodes_tour

