from render import Renderer


renderer = Renderer()

def mp_init(filename, caching, show=False):
	renderer.init(filename, caching, show)

def mp_func(npoints, nfreqs, show_freqs, filename):
	renderer.render(npoints, nfreqs, show_freqs, filename)