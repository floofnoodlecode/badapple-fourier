from render import Renderer


renderer = Renderer()

def mp_init(*args, **kwargs):
	renderer.init(*args, **kwargs)

def mp_func(*args, **kwargs):
	renderer.render(*args, **kwargs)