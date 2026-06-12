# -*- coding: utf-8 -*-
from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
devices = graph.get_input_devices()

print("Camaras DirectShow encontradas:")

for i, name in enumerate(devices):
    print(f"{i}: {name}")
