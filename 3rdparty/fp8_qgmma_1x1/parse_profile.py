import collections
import sys

colors = ["gray", "darkorange", "limegreen", "royalblue", "lightcoral", "bisque", "forestgreen", "blueviolet", "red", "gold", "cyan", "purple", "saddlebrown", "yellow", "slategray", "magenta", "peachpuff", "darkkhaki", "teal", "pink"]


profile = {}

filename = "profile.txt"
if len(sys.argv) > 1:
    filename = sys.argv[1]

for line in open(filename):
    if line.startswith("Tile"):
        tile_id = line.split(":")[0].split()[1]
        profile[tile_id] = {}
    else:
        toks = line.strip().split(": ")
        if (toks[0] != "sm_id"):
            timestamp = int(toks[1])
        profile[tile_id][toks[0]] = int(toks[1])

min_time = None
for tile in profile:
    if profile[tile]["sm_id"] == 0:
        start_time = profile[tile]["scheduler_fetch_start"]
        if min_time is None or start_time < min_time:
            min_time = start_time

for tile in profile:
    for key in profile[tile]:
        if key != "sm_id":
            assert(profile[tile][key] is not None)
            profile[tile][key] = profile[tile][key] - min_time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots()

max_time = 0

tile_offset = 0

totals = collections.defaultdict(float)

for tile_id in profile:
    entry = profile[tile_id]
    if entry["sm_id"] == 0:

        ax.add_patch(Rectangle((entry["scheduler_fetch_start"],tile_offset),entry["scheduler_fetch_complete"]-entry["scheduler_fetch_start"],0.4,facecolor="blue"))
        ax.add_patch(Rectangle((entry["dma_tile_wait_start"],tile_offset),entry["dma_tile_wait_complete"]-entry["dma_tile_wait_start"],0.4,facecolor="red"))
        ax.add_patch(Rectangle((entry["dma_tile_wait_complete"],tile_offset),entry["dma_loads_issued"]-entry["dma_tile_wait_complete"],0.4,facecolor="yellow"))
        ax.add_patch(Rectangle((entry["compute_tile_wait_start"],tile_offset+.25),entry["compute_tile_wait_complete"]-entry["compute_tile_wait_start"],0.4,facecolor="orange"))
        totals["tile_wait"] += entry["compute_tile_wait_complete"]-entry["compute_tile_wait_start"]
        ax.add_patch(Rectangle((entry["compute_tile_wait_complete"],tile_offset+.25),entry["compute_first_data_wait_complete"]-entry["compute_tile_wait_complete"],0.4,facecolor="green"))
        totals["first_data_wait"] += entry["compute_first_data_wait_complete"]-entry["compute_tile_wait_complete"]
        ax.add_patch(Rectangle((entry["compute_first_data_wait_complete"],tile_offset+.25),entry["epilogue_begin"]-entry["compute_first_data_wait_complete"],0.4,facecolor="pink"))
        totals["mainloop"] += entry["epilogue_begin"]-entry["compute_first_data_wait_complete"]
        ax.add_patch(Rectangle((entry["epilogue_begin"],tile_offset+.25),entry["epilogue_complete"]-entry["epilogue_begin"],0.4,facecolor="grey"))
        totals["epilogue"] += entry["epilogue_complete"]-entry["epilogue_begin"]

        if entry["epilogue_complete"] > max_time:
            max_time = entry["epilogue_complete"]
        tile_offset += 1

for k in ("tile_wait", "first_data_wait", "mainloop", "epilogue"):
    mean = float(totals[k])/tile_offset
    print("%s mean: %.02f"%(k,mean))

plt.xlim([0,max_time])
plt.ylim([0,tile_offset+1])
plt.show()

