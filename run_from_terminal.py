import main 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("TkAgg")  # Use an interactive backend for terminal execution
import imageio.v2 as imageio  # use v2 interface to avoid deprecation warnings
import os
import argparse

# Create parser
parser = argparse.ArgumentParser(description="Example script that takes a parameter.")
parser.add_argument('--param', type=int, default=-1, help='An integer parameter (default: -1)')
args = parser.parse_args()

main.init_simulation()

# Ensure key state exists
main.key_states[" "] = False

toggle = False
step = 0
image_files = []

saveFiles = False

os.makedirs("frames", exist_ok=True)

for cycle in range(3):
    toggle = not toggle
    main.key_states[" "] = toggle
    if toggle:
        main.key_states["ArrowRight"] = False
        main.key_states["ArrowLeft"] = True
    else:
        main.key_states["ArrowRight"] = True
        main.key_states["ArrowLeft"] = False

    for _ in range(10):
        step += 1
        main.update_frame()

        if saveFiles:
            # Plot the heatmap and save as image
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(main.T, cmap='RdBu', origin='lower')
            plt.colorbar(im, ax=ax, label='Temperature')
            ax.set_title(f"Step {step}")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            plt.tight_layout()
    
            filename = f"frames/heatmap_{args.param:02d}_{step:03d}.png"
            plt.savefig(filename)
            plt.close()
            image_files.append(filename)

if saveFiles:
    # Save to GIF
    with imageio.get_writer(f"temperature_evolution_{args.param:02d}.gif", mode='I', duration=0.1) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    print("✅ GIF saved as temperature_evolution.gif")

print(f"done: param: {args.param:02d}")