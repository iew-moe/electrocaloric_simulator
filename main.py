copyright_version = "© Stefan Mönch, v1.8c, CC BY-NC 4.0"

import numpy as np
import matplotlib
#matplotlib.use("Agg")  # Use non-interactive backend suitable for Pyodide
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke 
import io
import base64
import sys

 

try:
    from js import document, console, window, Plotly, Blob, URL
    from pyodide.ffi import create_proxy, to_js
    is_pyodide = True    
except ImportError: # Monkey-patch not used commands to do nothing and return nothing
    is_pyodide = False
    import matplotlib.pyplot as plt
    import logging
    logging.basicConfig(level=logging.INFO)

    class MockConsole:
        @staticmethod
        def log(message):
            print(message)  # Redirect to terminal
    console = MockConsole()

    class MockPlotly:
        @staticmethod
        def react(*args, **kwargs):
            pass  # Do nothing

        @staticmethod
        def newPlot(*args, **kwargs):
            pass  # Do nothing

        @staticmethod
        def update(*args, **kwargs):
            pass  # Do nothing

    Plotly = MockPlotly()

    class MockDocument:
        @staticmethod
        def getElementById(*args, **kwargs):
            return None  # Return None for all calls

        @staticmethod
        def createElement(*args, **kwargs):
            return None  # Return None for all calls

        @staticmethod
        def addEventListener(*args, **kwargs):
            pass  # Do nothing

    document = MockDocument()

    def to_js(value):
        return value  # Return the input value unchanged

from scipy.ndimage import binary_dilation
from scipy.ndimage import zoom
import cProfile
import pstats
import time
from math import ceil, sqrt, isnan
from scipy.ndimage import map_coordinates
import pickle


default_config = {
    # Simulation
    "nx": 360, "ny": 29, "dx": 1.0, "dy": 1.0, "dt": 0.1,
    "mu": 0.05, "rho": 1.0,
    
    # Material
    "c_p_fluid": 1.0, "c_p_solid": 1.0,
    "k_fluid": 0.2, "k_solid": 0.1,
    "n_diff": 50, "n_conv": 50,
    "dTad": 2.0,

    # Plates / Geometry
    "num_plates": 4,
    "plate_height": 4,
    "plate_spacing": 3
}

# old config until V1.5
# default_config = {
#     # Simulation
#     "nx": 360, "ny": 25, "dx": 1.0, "dy": 1.0, "dt": 0.1,
#     "mu": 0.05, "rho": 1.0,
    
#     # Material
#     "c_p_fluid": 1.0, "c_p_solid": 1.0,
#     "k_fluid": 0.5, "k_solid": 0.1,
#     "n_diff": 50, "n_conv": 50,
#     "dTad": 2.0,

#     # Plates / Geometry
#     "num_plates": 2,
#     "plate_height": 8,
#     "plate_spacing": 9
# }
 
#from wgpy_backends import get_backend_name
#from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel as WGPUElementwiseKernel
#from wgpy_backends.webgl.elementwise_kernel import ElementwiseKernel as WebGLElementwiseKernel

#backend = get_backend_name()
#_interpolation_kernel = None

# === Pressure Solver ===
def apply_pressure_boundary_conditions(p):
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    p[:, 0] = p[:, 1]
    p[:, -1] = 0
    return p

def pressure_poisson(p, u, v):
    b = (1 / dt) * ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                    (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))
    for _ in range(200):
        p_old = p.copy()
        p[1:-1, 1:-1] = (
            (p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 +
            (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2 -
            b * dx**2 * dy**2 ) / (2 * (dx**2 + dy**2))
        p = apply_pressure_boundary_conditions(p)
    return p

 

def calc_directional_block_mask(flowDir):
    # direction: +1 for right, -1 for left 
    if flowDir == +1:
        # Rightward flow: block left valves (-1)
        return (obstacle == 1) | (valve_mask == -1) | (iso_mask == 1)
    elif flowDir == -1:
        # Leftward flow: block right valves (+1)
        return (obstacle == 1) | (valve_mask == 1) | (iso_mask == 1)
    else:
        # Default: no valve blocking
        return (obstacle == 1)

# === Flow Solver ===
def solve_flow(u_in, v_in, p_in, skipSolve, flowDir):
 

    directional_block_mask = calc_directional_block_mask(flowDir)

    steps = 320

    if not skipSolve:
        u_in = np.zeros((ny, nx))
        v_in = np.zeros((ny, nx))
        p_in = np.zeros((ny, nx))

        for _ in range(steps):
            un, vn = u_in.copy(), v_in.copy()
            u_star = un.copy()
            v_star = vn.copy()

            u_star[1:-1,1:-1] += dt * mu * (
                (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2]) / dx**2 +
                (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1]) / dy**2)

            v_star[1:-1,1:-1] += dt * mu * (
                (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2]) / dx**2 +
                (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1]) / dy**2)

            u_star[directional_block_mask] = 0
            v_star[directional_block_mask] = 0
            p_in = pressure_poisson(p_in, u_star, v_star)

            u_in[1:-1,1:-1] = u_star[1:-1,1:-1] - dt * (p_in[1:-1,2:] - p_in[1:-1,:-2]) / (2*dx*rho)
            v_in[1:-1,1:-1] = v_star[1:-1,1:-1] - dt * (p_in[2:,1:-1] - p_in[:-2,1:-1]) / (2*dy*rho)

            u_in[:, 0] = flowDir
            u_in[:, -1] = flowDir
            v_in[0, :] = 0
            v_in[-1, :] = 0

            u_in[directional_block_mask] = 0
            v_in[directional_block_mask] = 0

            wall_mask = binary_dilation(directional_block_mask) & (~directional_block_mask)
            u_in[wall_mask] = 0
            v_in[wall_mask] = 0
    
    # Precompute streamlines after flow field update
    try:
        if 'stream' in globals():
            stream.lines.remove()
            stream.arrows.remove()
    except Exception as e:
        console.log(f"Streamline removal error: {e}")

    fig_tmp, ax_tmp = plt.subplots(figsize=fig_size, dpi=fig_stream_dpi)
    fig_tmp.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax_tmp.set_axis_off()
    #ax_tmp.set_xlim(0, nx)
    ax_tmp.set_xlim(xlim_min, xlim_max)
    ax_tmp.set_ylim(0, ny)
    #ax_tmp.invert_yaxis()

    if flowDir == 1:
        start_points = np.array([[3, y] for y in range(5, ny-5, 4)]) #ok
        #start_points = [[3,ny//2]]
        u_vals = np.array([abs(u_in[y, 3]) for _, y in start_points])
#         start_points = np.array([[piston_end, y] for y in range((ny//2)-3+7+3, ny-3, 1)])
#         #start_points = np.array([[x_inlet_idx, y] for y in range(0, ny//2, 2)])         # override
    else:
        start_points = np.array([[nx-4, y] for y in range(5, ny-4, 4)]) #ok
        #start_points = [[nx-4,ny//2-1]]
        u_vals = np.array([abs(u_in[y, nx-4]) for _, y in start_points])
#         start_points = np.array([[piston_start, y] for y in range(3, (ny//2)+3-7-2, 1)])
#         #start_points = np.array([[x_outlet_idx, y] for y in range(ny//2, ny-1, 2)])         # override
    

    # select start points
    mid_x = nx // 2
    #start_points = np.array([[mid_x, y] for y in range(ny) if obstacle[y, mid_x] == 0]) 
    # Evaluate abs(u_in) at each start point
    #u_vals = np.array([abs(u_in[y, mid_x]) for _, y in start_points])
    # Find max abs value among these points
    max_u = np.max(u_vals)
    # Use only those with abs(u_in) >= 10% of max
    threshold = 0.1 * max_u
    #filtered_start_points = np.array([pt for pt, uval in zip(start_points, u_vals) if uval >= threshold])
    #start_points = filtered_start_points
    #start_points = filtered_start_points[::2]

    stream = ax_tmp.streamplot(X, Y, u_in, v_in, color='white', linewidth=0.1,
                                start_points=start_points,
                                integration_direction='forward', density=2, broken_streamlines=False, arrowsize=0)
 
    # individual linewidth?? (slow...)
    is_indivStreamWidth = False
    
    streamline_plotly = []
    if is_indivStreamWidth:

        # Compute velocity magnitude across the whole field
        speed = np.sqrt(u_in**2 + v_in**2)

        # Global min and max for consistent mapping
        #vmin, vmax = np.min(speed), np.max(speed)
        vmin, vmax = 0, np.max(speed)

        # Visual width range
        min_width, max_width = 0.1, 3

        def map_speed_to_width(v):
            return min_width + (v - vmin) / (vmax - vmin) * (max_width - min_width)

        for line in stream.lines.get_paths():
            verts = line.vertices
            x_vals = verts[:, 0]
            y_vals = verts[:, 1]

            # Crop to visible region
            mask = (x_vals >= xlim_min) & (x_vals <= xlim_max -1 )
            x_crop = x_vals[mask]
            y_crop = y_vals[mask]

            if len(x_crop) < 3:
                continue  # Skip too-short or empty segments

            # Loop over line segments (pairs of points)
            for i in range(len(x_crop) - 1):
                x0, y0 = x_crop[i], y_crop[i]
                x1, y1 = x_crop[i + 1], y_crop[i + 1]

                # Midpoint of segment
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * (y0 + y1)

                # Convert to index in velocity grid
                xi = int(np.clip((xm - X.min()) / (X.max() - X.min()) * (X.shape[1] - 1), 0, X.shape[1] - 1))
                yi = int(np.clip((ym - Y.min()) / (Y.max() - Y.min()) * (Y.shape[0] - 1), 0, Y.shape[0] - 1))

                # Get local speed and map to width
                local_speed = speed[yi, xi]
                line_width = map_speed_to_width(local_speed)

                # Append each segment as a separate Plotly trace
                streamline_plotly.append({
                    'x': [x0, x1],
                    'y': [y0, y1],
                    'mode': 'lines',
                    'line': {
                        'color': 'white',
                        'width': line_width
                    },
                    'type': 'scatter',
                    'hoverinfo': 'skip',
                    'showlegend': False
                })
    else:
        for line in stream.lines.get_paths():
            verts = line.vertices
            x_vals = verts[:, 0]
            y_vals = verts[:, 1]

            # Crop to visible region
            mask = (x_vals >= xlim_min + 2) & (x_vals <= xlim_max -2)
            x_crop = x_vals[mask]
            mask = (y_vals >= 2) & (y_vals <= ny -2)
            y_crop = y_vals[mask]

            if len(x_crop) < 2:
                continue  # Skip too-short or empty segments

            streamline_plotly.append({
                'x': x_crop.tolist(),
                'y': y_crop.tolist(),
                'mode': 'lines',
                'line': {'color': 'white', 'width': 0.2},
                'type': 'scatter',
                'hoverinfo': 'skip',
                'showlegend': False
            })

    return u_in, v_in, p_in, streamline_plotly



# === Modify Grid at Click Location ===
def modify_obstacle_at_click(x, y, button=1):
    radius = 1.7
    for dy in range(-ceil(radius)-2, ceil(radius) + 1+2):
        for dx in range(-ceil(radius)-2, ceil(radius) + 1+2):
            px, py = x + dx, y + dy
            if sqrt((dx/3)**2 + dy**2) <= radius:            
                if 0 <= px < nx and 0 <= py < ny:
                    if button == 1:
                        obstacle[py, px] = 1
                        fluid_mask[py, px] = 0
                        u[py, px] = 0
                        v[py, px] = 0
                        T[py, px] = 0
                        console.log(f"🧱 Obstacle set at ({px}, {py})")
                    elif button == 2:
                        obstacle[py, px] = 0
                        fluid_mask[py, px] = 1
                        console.log(f"🧼 Obstacle removed at ({px}, {py})")

# === Coordinate Mapping ===
def get_sim_grid_coords(event):
    rect = document.getElementById("heatmap-plot").getBoundingClientRect()
    sim_width = xlim_max - xlim_min
    sim_height = ny

    rel_x = event.clientX - rect.left
    rel_y = event.clientY - rect.top

    grid_x = int(rel_x * sim_width / rect.width) + xlim_min
    grid_y = ny - 1 - int(rel_y * sim_height / rect.height)
    grid_x = min(max(grid_x, xlim_min), xlim_max - 1) # clamp to range, to avoid out of bound at borders
    grid_y = min(max(grid_y, 0), ny - 1)
    return grid_x, grid_y

def is_mouse_in_heatmap(event):
    """Check if the mouse event occurred within the heatmap-plot div."""
    rect = document.getElementById("heatmap-plot").getBoundingClientRect()
    x = event.clientX
    y = event.clientY
    return rect.left <= x <= rect.right and rect.top <= y <= rect.bottom

#=== Handlers ===
def on_mouse_down(event): 
    if isBuilderMode:
        if is_mouse_in_heatmap(event):
            modify_obstacle_at_click(*get_sim_grid_coords(event), event.buttons)

def on_mouse_move(event):
  #  console.log(f"Mouse moved ({grid_x}, {grid_y}), skip ({not mouse_is_down or event.buttons == 0})")
    global mouseX, mouseY
    
    if is_mouse_in_heatmap(event):
        mouseX, mouseY  = get_sim_grid_coords(event)
    else:
        mouseX = None
        mouseY = None
 
    if event.buttons == 0:
        return

    if isBuilderMode:
        if is_mouse_in_heatmap(event):
            modify_obstacle_at_click(*get_sim_grid_coords(event), event.buttons)


def on_mouse_up(event):
    return
    #global builder_mode_left
    #if isBuilderMode:
    #    if is_mouse_in_heatmap(event):
    #        global mouse_is_down, u, v, p, u2, v2, p2, streamline_plotly, streamline_plotly2
    #            
    #        u[:], v[:], p[:], streamline_plotly = solve_flow(u, v, p, False, 1)
    #        u2[:], v2[:], p2[:], streamline_plotly2 = solve_flow(u2, v2, p2, False, -1)



def on_key_down(event):

    global mouse_is_down, u, v, p, u2, v2, p2, streamline_plotly, streamline_plotly2  
    if event.key in key_states:
        if not key_states[event.key]:
            console.log(f"🔼 {event.key} key pressed (rising edge)")
        key_states[event.key] = True
    if event.key == "r": # Reset
        global T, first_space_press, inlet_history, outlet_history, qc_history, qc_mean_history, qc_integral, fluid_position_ist, last_time, elapsed_time, zmin, zmax, ec_avg_temp_history, pe_sim_data
        T[:, :] = 0.0
        first_space_press = True
        inlet_history = [0]
        outlet_history = [0]
        ec_avg_temp_history = [0]
        qc_history = [0]
        qc_mean_history = [0]
        qc_integral = [0]
        fluid_position_ist = 0

        pe_sim_data = {
            't': [],
            'i': [],
            'v': [],
            'vsw': [],
            'i0': 0.0,
            'v0': 0.0,
            't_last': 0.0,
            'mode': 'high'
        }


        last_time = time.time()
        elapsed_time = 0        
        #last_time = time.time()

        zmin = -0.5*dTad
        zmax = 0.5*dTad

        console.log("🧊 All reset to 0.0")
    if event.key == "l":
        # Collect your params/results
        data = {
#            "params": params,
            "u": u,
            "v": v,
            "p": p,
            "streamlines": streamline_plotly,
            "u2": u2,
            "v2": v2,
            "p2": p2,
            "streamlines2": streamline_plotly2,            
        }
        save_pickle_to_download(data, "solve_flow_result.pkl")
    if event.key == "b":
        
    #    global u, v, p, u2, v2, p2, streamline_plotly, streamline_plotly2
        # toggle builder mode
        global isBuilderMode

        if not isBuilderMode:
            console.log("🔨 Builder Mode: ON")
        else:
            console.log("🔨 Builder Mode: OFF")
            u[:], v[:], p[:], streamline_plotly = solve_flow(u, v, p, False, 1)
            u2[:], v2[:], p2[:], streamline_plotly2 = solve_flow(u2, v2, p2, False, -1)
        
        isBuilderMode = not isBuilderMode


def on_key_up(event):
    if event.key in key_states:
        console.log(f"🔽 {event.key} key released (falling edge)")
        key_states[event.key] = False



# control input from html
def on_toggle_mode(event):
    is_automatic = toggle_mode.checked
    global last_time, elapsed_time, fluid_position_ist
    last_time = time.time()
    #current_time = time.time()
    elapsed_time = 0
    #elapsed_time = float(slider_cycle.value)/4.0 # 1/4 phase as start to only pump half to the right at the start
    fluid_position_ist = 0    
    console.log(f"🔁 Automatic Mode: {'ON' if is_automatic else 'OFF'}")
    # Do something in Python based on mode
    # For example, disable simulation control logic when automatic is off

def on_toggle_invert(event):
    is_inverted = toggle_invert.checked
    console.log(f"🔄 Phase Inversion: {'ON' if is_inverted else 'OFF'}")
    # Toggle phase logic here

def on_slider_cycle(event):
    value = float(slider_cycle.value)
    console.log(f"⏱️ Cycle Time: {value:.1f} s")
    # Update internal cycle time

def on_slider_delay(event):
    value = float(slider_delay.value)
    console.log(f"⏳ Pump Delay: {value*100:.0f}%")
    # Update delay logic


def on_slider_load(event):
    global slider_load_value
    value = float(slider_load.value)
    slider_load_value = value
    console.log(f"🎚️ Load: {value*100:.0f}%")
    # Update load (hhx and chx)

def on_toggle_labels(event):
    global show_labels 
    show_labels = not event.target.checked
    if show_labels:
        console.log("📝 Labels enabled in simulation.")
    else:
        console.log("🙈 Labels hidden in simulation.")
    # Trigger redraw of the streamplot

    global u, v, p, u2, v2, p2, streamline_plotly, streamline_plotly2

    u[:], v[:], p[:], streamline_plotly = solve_flow(u, v, p, True, 1)
    u2[:], v2[:], p2[:], streamline_plotly2 = solve_flow(u2, v2, p2, True, -1)

def on_toggle_pause(event):
    global is_paused 
    is_paused = event.target.checked

def on_toggle_isSliders(event):
    global isSliders
    isSliders = event.target.checked

def on_toggle_buildermode(event):
    global isBuilderMode
    isBuilderMode = event.target.checked
    if isBuilderMode:
        console.log("🔨 Builder Mode: ON")
    else:
        console.log("🔨 Builder Mode: OFF")
        global mouse_is_down, u, v, p, u2, v2, p2, streamline_plotly, streamline_plotly2
        u[:], v[:], p[:], streamline_plotly = solve_flow(u, v, p, False, 1)
        u2[:], v2[:], p2[:], streamline_plotly2 = solve_flow(u2, v2, p2, False, -1)        
    # Update UI or logic based on builder mode

# === Semi-Lagrangian Advection ===
def semi_lagrangian_advection(T, u, v, dt):
    x_dep = X - u * dt
    y_dep = Y - v * dt
    x_dep = np.clip(x_dep, 0, nx - 1)
    y_dep = np.clip(y_dep, 0, ny - 1)
    # (I)
    #coords = [y_dep.ravel(), x_dep.ravel()]
    #T_interp = map_coordinates(T, coords, order=1, mode='nearest')
    
    # (II) could be faster
    x_flat = x_dep.ravel()
    y_flat = y_dep.ravel()
    T_interp = bilinear_interpolate_numpy(T, x_flat, y_flat)

    return T_interp.reshape(T.shape)

def compute_heat_flow(x, u_mod, v_mod):
    global T, u_mod_mult

    #console.log(f"XXX")
    if x <= 0 or x >= nx - 1:
        return 0.0, 0.0
    y_slice = slice(ny//2,ny)
    valid_y = fluid_mask[y_slice,x] # only through upper half (valve)
    #valid_y = fluid_mask[:, x]
    cp = c_p_fluid  # assuming fluid only for simplicity

    # Convective heat flow: ρ * cp * u * T * dy
    #Q_conv = np.sum(rho * cp * u[valid_y, x] * T[valid_y, x] * dy)
    #if key_states["ArrowRight"]:
    #    u_mod, v_mod = u, v
    #elif key_states["ArrowLeft"]:
    #    u_mod, v_mod = -u, -v
    #else:
    #    u_mod, v_mod = np.zeros_like(u), np.zeros_like(v)

    #u_mod = u_mod * u_mod_mult * n_convection_step
    #v_mod = v_mod * v_mod_mult * n_convection_step
    # only if pumped through that valve (otherwise no mass flow!)
    #if current_direction == 1:
    u_mod = u_mod * n_convection_step
    #else:
    #    u_mod = np.zeros_like(u_mod)
    #v_mod = v_mod * n_convection_step
    
    Q_conv = np.sum(rho * cp * u_mod_mult * u_mod[y_slice, x] * T[y_slice, x] * dy)

    # Diffusive heat flow: -k * ∂T/∂x * dy
    dT_dx = (T[:, x+1] - T[:, x-1]) / (2 * dx) * n_diffusion_steps
    k_map = np.where(piston_mask[:, x], 0.0, np.where(fluid_mask[:, x], k_fluid_x, k_solid_x))
    #Q_diff = -np.sum(k_map[valid_y] * dT_dx[valid_y] * dy)

    dT_dx_upper = dT_dx[y_slice][valid_y]
    k_map_upper = k_map[y_slice][valid_y]
    Q_diff = -np.sum(k_map_upper * dT_dx_upper * dy)

    return Q_conv, Q_diff

def get_automatic_clocks():
    if (not is_pyodide):
        return None, None
    if not toggle_mode.checked:
        return None, None  # manual mode → no clocks

    global last_time, elapsed_time
    current_time = time.time()
    elapsed_time += current_time - last_time
    last_time = current_time

    cycle_time = float(slider_cycle.value)
    pump_delay_ratio = float(slider_delay.value) / 100.0
    invert = toggle_invert.checked

    t = (elapsed_time *1.25) % cycle_time

    clk_heat = 1 if t < (cycle_time / 2) else 0
    if invert:
        clk_heat = 1- clk_heat

    # Delayed clock
    delay = pump_delay_ratio * cycle_time
    t2 = (t - delay + cycle_time) % cycle_time
    clk_flow = 1 if t2 < (cycle_time / 2) else 0
    #clk_flow = 0 if t2 < (cycle_time / 2) else 1

    

    return clk_heat, clk_flow

def update_heatmap(): 
    global mouseX, mouseY, zmin, zmax
    
    # Slice to visible region
    T_visible = T[:, xlim_min:xlim_max]
    x_vals = list(range(xlim_min, xlim_max))
    y_vals = list(range(ny))

    # Contour (drawn first, underneath)
    # Combine obstacle and valve regions, then cast to float for contour plotting
    #obstacle_to_plot = (obstacle[:, xlim_min:xlim_max].astype(bool) | (valve_mask[:, xlim_min:xlim_max] != 0)).astype(float)
    contour_trace = {
        'z': obstacle[:, xlim_min:xlim_max].tolist(),
        #'z': obstacle_to_plot.tolist(),
        'x': x_vals,
        'y': y_vals,
        'type': 'contour',
        'colorscale': [[0, 'rgba(0,0,0,0)'], [1, 'black']],  # Transparent → black
        'contours': {
            'start': 0.5,
            'end': 0.5,
            'size': 0.5,
            'coloring': 'lines'
        },
        'line': {
            'width': 2,
            'color': 'black'
        },
        'reversescale': True,
        'showscale': False,
        'hoverinfo': 'skip'
    }

    if current_direction == 1:
        piston_dash1 = 'dot'
        piston_dash2 = 'solid'
        piston_width1 = 0.5
        piston_width2 = 2
    elif current_direction == 0:
        piston_dash1 = 'dash'
        piston_dash2 = 'dash'
        piston_width1 = 1
        piston_width2 = 1
    else:  # current_direction == -1
        piston_dash1 = 'solid'
        piston_dash2 = 'dot'
        piston_width1 = 2
        piston_width2 = 0.5

    contour_valve1_to_plot = ((valve_mask[:, xlim_min:xlim_max] > 0)).astype(float)
    contour_valve1_trace = {
        #'z': obstacle[:, xlim_min:xlim_max].tolist(),
        'z': contour_valve1_to_plot.tolist(),
        'x': x_vals,
        'y': y_vals,
        'type': 'contour',
        'colorscale': [[0, 'rgba(0,0,0,0)'], [1, 'purple']],  # Transparent → black
        'contours': {
            'start': 0.5,
            'end': 0.5,
            'size': 0.5,
            'coloring': 'lines'
        },
        'line': {
            'width': piston_width1,
            'color': 'purple',
            'dash': piston_dash1
        },
        'reversescale': True,
        'showscale': False,
        'hoverinfo': 'skip'
    }
    contour_valve2_to_plot = (((-valve_mask[:, xlim_min:xlim_max]) > 0)).astype(float)
    contour_valve2_trace = {
        #'z': obstacle[:, xlim_min:xlim_max].tolist(),
        'z': contour_valve2_to_plot.tolist(),
        'x': x_vals,
        'y': y_vals,
        'type': 'contour',
        'colorscale': [[0, 'rgba(0,0,0,0)'], [1, 'cyan']],  # Transparent → black
        'contours': {
            'start': 0.5,
            'end': 0.5,
            'size': 0.5,
            'coloring': 'lines'
        },
        'line': {
            'width': piston_width2,
            'color': 'cyan',
            'dash': piston_dash2
        },
        'reversescale': True,
        'showscale': False,
        'hoverinfo': 'skip'
    }

    contour_iso_to_plot = (((iso_mask[:, xlim_min:xlim_max]) > 0)).astype(float)
    contour_iso_trace = {
        #'z': obstacle[:, xlim_min:xlim_max].tolist(),
        'z': contour_iso_to_plot.tolist(),
        'x': x_vals,
        'y': y_vals,
        'type': 'contour',
        'colorscale': [[0, 'rgba(0,0,0,0)'], [1, 'white']],  # Transparent → black
        'contours': {
            'start': 0.5,
            'end': 0.5,
            'size': 0.5,
            'coloring': 'lines'
        },
        'line': {
            'width': 2,
            'color': 'white'
        },
        'reversescale': True,
        'showscale': False,
        'hoverinfo': 'skip'
    }

    contour_hhx_mask_to_plot = (((hhx_mask[:, xlim_min:xlim_max]) > 0)).astype(float)
    contour_hhx_mask_trace = {
        #'z': obstacle[:, xlim_min:xlim_max].tolist(),
        'z': contour_hhx_mask_to_plot.tolist(),
        'x': x_vals,
        'y': y_vals,
        'type': 'contour',
        'colorscale': [[0, 'rgba(0,0,0,0)'], [1, 'red']],  # Transparent → black
        'contours': {
            'start': 0.5,
            'end': 0.5,
            'size': 0.5,
            'coloring': 'lines'
        },
        'line': {
            'width': 1,
            'color': 'white'
        },
        'reversescale': True,
        'showscale': False,
        'hoverinfo': 'skip'
    }
    
    contour_chx_mask_to_plot = (((chx_mask[:, xlim_min:xlim_max]) > 0)).astype(float)
    contour_chx_mask_trace = {
        #'z': obstacle[:, xlim_min:xlim_max].tolist(),
        'z': contour_chx_mask_to_plot.tolist(),
        'x': x_vals,
        'y': y_vals,
        'type': 'contour',
        'colorscale': [[0, 'rgba(0,0,0,0)'], [1, 'blue']],  # Transparent → black
        'contours': {
            'start': 0.5,
            'end': 0.5,
            'size': 0.5,
            'coloring': 'lines'
        },
        'line': {
            'width': 1,
            'color': 'white'
        },
        'reversescale': True,
        'showscale': False,
        'hoverinfo': 'skip'
    }
        


    # Piston line (*10 weil fluid_position in 0.1er Schritten variiert wird...)
    # Achtung: dt wird nicht korrekt berücksichtigt. dt = const, clk = const, aber fps is nicht konstant!
    contour_piston = {
    'x': [int(fluid_position_ist*n_convection_step*u_mod_mult*dt*10 - 1) % nx,int(fluid_position_ist*n_convection_step*u_mod_mult*dt*10 - 1) % nx],
    'y': [1, ny-2],  # or [min(y_vals), max(y_vals)]
    'mode': 'lines',
    'type': 'scatter',
    'line': {
        'color': 'white',
        'width': 1,
        'dash': 'dot'  # or 'solid', 'dot', etc.
    },
    #'name': f'x = {x0}',
    'hoverinfo': 'skip',
    'reversescale': True,
    #'showscale': False,
    'showlegend': False
    }

    zmin = min(-0.5*dTad, zmin)
    zmax = max(0.5*dTad, zmax)
    # adjust heatmap min/max color scaling
    if T.min() < zmin:
        zmin = T.min()
    else:
        zmin += 0.001 # 1mK per step max....

    if T.max() > zmax:
        zmax = T.max()
    else:
        zmax -= 0.001


    # Heatmap (drawn second, on top)
    heatmap_trace = {
        'z': T_visible.tolist(),
        'x': x_vals,
        'y': y_vals,
        'type': 'heatmap',
        'colorscale': 'RdBu',
        'zmin': zmin,
        'zmax': zmax,
        #'reversescale': True,
        'showscale': False,
        'hoverinfo': 'z',
        'zsmooth': 'fast', ## does it slow down animation?
        'hovertemplate': 'Temperature: %{z:.2f} K<extra></extra>'
    }

    heatmap_layout = {
        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
        'autosize': True,
        'height': 280,
        'dragmode': False,
        'automargin': False,

        'xaxis': {
        #    'range': [xlim_min, xlim_max],
            'autorange': True,
            #'fixedrange': True,
        #    'scaleanchor': 'y',
            'constrain': 'domain',
            'showline': False, 
            'showgrid': False, 
            'showticklabels': False,
            'visible': False,
            'zeroline': False,
            'title': '',
        },

        'yaxis': {
            'range': [ny, 0],  # Reversed manually
            'fixedrange': True,
            'autorange': False,
            'showline': False,
            'showgrid': False,
            'showticklabels': False,
            'visible': False,
            'zeroline': False,
            'title': '',
        }
    }

    # # this works, but slows down visualization a lot...
    # obstacle_visible = obstacle[:, xlim_min:xlim_max]
    # obstacle_mask = np.where(obstacle_visible == 1, 1.0, np.nan)
    # # add hhx and chx to mask
    # obstacle_mask[hhx_mask[:, xlim_min:xlim_max] > 0] = 0.5 # less dark
    # obstacle_mask[chx_mask[:, xlim_min:xlim_max] > 0] = 0.5 # less dark
    # # add valve to mask
    # obstacle_mask[valve_mask[:, xlim_min:xlim_max] != 0] = 0.5 # less dark  
    # # add iso to mask
    # obstacle_mask[iso_mask[:, xlim_min:xlim_max] != 0] = 0.5 # less dark


    # obstacle_mask_trace = {
    #     'z': obstacle_mask.tolist(),
    #     'x': x_vals,
    #     'y': y_vals,
    #     'type': 'heatmap',
    #     'colorscale': [[0, 'rgba(1,1,1,0)'], [1, 'rgba(1,1,1,0.19)']],  # 19% transparent white
    #     'showscale': False,
    #     'hoverinfo': 'skip',
    #     'zmin': 0,
    #     'zmax': 1
    # }
    

    if num_plates % 2 == 0:
        y_fluid_pos = ny/2-0.87
        y_material_pos = y_fluid_pos - (plate_height + plate_spacing)/2-0.87
    else:
        y_fluid_pos = ny/2 + (plate_height + plate_spacing)/2-0.87
        y_material_pos = ny/2-0.87
    # put Text Annotation (workaround: use labeled scatterplot...)
    trace1 = { 
        'type': 'scatter',
        'x': [int((xlim_min + xlim_max) / 2)],
        #'y': [int(ny / 2) - 2],
        'y': [y_fluid_pos],
        #'text': ["<b>heat transfer fluid</b> (<i>silicone oil</i> or <i>water</i>)"],
        'text': ["<b>heat transfer fluid</b>"],
        'mode': 'text',
        'textposition': 'middle center',
        'hoverinfo': 'skip',
        'showlegend': False,
        'cliponaxis': False
    }

    trace2 = {
        'type': 'scatter',
        'x': [int((xlim_min + xlim_max) / 2)],
        #'y': [4],
        'y': [y_material_pos],
        #'text': ["<b>electrocaloric material</b> (<i>PVDF polymer</i> or <i>PST ceramic</i>)"],
        'text': ["<b>electrocaloric material</b>"],
        #'text': ['<span style="background-color:rgba(255,255,255,0.7); padding:2px;"><b>electrocaloric material</b> (<i>PVDF polymer</i> or <i>PST ceramic</i>)</span>'],
        'mode': 'text',
        'textposition': 'middle center',
        'hoverinfo': 'skip',
        'showlegend': False,
        'cliponaxis': False
    }
 
    trace3 = {
        'type': 'scatter',
        #'x': [xlim_min + 22],  # Adjust for the correct placement
        'x': [(piston_end + x_valve1)//2-1],  # Adjust for the correct placement
        'y': [int(3 * ny / 4)],
        'text': ["<b>cool<br>side<br><i>T</i><sub>C</sub></b>"],
        'mode': 'text',
        'textposition': 'middle left',
        #'textangle': 90,  # Rotate text by 90 degrees
        'hoverinfo': 'skip',
        'showlegend': False,
        'cliponaxis': False
    }

    trace4 = {
        'type': 'scatter',
        #'x': [xlim_max - 22],  # Adjust for the correct placement
        'x': [(x_valve2+piston_start)//2+2],
        'y': [int(3* ny / 4)],
        'text': ["<b>hot<br>side<br><i>T</i><sub>H</sub></b>"],
        'mode': 'text',
        'textposition': 'middle right',
        #'textangle': 90,  # Rotate text by 90 degrees
        'hoverinfo': 'skip',
        'showlegend': False,
        'cliponaxis': False
    }

    # Adding copyright text
    trace5 = {
        'type': 'scatter',
        #'x': [int((xlim_min + xlim_max) / 2)],
        'x': [int(nx*0.95-1)],
        'y': [2],  # Adjust for the correct placement
        'text': [f'{copyright_version}'],
        'mode': 'text',
        'textposition': 'bottom left',
        'showlegend': False,
        'cliponaxis': False,        
        'hoverinfo': 'skip',
        'yaxis_title': {
            'text': 'temp.'
        }
    }


    if not ((mouseX == None) or (mouseY == None)):
        T_atCursor = T[int(mouseY),int(mouseX)] 
        xPos = [int(np.clip(mouseX, nx*0.06, nx*0.94))]
        yPos = [int(np.clip(mouseY, 1, ny -3))]
    else:
        T_atCursor = None
        xPos = [int((xlim_min + xlim_max) / 2)]
        yPos = [ny-4]
    
    trace6 = {
        'type': 'scatter',
        #'x': ,
        #'y': ,  # Adjust for the correct placement
        'x': xPos,
        'y': yPos,  # Adjust for the correct placement
        'text':  [f"{(T_atCursor+T_show_offset):.2f} °C" if T_atCursor is not None else ''],
        'mode': 'text',
        'textposition': 'top center',
        'showlegend': False,
        'cliponaxis': False,        
        'hoverinfo': 'skip'
    }

    # builder mode text
    trace7 = {
        'type': 'scatter',
        'x': [int((xlim_min + xlim_max) / 2)],
        #'y': [4],
        'y': [ny//8],
        #'text': ["<b>electrocaloric material</b> (<i>PVDF polymer</i> or <i>PST ceramic</i>)"],
        'text': ["<b>BUILDER MODE ('b' to recalculate flow)</b>"],
        #'text': ['<span style="background-color:rgba(255,255,255,0.7); padding:2px;"><b>electrocaloric material</b> (<i>PVDF polymer</i> or <i>PST ceramic</i>)</span>'],
        'mode': 'text',
        'textposition': 'middle center',
        'hoverinfo': 'skip',
        'showlegend': False,
        'cliponaxis': False
    }   

 
    
    if show_labels:
        if not isBuilderMode:
            dat = [heatmap_trace] + (streamline_plotly if current_direction >= 0 else streamline_plotly2) + [contour_piston, contour_trace, contour_chx_mask_trace, contour_hhx_mask_trace, contour_iso_trace, contour_valve1_trace, contour_valve2_trace, trace1, trace2, trace3, trace4, trace5, trace6]
        else:
            dat = [heatmap_trace] + (streamline_plotly if current_direction >= 0 else streamline_plotly2) + [contour_piston, contour_trace, contour_chx_mask_trace, contour_hhx_mask_trace, contour_iso_trace, contour_valve1_trace, contour_valve2_trace, trace1, trace2, trace3, trace4, trace5, trace6, trace7]
    #    dat = [heatmap_trace] +  [contour_trace, trace1, trace2, trace3, trace4, trace5, trace6]
    else:
        dat = [heatmap_trace] + (streamline_plotly if current_direction >= 0 else streamline_plotly2) + [contour_piston, contour_trace, contour_chx_mask_trace, contour_hhx_mask_trace, contour_iso_trace, contour_valve1_trace, contour_valve2_trace, trace5]

        
    Plotly.react(
        document.getElementById("heatmap-plot"),
        to_js(dat),  # First = behind
        to_js(heatmap_layout),
        to_js({'displayModeBar': False, 'staticPlot': True})  # config
    )

    # Only attach once (mouse interface)
    #if not hasattr(window, "_plotly_click_attached"):
    #    click_proxy = create_proxy(handle_plotly_click)
    #    document.getElementById("heatmap-plot").on("plotly_click", click_proxy)
    #    window._plotly_click_attached = True
        


# === Time-Domain Plot of Temperature History ===

def update_temperature_graph():
    #global step_counter
    #if step_counter % 2 != 0:
    #    return
    
    useProfiler = False
    if useProfiler:
        pr = cProfile.Profile()
        pr.enable()
        update_temperature_graph_noProfile()
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')  # or 'tottime'
        ps.print_stats(20)  # Top 20 slowest function calls
        console.log("📊 update_temperature_graph profile:\n" + s.getvalue())
    else:        
        update_temperature_graph_noProfile()


def update_temperature_graph_noProfile():
    if len(inlet_history) < 2:
        return

    x_data = list(range(len(inlet_history)))

    # --- Plot 1: Temperature History ---
    temp_data = [
        {
            'x': x_data,
            'y': [x + T_show_offset for x in inlet_history],
            'name': r'cool-side temperature <i>T</i><sub>C</sub> [°C]',
            'line': {'color': 'blue'},
            'type': 'scatter'
        },
        {
            'x': x_data,
            'y': [x + T_show_offset for x in outlet_history],
            'name': r'hot-side temperature <i>T</i><sub>H</sub> [°C]',
            'line': {'color': 'red'},
            'type': 'scatter'
        }
    ]

    temp_layout = {
        'margin': {'l': 30, 'r': 10, 't': 10, 'b': 20},
        'showlegend': True,
        'autosize': True,        
        #'font' : {'size': 1400},
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5,
        },
        'xaxis': {'visible': True, 'title': 'time'},
        'yaxis': {'visible': True, 'title': 'temperature [°C]'},
        #'xaxis': r'time <i>t</i>',
        #'yaxis': r'temperature <i>T</i> [°C]',
        'height': 200
    }

    Plotly.react(document.getElementById("graph-plot"),
        to_js(temp_data), 
        to_js(temp_layout))

    # --- Plot 2: Cooling Power and Energy ---
    cumulative_qc = np.cumsum(qc_history) * dt
    qc_integral.clear()
    qc_integral.extend(cumulative_qc.tolist()) 
     
    # --- Plot 1: Cooling Power ---
    power_data = [
        {
            'x': x_data,
            'y': qc_history,
            'name': 'Cooling power <i>Q&#775;</i><sub>C</sub> [W]',
            'line': {'color': 'green'},
            'type': 'scatter'
        },
      {
            'x': x_data,
            'y': qc_mean_history,
            'name': 'Average cooling power <i>Q&#775;</i><sub>C,AVG</sub> [W]',
            'line': {'color': 'green', 'dash': 'dot', 'width': 1},
            'type': 'scatter'
        }
    ]

    power_layout = {
        'margin': {'l': 30, 'r': 40, 't': 10, 'b': 20},
        'autosize': True,      
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5
        },
        'xaxis': {
            'showgrid': False
        },
        'yaxis': {
            'title': 'Q̇<sub>C</sub> [W]',
            'side': 'left'
        },
        'height': 200
    }

    Plotly.react(document.getElementById("power-plot"),
                to_js(power_data),
                to_js(power_layout))


    # --- Plot 2: Cooling Energy ---
    energy_data = [
        {
            'x': x_data,
            'y': qc_integral,
            'name': 'Cooling energy <i>E</i><sub>C</sub> [J]',
            'line': {'color': 'orange', 'dash': 'dot'},
            'type': 'scatter'
        }
    ]

    energy_layout = {
        'margin': {'l': 30, 'r': 40, 't': 10, 'b': 20},
        'autosize': True,      
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5
        },
        'xaxis': {
            'showgrid': False
        },
        'yaxis': {
            'title': 'E<sub>C</sub> [J]',
            'side': 'left'
        },
        'height': 200
    }

    Plotly.react(document.getElementById("energy-plot"),
                to_js(energy_data),
                to_js(energy_layout))
    
    # --- Plot: ΔT vs. Q̇c ---
    dT_vs_Qc_data = [
        {
            'x': [0, 0] + qc_mean_history,
            'y': [0, float('nan')] + [out - inn for out, inn in zip(outlet_history, inlet_history)],
            'name': 'Temperature span Δ<i>T</i><sub>SPAN</sub> vs. Cooling power <i>Q&#775;</i><sub>C</sub>',
            'mode': 'lines',
            'line': {
                'color': 'purple',
                'width': 1.5  # Thinner line; default is 2
            },
            'type': 'scatter'
        },
        {
            'x': [qc_mean_history[-1]],  # Wrapped in a list
            'y': [outlet_history[-1] - inlet_history[-1]],  # Wrapped in a list
            'mode': 'markers',
            'marker': {'color': 'purple', 'size': 1, 'size_max': 48},  
            'type': 'scatter',
            'showlegend': False
        }
    ]


    dT_vs_Qc_layout = {
        'margin': {'l': 30, 'r': 40, 't': 10, 'b': 20},
        'autosize': True,
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5
        },
        'xaxis': {
            'title': 'Average cooling power <i>Q&#775;</i><sub>C</sub> [W]',
            'showgrid': False
        },
        'yaxis': {
            'title': 'Δ<i>T</i> [K]',
            'side': 'left'
        },
        'height': 200
    }

    Plotly.react(
        document.getElementById("dT_vs_Qc_plot"),
        to_js(dT_vs_Qc_data),
        to_js(dT_vs_Qc_layout)
    )


    # rolling data (viewer can get dizzy)
    #     # --- Plot 4+5: PE Electrical Combined Plot ---
    # # Voltage plot
    # pe_voltage_data = [{
    #     'x': pe_sim_data['t'],
    #     'y': pe_sim_data['v'],
    #     'name': 'Electrocaloric Capacitor Voltage [V]',
    #     'line': {'color': 'blue'},
    #     'type': 'scatter'
    # }]

    # pe_voltage_layout = {
    #     'margin': {'l': 40, 'r': 40, 't': 10, 'b': 20},
    #     'autosize': True,
    #     'showlegend': True,
    #     'legend': {
    #         'orientation': 'h',
    #         'yanchor': 'bottom',
    #         'y': 1.02,
    #         'xanchor': 'center',
    #         'x': 0.5
    #     },
    #     'xaxis': {'title': 'Time [s]'},
    #     'yaxis': {'title': 'Electrocaloric Capacitor Voltage [V]'},
    #     'height': 200
    # }

    # Plotly.react(
    #     document.getElementById("pe_voltage-plot"),
    #     to_js(pe_voltage_data),
    #     to_js(pe_voltage_layout)
    # )

    # # Current plot
    # pe_current_data = [{
    #     'x': pe_sim_data['t'],
    #     'y': pe_sim_data['i'],
    #     'name': 'Electrocaloric Capacitor Charging Current [A]',
    #     'line': {'color': 'red'},
    #     'type': 'scatter'
    # }]

    # pe_current_layout = {
    #     'margin': {'l': 40, 'r': 40, 't': 10, 'b': 30},
    #     'autosize': True,
    #     'showlegend': True,
    #     'legend': {
    #         'orientation': 'h',
    #         'yanchor': 'bottom',
    #         'y': 1.02,
    #         'xanchor': 'center',
    #         'x': 0.5
    #     },
    #     'xaxis': {'title': 'Time [s]'},
    #     'yaxis': {'title': 'Electrocaloric Capacitor Charging Current [A]'},
    #     'height': 200
    # }

    # Plotly.react(
    #     document.getElementById("pe_current-plot"),
    #     to_js(pe_current_data),
    #     to_js(pe_current_layout)
    # )

    #split-line approach 
    # Parameters# Parameters
    window = 5.5
    gap_frac = 0  # not used any more
    gap_duration = window * gap_frac
    gap_start = window - gap_duration / 2
    gap_end = gap_duration / 2

    # Modulo wrapped time
    t_wrapped = np.array(pe_sim_data['t']) % window
    v_arr = np.array(pe_sim_data['v'])
    i_arr = np.array(pe_sim_data['i'])
    
    # Mask visible data (excluding gap)
    mask = (t_wrapped >= gap_end) | (t_wrapped < gap_start)
    t_visible = t_wrapped[mask]
    v_visible = v_arr[mask]
    i_visible = i_arr[mask]

    # Find index of wrap-around (minimum t_wrapped value)
    if len(t_visible) > 0:
        wrap_idx = np.argmin(t_visible)

        # Insert NaN just before the wrap
        t_plot = np.insert(t_visible, wrap_idx, np.nan)
        v_plot = np.insert(v_visible, wrap_idx, np.nan)
        i_plot = np.insert(i_visible, wrap_idx, np.nan)
    else:
        t_plot = t_visible
        v_plot = v_visible
        i_plot = i_visible

    t_dummy = np.array([-0.1, -0.05, window + 0.05, window + 0.1])
    v_dummy = np.array([0.0, np.nan, np.nan, 0.0])
    i_dummy = np.array([0.0, np.nan, np.nan, 0.0])

    t_plot = np.concatenate((t_dummy[[0,1]], t_plot, t_dummy[[2,3]]))
    v_plot = np.concatenate((v_dummy[[0,1]], v_plot, v_dummy[[2,3]]))
    i_plot = np.concatenate((i_dummy[[0,1]], i_plot, i_dummy[[2,3]]))

    # Plot 4: PE Voltage
    pe_voltage_data = [{
        'x': t_plot,
        'y': v_plot,
        'name': 'Electrocaloric Capacitor Voltage [V]',
        'line': {'color': 'blue'},
        'type': 'scatter'
    }]

    pe_voltage_layout = {
        'margin': {'l': 40, 'r': 40, 't': 10, 'b': 20}, 
        'autosize': True,        
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5
        },
        'xaxis': {
            'title': 'Time [s]',   
            'range': [0, window],
            'autorange': False, 
            'fixedrange': True,
            'dtick': 1
        },
        'yaxis': {'title': 'Electrocaloric Capacitor Voltage [V]',
                  'autorange': True},
        'height': 160,
        'shapes': [{
            'type': 'line',
            'x0': 0,
            'x1': 0,
            'y0': 0,
            'y1': 1,
            'yref': 'paper',
            'line': {'color': 'gray', 'width': 1, 'dash': 'dot'}
        }]
    }

    Plotly.react(
        document.getElementById("pe_voltage-plot"),
        to_js(pe_voltage_data),
        to_js(pe_voltage_layout)
    )

    # Plot 5: PE Current
    pe_current_data = [{
        'x': t_plot,
        'y': i_plot,
        'name': 'Electrocaloric Capacitor Charging Current [A]',
        'line': {'color': 'red'},
        'type': 'scatter'
    }]

    pe_current_layout = {
        'margin': {'l': 40, 'r': 40, 't': 10, 'b': 30}, 
        'autosize': True,        
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'center',
            'x': 0.5
        },
        'xaxis': {
            'title': 'Time [s]', 
            'range': [0, window],
            'autorange': False, 
            'fixedrange': True,
            'dtick': 1
        },
        'yaxis': {'title': 'Electrocaloric Capacitor Charging Current [A]',
                  'autorange': True},
        'height': 160,
        'shapes': [{
            'type': 'line',
            'x0': 0,
            'x1': 0,
            'y0': 0,
            'y1': 1,
            'yref': 'paper',
            'line': {'color': 'gray', 'width': 1, 'dash': 'dot'}
        }]
    }

    Plotly.react(
        document.getElementById("pe_current-plot"),
        to_js(pe_current_data),
        to_js(pe_current_layout)
    )




# === Update Function ===
def update_frame():
    if is_paused:
        return
    
    useProfiler = False
    if useProfiler:
        pr = cProfile.Profile()
        pr.enable()
        update_frame_noProfile()
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(15)  # top 15 slowest calls
        console.log(s.getvalue())
    else:
        update_frame_noProfile()

def update_frame_noProfile():
    global T, u, v, p, u2, v2, p2, cf, step_counter, space_previous, first_space_press, stream, fluid_position_ist, isRemoteControlled, efield_soll, efield_ist, last_frame_time, valve_mask, current_direction, hhx_mask, chx_mask, slider_load_value, Q_chx

    clk_heat, clk_flow = get_automatic_clocks()

    if isSliders:
        fluid_position_slider_measured = float(document.getElementById("param-fluid_position").value)
        #efield_slider =  bool(float(document.getElementById("param-e_field").value) <= 0.5)
        # 1 - x weil der slider "von oben nach unten" ist...
        efield_soll = 1.0 - float(document.getElementById("param-e_field").value) # ist zw. 0 <= x <= 1
    if isRemoteControlled:
        # Hier die Werte von der RT Box abfragen (Pascal)
        fluid_position_slider_measured = -1 # Wertebereich -1 bis 1
        efield_soll = 1 # Wertebereich 0 bis 1 (wird bisher auf 0/1 geklippt und kann zukünftig auch noch in feineren Schritten implementiert werden)

    # Flow direction (automatic mode)
    if clk_flow == 1:
        u_mod, v_mod = u, v
        current_direction = 1

        fluid_position_ist += 0.1

        if fluid_position_ist > 0.2:     # hack to visually solve dt ≠ 1/fps issue....
            fluid_position_ist -= ((fluid_position_ist-0.2)**2)*0.02 # quadratic damping

    elif clk_flow == 0:
        #u_mod, v_mod = -u, -v
        u_mod, v_mod = u2, v2
        current_direction = -1

        fluid_position_ist -= 0.1

        if fluid_position_ist < -0.2:   # hack to visually solve dt ≠ 1/fps issue....
            fluid_position_ist +=  ((fluid_position_ist+0.2)**2)*0.02 # quadratic damping
    else:
        if isSliders:
            # remote mode overriden by sliders  
    
            if fluid_position_slider_measured > fluid_position_ist + 0.05: # more right
                fluid_position_ist += 0.1
                u_mod, v_mod = u, v
                current_direction = 1
            elif  fluid_position_slider_measured < fluid_position_ist - 0.05: #more left
                fluid_position_ist -= 0.1
                #u_mod, v_mod = -u, -v
                u_mod, v_mod = u2, v2
                current_direction = -1
            else:
                u_mod, v_mod = np.zeros_like(u), np.zeros_like(v)
        else:
            # manual mode
            if key_states["ArrowRight"]:
                u_mod, v_mod = u, v
                current_direction = 1
            elif key_states["ArrowLeft"]:
                #u_mod, v_mod = -u, -v
                u_mod, v_mod = u2, v2 
                current_direction = -1
            else:
                u_mod, v_mod = np.zeros_like(u), np.zeros_like(v)
                current_direction = 0

    u_mod = u_mod * u_mod_mult
    v_mod = v_mod * v_mod_mult

    # Heat input (automatic)
    if clk_heat is not None:
        efield_soll = bool(clk_heat)
    elif isSliders: # remote mode overritten by sliders
          #efield_soll = efield_slider
          pass
    else: #manual
        efield_soll = float(key_states[" "]) # 0/1 digital

    heat_adjustment = 0.0

    if efield_soll != efield_ist:
        heat_adjustment = dTad * (efield_soll - efield_ist)
        efield_ist = efield_soll

    #if space_current != space_previous:
    #    if space_current:
    #        console.log("🔼 Space key pressed (rising edge)")
    #        if first_space_press:
    #            heat_adjustment = dTad / 2.0
    #            first_space_press = False
    #        else:
    #            heat_adjustment = dTad
    #    else:
    #        console.log("🔽 Space key released (falling edge)")
    #        heat_adjustment = -dTad
    #space_previous = space_current

    # attempt to fix "crash after 6 minutes"
    if isnan(heat_adjustment):
        console.log(f"❌❌❌ heat_adjustment was NaN, set to 0 to avoid crash!")
        heat_adjustment = 0.0
    
    if heat_adjustment != 0.0:
        if first_space_press:
            heat_adjustment *= 0.5   # check if it still works with RTBox
            first_space_press = False
        console.log(f"dTad={dTad}, efield_soll={efield_soll}, efield_ist={efield_ist}")
        console.log(f"🌡️ Heat {'injected' if heat_adjustment > 0 else 'removed'}: {heat_adjustment:+.1f} K")
        T[~fluid_mask] += heat_adjustment

    T_new = T.copy()
    cp_map = np.where(fluid_mask[1:-1,1:-1], c_p_fluid, c_p_solid)
    count_diff = 0
    count_conv = 0
    for _ in range(total_steps): # weighted interleaving  
        ratio_diff = count_diff / n_diffusion_steps if n_diffusion_steps > 0 else float('inf')
        ratio_conv = count_conv / n_convection_step if n_convection_step > 0 else float('inf')

        if ratio_diff <= ratio_conv:
            kx_map = np.where(piston_mask[1:-1,1:-1], 0.0,
                              np.where(fluid_mask[1:-1,1:-1], k_fluid_x, k_solid_x))
            ky_map = np.where(piston_mask[1:-1,1:-1], 0.0,
                              np.where(fluid_mask[1:-1,1:-1], k_fluid_y, k_solid_y))
            T_new[1:-1, 1:-1] += dt / cp_map * (
                kx_map * (T_new[1:-1,2:] - 2*T_new[1:-1,1:-1] + T_new[1:-1,:-2]) / dx**2 +
                ky_map * (T_new[2:,1:-1] - 2*T_new[1:-1,1:-1] + T_new[:-2,1:-1]) / dy**2
            )
            count_diff += 1
        else:
            if use_convection:
                T_new = semi_lagrangian_advection(T_new, u_mod, v_mod, dt)

            #HHX heat flow
            # Calculate the heat flow for each HHX cell (set Thhx to +1K over reference)
            R_th_hhx = 2.5 # 2 is too low (instable oscillations)
            C = rho * c_p_fluid * dx * dy
            alpha = np.exp(-dt / (R_th_hhx * C)) # exponential (RC!)
            T_hhx = 0.0
            #Q_hhx = (T[hhx_mask] - T_hhx) / R_th_hhx  # linear
            #Q_hhx = C * (T[hhx_mask] - T_new[hhx_mask]) / dt # exponential
            # Calculate temperature change for each HHX cell  
            #dT_hhx = -Q_hhx * dt / (rho * c_p_fluid * dx * dy)   
            T_new[hhx_mask] = T_hhx + (T_new[hhx_mask] - T_hhx) * alpha

            # CHX heat flow (exponential RC, like HHX)
             # R worse than HHX
            #R_th_chx = R_th_hhx * (10 - 9 * slider_load_value)
            R_th_chx = R_th_hhx * (50 - 49 * slider_load_value)
            C = rho * c_p_fluid * dx * dy
            alpha_chx = np.exp(-dt / (R_th_chx * C))
            T_chx = 0.0

            # Exponential heat flow (average over dt)
            global Q_hhx, Q_chx
            Q_hhx = (T_new[hhx_mask] - T_hhx) / R_th_hhx
            Q_chx = (T_new[chx_mask] - T_chx) / R_th_chx

            # Exponential temperature update
            T_new[chx_mask] = T_chx + (T_new[chx_mask] - T_chx) * alpha_chx


            #console.log(f"Q_hhx_mean: {Q_hhx.mean():.4f}, Q_chx_mean: {Q_chx.mean():.4f}  ")

            count_conv += 1

          #zero-gradient (Neumann) BCs:
        T_new[:, 0] = T_new[:, 1] # x direction
        T_new[:, -1] = T_new[:, -2] # x direction
        T_new[0, :] = T_new[1, :] # y direction
        T_new[-1, :] = T_new[-2, :] # y direction
        # Looping BCs (x-direction) - overrides Neumann BC
        T_new[:, 0] = T_new[:, -2]     # inlet from outlet
        T_new[:, -1] = T_new[:, 1]     # ghost cell from interior

    T = T_new
    #T = np.clip(T, -1.1, 1.1)
    cf.set_data(T)


    if (is_pyodide):
        # Update obstacle contour
        global obstacle_contour
        for coll in list(obstacle_contour.collections):
            try:
                coll.remove()
            except ValueError:
                pass
        obstacle_contour = ax.contour(((obstacle.astype(bool) | (valve_mask != 0)).astype(float)), levels=[0.25], colors='black', linewidths=1.0)
        #obstacle_contour = ax.contour((((valve_mask != 0)).astype(float)), levels=[0.25], colors='black', linewidths=1.0)

        #buf = io.BytesIO()
        #fig.savefig(buf, format="png", pad_inches=0)
        #buf.seek(0)
        #img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        #buf.close()

        #img_element.src = f"data:image/png;base64,{img_base64}"

        update_heatmap()


    # === Inlet and Outlet Average Temperatures ===
   
    
    # Use fluid_mask to restrict to fluid cells

    # variation 1: use inlet/outlet (at valve) temp
    inlet_temp = np.mean(T[ny//2:, x_inlet_idx][fluid_mask[ny//2:, x_inlet_idx]])
    outlet_temp = np.mean(T[ny//2:, x_outlet_idx][fluid_mask[ny//2:, x_outlet_idx]])

    # variation 2: use heat exchanger temp
    inlet_temp = np.mean(T_new[chx_mask])
    outlet_temp = np.mean(T_new[hhx_mask])

    #ec_avg_temp = np.mean(T[obstacle]) # todo
    ec_avg_temp = 0
    
    Tspan = outlet_temp - inlet_temp
    Q_conv_in, Q_diff_in = compute_heat_flow(x_inlet_idx - 2, u_mod, v_mod)
    Q_total_in = Q_conv_in + Q_diff_in



    # PE simulator update
    Vdc = 1150
    L = 8000000e-6         # Inductance (H) increased for better visualization (!)
    C = 180e-6          # Capacitance (F)
 
    pe_t_start = pe_sim_data['t_last']
    pe_Tseg = time.time() - last_frame_time
    pe_v_set = efield_soll * Vdc
    pe_ipeak, pe_ivalley = [1, -1]
    if pe_sim_data['v0'] < efield_soll * Vdc:
        pe_ipeak, pe_ivalley = [1,-0.1]
    else:
        pe_ipeak, pe_ivalley = [0.1, -1]
    pe_results = run_hcc_segment(pe_sim_data['i0'], pe_sim_data['v0'], pe_Tseg, pe_v_set, pe_ipeak, pe_ivalley, Vdc, L, C)
    pe_t_seg, pe_i_seg, pe_v_seg, pe_vsw_seg = pe_results

    append_segment_to_sim(pe_sim_data, pe_t_seg, pe_i_seg, pe_v_seg, pe_vsw_seg, pe_t_start)



    current_time = time.time()
    frame_duration = current_time - last_frame_time
    fps = 1.0 / frame_duration if frame_duration > 0 else 0
    last_frame_time = current_time

    #console.log(f"Step: {step_counter} | Ti = {inlet_temp:.4f} | Qin_conv: {Q_conv_in:.4f}, Qin_diff: {Q_diff_in:.4f}, Qin_total: {Q_total_in:.4f}")
    console.log(f"Step: {step_counter} | Ti = {inlet_temp:.4f} | Qin_conv: {Q_conv_in:.4f}, Qin_diff: {Q_diff_in:.4f}, Qin_total: {Q_total_in:.4f} | FPS: {fps:.1f} ")

    inlet_history.append(inlet_temp)
    outlet_history.append(outlet_temp)
    ec_avg_temp_history.append(ec_avg_temp)

    #qc_history.append(Q_total_in)
    qc_history.append(-np.mean(Q_chx)*10000.0)
    qc_mean_history.append(np.mean(qc_history[-60:]))  # Mean of last 10 values

    # Optional: trim to last N values for performance/clarity
    max_points = 1000
    if len(inlet_history) > max_points:
        inlet_history.pop(0)
        outlet_history.pop(0)
        qc_history.pop(0)
        qc_mean_history.pop(0)
        ec_avg_temp_history.pop(0)
        
    update_temperature_graph()

    step_counter += 1

 
    #console.log(f"Step: {step_counter} | Tspan (To - Ti) = {Tspan:.4f} | Ti = {inlet_temp:.4f} | To = {outlet_temp:.4f}")

   # Q1_conv, Q1_diff = compute_heat_flow(x_inlet_idx)
   # Q2_conv, Q2_diff = compute_heat_flow(x_outlet_idx)

    #Q_avg_conv = (Q1_conv + Q2_conv) / 2
    #Q_avg_diff = (Q1_diff + Q2_diff) / 2
    #Q_avg_total = Q_avg_conv + Q_avg_diff

    #console.log(f"Step: {step_counter} | Tspan: {Tspan:.4f} | Qconv: {Q_avg_conv:.4f}, Qdiff: {Q_avg_diff:.4f}, Qtotal: {Q_avg_total:.4f}")

def bilinear_interpolate_numpy(T, x, y):
    ny, nx = T.shape

    # Convert to int indices
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clip only once per axis
    x0 = np.maximum(0, np.minimum(nx - 1, x0))
    x1 = np.maximum(0, np.minimum(nx - 1, x1))
    y0 = np.maximum(0, np.minimum(ny - 1, y0))
    y1 = np.maximum(0, np.minimum(ny - 1, y1))

    # Compute weights once
    dx = x - x0
    dy = y - y0
    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy)
    wc = (1 - dx) * dy
    wd = dx * dy

    # Flatten index helpers
    flat_index = lambda i, j: i * nx + j

    # Flatten T to 1D to support fast advanced indexing
    T_flat = T.ravel()

    idx_a = flat_index(y0, x0)
    idx_b = flat_index(y0, x1)
    idx_c = flat_index(y1, x0)
    idx_d = flat_index(y1, x1)

    result = wa * T_flat[idx_a] + wb * T_flat[idx_b] + wc * T_flat[idx_c] + wd * T_flat[idx_d]

    return result


### Power electronics simulator

# Function to calculate time to next switch event
def next_switch_time(i0, v0, Vsw, I_target, L, C): 
    omega = 1 / np.sqrt(L * C)

    A = i0
    B = (Vsw - v0) / (omega * L)
    I_amp = np.sqrt(A**2 + B**2)
    phi = np.arctan2(A, B)

    if np.abs(I_target) > I_amp:
        return None, A, B, I_amp, phi, None  # Can't reach target

    target_angle = np.arcsin(I_target / I_amp)
    theta_options = [
        target_angle,
        np.pi - target_angle,
        2 * np.pi + target_angle,
        2 * np.pi - target_angle
    ]
    t_candidates = []
    for theta in theta_options:
        while theta - phi <= 0:
            theta += 2 * np.pi
        t_candidates.append((theta - phi) / omega)
    t_pos = [tc for tc in t_candidates if tc > 1e-12]
    t_next = min(t_pos) if t_pos else None
    return t_next, A, B, I_amp, phi, t_candidates

# Function to run a single HCC simulation step
def run_hcc_segment(i0, v0, Tseg, v_set, I_peak, I_valley, Vin, L, C):
    omega = 1 / np.sqrt(L * C)

    t = 0.0
    t_log = [t]
    i_log = [i0]
    v_log = [v0]
    vsw_log = [(t, Vin)]
    
    if i0 == 0:
        if abs(I_peak) > abs(I_valley):
            mode = 'high'
        else:
            mode = 'low'
    else:
        if i0 < 0:
            mode = 'high'
        else:
            mode = 'low'

    while t < Tseg:
        Vsw = Vin if mode == 'high' else 0
        I_target = I_peak if mode == 'high' else I_valley
        dt, A, B, I_amp, phi, t_candidates = next_switch_time(i0, v0, Vsw, I_target, L, C)

        

        

        if dt is None or (
            (abs(I_peak) > abs(I_valley) and v0 > v_set) or
            (abs(I_peak) < abs(I_valley) and v0 < v_set)
        ):
            print("No further switching possible. Holding last operating point until Tsim.")
            t_log.append(t)
            i_log.append(0.0)
            v_log.append(v0)
            vsw_log.append((t, v0))
            t_log.append(Tseg)
            i_log.append(0.0)
            v_log.append(v0)
            vsw_log.append((Tseg, v0))
            break

        N_points_per_Tres = 100 # for a smooth sinus select how many points should be rendered for one Tres
        T_res = 2 * np.pi / omega
        N_segment = max(3, int(N_points_per_Tres * dt / T_res))
        t_vals = np.linspace(0, dt, N_segment)
        iL = A * np.cos(omega * t_vals) + B * np.sin(omega * t_vals)
        vC = v0 + (A / (C * omega)) * np.sin(omega * t_vals) + (B / (C * omega)) * (1 - np.cos(omega * t_vals))

        t_log.extend((t_vals[1:] + t).tolist())
        i_log.extend(iL[1:].tolist())
        v_log.extend(vC[1:].tolist())
        vsw_log.append((t, Vsw))

        t += dt
        i0 = iL[-1]
        v0 = vC[-1]
        mode = 'low' if mode == 'high' else 'high'

    return t_log, i_log, v_log, vsw_log

# Function to initialize/reset simulation state
def initialize_simulation():
    return {
        'i0': 0.0,              # Initial inductor current (A)
        'v0': 0.0,              # Initial capacitor voltage (V)
        't': 0.0,               # Initial time (s)
        't_log': [],
        'i_log': [],
        'v_log': [],
        'vsw_log': [],
        'mode': 'high'
    }


def append_segment_to_sim(sim, t_seg, i_seg, v_seg, vsw_seg, t_start):
    # Calculate absolute time
    t_abs = [t_start + ts for ts in t_seg]
        
    sim['t'].extend([t_start + ts for ts in t_seg])
    sim['i'].extend(i_seg)
    sim['v'].extend(v_seg)
    sim['vsw'].extend([(t_start + ts, val) for ts, val in vsw_seg])
    sim['t_last'] = t_start + t_seg[-1]
    sim['i0'] = i_seg[-1]
    sim['v0'] = v_seg[-1]

    max_age = 5
    # Determine cutoff time
    cutoff = t_abs[-1] - max_age

    # Find valid indices (newer than cutoff)
    valid_idx = next((i for i, t in enumerate(sim['t']) if t >= cutoff), len(sim['t']))
    maxlen = len(sim['t']) - valid_idx  # Only keep the last max_age seconds

    for key in ['t', 'i', 'v']:
        sim[key] = sim[key][-maxlen:]

    # For vsw, use the time value in the tuple
    sim['vsw'] = [item for item in sim['vsw'] if item[0] >= cutoff]

### Startup/Initialisation of program
# Default config should already be defined before this

def init_simulation(config=default_config):
    global nx, ny, dx, dy, dt, mu, rho
    global c_p_fluid, c_p_solid, k_fluid_x, k_fluid_y, k_solid_x, k_solid_y
    global use_convection, n_diffusion_steps, n_convection_step, u_mod_mult, v_mod_mult, dTad
    global total_steps, fig_size, fig_dpi, fig_graph_dpi, fig_stream_dpi
    global mouseX, mouseY, mouse_is_down
    global obstacle, fluid_mask, T, u, v, p
    global x_inlet_idx, x_outlet_idx, xlim_min, xlim_max
    global margin, plate_height, plate_spacing, num_plates
    global start_y, Y, X, fig, ax, cf, obstacle_contour
    global streamline_plotly, streamline_plotly2
    global inlet_history, outlet_history, qc_history, qc_mean_history, qc_integral, ec_avg_temp_history
    global step_counter, show_labels, key_states, isBuilderMode
    global space_previous, first_space_press
    global zmin, zmax
    global isSliders, fluid_position_ist, isRemoteControlled, efield_ist, efield_soll, slider_load_value, Q_hhx, Q_chx
    global last_frame_time
    global piston_mask, piston_start, piston_end
    global iso_mask, hhx_mask, chx_mask
    global valve_mask, x_valve1, x_valve2
    global u2,v2,p2 # for left low (if valves are used, to calculate two velocity fields)
    global current_direction
    global T_show_offset
    global pe_sim_data 
    
    pe_sim_data = {
        't': [],
        'i': [],
        'v': [],
        'vsw': [],
        'i0': 0.0,
        'v0': 0.0,
        't_last': 0.0,
        'mode': 'high'
    }

    current_direction = 0 # for selection of which streamline_plot to show
    T_show_offset = 20.0 # just for displaying... internaly calculated starting with 0
    
    # Timekeeping
    import time
    global last_time, elapsed_time, is_paused
    last_time = time.time()
    elapsed_time = 0.0
    is_paused = True

    last_frame_time = time.time()
    
    # Assign from config
    nx = config["nx"]
    ny = config["ny"]
    dx = dy = config.get("dx", 1.0)
    dt = config["dt"]
    mu = config["mu"]
    rho = config["rho"]

    c_p_fluid = config["c_p_fluid"]
    c_p_solid = config["c_p_solid"]
    k_fluid_x = k_fluid_y = config["k_fluid"]
    k_solid_x = k_solid_y = config["k_solid"]
    dTad = config["dTad"]

    n_diffusion_steps = config["n_diff"]
    n_convection_step = config["n_conv"]
    total_steps = n_diffusion_steps + n_convection_step

    u_mod_mult = config.get("u_mod_mult", 0.5)
    v_mod_mult = config.get("v_mod_mult", 0.5)
    use_convection = True

    fig_size = (4, 1)
    fig_dpi = 150
    fig_graph_dpi = 200
    fig_stream_dpi = 200

    isBuilderMode = False
    slider_load_value = 0.0
    Q_hhx = 0.0
    Q_chx = 0.0


    Y, X = np.mgrid[0:ny, 0:nx] # Represents the grid coordinates for a domain of size (ny, nx)

    # Geometry
    plate_height = config["plate_height"]
    plate_spacing = config["plate_spacing"]
    num_plates = config["num_plates"]

    margin = nx // 10 + nx // 4
    start_y = (ny - (num_plates * plate_height + (num_plates - 1) * plate_spacing)) // 2

    zmin = -1
    zmax = 1 # inital scaling

    # Arrays
    #T = np.full((ny, nx), 22.0) # initial temperature = "roomtemperature"
    T = np.zeros((ny, nx))
    u = np.zeros((ny, nx))
    u2 = np.zeros((ny, nx))  
    v = np.zeros((ny, nx))
    v2 = np.zeros((ny, nx))
    p  = np.zeros((ny, nx))
    p2 = np.zeros((ny, nx))
    obstacle = np.zeros((ny, nx))

    if True: # parallel plates
        for i in range(num_plates):
            y_start = start_y + i * (plate_height + plate_spacing)
            for dy_ in range(plate_height):
                y = y_start + dy_
                if 0 <= y < ny:
                    obstacle[y, margin:nx - margin] = 1
    elif False: # random initialization
        fill_probability = 0.05  # adjust between 0 (none) and 1 (full)

        for y in range(1,ny-1):
            for x in range(margin, nx - margin-1):
                if np.random.rand() < fill_probability:
                    obstacle[y, x] = 1
                    obstacle[y+1, x] = 1
                    obstacle[y+1, x+1] = 1
                    obstacle[y, x+1] = 1
    else: # vertical plates
        x_start = margin
        x_end = nx - margin
        plate_index = 0

        x = x_start
        while x + plate_height <= x_end:
            for dx in range(plate_height):
                x_pos = x + dx
                if x_pos >= nx:  # safety
                    break

                if plate_index % 2 == 0:
                    # Even plates → gap at top
                    for y in range(plate_spacing, ny):
                        obstacle[y, x_pos] = 1
                else:
                    # Odd plates → gap at bottom
                    for y in range(ny - plate_spacing):
                        obstacle[y, x_pos] = 1

            x += plate_height + plate_spacing
            plate_index += 1

    fluid_mask = (obstacle == 0)

    x_inlet_idx = margin - 8
    x_outlet_idx = nx - margin + 8

    # show only regenerator and inlet/outlet
    xlim_min = x_inlet_idx - 8
    xlim_max = x_outlet_idx + 8

    # show all
    xlim_min = 0
    xlim_max = nx

    # Define piston region
    piston_end = x_inlet_idx // 2
    piston_start = nx - (nx - x_outlet_idx) // 2 
    piston_mask = np.zeros((ny, nx), dtype=bool)
    #piston_mask[:, 0:piston_end] = True
    #piston_mask[:, piston_start:nx] = True



    # Define valves
    valve_mask = np.zeros((ny, nx), dtype=int)
    # Valve positions
    x_valve1 = x_inlet_idx + 3
    x_valve2 = x_outlet_idx - 3
    # Upper and lower halves
    y_half = ny // 2
        
    # At x_valve1: upper half is right valve (+1), lower half is left valve (-1)
    valve_mask[:y_half, x_valve1:x_valve1+1] = +1   # right valve (blocks leftward flow)
    valve_mask[y_half:, x_valve1:x_valve1+1] = -1   # left valve (blocks rightward flow)

    # At x_valve2: upper half is left valve (-1), lower half is right valve (+1)
    valve_mask[:y_half, x_valve2-1:x_valve2] = -1   # left valve (blocks rightward flow)
    valve_mask[y_half:, x_valve2-1:x_valve2] = +1   # right valve (blocks leftward flow)

    # Define iso region
    iso_mask = np.zeros((ny, nx), dtype=bool)
    iso_mask[y_half-1:y_half+1, (piston_end + x_valve1)//2:x_valve1] = 1 # horizontal separations
    iso_mask[y_half-1:y_half+1, x_valve2:(x_valve2+piston_start)//2] = 1
    #iso_mask[:y_half-3+7, piston_end-1:piston_end+1] = 1
    iso_mask[y_half+3-7:, piston_end-1:piston_end+1] = 1
    #iso_mask[y_half+3+5:, piston_end-1:piston_end+1] = 1
    #iso_mask[:y_half-3-5, piston_start-1:piston_start+1] = 1
    iso_mask[y_half+3-7:, piston_start-1:piston_start+1] = 1

    # Hot side heat exchanger (HHX) mask (covers 25% to 75% in y-direction between valve and piston, and in x-direction 25%-75% of upper valve)
    hhx_mask = np.zeros((ny, nx), dtype=bool)
    #hhx_mask[ny*9//16:ny*15//16, (x_valve2+1) + ((piston_start-1)-(x_valve2+1))*1//4 : (x_valve2+1) + ((piston_start-1)-(x_valve2+1))*3//4] = 1
    # Calculate HHX bounding box
    #hhx_y_start = ny * 10 // 16
    #hhx_y_end   = ny * 16 // 16
    hhx_y_end = ny
    hhx_y_start = ny - 12
    x_span = (piston_start - 1) - (x_valve2 + 1) 
    #x_span = ((x_span + 6) // 7) * 7  # next integer multiple of 7 greater than or equal to x_span 
    hhx_x_start = (x_valve2 + 1) + 3 # was: x_span * 1//4
    #hhx_x_end   = (x_valve2 + 1) + x_span * 3//4
    #y_span = hhx_y_end - hhx_y_start

    hhx_mask = np.zeros((ny, nx), dtype=bool)
    # # Create HHX mask (rectangle)
    # #hhx_mask[hhx_y_start:hhx_y_end, hhx_x_start:hhx_x_end] = True
    # # Divide x-span into 7 equal segments and fill 1st, 3rd, 5th, 7th
    # segment_width = (hhx_x_end - hhx_x_start) // 7
    # for i in range(1-1, 8-1, 2):  # 0-based indices for 1st, 3rd, 5th, 7th
    #     start = hhx_x_start + i * segment_width
    #     end = start + segment_width
    #     hhx_mask[hhx_y_start:hhx_y_end, start:end] = True
    # for i in range(2-1, 7-1, 4):  # lower connections
    #     start = hhx_x_start + i * segment_width
    #     end = start + segment_width
    #     hhx_mask[hhx_y_start:(hhx_y_start + y_span//7), start:end] = True
    # #for i in range(3, 4, 1):  # upper connection
    # i = 3
    # start = hhx_x_start + i * segment_width
    # end = start + segment_width  
    # hhx_mask[hhx_y_end - y_span//7:(hhx_x_end), start:end] = True

    # Original 7x7 binary pattern
    # hhx_pattern = np.array([
    #     [1, 0, 0, 0, 0, 0, 1],
    #     [1, 0, 1, 1, 1, 0, 1],
    #     [1, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 1, 0, 1, 0, 1],
    #     [1, 1, 1, 0, 1, 1, 1]
    # ], dtype=bool)
    hhx_pattern = np.array([
  #      [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],        
        [1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1],        
        [1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ], dtype=bool)
    # Mirror the pattern vertically (flip along Y-axis)
    hhx_pattern = hhx_pattern[::-1]
    
    # Calculate HHX region dimensions
    #hhx_height = hhx_y_end - hhx_y_start
    #hhx_width = hhx_x_end - hhx_x_start
    # Scale pattern to fit the HHX region using nearest-neighbor
    #scale_y = hhx_height / 11
    #scale_x = hhx_width / 7
    #scaled_pattern = zoom(hhx_pattern.astype(float), (scale_y, scale_x), order=0) > 0.25
    # Assign to hhx_mask
    #hhx_mask[hhx_y_start:hhx_y_end, hhx_x_start:hhx_x_end] = scaled_pattern[:hhx_height, :hhx_width]

    hhx_pattern = np.repeat(hhx_pattern, 3, axis=1) #  scale by integer factor in x direction (y|x !!)

    console.log(
        f"hhx_y_start: {hhx_y_start}, hhx_y_end: {hhx_y_start + hhx_pattern.shape[0]}, "
        f"hhx_x_start: {hhx_x_start}, hhx_x_end: {hhx_x_start + hhx_pattern.shape[1]}, "
        f"hhx_pattern.shape: {hhx_pattern.shape}")

    # No zoom: just place the pattern at the start position
    hhx_mask[hhx_y_start:hhx_y_start + hhx_pattern.shape[0], hhx_x_start:hhx_x_start + hhx_pattern.shape[1]] = hhx_pattern


    # CHX is a morrowed version
    chx_mask = np.fliplr(hhx_mask)


    # Boundary conditions
    u[:, 0] = 1.0 #right flow
    u[:, -1] = u[:, -2]

    u2[:, 0] = -1.0 #left flow
    u2[:, -1] = u[:, -2]

    # Initialize history
    inlet_history = [0]
    outlet_history = [0]
    qc_history = [0]
    qc_mean_history = [0]
    qc_integral = [0]
    streamline_plotly = []
    ec_avg_temp_history = [0]

    # Plot
    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    obstacle_contour = ax.contour(((obstacle.astype(bool) | (valve_mask != 0)).astype(float)), levels=[0.25], colors='black', linewidths=1.0)
    #obstacle_contour = ax.contour((((valve_mask != 0)).astype(float)), levels=[0.25], colors='black', linewidths=1.0)
    cf = ax.imshow(T, cmap='coolwarm', origin='lower', interpolation='bicubic',
                   vmin=min(-1, min(min(inlet_history), min(outlet_history))),
                   vmax=max(1, max(max(inlet_history), max(outlet_history))))
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(0, ny)
    ax.axis('off')

    # Mouse + keyboard
    mouseX = mouseY = None
    mouse_is_down = False
    show_labels = True
    step_counter = 0
    space_previous = False
    first_space_press = True
    key_states = {"ArrowLeft": False, "ArrowRight": False, " ": False, "r": False, "b": False}
    isSliders = False
    fluid_position_ist = 0.0
    isRemoteControlled = False
    efield_ist = 0
    efield_soll = 0

    # Solve initial flow

    # calculate initial flow
    if (config != default_config) or (not is_pyodide):
        u2[:], v2[:], p2[:], streamline_plotly2 = solve_flow(u2, v2, p2, False, -1)
        u[:], v[:], p[:], streamline_plotly = solve_flow(u, v, p, False, 1)
    # load precalculated flow from file (default for web deployment)
    # if sim parameters are changed, using "l" key a new pkl file can be downloaded and saved to the Repo
    else:
    #    u2[:], v2[:], p2[:], streamline_plotly2 = solve_flow(u2, v2, p2, True, -1) # to initialize everything else
    #    u[:], v[:], p[:], streamline_plotly = solve_flow(u, v, p, True, 1)
    #    data = await load_pickle_from_relative_url("solve_flow_result.pkl")
        file_like = io.BytesIO(bytes(pickle_bytes))  # `pickle_bytes` is set by JS
        data = pickle.load(file_like)
        u2, v2, p2, streamline_plotly2 = (data[k] for k in ['u2', 'v2', 'p2', 'streamlines2'])
        u, v, p, streamline_plotly = (data[k] for k in ['u', 'v', 'p', 'streamlines'])



    is_paused = False

        
#async def load_pickle_from_relative_url(filename):
#    # Use a relative path (e.g. "solve_flow_result.pkl" or "data/solve_flow_result.pkl")
#    resp = await pyodide.http.open_url(filename)
#    data = resp.read()  # returns bytes
#    obj = pickle.loads(data)
#    return obj

def save_pickle_to_download(data, filename="result.pkl"):
    # Pickle the data to bytes
    bytes_io = io.BytesIO()
    pickle.dump(data, bytes_io)
    bytes_io.seek(0)
    # Convert to JS Uint8Array
    uint8 = __import__("pyodide").ffi.to_js(bytes_io.getvalue())
    blob = Blob.new([uint8], { "type": "application/octet-stream" })
    url = URL.createObjectURL(blob)
    # Create a download link and click it
    link = document.createElement("a")
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
    
def register_handlers():
    global img_element, stream_img_element, graph_img_element, power_img_element
    global toggle_mode, toggle_invert, slider_cycle, slider_delay, toggle_labels, toggle_puase, toggle_isSliders
    global slider_efield, slider_fluidposition, slider_load, toggle_buildermode

    img_element = document.getElementById("mpl-canvas")
    stream_img_element = document.getElementById("heatmap-plot")
    graph_img_element = document.getElementById("graph-canvas")
    power_img_element = document.getElementById("power-canvas")

    toggle_mode = document.getElementById("toggle-mode")
    toggle_invert = document.getElementById("toggle-invert")
    slider_cycle = document.getElementById("slider-cycle")
    slider_delay = document.getElementById("slider-delay")
    slider_load = document.getElementById("slider-load")
    toggle_labels = document.getElementById("toggle-labels")
    toggle_puase = document.getElementById("toggle-pause")
    toggle_isSliders = document.getElementById("toggle-isSliders")
    slider_efield = document.getElementById("param-e_field")
    slider_fluidposition = document.getElementById("param-fluid_position")
    toggle_buildermode = document.getElementById("toggle-buildermode")

    heatmap_div = stream_img_element
    heatmap_div.addEventListener("contextmenu", create_proxy(lambda e: e.preventDefault()))
    heatmap_div.addEventListener("dragstart", create_proxy(lambda e: e.preventDefault()))
    heatmap_div.addEventListener("mousedown", create_proxy(lambda e: e.preventDefault()))

    window.addEventListener("mousedown", create_proxy(on_mouse_down)) 
    window.addEventListener("mousemove", create_proxy(on_mouse_move))
    window.addEventListener("mouseup", create_proxy(on_mouse_up))

    window.addEventListener("keydown", create_proxy(on_key_down))
    window.addEventListener("keyup", create_proxy(on_key_up))

    toggle_mode.addEventListener("change", create_proxy(on_toggle_mode))
    toggle_invert.addEventListener("change", create_proxy(on_toggle_invert))
    slider_cycle.addEventListener("input", create_proxy(on_slider_cycle))
    slider_delay.addEventListener("input", create_proxy(on_slider_delay))
    slider_load.addEventListener("input", create_proxy(on_slider_load))
    toggle_labels.addEventListener("change", create_proxy(on_toggle_labels))
    toggle_puase.addEventListener("change", create_proxy(on_toggle_pause))
    toggle_isSliders.addEventListener("change", create_proxy(on_toggle_isSliders))
    toggle_buildermode.addEventListener("change", create_proxy(on_toggle_buildermode))


def read_config_from_html():
    config = default_config.copy()

    def get_val(id, cast_func, key=None):
        el = document.getElementById(id)
        if el and el.value != "":
            config[key or id.replace("param-", "")] = cast_func(el.value)

    get_val("param-nx", int)
    get_val("param-ny", int)
    get_val("param-mu", float)
    get_val("param-cpfluid", float, "cp_fluid")
    get_val("param-cpsolid", float, "cp_solid")
    get_val("param-kfluid", float, "k_fluid")
    get_val("param-ksolid", float, "k_solid")
    get_val("param-ndiff", int, "n_diff")
    get_val("param-nconv", int, "n_conv")
    get_val("param-dtad", float, "dTad")
    get_val("param-plateheight", int, "plate_height")
    get_val("param-platespacing", int, "plate_spacing")
    get_val("param-numplates", int, "num_plates")

    return config

def restart_simulation_from_html():
    config = read_config_from_html()
    init_simulation(config)
    register_handlers()

def startRemoteControl_from_html():
    console.log(f"Initialize RT Box + RPC interface")
    # Hier Initialisierung für RT Box ergänzen (Pascal)
 
    isRemoteControlled = True
    return

console.log(f"Is Pyodide?: {is_pyodide}")
if __name__ == "__main__":
    init_simulation()
    register_handlers()
