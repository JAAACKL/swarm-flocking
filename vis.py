import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

# --- Create the main figure ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.35)  # Adjust for slider space

# --- parameters (default values) ---
eps = 0.6    # σ‑norm sharpness
h = 0.2      # bump flat‑top fraction of r
a = 5.0      # attraction coefficient
b = 5.0      # repulsion coefficient 
c = abs(a - b) / (2 * np.sqrt(a * b))
d = 7.0      # desired inter‑agent spacing
r = 7.0 * 1.2      # interaction cut‑off range
d_beta = 10.0      # desired inter‑agent spacing for β

# View mode and show flags
view_mode = 'Both'  # 'Both', 'Gradient', 'Potential'
show_phi_alpha = True  # Show agent-agent gradient
show_psi_alpha = True  # Show agent-agent potential
show_psi_beta = True  # Show agent-agent potential
show_phi_beta = True  # Show agent-agent gradient
# --- helper functions ---
def sigma_grad(x):
    """σ‑norm of scalar distance x."""
    return x / (1 + eps * sigma_norm(x))

def sigma_norm(z):
    """σ‑norm of scalar distance z."""
    return (np.sqrt(1 + eps * np.linalg.norm(z)**2) - 1) / eps

def sigma1(x):
    return x / np.sqrt(1 + x**2)

def sigma1_vec(z_vec):
    norm = np.linalg.norm(z_vec)
    return z_vec / np.sqrt(1 + norm**2)

def bump(z):
    """
    Finite‑range window ρ(z):
      = 1           , if 0 ≤ z < h
      = 0.5*(1+cos(pi*(z-h)/(1-h))) , if h ≤ z ≤ 1
      = 0           , if z > 1
    """
    return np.where(
        z < h,
        1.0,
        np.where(
            z <= 1.0,
            0.5 * (1 + np.cos(np.pi * (z - h) / (1 - h))),
            0.0
        )
    )

def phi(z):
    """Attractive/repulsive core φ as in Eq. (16), with shift c."""
    return 0.5 * ((a + b) * sigma1(z + c) + (a - b))

def phi_alpha(z):
    """finite-range core phi_alpha(z) = bump(z/r) * phi(z - sigma_norm(d))"""
    return bump(z / sigma_norm(r)) * phi(z - sigma_norm(d))

def phi_beta(z):
    """
    φ_β(z) = bump(z/d_beta) * (σ1(z - d_beta) - 1)
    where z = ||z_vec||, d_beta = d_beta
    """
    return bump(z / sigma_norm(d_beta)) * (sigma1(z - sigma_norm(d_beta)) - 1.0)

def psi_alpha(z):
    """pairwise potential psi_alpha(z) = ∫_d^z phi_alpha(s) ds"""
    """
    Pairwise potential ψₐ(z) = ∫_d^z φₐ(s) ds,
    shifted so that ψₐ(d) = 0.
    """
    z = np.array(z)
    # 1) evaluate the gradient at all sample points
    phi_vals = phi_alpha(z)
    # 2) do a cumulative trapezoidal integral from z[0] to z[i]
    psi = np.zeros_like(z)
    psi[1:] = np.cumsum((phi_vals[:-1] + phi_vals[1:]) / 2.0 * np.diff(z))
    # 3) subtract the constant so that ψₐ(d) = 0
    #    find the first index where z >= d
    idx = np.searchsorted(z, d)
    psi -= psi[idx]
    return psi

def psi_beta(z):
    """pairwise potential psi_beta(z) = ∫_d^z phi_beta(s) ds"""
    """
    Pairwise potential ψₐ(z) = ∫_d^z φₐ(s) ds,
    shifted so that ψₐ(d) = 0.
    """
    z = np.array(z)
    # 1) evaluate the gradient at all sample points
    phi_vals = phi_beta(z)
    # 2) do a cumulative trapezoidal integral from z[0] to z[i]
    psi = np.zeros_like(z)
    psi[1:] = np.cumsum((phi_vals[:-1] + phi_vals[1:]) / 2.0 * np.diff(z))
    # 3) subtract the constant so that ψₐ(d) = 0
    #    find the first index where z >= d
    idx = np.searchsorted(z, d)
    psi -= psi[idx]
    return psi
# --- Create the main figure ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.35)  # Adjust for slider space

# --- Initialize the plot ---
z_values = np.linspace(0, r * 1.2, 500)
phi_line, = ax.plot([], [], 'b-', lw=2, label='φ_α (Gradient)')
psi_line, = ax.plot([], [], 'r-', lw=2, label='ψ_α (Potential)')
psi_beta_line, = ax.plot([], [], 'g-', lw=2, label='ψ_β (Potential)')
phi_beta_line, = ax.plot([], [], 'y-', lw=2, label='φ_β (Gradient)')
ax.set_xlabel('Distance (z)')
ax.set_ylabel('Value')
ax.set_xlim(0, r * 1.2)
ax.set_ylim(-1.5, 1.5)
ax.grid(True)
ax.legend(loc='upper right')
ax.set_title('Interactive Agent Interaction Functions')

# --- Create sliders ---
slider_params = {
    'eps': {'label': 'eps (σ-norm sharpness)', 'valmin': 0.01, 'valmax': 5.0, 'valinit': eps},
    'h': {'label': 'h (bump flat-top fraction)', 'valmin': 0.0, 'valmax': 1.0, 'valinit': h},
    'a': {'label': 'a (attraction coef.)', 'valmin': 0.0, 'valmax': 50.0, 'valinit': a},
    'b': {'label': 'b (repulsion coef.)', 'valmin': 0.0, 'valmax': 50.0, 'valinit': b},
    'd': {'label': 'd (desired spacing)', 'valmin': 0.1, 'valmax': 50.0, 'valinit': d},
    'r': {'label': 'r (cutoff range)', 'valmin': 0.1, 'valmax': 100.0, 'valinit': r},
    'd_beta': {'label': 'd_beta (desired spacing for β)', 'valmin': 0.1, 'valmax': 100.0, 'valinit': d_beta},
}

sliders = {}
for i, (param_name, param_info) in enumerate(slider_params.items()):
    slider_pos = plt.axes([0.25, 0.05 + 0.03 * i, 0.65, 0.02])
    sliders[param_name] = Slider(
        slider_pos, 
        param_info['label'], 
        param_info['valmin'], 
        param_info['valmax'], 
        valinit=param_info['valinit']
    )

# --- Add radio buttons for view mode ---
view_radio_ax = plt.axes([0.025, 0.7, 0.15, 0.15])
view_radio = RadioButtons(view_radio_ax, ('Both', 'Gradient', 'Potential'))

# --- Add checkbuttons for showing functions ---
show_check_ax = plt.axes([0.025, 0.5, 0.15, 0.15])
show_check = CheckButtons(
    show_check_ax, 
    ('φ_α (Gradient)', 'ψ_α (Potential)'), 
    (show_phi_alpha, show_psi_alpha)
)

# --- Add reset button ---
reset_ax = plt.axes([0.025, 0.35, 0.15, 0.04])
reset_button = Button(reset_ax, 'Reset Parameters')

# --- Update function ---
def update(val=None):
    global eps, h, a, b, c, d, r, d_beta, show_phi_alpha, show_psi_alpha, show_psi_beta, show_phi_beta, view_mode
    
    # Update parameters based on slider values
    eps = sliders['eps'].val
    h = sliders['h'].val
    a = sliders['a'].val
    b = sliders['b'].val
    c = abs(a - b) / (2 * np.sqrt(a * b))
    d = sliders['d'].val
    r = sliders['r'].val
    d_beta = sliders['d_beta'].val
    
    # Update x-axis range dynamically based on r
    x_max = r * 1.2
    z_values = np.linspace(0, x_max, 500)
    ax.set_xlim(0, x_max)
    
    # Calculate values
    phi_values = phi_alpha(z_values)
    psi_values = psi_alpha(z_values)
    psi_beta_values = psi_beta(z_values)
    phi_beta_values = phi_beta(z_values)
    # Update plots based on view mode and show flags
    if show_phi_alpha and (view_mode in ['Both', 'Gradient']):
        phi_line.set_data(z_values, phi_values)
        phi_line.set_visible(True)
    else:
        phi_line.set_visible(False)
        
    if show_psi_alpha and (view_mode in ['Both', 'Potential']):
        psi_line.set_data(z_values, psi_values)
        psi_line.set_visible(True)
    else:
        psi_line.set_visible(False)
    
    if show_psi_beta and (view_mode in ['Both', 'Potential']):
        psi_beta_line.set_data(z_values, psi_beta_values)
        psi_beta_line.set_visible(True)
    else:
        psi_beta_line.set_visible(False)

    if show_phi_beta and (view_mode in ['Both', 'Gradient']):
        phi_beta_line.set_data(z_values, phi_beta_values)
        phi_beta_line.set_visible(True)
    else:
        phi_beta_line.set_visible(False)

    phi_beta_line.set_visible(False)
    psi_beta_line.set_visible(False)
        
    
    # Auto-adjust y-axis limits if needed
    all_values = []
    if phi_line.get_visible():
        all_values.extend(phi_values)
    if psi_line.get_visible():
        all_values.extend(psi_values)
    if psi_beta_line.get_visible():
        all_values.extend(psi_beta_values)
    if phi_beta_line.get_visible():
        all_values.extend(phi_beta_values)
    
    if all_values:
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        padding = 0.1 * (max_val - min_val)
        ax.set_ylim(min_val - padding, max_val + padding)
    
    fig.canvas.draw_idle()

# --- Connect callbacks ---
def view_mode_callback(label):
    global view_mode
    view_mode = label
    update()

def show_callback(label):
    global show_phi_alpha, show_psi_alpha
    if label == 'φ_α (Gradient)':
        show_phi_alpha = not show_phi_alpha
    elif label == 'ψ_α (Potential)':
        show_psi_alpha = not show_psi_alpha
    update()

def reset_callback(event):
    for param_name, slider in sliders.items():
        slider.reset()
    update()

# Connect callbacks
for slider in sliders.values():
    slider.on_changed(update)
view_radio.on_clicked(view_mode_callback)
show_check.on_clicked(show_callback)
reset_button.on_clicked(reset_callback)

# Initial update
update()

# Show the plot
plt.show()
