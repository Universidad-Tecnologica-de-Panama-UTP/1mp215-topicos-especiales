import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm

# Initial parameters
mu_init = 0
sigma_init = 1

# Prepare figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)  # Leave space for sliders

# Create the x range
x = np.linspace(-10, 10, 1000)
y = norm.pdf(x, mu_init, sigma_init)

# Plot the initial normal distribution
[line] = ax.plot(x, y, lw=2)
ax.set_title("Normal Distribution")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.grid(True)

# Set axis limits
ax.set_xlim(-10, 10)
ax.set_ylim(0, 0.5)

# Create sliders
# Slider for mu (mean)
ax_mu = plt.axes([0.1, 0.1, 0.8, 0.03])  # [left, bottom, width, height]
slider_mu = Slider(ax_mu, 'Mean (μ)', -10, 10, valinit=mu_init)

# Slider for sigma (std dev)
ax_sigma = plt.axes([0.1, 0.15, 0.8, 0.03])
slider_sigma = Slider(ax_sigma, 'Std Dev (σ)', 0.1, 5, valinit=sigma_init)

# Update function
def update(val):
    mu = slider_mu.val
    sigma = slider_sigma.val
    y = norm.pdf(x, mu, sigma)
    line.set_ydata(y)
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_mu.on_changed(update)
slider_sigma.on_changed(update)

# Show window
plt.show()
