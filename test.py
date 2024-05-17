import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import tau
from scipy.integrate import quad_vec
from tqdm import tqdm
import matplotlib.animation as animation

def interpolate(t, t_data, x_data, y_data):
    return np.interp(t, t_data, x_data + 1j * y_data)

image_path = "./images/nepal.png"
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image_gray, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = np.array(contours[0])

x_data, y_data = contours[:, :, 0].reshape(-1, ).astype(float), -contours[:, :, 1].reshape(-1, ).astype(float)
x_data -= np.mean(x_data)
y_data -= np.mean(y_data)

plt.plot(x_data, y_data)
plt.title("Contour Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

t_data = np.linspace(0, tau, len(x_data))
order = 100

print("Generating Fourier coefficients...")
fourier_coeffs = []
with tqdm(total=(order * 2 + 1), desc="Progress") as progress_bar:
    for n in range(-order, order + 1):
        coef = 1 / tau * quad_vec(lambda t: interpolate(t, t_data, x_data, y_data) * np.exp(-n * t * 1j), 0, tau, limit=100, full_output=1)[0]
        fourier_coeffs.append(coef)
        progress_bar.update(1)
print("Completed generating coefficients.")

fourier_coeffs = np.array(fourier_coeffs)
np.save("fourier_coeffs.npy", fourier_coeffs)

draw_x, draw_y = [], []

fig, ax = plt.subplots()
circles = [ax.plot([], [], 'r-')[0] for _ in range(-order, order + 1)]
circle_lines = [ax.plot([], [], 'b-')[0] for _ in range(-order, order + 1)]
drawing, = ax.plot([], [], 'k-', linewidth=2)
orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)

ax.set_xlim(np.min(x_data) - 200, np.max(x_data) + 200)
ax.set_ylim(np.min(y_data) - 200, np.max(y_data) + 200)
ax.set_axis_off()
ax.set_aspect('equal')

print("Compiling animation...")
frames = 300
with tqdm(total=frames, desc="Animation Progress") as progress_bar:
    def sort_coeffs(coeffs):
        sorted_coeffs = [coeffs[order]]
        for i in range(1, order + 1):
            sorted_coeffs.extend([coeffs[order + i], coeffs[order - i]])
        return np.array(sorted_coeffs)

    def animate_frame(i, time, coeffs, progress_bar):
        t = time[i]
        exp_term = np.array([np.exp(n * t * 1j) for n in range(-order, order + 1)])
        coeffs = sort_coeffs(coeffs * exp_term)
        x_coeffs = np.real(coeffs)
        y_coeffs = np.imag(coeffs)
        center_x, center_y = 0, 0
        for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
            r = np.linalg.norm([x_coeff, y_coeff])
            theta = np.linspace(0, tau, num=50)
            x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
            circles[i].set_data(x, y)
            x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
            circle_lines[i].set_data(x, y)
            center_x, center_y = center_x + x_coeff, center_y + y_coeff
        draw_x.append(center_x)
        draw_y.append(center_y)
        drawing.set_data(draw_x, draw_y)
        orig_drawing.set_data(x_data, y_data)
        progress_bar.update(1)

    time = np.linspace(0, tau, num=frames)
    animation_frames = animation.FuncAnimation(fig, animate_frame, frames=frames, fargs=(time, fourier_coeffs, progress_bar), interval=5)
    plt.show()
    print("Animation compilation completed.")
