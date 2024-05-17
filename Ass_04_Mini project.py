# Import necessary libraries
import cv2
import numpy as np
from scipy.signal import convolve2d
import pygame
import pygameZoom
import argparse
import time
import os

# Import Cython functions
from cython_functions import discrete_fourier_transform

# Define constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define functions

def canny_edge_detector(image, kernel_size=3, sigma=1):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Sobel operator
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Convolve the image with Sobel filters
    gradient_x = convolve2d(blurred, sobel_x, mode='same')
    gradient_y = convolve2d(blurred, sobel_y, mode='same')

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Compute gradient direction
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Non-maximum suppression
    gradient_magnitude_suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # Apply double thresholding and edge tracking by hysteresis
    edges = hysteresis_threshold(gradient_magnitude_suppressed, 50, 150)

    return edges.astype(np.uint8)


def non_maximum_suppression(gradient_magnitude, gradient_direction):
    rows, cols = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction = gradient_direction[i, j]

            # Define neighboring pixels based on gradient direction
            if (0 <= direction < np.pi / 8) or (15 * np.pi / 8 <= direction <= 2 * np.pi):
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
            elif (np.pi / 8 <= direction < 3 * np.pi / 8) or (9 * np.pi / 8 <= direction < 11 * np.pi / 8):
                neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
            elif (3 * np.pi / 8 <= direction < 5 * np.pi / 8) or (11 * np.pi / 8 <= direction < 13 * np.pi / 8):
                neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
            else:
                neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]

            # Suppress non-maximum pixels
            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]

    return suppressed


def hysteresis_threshold(gradient_magnitude, low_threshold, high_threshold):
    rows, cols = gradient_magnitude.shape
    strong_edges = (gradient_magnitude > high_threshold)
    weak_edges = (gradient_magnitude >= low_threshold) & (gradient_magnitude <= high_threshold)

    # Perform edge tracking by hysteresis
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j]:
                neighborhood = gradient_magnitude[i - 1:i + 2, j - 1:j + 2]
                if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                    strong_edges[i, j] = True

    return strong_edges


def extract_points(image):
    edges = canny_edge_detector(image)
    points = np.column_stack(np.where(edges > 0))
    return points


def draw_epicycles(screen, fourier_series, t, path, image_visibility, background=None):
    # Draw background image if specified
    if image_visibility == 'VISIBLE':
        screen.blit(background, (0, 0))

    # Draw path
    pygame.draw.aalines(screen, WHITE, False, path, 1)

    # Draw epicycles
    x, y = 0, 0
    for i, (a, b, frequency) in enumerate(fourier_series):
        prev_x, prev_y = x, y
        x += a * np.cos(frequency * t + b)
        y += a * np.sin(frequency * t + b)
        pygame.draw.line(screen, WHITE, (prev_x, prev_y), (x, y), 1)
        pygame.draw.circle(screen, WHITE, (int(prev_x), int(prev_y)), int(a), 1)

    return x, y


def main(image_path, static_path, reset_path, image_visibility, save_as_video, custom_recording, cycle_duration):
    # Initialize Pygame
    pygame.init()

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract points from image
    points = extract_points(image)

    # Convert points to Fourier series
    fourier_series = discrete_fourier_transform(points)

    # Initialize Pygame window
    screen_width, screen_height = 800, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Image to Fourier Series")

    # Load background image if specified
    if image_visibility == 'VISIBLE':
        background = pygame.image.load(image_path)
        background = pygame.transform.scale(background, (screen_width, screen_height))

    # Initialize variables
    path = []
    t = 0
    running = True
    recording = False

    # Start video recording if specified
    if save_as_video:
        filename = f"ImageToFourierSeries-{int(time.time())}.mp4"
        frame_rate = 30
        frame_size = (screen_width, screen_height)
        video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, frame_size)

    # Main loop
    while running:
        screen.fill(BLACK)

        # Draw epicycles and track path
        x, y = draw_epicycles(screen, fourier_series, t, path, image_visibility)
        path.append((x + screen_width / 2, y + screen_height / 2))

        # Save frame for video recording
        if save_as_video:
            pygame_image = pygame.surfarray.array3d(screen)
            pygame_image = np.flip(pygame_image, axis=0)
            pygame_image = cv2.cvtColor(pygame_image, cv2.COLOR_RGB2BGR)
            video_writer.write(pygame_image)

        # Update display
        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Update time
        t += 0.01

        # Reset path if specified
        if reset_path:
            path = []

        # End video recording if specified and one cycle is completed
        if save_as_video and not custom_recording and t >= 2 * np.pi:
            running = False

        # Control cycle duration
        pygame.time.delay(int(cycle_duration * 1000))

    # Release resources
    pygame.quit()

    # Release video writer if specified
    if save_as_video:
        video_writer.release()

    # Remove video file if it's empty (no frames recorded)
    if save_as_video and os.path.getsize(filename) == 0:
        os.remove(filename)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image to Fourier Series")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--static_path", action="store_true", help="Keep path static and not lose color over time")
    parser.add_argument("--reset_path", action="store_true", help="Reset path every cycle")
    parser.add_argument("--image_visibility", choices=['VISIBLE', 'NOT_VISIBLE'], default='NOT_VISIBLE',
                        help="Visibility of the background image")
    parser.add_argument("--save_as_video", action="store_true", help="Save result to MP4 file")
    parser.add_argument("--custom_recording", action="store_true",
                        help="End video recording only when window is closed, instead of one full cycle")
    parser.add_argument("--cycle_duration", type=float, default=30,
                        help="Duration of one cycle in seconds (default is 30 seconds)")
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args.image_path, args.static_path, args.reset_path, args.image_visibility, args.save_as_video,
         args.custom_recording, args.cycle_duration)
