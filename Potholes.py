import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Conversion factor from pixels to millimeters
pixels_to_mm = 0.0011  # Example: 1 pixel = 0.0011 mm (replace this with your actual value)

def detect_potholes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def measure_pothole_size(contours):
    max_width = 0
    max_height = 0

    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Increased threshold to filter larger contours
            # Fit a bounding box around the contour
            _, _, w, h = cv2.boundingRect(contour)
            if w > max_width:
                max_width = w
            if h > max_height:
                max_height = h

    return max_width, max_height

def main(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize the video writer
    video_writer = cv2.VideoWriter(os.path.join(output_dir, 'detected_potholes.mp4'), fourcc, fps, (width, height))

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        contours = detect_potholes(frame)
        max_width, max_height = measure_pothole_size(contours)

        # Convert max width and height from pixels to millimeters
        max_width_mm = max_width * pixels_to_mm
        max_height_mm = max_height * pixels_to_mm

        # Print the pothole size in the terminal
        print(f'Max pothole size in current frame: {max_width_mm:.2f} mm x {max_height_mm:.2f} mm')

        overlay = frame.copy()

        # Draw contours with a thick red line and annotate the size
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                hull = cv2.convexHull(contour)
                cv2.polylines(overlay, [hull], isClosed=True, color=(0, 0, 255), thickness=5)  # Broad red polyline, thickness 5
                x, y, w, h = cv2.boundingRect(contour)
                width_mm = w * pixels_to_mm
                height_mm = h * pixels_to_mm
                cv2.putText(overlay, f'{width_mm:.2f} mm x {height_mm:.2f} mm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Add transparency to the overlay
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Save the frame with detected potholes
        output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(output_path, frame)

        # Write the frame to the video writer
        video_writer.write(frame)

        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame with detected potholes
        ax.imshow(frame_rgb)
        plt.draw()
        plt.pause(0.001)
        ax.clear()

        frame_count += 1

    cap.release()
    video_writer.release()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    video_path = 'pothole.mp4'  # Replace with the path to your video file
    output_dir = 'pothole_frames'  # Directory to save the annotated frames and video
    main(video_path, output_dir)
