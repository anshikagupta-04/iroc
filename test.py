import cv2
import numpy as np

# Open a video capture object (0 corresponds to the default camera, adjust as needed)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection (using Canny edge detector in this case)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on their areas (customize as needed)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # Loop over the filtered contours
    for contour in filtered_contours:
        # Fit a bounding box to the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the size of the cube (assuming it is a square face)
        cube_size = max(w, h)

        # Display the size information
        cv2.putText(frame, f'Cube Size: {cube_size:.2f} pixels', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Cube Size Measurement', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
