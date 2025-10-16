import cv2
import numpy as np
from cube_color_classifier import CubeColorClassifier  # Import ML classifier

class CubeScanner:
    def __init__(self):
        self.cube = {face: [['']*3 for _ in range(3)] for face in ['U','D','F','B','L','R']}
        self.cap = None
        self.classifier = CubeColorClassifier()  # Initialize ML classifier

    def get_color(self, avg_hsv):
        """Use ML classifier instead of static HSV ranges"""
        return self.classifier.predict_color(avg_hsv)

    def detect_colors(self, frame):
        h, w, _ = frame.shape

        # Define the central square (same as your overlay)
        size = min(h, w) // 2
        top_left_x = (w - size) // 2
        top_left_y = (h - size) // 2
        cell_size = size // 3
        accuracy = 1  # pixel step

        # Convert frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        grid = [['' for _ in range(3)] for _ in range(3)]

        # Loop through 3x3 grid cells
        for row in range(3):
            for col in range(3):
                cell_x = top_left_x + col * cell_size
                cell_y = top_left_y + row * cell_size

                hsv_sum = np.array([0, 0, 0], dtype=np.float32)
                count = 0

                for y in range(cell_y, cell_y + cell_size, accuracy):
                    for x in range(cell_x, cell_x + cell_size, accuracy):
                        hsv_sum += hsv_frame[y, x]
                        count += 1

                avg_hsv = (hsv_sum / count).astype(int)
                color_letter = self.get_color(avg_hsv)

                grid[row][col] = color_letter
                print(f"Center cell average HSV: {tuple(avg_hsv)} -> {color_letter}")

        return grid

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame

    def draw_overlay(self, frame):
        overlay = frame.copy()
        h, w, _ = frame.shape
        size = min(h, w) // 2
        top_left = ((w - size) // 2, (h - size) // 2)
        bottom_right = ((w + size) // 2, (h + size) // 2)
        cell_size = size // 3

        for i in range(4):
            y = top_left[1] + i * cell_size
            cv2.line(overlay, (top_left[0], y), (bottom_right[0], y), (0,0,0), 2)
            x = top_left[0] + i * cell_size
            cv2.line(overlay, (x, top_left[1]), (x, bottom_right[1]), (0,0,0), 2)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def capture_face(self, face_name):
        while True:
            frame = self.get_frame()
            frame_with_overlay = self.draw_overlay(frame)
            cv2.imshow(f'Scan {face_name}', frame_with_overlay)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                self.cube[face_name] = self.detect_colors(frame)
                print(f"{face_name} face captured:")
                for row in self.cube[face_name]:
                    print(row)
                cv2.destroyWindow(f'Scan {face_name}')
                break

    def scan_cube(self):
        self.setup_camera()
        for face in ['U','R','F','D','L','B']:
            print(f"Please scane the {face} face.")
            self.capture_face(face)
        self.cap.release()
        cv2.destroyAllWindows()
        print("All faces captured!")