import cv2
import numpy as np

class CubeScanner:
    def __init__(self):
        self.cube = {face: [['']*3 for _ in range(3)] for face in ['U','D','F','B','L','R']}
        self.cap = None
        self.color_ranges = {
            'R': ((0, 100, 100), (10, 255, 255)),
            'G': ((50, 100, 100), (70, 255, 255)),
            'B': ((100, 150, 0), (140, 255, 255)),
            'Y': ((20, 100, 100), (30, 255, 255)),
            'O': ((10, 100, 100), (20, 255, 255)),
            'W': ((0, 0, 200), (180, 30, 255)),
        }
        
    def detect_colors(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_grid = [['' for _ in range(3)] for _ in range(3)]
        h, w, _ = frame.shape
        cell_h, cell_w = h // 3, w // 3

        for i in range(3):
            for j in range(3):
                y = i * cell_h + cell_h // 2
                x = j * cell_w + cell_w // 2
                pixel = hsv[y, x]

                for color, (lower, upper) in self.color_ranges.items():
                    lower = np.array(lower)
                    upper = np.array(upper)
                    if cv2.inRange(np.array([[pixel]]), lower, upper)[0][0] != 0:
                        detected_grid[i][j] = color
                        break
        return detected_grid

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

        # Define the guide square in the center
        size = min(h, w) // 2  # size of the outer square
        top_left = ((w - size) // 2, (h - size) // 2)
        bottom_right = ((w + size) // 2, (h + size) // 2)

        # Draw 3x3 grid
        cell_size = size // 3
        for i in range(4):
            # horizontal lines
            y = top_left[1] + i * cell_size
            cv2.line(overlay, (top_left[0], y), (bottom_right[0], y), (0,0,0), 2)
            # vertical lines
            x = top_left[0] + i * cell_size
            cv2.line(overlay, (x, top_left[1]), (x, bottom_right[1]), (0,0,0), 2)

        # Add semi-transparent effect
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def capture_face(self, face_name):
        while True:
            frame = self.get_frame()
            frame_with_overlay = self.draw_overlay(frame)
            cv2.imshow(f'Scan {face_name}', frame_with_overlay)

            if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save face
                self.cube[face_name] = self.detect_colors(frame)
                print(f"{face_name} face captured:")
                for row in self.cube[face_name]:
                    print(row)
                cv2.destroyWindow(f'Scan {face_name}')
                break
            
    def scan_cube(self):
        self.setup_camera()
        for face in ['U','R','F','D','L','B']:
            self.capture_face(face)
        self.cap.release()
        cv2.destroyAllWindows()
        print("All faces captured!")