# cube.py
import cv2
import numpy as np
from color_classifier import CubeColorClassifier # If you want to change the way that the color is extracted from the image create a new object that can do that with the frame then create the predict function you can read through mine in the color_classifier.py
from collections import Counter

class CubeScanner:
    def __init__(self):
        # Cube dictionary: 6 faces, each a 3x3 grid
        self.cube = {face: [['']*3 for _ in range(3)] for face in ['U','D','F','B','L','R']}
        self.cap = None
        self.classifier = CubeColorClassifier()  # Load existing data if present

    def get_color(self, avg_hsv, training=False):
        if training:
            print(f"Detected HSV: {avg_hsv}")
            color = input("Enter the correct color letter (R/G/B/O/Y/W): ").upper()
            # color = "O" just comment out the line about in order to train it quickly if you have a solved cube so that you dont have to identify each individual pieces sticker color
            self.classifier.add_training_sample(avg_hsv, color)
            return color
        else:
            return str(self.classifier.predict(avg_hsv))

    def detect_colors(self, frame, training=False):
        h, w, _ = frame.shape
        size = min(h, w) // 2
        top_left_x = (w - size) // 2
        top_left_y = (h - size) // 2
        cell_size = size // 3
        accuracy = 1

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grid = [['' for _ in range(3)] for _ in range(3)]

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
                color_letter = self.get_color(avg_hsv, training)
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

    def capture_face(self, face_name, training=False):
        while True:
            frame = self.get_frame()
            frame_with_overlay = self.draw_overlay(frame)
            cv2.imshow(f'Scan {face_name}', frame_with_overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # print("Capturing 6 frames for color averaging...")
                grids = []
                for _ in range(6):
                    f = self.get_frame()
                    grid = self.detect_colors(f, training)
                    grids.append(grid)
                    cv2.waitKey(100)  # ~100ms between frames

                # Compute the most common color for each cell
                final_grid = [['' for _ in range(3)] for _ in range(3)]
                for row in range(3):
                    for col in range(3):
                        cell_colors = [grids[i][row][col] for i in range(6)]
                        final_grid[row][col] = Counter(cell_colors).most_common(1)[0][0]

                self.cube[face_name] = final_grid
                print(f"{face_name} face captured:")
                for row in final_grid:
                    print(row)

                cv2.destroyWindow(f'Scan {face_name}')
                if(not training):
                    correct_check = input("Is this correct (y/n):").lower()
                    if(correct_check == "y"):
                        break
                else:
                    break

    def scan_cube(self):
        # mode = input("Enter mode (train/solve): ").strip().lower() If you want to train the model uncomment this line to prompt for the mode or change the variable below to the "train" mode
        mode = "solve"  # or "train"
        training = mode == "train"

        self.setup_camera()
        face_order = ['U','R','F','D','L','B']

        for face in face_order:
            print(f"Please scan the {face} face. Press 's' to capture.")
            self.capture_face(face, training)

        self.cap.release()
        cv2.destroyAllWindows()
        print("All faces captured!")

        # Define valid cube colors
        valid_colors = {'R', 'G', 'B', 'Y', 'W', 'O'}

        # Auto correction after scan
        all_detected_colors = set(c for face in face_order for row in self.cube[face] for c in row)
        print("\nDetected colors before correction:", all_detected_colors)

        # Detect invalid colors
        invalid_colors = [c for c in all_detected_colors if c not in valid_colors]
        if invalid_colors:
            print("Invalid colors detected:", invalid_colors)
            print("Attempting automatic correction...")

            # Replace invalid ones with the most common valid color
            all_valid = [c for face in face_order for row in self.cube[face] for c in row if c in valid_colors]
            if all_valid:
                most_common = max(set(all_valid), key=all_valid.count)
            else:
                most_common = 'W'  # default fallback

            for face in face_order:
                for i in range(3):
                    for j in range(3):
                        if self.cube[face][i][j] not in valid_colors:
                            self.cube[face][i][j] = most_common

            print(f"Replaced invalid colors with '{most_common}'")

        print("Detected colors after correction:",
            {c for face in face_order for row in self.cube[face] for c in row})

        if training:
            # Save the new training data alongside existing data
            self.classifier.save_model()
            print("Training data saved and model updated!")
        else:
            print("Cube scanned successfully and corrected!")
