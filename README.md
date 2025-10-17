# Rubik's Cube Solver

## Preview
<!-- <img src="https://raw.githubusercontent.com/BenjaminBurnell/Lyrify/refs/heads/main/assets/ezgif-4e32b50b9ea37f.gif"></img> -->

## Overview

The **Rubik's Cube Solver** is a project that merges **computer vision** and **machine learning** to get the positions of all pieces on a physical 3x3 Rubik's Cube and give you the moves to solve it in real-time. It uses your webcam to scan the cube's state, a custom-trained **K-Nearest Neighbors (KNN)** model for accurate color classification, and the **Kociemba algorithm** to produce the optimal, shortest solution path.

---

## Features

* **Real-time Webcam Scanning** — Utilizes **OpenCV** (`cv2`) to capture the state of all six faces of your cube directly from a live video feed.
* **Intelligent Color Classification** — Employs a **K-Nearest Neighbors (KNN)** model (stored in `cube_color_model.pkl`) to accurately identify Red, Green, Blue, Yellow, White, and Orange stickers. This model uses **HSV** (Hue, Saturation, Value) data, which is more robust to lighting changes.
* **Optimal Solving Algorithm** — Integrated with the renowned **Kociemba two-phase algorithm** to find a near-minimal solution string.
* **Custom Retraining** — Users can easily generate new training data and retrain the color model to adapt to different lighting conditions or unique cube shades.
* **Modular Design** — Cleanly separated logic into `cube.py` (scanner/vision), `color_classifier.py` (ML model), and `cube_solver.py` (algorithm handling).

---

## How It Works

This project operates through a seamless three-part pipeline:

1.  **Scanning (`cube.py`)**: The `CubeScanner` opens the webcam. The user presents each of the six faces, and **OpenCV** extracts the average **HSV** values from **multiple frames** for the 9 stickers on the face, increasing accuracy.
2.  **Classification (`color_classifier.py`)**: These HSV values are passed to the `CubeColorClassifier`, which uses the pre-trained **KNN model** to predict the corresponding color (R, G, B, Y, W, O) for all 54 stickers.
3.  **Solving (`cube_solver.py`)**: The recognized 54-character state string is compiled and fed to the **Kociemba library**, which calculates the sequence of moves required to solve the cube in the fewest possible steps.

---

## Setup

These instructions will guide you through setting up and running the solver on your local machine.

### Prerequisites

* **Python 3.x**
* A working **webcam**

### 1. Clone the repository

```bash
git clone https://github.com/BenjaminBurnell/Cube-Solver
cd Cube-Solver
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows, use: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the main program

Execute the main script. Follow the on-screen prompts to present the faces to your webcam.

```bash
python main.py
```

## Retraining the Color Classifier

If you encounter color misclassification (e.g., orange being detected as red) due to unique cube shades or extreme lighting, you can easily retrain the model:

1.  **Delete the existing data files:**
    ```bash
    rm cube_color_model.pkl cube_training_data.pkl
    ```
2.  **Rerun the main script:** The `cube.py` module will detect the missing files and enter **training mode**, prompting you to manually identify the color of each sticker during the scan. This process generates new, custom training data specific to your environment.

---

## Built With

* **Python 3.x** — Core language for all components.
* **OpenCV (`cv2`)** — For webcam access, frame capture, and **HSV** color extraction.
* **scikit-learn (KNN)** — Provides the robust K-Nearest Neighbors algorithm for color classification.
* **Kociemba** — The library implementing the optimal two-phase algorithm for solving the Rubik's Cube state.
* **Joblib & Pickle** — Used for efficient serialization and deserialization of the trained machine learning model and training data.

---

## Future Enhancements

* Implement a **Graphical User Interface (GUI)** using Tkinter or PyQt for a more intuitive user experience.
* Add a **move visualization** feature to show the solution steps on a virtual cube.
* Enhance **Color Calibration** using a simple initial calibration pattern to dynamically adjust to different lighting conditions.

---

## License

MIT License © 2025 **Benjamin Burnell**

You’re free to use, modify, and distribute this project under the terms of the MIT License.

---

## Credits

Developed by **Benjamin Burnell**  
Powered by **OpenCV**, **scikit-learn**, and the Python **Kociemba** library.
