# cube_solver.py
from collections import Counter
import sys
import kociemba

VALID_COLORS = {'R', 'G', 'B', 'Y', 'W', 'O'}  # your scanned color letters

# Kociemba expects face letters in this order: U, R, F, D, L, B
KOCIEMBA_FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']

class CubeSolver:
    def __init__(self):
        # lazy import so module can be imported even if kociemba not installed
        try:
            self.kociemba = kociemba
        except Exception:
            self.kociemba = None

    def _collapse_sticker(self, sticker):
        """
        sticker may be:
          - a single color letter (e.g. 'R')
          - or an iterable/list of candidates (e.g. ['R','R','O','R','?'])
        Return a single color letter (most common). If unknown, return None.
        """
        if isinstance(sticker, (list, tuple)):
            # filter to known colors, then take most common
            filtered = [s for s in sticker if s in VALID_COLORS]
            if not filtered:
                # try to pick most common even if invalid values present
                cand = Counter(sticker).most_common(1)
                return cand[0][0] if cand else None
            return Counter(filtered).most_common(1)[0][0]
        else:
            return sticker if sticker in VALID_COLORS else None

    def _normalize_cube_state(self, cube_state):
        """
        Return a cleaned copy of cube_state where each sticker is a single color letter.
        If a sticker is missing/unknown, it becomes None.
        """
        norm = {}
        for face, grid in cube_state.items():
            norm_grid = []
            for row in grid:
                norm_row = []
                for sticker in row:
                    norm_row.append(self._collapse_sticker(sticker))
                norm_grid.append(norm_row)
            norm[face] = norm_grid
        return norm

    def _validate_and_build_map(self, norm_state):
        """
        Build {scanned_color -> Kociemba face letter} mapping using centers.
        Validate centers contain the 6 unique colors. If not unique/missing, return (None, error_msg)
        """
        # get centers
        try:
            centers = {face: norm_state[face][1][1] for face in KOCIEMBA_FACE_ORDER}
        except KeyError as e:
            return None, f"Missing face in cube_state: {e}"

        # Check for missing center readings
        if any(c is None for c in centers.values()):
            return None, f"One or more center stickers are unknown: {centers}"

        # Check centers are among VALID_COLORS
        center_set = set(centers.values())
        if not center_set.issubset(VALID_COLORS):
            return None, f"Invalid center colors detected: {centers}"

        # Ideally we want 6 unique center colors
        if len(center_set) != 6:
            # attempt to fix by picking most common color overall among centers, but prefer requiring unique
            return None, f"Center colors are not unique (found: {center_set}). Make sure each face center is a distinct color."

        # Map scanned color letter -> Kociemba face letter
        color_to_face = {centers[face]: face for face in KOCIEMBA_FACE_ORDER}
        return color_to_face, None

    def convert_to_kociemba_string(self, cube_state):
        """
        Converts cube_state (dict of 6 faces each 3x3) into Kociemba 54-char string.
        Returns (kociemba_string, None) on success, or (None, error_message) on failure.
        """
        # 1) Collapse lists into single value if necessary
        norm = self._normalize_cube_state(cube_state)

        # 2) Build mapping from scanned color -> face letter (U/R/F/D/L/B) using centers
        color_map, err = self._validate_and_build_map(norm)
        if err:
            return None, err

        # 3) Build the 54-character facelet string
        kociemba_chars = []
        for face in KOCIEMBA_FACE_ORDER:
            grid = norm[face]
            for row in grid:
                for sticker in row:
                    if sticker is None:
                        return None, f"Found unknown sticker on face {face}: {grid}"
                    if sticker not in color_map:
                        return None, f"Scanned color '{sticker}' on face {face} does not match any center color."
                    kociemba_chars.append(color_map[sticker])

        if len(kociemba_chars) != 54:
            return None, "Internal error: resulting facelet length != 54"

        return ''.join(kociemba_chars), None

    def solve_cube(self, cube_state):
        """
        Returns a tuple: (solution_string, error_message)
        If solution_string is not None then error_message is None.
        """
        # if self.kociemba is None:
        #     return None, ("kociemba package not installed. Install with:\n"
        #                   "    pip install kociemba")

        kociemba_string, err = self.convert_to_kociemba_string(cube_state)
        if err:
            return None, f"Conversion error: {err}"

        # sanity-check string length
        if not kociemba_string or len(kociemba_string) != 54:
            return None, f"Invalid Kociemba string: {kociemba_string!r}"

        try:
            solution = self.kociemba.solve(kociemba_string)
            return solution, None
        except Exception as exc:
            return None, f"kociemba failed: {exc}"
