# main.py
import cube
import cube_solver

# NOTE: The cube_color_classifier.py isn't used its just extra data.

if __name__ == "__main__":
    my_cube = cube.CubeScanner()
    my_cube_solver = cube_solver.CubeSolver()
    
    my_cube.scan_cube()
    
    print(my_cube.cube)
    
    # my_cube.cube = {'U': [['Y', 'G', 'B'], ['G', 'W', 'G'], ['R', 'W', 'W']], 'D': [['O', 'W', 'O'], ['O', 'Y', 'R'], ['B', 'R', 'B']], 'F': [['G', 'O', 'O'], ['Y', 'B', 'G'], ['Y', 'R', 'Y']], 'B': [['Y', 'Y', 'R'], ['O', 'G', 'W'], ['W', 'B', 'W']], 'L': [['G', 'W', 'W'], ['B', 'R', 'B'], ['O', 'Y', 'G']], 'R': [['G', 'R', 'R'], ['O', 'O', 'B'], ['B', 'Y', 'R']]}
    
    solution = my_cube_solver.solve_cube(my_cube.cube)
    print(f"Solution to solve the cube {solution}")
    # my_cube.get_frame()
    # print(my_cube)