import os
import numpy as np

def load_sudoku_puzzles(folder_path):
    sudoku_boards = []
    board_names = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                board = []
                for line in file:
                    # Handle both space-separated and non-space-separated formats
                    if ' ' in line.strip():
                        row = [int(char) if char.isdigit() else -1 for char in line.strip().split()]
                    else:
                        row = [int(char) if char.isdigit() else -1 for char in line.strip()]
                    board.append(row)
                sudoku_boards.append(np.array(board))
                board_names.append(os.path.splitext(file_name)[0])

    return sudoku_boards, board_names


def is_valid(board, row, col, num):
    for i in range(9):
        if board[row, i] == num or board[i, col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i, j] == num:
                return False
    return True

def fill_board(board):
    for row in range(9):
        for col in range(9):
            if board[row, col] == -1:
                nums = np.random.permutation(range(1, 10))
                for num in nums:
                    if is_valid(board, row, col, num):
                        board[row, col] = num
                        if fill_board(board):
                            return True
                        board[row, col] = -1
                return False
    return True

def generate_random_sudoku_with_holes(hole_count=50):
    board = np.full((9, 9), -1)

    fill_board(board)

    holes = np.random.choice(81, hole_count, replace=False)
    for hole in holes:
        row, col = divmod(hole, 9)
        board[row, col] = -1

    return board


if __name__ == "__main__":
    print(len(load_sudoku_puzzles("./sudoku_puzzles")[0]))