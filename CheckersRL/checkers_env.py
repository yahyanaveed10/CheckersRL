
import numpy as np


class checkers_env:

    def __init__(self, board, player):

        self.board = board
        self.player = player

    def reset(self):
        self.board = np.array([[1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, -1, 0, -1, 0, -1],
                      [-1, 0, -1, 0, -1, 0]], dtype=int)
        self.player = 1
    def initialize_board(self):
        # 1 and -1 represent the pieces of two players 1 and -1
        board = np.zeros((6, 6),dtype=int)
        for i in range(2):
            for j in range(0, 6, 2):
                board[i][j + (i % 2)] = 1
                board[6 - i - 1][j + (i % 2)] = -1
        return board

    def possible_pieces(self, player):
        positions = []
        for i, row in enumerate(self.board):
            for j, value in enumerate(row):
                if value == player:
                    positions.append([i,j])
        return positions

    def possible_actions(self, player):
        def is_valid_position(x, y):
            return 0 <= x < 6 and 0 <= y < 6
        actions = []
        starters = self.possible_pieces(player)
        directions = [(1, -1), (1, 1)] if player == 1 else [(-1, -1), (-1, 1)]
        for x,y in starters:
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                if is_valid_position(nx, ny):
                    if self.board[nx][ny] == 0:
                    # one-step
                        actions.append([x, y, nx, ny])
                    elif self.board[nx][ny] == -player:
                    # one jump
                        jx, jy = x+2*dx, y+2*dy
                        if is_valid_position(jx, jy):
                            if self.board[jx][jy] == 0:
                                actions.append([x, y, jx, jy])
        return actions

    def get_piece(self, action):
        if action[2] - action[0] > 1:
            # jump
            self.board[(action[0]+action[2])/2][(action[1]+action[3])/2] = 0
    def count_board(self,board):
        positive_count = 0
        negative_count = 0
        # Loop through each element in the array
        for row in board:
            for cell in row:
                if cell > 0.0:
                    positive_count += 1
                if cell < 0.0:
                    negative_count += 1
        return positive_count,negative_count
    def game_winner(self, board):
        #positive_count,negative_count = self.count_board(board)
        if np.sum(board < 0.0) == 0:
            return 1
        elif np.sum(board > 0.0) == 0:
            return -1
        elif len(self.possible_actions(-1)) == 0:
            return -1
        elif len(self.possible_actions(1)) == 0:
            return 1
        else:
            return 0

    def step(self, action, player):
        if type(action[0] == int) or type(action[0] == float) :
         row1, co1, row2, co2 = action
        else:
            row1, co1, row2, co2 = action[0]
        if action in self.possible_actions(player):
            self.board[row1][co1] = 0
            self.board[row2][co2] = player
            self.get_piece(action)
            if self.game_winner(self.board) == player:
                reward = 1
            else:
                reward = 0
        else:
            reward = 0

        return [self.board, reward]

    def render(self):
        for row in self.board:
            for square in row:
                if square == 1:
                    piece = "|0"
                elif square == -1:
                    piece = "|X"
                else:
                    piece = "| "
                print(piece, end='')
            print("|")
    def is_terminal(self, board):
        """
checks if the state reaches the end
        """
        winner = self.game_winner(board)
        if winner != 0:
            return True

        # no possible actions for both players
        return len(self.possible_actions(1)) == 0 and len(self.possible_actions(-1)) == 0
