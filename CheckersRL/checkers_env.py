
import numpy as np


class checkers_env:

    def __init__(self, board, player):

        self.board = board
        self.player = player
        self.count = 0

    def reset(self):
        self.board = np.array([[0, 1, 0, 1, 0, 1],
                      [1, 0, 1, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, -1, 0, -1, 0, -1],
                      [-1, 0, -1, 0, -1, 0]], dtype=int)
        self.player = 1
    def initialize_board(self):
        # 1 and -1 represent the pieces of two players 1 and -1
        self.count = 0
        board = np.zeros((6, 6),dtype=int)
        for i in range(2):
            for j in range(0, 6, 2):
                board[i][j + ((i+1) % 2)] = 1
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
        elif len(self.possible_actions(-1)) == 0 and len(self.possible_actions(1)) == 0:
            #print( "-1 pieces:", np.sum(board < 0.0))
            #print( "1 pieces:", np.sum(board > 0.0))
            #print("1 won", np.sum(board < 0.0) < np.sum(board > 0.0))
            if np.sum(board < 0.0) < np.sum(board > 0.0):
                return 1
            elif np.sum(board < 0.0) > np.sum(board > 0.0):
                return -1
            else:
                return 0
        #elif (len(self.possible_actions(-1)) == 0 and len(self.possible_actions(1)) == 1) or (len(self.possible_actions(-1)) == 1 and len(self.possible_actions(1)) == 0):
        #    if np.sum(board < 0.0) < np.sum(board > 0.0):
        #        return 1
         #   elif np.sum(board < 0.0) > np.sum(board > 0.0):
        #        return -1
        elif (len(self.possible_actions(self.player)) == 0 and len(self.possible_actions(-self.player)) > 1):
            return -self.player
        elif (len(self.possible_actions(-self.player)) == 0 and len(self.possible_actions(self.player)) > 1):
            return self.player
        #elif (len(self.possible_actions(-1)) > 1 and len(self.possible_actions(1)) ==0 ):
        #    return -1
       # elif len(self.possible_actions(-1)) == 0:
        #    return -1
        #elif len(self.possible_actions(1)) == 0:
        #    return 1
        else:
            return 0

    def step(self, action, player):
        reward = 0
        if type(action[0] == int) or type(action[0] == float) :
         row1, co1, row2, co2 = action
        #elif isinstance(action,list):
        #    row1, co1, row2, co2 = action[0]
        else:
            row1, co1, row2, co2 = action[0]
        if action in self.possible_actions(player):
            if self.piece_captured(self.board, action):
                reward += 2
                #print("piece captured!")
            self.board[row1][co1] = 0
            self.board[row2][co2] = player
            #self.get_piece(action)
            if self.game_winner(self.board) == player:
                #print("Player :",player, " won")
                reward += 12
            elif self.game_winner(self.board) == -player:
                #print("Player :",-player, " won")
                reward += -12
        else:
            print(self.possible_actions(player),"player: ", player)
            print(action)
            reward += -3
        self.count += reward
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
        #winner = self.game_winner(board)
        #if winner != 0:
         #   return True

        # no possible actions for both players
        self.board=board
        #print(len(self.possible_actions(1)) == 0 and len(self.possible_actions(-1)) == 0)
        return len(self.possible_actions(1)) == 0 and len(self.possible_actions(-1)) == 0
    def piece_captured(self, board, action):
     """
    Check if an opponent's piece is captured after a move.
    """
     if type(action[0] == int) or type(action[0] == float):
         start_row, start_col, end_row, end_col = action
     else:
         start_row, start_col, end_row, end_col = action[0]

    # Check if the move is a capture move (jump over opponent's piece)
     if abs(end_row - start_row) == 2 and abs(end_col - start_col) == 2:
        # Calculate the position of the captured piece
        captured_row = (start_row + end_row) // 2
        captured_col = (start_col + end_col) // 2

        # Check if there is an opponent's piece at the captured position
        if board[captured_row][captured_col] == -self.player:
            self.board[captured_row][captured_col] = 0
            return True

     return False