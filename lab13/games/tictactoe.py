import os
import numpy

from abstract_game import AbstractGame

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 0     ## player O as 0 and player X as 1

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.player

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 1     ## player O as 1 and player X as -1
        return self.get_observation()

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1, 0)
        board_player2 = numpy.where(self.board == -1, 1, 0)
        board_to_play = numpy.full((3, 3), self.player)
        return numpy.array([board_player1, board_player2, board_to_play], dtype="int32")
        
    def step(self, action):
        # action is a number 0-8 for declare row and column of position in the board
        row = action // 3
        col = action % 3

        # Check that the action is illegal action, unless the player should loss from illegal action.
        if not (action in self.legal_actions()):
            return self.get_observation(), -1, True

        # input the action of current player into the board
        self.board[row, col] = self.player

        # Check that the game is finished in 2 condition: have a winner, or no any moves left
        have_win = self.have_winner()
        done = have_win or len(self.legal_actions()) == 0

        # If have the winner, the current player should be a winner.
        reward = 1 if have_win else 0

        # change current player
        self.player *= -1

        return self.get_observation(), reward, done

    def legal_actions(self):
        legal = []
        for i in range(9):
            row = i // 3
            col = i % 3
            if self.board[row, col] == 0:
                legal.append(i)
        return legal

    def have_winner(self):
        # Horizontal and vertical checks
        for i in range(3):
            if (self.board[i, :] == self.player * numpy.ones(3, dtype="int32")).all():
                return True
            if (self.board[:, i] == self.player * numpy.ones(3, dtype="int32")).all():
                return True

        # Diagonal checks
        if (
            self.board[0, 0] == self.player
            and self.board[1, 1] == self.player
            and self.board[2, 2] == self.player
        ):
            return True
        if (
            self.board[2, 0] == self.player
            and self.board[1, 1] == self.player
            and self.board[0, 2] == self.player
        ):
            return True

        return False

    def render(self):
        """
        Display the game observation.
        """
        pass

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        pass

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        pass

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        
        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        pass