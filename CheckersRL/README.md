# Checkers
Implemented Q learning algorithm.
Main method runs the algorithm with a twist of using optimal policies in teh end to help the algorithm, because due to too many states its nearly difficult to award a reward for the initial moves. Thus the optimal policy method in solver.py is used to check which sequence of actions could lead to the highest rewards from all the games it played.
The main method uses episodes and number of games to track when to use the optimal policy. It is preseted that in the last game of the episode it uses the optimal policy to test what the q learning algorithm has learned and which creates the graph.
So per episode, the last game is tests the algorithm.
The classes related to Neural1Func and NNsolver and CNN are not used neither are they implemented.