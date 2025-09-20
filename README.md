# Multi-Agent Pacman Project

This project is based on the **Multi-Agent Pacman project** from the University of California, Berkeley. 
The original project was designed for educational purposes in the course *CS 188: Introduction to Artificial Intelligence*. 
My work involves extending and optimizing the baseline agents, implementing additional algorithms, and comparing their performance.

## Modified Files

- **moveOrdering** – Move ordering algorithms  
- **multiAgents** – Various baseline agents  
- **multiAgentsHelper** – Utility functions used by agents (A*, BFS)  
- **otherAgents** – Optimized AlphaBeta agents  
- **transpositionTable** – Transposition table implementation  
- **cacheCapsuleAgents** – (Incomplete) Agent that computes a path and follows it  

## Agents Overview

This project includes several AI agents, each implementing a different search strategy and optimizations:

- **Minimax Agent**  
  The baseline agent using the standard Minimax algorithm. It explores all possible moves to a given depth without any pruning or move ordering.

- **AlphaBeta Agent**  
  Implements the Minimax algorithm with **Alpha-Beta pruning**. This optimization reduces the number of nodes explored by pruning branches that cannot influence the final decision, improving performance significantly.

- **PVS (Principal Variation Search) Agent**  
  A further optimized version of Alpha-Beta. PVS assumes that the first move considered is likely the best, allowing faster pruning of subsequent moves in the search tree.

- **PVSCached Agent**  
  Builds on PVS with additional optimizations, including **transposition tables** to store previously computed game states. This agent achieves the best performance among all, significantly reducing redundant calculations.
 

## Real-Time Performance Comparison

This section showcases a real-time comparison between the **Minimax Agent** and the **PVSCached Agent**, which is the most optimized agent in this project.

Both agents are configured with a search depth of 4.

![Minimax Agent](comparison/minimax_4.gif)
![PVSCached Agent](comparison/pvs_4.gif)

## Full Demo

Full demonstrations of all agents are available in the [demo folder](./full_demo/).  
There, you can observe the differences in execution speed and responsiveness between the various agents.

## Agent Statistics

Located in the [results](./results/):  

- **stats.txt** – View and compare the scores and execution times of different agents  
- **AlphaBeta_no_TT_vs_TT/** – Compare performance with and without a transposition table  
- **AlphaBeta_vs_PVSCached/** – Compare execution speed between the optimized AlphaBeta and the baseline version  
- **PVS_no_order_vs_order/** – Compare the impact of move ordering in AlphaBeta with PVS (Principal Variation Search) 