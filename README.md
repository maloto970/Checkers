<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Checkers-playing AI based on reinforcement learning and AlphaGo

Final project for the Building AI course

## Summary

The ultimate benchmark of any AI has since long been how it compares against humans. Games have thus historically been popular subjects for evaluating and implementing AI algorithms and ideas.
All games aren't created equal, however. Some games won't offer enough complexity for the 'AI' to even reasonable have earnt the title AI, example being 3x3 tic-tac-toe solveable with
a few easy rules (if-statements). Hence the interest in games such as Chess where you would resort to search algorithms such as MinMax. MinMax builds parts of the game tree, all possible outcomes from current
position, which is used to recursively determine the best move available based on what it leads to further down the tree. A few years ago, Google took the game-playing
AI field to the next level by beating the Go world champion. Go requires what, compared to MinMax "brute-force" algorithms, may informally be recognized as closer to an intuitive understanding
of the game due to the unmanageable game tree size. This project experiments with the solution used by Google to pick apart each of its components (Neural networks and tree search algorithms)
with the purpose of gaining a better understanding of where the "intuition" comes from: can the neural network learn to play a game on its own, at least for the simpler game Checkers? Which
parts of the tree search algorithm (MCTS) can be trimmed to still achieve superhuman performance? At the very least, this project may give the practical insight of when less advanced
AI solutions, MinMax in this case, beats even the very latest method from Google - motivating how crucial it is to understand your problem before applying method.


## Background

This idea is on the more theoretical side but may resolve some unknowns at least for myself. Neural networks are mystified things even for some AI experts, due to its black-box nature. Combined
with the remarkable results in many areas (e.g. 100% realistic synthesis of human faces) it's understandable the method is sometimes treated as something supernatural or even magic.
Recently I believe the research has begun to settle on the fact that neural network not at all are capable of "thinking" or "planning", but rather are pattern learning machines. 
I want observe this myself, since I believe it to be healthy experiencing any method or technology fail as much as succeed. Demystifying neural networks may be achieved in this project
by gradually moving away from the proven AlphaGo-model, leaving more responsibility for the neural network as opposed to the classic AI-algorithm of Monte Carlo Tree Search.


## How is it used?

Ideally the project is built with the use-case in mind. This would involve the ability to control which AlphaGo components to switch off or include. Also, observing the model train and 
develop an understanding would be a nice feature, possibly highlighting interesting games along the way (by some measure, e.g. evaluated by a Checkers AlphaGo-trained model).

## Challenges

What does your project _not_ solve? Which limitations and ethical considerations should be taken into account when deploying a solution like this?

Since the project is more suited for educational purposes, there are few ethical considerations. Perhaps it's important to not claim observations from the program to be proof of anything.
The project does not solve whether or not neural networks on their own can build an intuition for games, but rather helps demystifying them and giving credit to old school AI algorithms (technically speaking, neural networks are also old school, i know...).

## What next?

Future steps could involve the expansion to more games, building a game-independent library to train AIs on user-provided games.

## Acknowledgments

* Google AlphaGo
* IBM DeepBlue
