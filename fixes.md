**P1** my agent has a problem called [[Distribution Shift]], which is basically train on a subject and tesst on another. Of course, this is not fair. the teacher has full vision, while my agent has partial

more data can help [[overfitting]], but more data from the same wrong distribution won't help

i need to learn more on [[Convolutional Neural Networks]].

**P2** My agent also have an [[overfitting]] problem since during the flatten process of the CNN, it flatten the whole environment into individual grids, and grids alone are similiar. For example, a task at (1,2) is treated the same as a task at (4,2). therefore, the model is likely to only learn "tasks", and not "taks A", task "B",... so it overfits instead of generalizes

*there should be a seperate weight for every grid*

**P3** during training, i do not disclose the agent's own position, it is like using a map but not knowing where I currently am. The agent also does not know anything about its own direction.

this creates a situation when my model cannot be more accurate than the input itself. It is like given the pronunciation of a word, and I cannot know what that word is meant without knowing the context. I only know how to spell that word.

*training an agent should give all information about the agent itself as the input*

**P4** rewards are shared. when a teammate completes a task, the whole team gets +1 reward so it is hard for our agent to learn from rewards since the game could be complete by teammates when the agent is wandering around. it is best to be able to *distinguish between your own reward and the team's reward*

**P5** my input needs to be more detailed:
- teammates : should be levels for grids with teammates and 0 for grids without teammates *how to know where i am?*
- tasks: same as teammates.
- consider changing the inputs to the same as those in the branch posturop1010.
