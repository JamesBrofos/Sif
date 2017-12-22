# Sif
A Library for inference with Gaussian processes.


## Planned Features

1. Support for both MAP and fully Bayesian inference over the kernel hyperparameters. This could again be accomplished via elliptical slice sampling; but is there something more general?
2. Ability to ascend the gradient with respect to the input (x) even when Bayesian posterior samples are obtained.
