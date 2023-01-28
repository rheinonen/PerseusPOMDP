# PerseusPOMDP
This is an implementation of the Perseus solver (Spaan and Vlassis 2005) for POMDPs, specifically written for the olfactory navigation problem.
## Basic classes
A POMDP is defined by an environment class, which should implement, at a minimum, the following attributes:
- `gamma` (float)
- `numactions` (integer)
- `numobs` (integer)
- `obs` (list of possible observations)
- `actions` (list of possible actions)

and the following functions:
- `stepInTime()`: allows for dynamic environments, for our purposes usually trivial
- `get_g(alpha,action)`: implements Eq. 10 in Spaan and Vlassis
- `getObs(state)` 
- `getReward(state,action)`

It is often useful to also implement:
- `transition(state,action)`: yields a new state after the agent takes an action
- `transition_function(belief,action)`: returns obs probabilities ${\rm Pr}(o|b,a)$ and Bayes updates $\tau(b,a,o)$. used primarily for backup prioritization (Shani et al. 2006).

A number of environments are implemented in environment.py, most notably SimpleEnv2D, which is the basic class for 2D olfactory search. Other benchmark POMDPs (TigerGrid, Bayesian Bandits, and Tag) are also implemented but have not been tested recently, so no guarantees are made concerning their continued functionality.

To explore the environment, you need an Agent class member, which takes actions, makes observations, and maintains a belief over states. A function `updateBelief(obs)' must be implemented on a case-by-case basis in order to tell the agent how to do the Bayesian update.

The Agent needs a policy. Policy classes are implemented in policy.py and include many heuristics, as well as policies which are greedy with respect to a value function. The basic function to implement for a policy is `getAction(belief)`.

## Perseus
Perseus is implemented in perseus_redux.py. The basic object is a ValueFunction, which consists at its core of a collection of alpha-vectors, called `alpha_vec` in the code. The function `computeOptimal()` performs some number of backup iteration steps, terminating either by some choice of convergence tolerance or simply for a specified number of iterations. It is also possible to load previously computed value functions, either using `load(file)` (from a pickled ValueFunction object) or `load_alphas(list)` (from a list of alpha_vec objects).

## Usage
Two examples of usage are given in main.py, used to construct policies in an isotropic environment with multiple possible detections, and main_windy.py, used to construct policies in a windy environment with one possible detection.

## POMDP files
A few 2D search problems have also been included as .pomdp files, using Tony Cassandra's format. These problems are benchmarked in Loisy and Heinonen (2023). 
