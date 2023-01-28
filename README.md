# PerseusPOMDP
This is an implementation of the Perseus solver (Spaan and Vlassis 2005) for POMDPs, specifically written for the olfactory navigation problem.
## Basic classes
A POMDP is defined by an environment class, which should implement:
1. get_reward(), a function that returns the reward as a function of the state and action
2. 
