from action import Action
import numpy as np


# USER INPUTS:

arrLower = [3.5, 4.5, 5.4]
arrUpper = [10.5, 20.4, 30.5]

arrNumClasses = [4, 5, 10]

arrContAction = [10.5, 10.5, 30.5]

# Discretize actions

print("Continuous action values: ", arrContAction)

action = Action(
    arrLower=arrLower,
    arrUpper=arrUpper,
    arrNumClasses=arrNumClasses)
arrDiscAction = action.descretize(arrContAction)

print("Discretized action values: ", arrDiscAction)
