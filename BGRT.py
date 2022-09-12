'''
This class is an implementation of the Binary Guided Random Testing from:
Wei-Fan Chiang, Ganesh Gopalakrishnan, Zvonimir Rakamaric, and Alexey Solovyev. "Efficient search for inputs causing high floating-point errors". In PPoPP, 2014.
This implementation is a modification of https://github.com/tanmaytirpankar/Seesaw/blob/code_revamp/src/RandomTesting.py by Tanmay Tirpankar
'''

from sympy import Symbol, sin, parse_expr
from sympy.utilities import lambdify
from random import uniform, random
from sys import float_info
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np

class RandomTesting(object):
    """
    Class for randomized testing.
    Attributes
    ----------
    input_intervals : dict
            Dictionary(Symbol -> list[]) of input intervals.
    function :
        Symbolic Expression
    target_value : float
        Output value of the symbolic expression for which we need to find inputs.
    distant_value : float
        Value supposedly furthest away from target value
    Methods
    -------
    Notes
    -----
    It is assumed that the parameters to symbolic expression and the input variables match in terms of the count as
    well as names.
    """

    def __init__(self, input_intervals, function, target_value=0, distant_value=float_info.max):
        """
        Class initializer
        Parameters
        ----------
        input_intervals : 2D nparray
        target_value : float
            Output value of the function for which we need to find inputs.
        distant_value : float
            Value supposedly furthest away from target value
        Returns
        -------
        None
        """
        self.initial_configuration = input_intervals
        self.target_value = target_value
        self.distant_value = distant_value
        self.function = function

    # def __str__(self):
    #     """
    #     String representation of this class
    #     Example:
    #     >>> print(RandomTesting({Symbol('x'): [1,2], Symbol('y'): [2,3]}, sin(Symbol('x'))/Symbol('x'), target_value=0))
    #     Inputs  : {x: [1, 2], y: [2, 3]}
    #     Function: sin(x)/x
    #     Target  : 0
    #     Distant : 1.7976931348623157e+308
    #     """
    #     returning_str = "Inputs".ljust(8) + ": " + str(self.initial_configuration) + '\n'
    #     returning_str += "Function".ljust(8) + ": " + str(self.function) + '\n'
    #     returning_str += "Target".ljust(8) + ": " + str(self.target_value) + '\n'
    #     returning_str += "Distant".ljust(8) + ": " + str(self.distant_value)
    #
    #     return returning_str


class BinaryGuidedRandomTesting(RandomTesting):
    """
    Class for Binary Guided Randomized Testing
    Attributes
    ----------
    sampling_factor: int
        Number of times to sample from the input interval and evaluate.
    termination_criteria_iterations : int
        Number of iterations as the termination criteria of for this algorithm
    configuration_generation_factor : int
        Number of times to partition and generate new configurations. (Approximately at most 3 * number of input
        variables. No point keeping this too high due to redundant configuration possibility)
    restart_probability : float
        Probability of starting search from initial configuration.
    Methods
    -------
    binary_guided_random_testing
    generate_configurations
    partition_input_box
    evaluate
    print_output
    """
    def __init__(self, input_intervals, function, target_value=0, distant_value=0,
                 sampling_factor=10, termination_criteria_iterations=100, configuration_generation_factor=2,
                 restart_probability=0.05):
        """
        Class initializer
        Paramters
        ---------
        sampling_factor: int
            Number of times to sample from the input interval and evaluate.
        termination_criteria_iterations : int
            Number of iterations as the termination criteria of for this algorithm
        configuration_generation_factor : int
            Number of times to partition and generate new configurations. (Approximately at most 3 * number of input
            variables. No point keeping this too high due to redundant configuration possibility)
        restart_probability : float
            Probability of starting search from initial configuration. (Set to something less than 0.05 as you dont want
            restarts too often)
        Returns
        -------
        """
        super().__init__(input_intervals, function, target_value, distant_value)
        self.sampling_factor = sampling_factor
        self.termination_criteria_iterations = termination_criteria_iterations
        self.configuration_generation_factor = configuration_generation_factor
        self.restart_probability = restart_probability

    # def __str__(self):
    #     """
    #     String representation of this class
    #     Binary Guided Random Testing:
    #     Inputs  : {x: [1, 2], y: [2, 3]}
    #     Function: sin(x)/x
    #     Target  : 0
    #     Distant : 1.7976931348623157e+308
    #     Sampling: 10
    #     Iters   : 100
    #     New Conf: 2
    #     Restart : 0.05
    #     """
    #     returning_str = "Binary Guided Random Testing:\n"
    #     returning_str += super().__str__() + '\n'
    #     returning_str += "Sampling".ljust(8) + ": " + str(self.sampling_factor) + '\n'
    #     returning_str += "Iters".ljust(8) + ": " + str(self.termination_criteria_iterations) + '\n'
    #     returning_str += "New Conf".ljust(8) + ": " + str(self.configuration_generation_factor) + '\n'
    #     returning_str += "Restart".ljust(8) + ": " + str(self.restart_probability)
    #
    #     return returning_str

    def binary_guided_random_testing(self):
        """
        Gives the narrowest box and corresponding best values found from the given inputs for which symbolic expression
        gives value closest to target_value
        Unguided Random Testing method as implemented in "Efficient Search for Inputs Causing High Floating-point
        Errors"
        Returns
        -------
        (list, list)
            A tuple of a list of boxes for which symbolic expression gives best value closest to target_value and
            another list of best values found in those intervals
        """
        return_values_list = []
        return_input_interval_list = []

        # Initial Setup
        best_value = self.distant_value
        largest_difference = abs(self.distant_value-self.target_value)
        best_configuration = self.initial_configuration

        # Looping till termination criteria reached
        for i in range(self.termination_criteria_iterations):
            # Generating new configurations
            new_configurations = self.generate_configurations(best_configuration)

            # Evaluating symbolic expression for each configuration to find the configuration giving closest value to
            # target
            for input_configuration in new_configurations:
                new_value = self.evaluate(input_configuration)

                # If better value found, record it and the corresponding configuration
                if abs(new_value-self.target_value) > largest_difference:
                    largest_difference = abs(new_value-self.target_value)
                    best_value = new_value
                    best_configuration = deepcopy(input_configuration)

            # Restarting with some probability to allow exploring other intervals and not getting stuck in a rabbit
            # hole.
            # Add the best value and configuration to list before restarting
            if random() < self.restart_probability:
                # print("Restarting from initial configuration:")
                return_input_interval_list.append(deepcopy(best_configuration))
                return_values_list.append(best_value)
                best_configuration = deepcopy(self.initial_configuration)
                best_value = self.distant_value

        return_input_interval_list.append(deepcopy(best_configuration))
        return_values_list.append(best_value)
        # print(return_input_interval_list)
        # print(return_values_list)
        return return_input_interval_list, return_values_list

    def generate_configurations(self, input_configuration):
        """
        Generates input configurations by dividing input set, splitting into upper and lower and permuting intervals.
        Parameters
        ----------
        input_configuration : a 2D nparray with the lower bounds for each parameter in input_configuration[0]
        and upper bound in input_configuration[1]
        Returns
        -------
        ndarray
            A list of new input configurations
        """
        new_confs = []

        lower_bound = input_configuration[0]
        upper_bound = input_configuration[1]
        new_bound = np.mean(input_configuration, 0)

        for i in range(2 ** input_configuration[0].shape[0]):
            random_conf = np.random.rand(*new_bound.shape) > 0.5
            new_lower_bound = np.where(random_conf, new_bound, lower_bound)
            new_upper_bound = np.where(random_conf, upper_bound, new_bound)
            new_conf = np.array([new_lower_bound, new_upper_bound])
            new_confs.append(new_conf)

        return np.unique(new_confs, return_index=False, return_inverse=False, return_counts=False, axis=0)

    def evaluate(self, input_intervals):
        """
        Evaluates the symbolic expression by drawing inputs from the given input intervals randomly and returns result
        closest to target. The inputs are drawn from the half-open interval [low, high + 2**-53)
        For a set of variables X, selects inputs for var_i in X such that var_i_lower<=var_i<var_i_upper+2**-53
        Parameters
        ----------
        input_intervals : 2D nparray
        function: the function to evaluate
        Returns
        -------
        float
            Returns the closest value to target value obtained by random sampling
        """
        best_value = self.distant_value
        largest_difference = abs(self.distant_value - self.target_value)

        # Random sampling sampling_factor-1 more times
        for i in range(self.sampling_factor):
            # Values are drawn from the half-open interval [low, high + 2**-53)
            parameter_values = np.random.uniform(low=input_intervals[0], high=input_intervals[1])
            new_value = self.function(parameter_values)
            if abs(new_value-self.target_value) > largest_difference:
                largest_difference = abs(new_value-self.target_value)
                best_value = new_value
                # print(final_difference)

        return best_value

    def print_output(self, narrowed_inputs, best_values_found):
        """
        Prints the output of BGRT.
        Parameters
        ----------
        narrowed_inputs : list
            List of input configurations that may or may not be better than initial input configuration.
        best_values_found : list
            List of best values found for the given function on random sampling from the corresponding configuration
            from list of input configurations
        Returns
        -------
            Nothing
        """
        assert len(narrowed_inputs) == len(best_values_found)
        for j in range(len(narrowed_inputs)):
            print(str(j).rjust(3) + ": value: " + str(best_values_found[j]).rjust(23) + ", [", end='')
            for key, val in narrowed_inputs[j].items():
                print(str(key) + ": [" + str(val[0]).ljust(23) + ", " + str(val[1]).ljust(23) + "], ", end='')
            print(']')
        print()