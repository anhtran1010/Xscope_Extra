#!//usr/bin/env python3
import logging
from functools import partial
from utils import *
from test_function import *
import torch
from botorch.models import SingleTaskGP
from botorch.utils import standardize
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from BGRT import BinaryGuidedRandomTesting

# verbose = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
verbose = True
CUDA_LIB = ''
bo_iterations = 25  # number of iteration
logging.basicConfig(filename='Xscope.log', level=logging.INFO)
logger = logging.getLogger(__name__)
BATCH_SIZE = 3
NUM_RESTARTS = 5
RAW_SAMPLES = 512

# ----- Status variables ------
found_inf_pos = False
found_inf_neg = False
found_under_pos = False
found_under_neg = False
# -----------------------------
def data_initialization(obj_func, batch_shape, sample_dim):
    initial_X = torch.cuda.DoubleTensor(batch_shape, sample_dim).uniform_(-1, 1)
    initial_Y = []
    for x in initial_X:
        initial_Y.append(obj_func(x))
    initial_Y = torch.cuda.DoubleTensor(initial_Y).unsqueeze(-1)
    initial_X = initial_X.to(device=device)
    return initial_X, initial_Y

def initialize():
    global found_inf_pos, found_inf_neg, found_under_pos, found_under_neg
    found_inf_pos = False
    found_inf_neg = False
    found_under_pos = False
    found_under_neg = False

def set_max_iterations(n: int):
    global bo_iterations
    bo_iterations = n

test_func = TestFunction()
result_logger = ResultLogger()

# ----------------------------------------------------------------------------
# Results Checking
# ----------------------------------------------------------------------------

def run_optimizer(bounds, func, new_max, exp_name):
    global trials_to_trigger, trials_so_far
    num_fail = 0
    trials_so_far = 0
    trials_to_trigger = -1
    train_X, train_Y = data_initialization(func, 10, bounds.shape[-1])
    train_Y = standardize(train_Y)
    best_Y = train_Y.max()

    # if are_we_done(func, 0.0, exp_name):
    #   return

    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    if verbose: print('BO opt...')
    sampler = SobolQMCNormalSampler(1024)
    qEI = qExpectedImprovement(gp, best_Y, sampler)

    for i in range(bo_iterations):
        print("iteration: ", i)
        trials_so_far += 1
        try:
            new_Y = []
            for bound in bounds:
                candidates, _ = optimize_acqf(
                    acq_function=qEI,
                    bounds=bound,
                    q=BATCH_SIZE,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                    options={"batch_limit": 3, "maxiter": 100},
                )

                new_x = candidates.detach()
                train_X = torch.cat([train_X, new_x])
                for x in new_x:
                    new_Y.append(func(x))

            new_Y = torch.cuda.DoubleTensor(new_Y).unsqueeze(-1)
            train_Y = torch.cat([train_Y, new_Y])
            gp = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)
        except Exception as e:
            result_logger.save_results(train_Y.max(), exp_name)
            result_logger.update_runs_table(exp_name)
            break
            # if isinstance(e, ValueError):
            #     num_fail += 1
            #     optimizer._space._target[-1] /= (10 ** num_fail)
            # if verbose: print("Oops!", e.__class__, "occurred.")
            # if verbose: print(e)
            # if verbose: logging.exception("Something awful happened!")

    if verbose: print(train_Y.max())
    result_logger.save_results(train_X[train_Y.argmax()], exp_name)

def optimize(shared_lib: str, input_type: str, num_inputs: int, splitting: str, new_max: float, optimizer: str="BGRT"):
    test_func.set_kernel(shared_lib)
    logger.info("Max value to replace: {}".format(str(new_max)))
    if input_type != "exp" and input_type != "fp":
        print('Invalid input type!')
        exit()

    assert num_inputs >= 1 and num_inputs <= 3

    funcs = ["max_inf", "min_inf", "max_under", "min_under"]
    exp_name = [shared_lib, input_type, splitting]
    logging.info('|'.join(exp_name))
    for f in funcs:
        initialize()
        g = partial(test_func.function_to_optimize, num_input=num_inputs, func_type=f, mode=input_type)
        for b in bounds(split=splitting, num_input=num_inputs, input_type=input_type):
            if optimizer == "BO":
                run_optimizer(b, g, new_max, '|'.join(exp_name))
            else:
                bgrt = BinaryGuidedRandomTesting(b,g)
                narrowed_inputs, best_values_found = bgrt.binary_guided_random_testing()
                bgrt.print_output(narrowed_inputs, best_values_found)


# -------------- Results --------------
def print_results(shared_lib: str, number_sampling, range_splitting):
    result_logger.print_result(shared_lib, number_sampling, range_splitting, logger)

# -------------------------------------------------------
if __name__ == '__main__':
    optimize()
