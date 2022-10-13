import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood
from torch.optim import Adam
from gpytorch.mlls import ExactMarginalLogLikelihood
from utils import *
import logging
from botorch.optim.fit import fit_gpytorch_torch
logging.basicConfig(filename='Xscope.log', level=logging.INFO)
logger = logging.getLogger(__name__)
max_normal = 1e+307

class ExactGPModel(gpytorch.models.ExactGP):
    """
    Class for Gaussian Process Model
    Attributes
    ----------
    train_x: a Torch tensor with shape n x d
        The training data
    train_x: a Torch tensor with shape n x 1
        The label
    likelihood : The likelihood function
        A function that approximate the likelihood of the GP given new observed datapoint
    Methods
    -------
    forward
    """

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MaternKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BaysianOptimization():
    """
    Class for Baysian Optimization for inputs that trigger FP-exceptions Model
    Attributes
    ----------
    eval_func: GPU function
        The function that are being tested on
    iteration: int
        How many time will the BO be run for
    batch_size: int
        How many time we run the GP before we update it
    acquisition_function: string
        A string indicating which acquisition function to use
        As of right now, we only accept UCB, IP, EI.
    likelihood : The likelihood function
        A function that approximate the likelihood of the GP given new observed datapoint
    bounds: a Tensor with shape 2xd
        The bound of each of the input parameter
    initial_sample: int
        Number of initial data points to sample.
    device:
        The device that the BO will be run on
    Methods
    -------
    initialize_data
    """

    def __init__(self, eval_func, iteration=25, batch_size=10, acquisition_function='ei',
                 likelihood_func=GaussianLikelihood(), bounds=None, initial_sample=1024, device=torch.device("cuda")):
        self.eval_func = eval_func
        self.iteration = iteration
        self.batch_size = batch_size
        self.acquisition_function = acquisition_function
        self.bounds = bounds
        self.device = device
        self.likelihood = likelihood_func.to(device=self.device)

        self.results = {}
        self.exception_induced_params = {}
        self.error_types = ["max_inf", "min_inf", "max_under", "min_under", "nan"]
        for type in self.error_types:
            self.results[type] = 0
            self.exception_induced_params[type] = []

        # initialize training data and model
        self.train_x, self.train_y = self.initialize_data(initial_sample)
        self.y_max = self.train_y.max()
        self.GP = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(device=self.device, dtype=dtype)

        self.acq = UtilityFunction(self.acquisition_function, kappa=2.5, xi=0.1e-1)

        self.optim = Adam(self.GP.parameters(), lr=0.1)
        self.mll = ExactMarginalLogLikelihood(self.GP.likelihood, self.GP)

        self.sampler = torch.distributions.uniform.Uniform(self.bounds[0], self.bounds[1])

        # Result report


    def initialize_data(self, num_sample=10):
        """
        :param
        eval_func: GPU function
            The function that are being tested on
        num_sample: int
            The number of sample to initialize
        :return: Tuple
            A tuple containing the training data
        """
        initial_X = self.sampler.sample((num_sample,)).to(device=device, dtype=dtype)
        initial_Y = torch.zeros(num_sample).to(device=device, dtype=dtype)
        for index, x in enumerate(initial_X):
            new_cadidate_target = self.eval_func(x)
            new_cadidate_target = torch.as_tensor(new_cadidate_target, device=self.device, dtype=dtype)
            if self.check_exception(x, new_cadidate_target):
                logger.info("parameter {} caused floating point error {}".format(x, new_cadidate_target))
                continue

        return initial_X, initial_Y

    def suggest_new_candidate(self, n_warmup=10000, n_iter=10):
        """
            A function to find the maximum of the acquisition function
            It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
            optimization method. First by sampling `n_warmup` (1e5) points at random,
            and then running L-BFGS-B from `n_iter` (250) random starting points.
            Parameters
            ----------
            :param n_warmup:
                number of times to randomly sample the acquisition function
            :param n_iter:
                number of times to run scipy.minimize
            Returns
            -------
            :return: x_max, The arg max of the acquisition function.
            """

        # Warm up with random points
        x_tries = self.sampler.sample((n_warmup,)).to(device=device, dtype=dtype)
        self.likelihood.eval()
        self.GP.eval()
        ys = self.GP.likelihood(self.GP.forward(x_tries)).mean
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()

        # Explore the parameter space more throughly
        x_seeds = self.sampler.sample((n_iter,)).to(device=device, dtype=dtype)

        to_minimize = lambda x: -self.acq.forward(self.GP, self.GP.likelihood, x.reshape(1, -1),  y_max=self.y_max)

        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            x_try.require_grad = True
            optimizer = torch.optim.LBFGS([x_try], lr=1e-5)
            for _ in range(10):
                def closure():
                    loss = to_minimize(x_try)
                    optimizer.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        loss[:] = loss.clamp(self.bounds[0], self.bounds[1])
                    return loss
                loss = closure()
                if not torch.isfinite(loss):
                    continue
                optimizer.step(closure)
                if max_acq is None or -torch.squeeze(loss) >= max_acq:
                    x_max = x_try
                    max_acq = -torch.squeeze(loss)

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return torch.clip(x_max, self.bounds[0], self.bounds[1])

    def check_exception(self, param, val):
        # Infinity
        posinf = torch.isposinf(val)
        neginf = torch.isneginf(val)
        nan = torch.isnan(val)
        if posinf.any():
            inf_indices = torch.nonzero(val)
            inf_induced_param = torch.index_select(param, 0, inf_indices)
            self.results["min_inf"] += len(inf_indices)
            self.exception_induced_params["min_inf"].append(inf_induced_param)
            return True
        if neginf:
            inf_indices = torch.nonzero(val)
            inf_induced_param = torch.index_select(param, 0, inf_indices)
            self.results["max_inf"] += len(inf_indices)
            self.exception_induced_params["max_inf"].append(inf_induced_param)
            return True

        # Subnormals
        if torch.isfinite(val):
            if val > -2.22e-308 and val < 2.22e-308:
                if val != 0.0 and val != -0.0:
                    if val < 0.0:
                        self.results["min_under"] += 1
                        self.exception_induced_params["min_under"].append(param)
                    else:
                        self.results["max_under"] += 1
                        self.exception_induced_params["max_under"].append(param)
                    return True

        if nan:
            nan_indices = torch.nonzero(val)
            nan_induced_param = torch.index_select(param, 0, nan_indices)
            self.results["nan"] += len(nan_indices)
            self.exception_induced_params["nan"].append(nan_induced_param)
            return True
        return False

    def train(self):
        new_train_x = []
        new_train_y = []
        self.mll, _ = fit_gpytorch_torch(self.mll)
        for i in range(self.iteration):
            # Set the gradients from previous iteration to zero
            if i%self.batch_size==0 and i!=0:
                new_train_x = torch.stack(new_train_x)
                new_train_y = torch.stack(new_train_y).unsqueeze(-1)
                self.mll.model.set_train_data(new_train_x, new_train_y, strict=False)
                self.mll, _ = fit_gpytorch_torch(self.mll)
                new_train_x = []
                new_train_y = []

            new_candidate = self.suggest_new_candidate()
            new_cadidate_target = self.eval_func(new_candidate)
            new_cadidate_target = torch.as_tensor(new_cadidate_target).to(dtype=dtype)
            if self.check_exception(new_candidate, new_cadidate_target):
                logger.info("parameter {} caused floating point error {}".format(new_candidate, new_cadidate_target))
                break
            new_train_x.append(new_candidate)
            new_train_y.append(new_cadidate_target)

        for type in self.error_types:
            self.exception_induced_params[type] = torch.cat(self.exception_induced_params[type]).flatten()
