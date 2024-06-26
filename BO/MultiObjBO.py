from global_init import *
from BO.init import *
from models.multiGP import MultitaskGPModel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.pareto import is_non_dominated


class MultiObjectiveBO(bo_base):
    """
    Class for Baysian Optimization for inputs that trigger FP-exceptions Model
    for multi-objectives functions.
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
    def __init__(self, eval_func: TestFunction, num_task, iteration=50, batch_size=5, acquisition_function='qehvi', bounds: Input_bound=None, device=torch.device("cuda")):
        super(MultiObjectiveBO, self).__init__(eval_func, iteration, batch_size, acquisition_function, bounds, device)
        self.num_task = num_task
        # initialize training data and model
        self.initialize_data()
        self.initialize_model()

    def initialize_data(self, normalize=False):
        """
        :param
        normalize: bool
            If the initial data need to be normalize
        :return: Tuple
            A tuple containing the training data
        """
        initial_x = self.bounds_object.bounds_sampler(1)
        if normalize:
            x_min, x_max = initial_x.min(), initial_x.max()
            new_min, new_max = -1e+100, 1e+100
            initial_x = (initial_x - x_min)/(x_max - x_min)*(new_max - new_min) + new_min
        self.train_x, self.train_y = self.evaluate_candidates(initial_x)
        self.ref_point = self.compute_reference_point

    def compute_reference_point(self):
        pareto_mask = is_non_dominated(self.train_y)
        pareto_Y = self.train_y[pareto_mask]
        ref_points = infer_reference_point(pareto_Y)
        return ref_points

    def initialize_model(self, state_dict=None):
        self.likelihood = MultitaskGaussianLikelihood(num_task = self.num_task).to(device=self.device, dtype=dtype)
        self.GP = MultitaskGPModel(self.train_x, self.train_y, self.likelihood).to(device=self.device, dtype=dtype)
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.GP)
        if state_dict is not None:
            self.GP.load_state_dict(state_dict)
        fit_mll(self.mll, options={"disp": False, "lr": 0.005}, approx_mll=True)

    def evaluate_candidates(self, candidates):
        targets = torch.empty((self.bounds_object.num_bounds,self.num_task), dtype=dtype, device=self.device)
        for i,x in enumerate(candidates):
            new_candidate_targets = self.eval_func.eval(x[0])
            new_candidate_targets = torch.as_tensor(new_candidate_targets, device=self.device, dtype=dtype)
            for j,target in enumerate(new_candidate_targets):
                target, exception_found = self.check_exception(x, target)
                if exception_found:
                    self.exception_per_bounds[i] += 1
                    print("Input belong to bound: ", self.bounds_object.bounds[i])
                new_candidate_targets[j] = target
            targets[i] = new_candidate_targets
        return candidates, targets

    def suggest_new_candidate(self, n_warmup=5000, n_samples=10):
        """
            A function to find the maximum of the acquisition function
            It uses a combination of random sampling (cheap) and the 'AdamW'
            optimization method. First by sampling `n_warmup` (500) points per bound at random,
            and then running AdamW from `n_samples` (5) random starting points per bound.
            Parameters
            ----------
            :param n_warmup:
                number of times to randomly sample the acquisition function
            :param n_samples:
                number of samples to try
            Returns
            -------
            :return: x_max, The arg max of the acquisition function.
            """
        # Warm up with random points
        #x_tries has shape: B*n_warmup x D
        self.likelihood.eval()
        self.GP.eval()
        x_tries = []
        n_warmup = max(n_warmup, round(100000/self.bounds_object.num_bounds))
        warmup_x = self.bounds_object.bounds_sampler(n_warmup)
        with torch.no_grad():
            posterior = self.GP.likelihood(self.GP(warmup_x))
            mean = posterior.mean
            while mean.ndim > 2:
                # take average over batch dims
                mean = mean.mean(dim=0)
            f_pred = mean
            n_best = max(1,  round(warmup_x.shape[1] * 0.05))
            best_idcs = torch.topk(f_pred, n_best).indices.unsqueeze(-1)
            best_X = torch.gather(warmup_x, 1, best_idcs.repeat(1,1,warmup_x.shape[-1]))
        for X, bound in zip(best_X, self.bounds_object.bounds):
            x_tries.append(sample_points_around_best(X, n_discrete_points=n_samples, sigma=1e-3, bounds=bound))
        x_tries = torch.stack(x_tries, dim=0)
        candidates = self.thorough_space_exploration(x_tries)
        return candidates.detach()

    def train(self):
        print("Begin BO")
        start_fitting = time.time()
        for i in range(self.iteration):
            if i % self.batch_size == 0 and i != 0:
                old_state_dict = self.mll.model.state_dict()
                self.initialize_model(state_dict=old_state_dict)
            new_candidates = self.suggest_new_candidate()
            new_candidates, new_targets = self.evaluate_candidates(new_candidates)
            self.train_x = torch.cat([self.train_x, new_candidates], dim=1)
            self.train_y = torch.cat([self.train_y, new_targets], dim=1)
            assert self.train_x.shape[0] == self.train_y.shape[0], f"shape mismatch, got {self.train_x.shape[0]} for training data but {self.train_y.shape[0]} for testing data"
            self.acq.update_params
        
        self.best_interval()
        print("Fitting time: ", time.time() - start_fitting)