import numpy as np


class Minimize(object):
    """
    Class based minimization of scalar functions of one or more variables::
        minimize f(x) subject to
        g_i(x) >= 0,  i = 1,...,m
        h_j(x)  = 0,  j = 1,...,p
    where x is a vector of one or more variables.
    ``g_i(x)`` are the inequality constraints.
    ``h_j(x)`` are the equality constraints.
    Optionally, the lower and upper bounds for each element in x can also be
    specified using the `bounds` argument.
    """
    opt_options = {'bounds': None, 'x0': None}

    """
    Implementation notes
    --------------------
    1. Solvers that inherit Optimizer should use Optimizer.func, Optimizer.jac,
        and Optimizer.hess to make sure that function counting is carried out
    correctly.
    2. Solvers that inherit Optimizer should ensure that they call
        `super(InheritingOptimizer, self).__init__(func)` for correct
        functioning (i.e. attributes and methods are setup correctly).
    3. No evaluation of functions are expected to happen in __init__, they
        should only happen in __next__.
    4. The result property (if overridden) should include
        ```
        if self.fun is None:
            self.fun = self.func(self.x)
        ```
        because if maxiter is set to 0 then the function will have never been
        evaluated and self.fun will be None. Some optimizers don't provide x0,
        only doing that after first iteration. In this case self.x will be None
        before the first iteration is done. 
    5. Solver hyper_parameters should be exposed in the `_hyper` dictionary,
        which should be set in the inheriting Optimizer. This `_hyper`
        dictionary is visible from the `hyper_parameters` property. Only make
        visible attributes that you expect a user to change.
    6. To stop iteration the __next__ method should raise a StopIteration
        exception, which is caught in Optimizer.__call__.
    7. The user can halt evaluation by raising StopIteration in Function and
        in the callback. Doing so in the callback is not considered an error
        state (they can raise a different error to signify that), doing so in
        the Function is considered an error state.
    7. Subclassing Optimizers should use the Optimizer.func, Optimizer.grad,
        Optimizer.hess and Optimizer.func_and_grad callable attributes, rather
        than calling the Function directly. This is because the base Optimizer
        class keeps track of function calls via closures.
    8. After each iteration in ``solve`` a user can provide a callback function
        to receive status updates. These updates are provided as an
        intermediate `OptimizeResult` - not a final solution. If an
        Optimizer would like to provide more fields to this intermediate result
        they should override the `Optimizer_callback` method. The callback can
        raise StopIteration to halt.
    9. After adding a scalar Optimizer it should be added to the parameterised
         tests in `tests.test_optimize::Test_Optimizer`.
    """
    def __init__(self, func, *args, **options):
        self.opt_options.update(options)
        if not isinstance(func, Function):
            raise ValueError("`func` needs to be a Function object.")

        self._func = func
        self._njev = np.zeros((1,), dtype=int)
        self._nfev = np.zeros((1,), dtype=int)
        self._nhev = np.zeros((1,), dtype=int)

        # wrap the constituent methods to permit counting of the number
        # of function calls
        def fgh_call(f, g, h, obj, meth):
            def wrapper(x, **kwds):
                val = meth(x, **kwds)
                f[0] += obj.f_calls
                g[0] += obj.g_calls
                h[0] += obj.h_calls
                return val

            return wrapper

        self.func = fgh_call(self._nfev, self._njev, self._nhev,
                             self._func, self._func.func)
        self.grad = fgh_call(self._nfev, self._njev, self._nhev,
                             self._func, self._func.grad)
        self.hess = fgh_call(self._nfev, self._njev, self._nhev,
                             self._func, self._func.hess)
        self.func_and_grad = fgh_call(self._nfev, self._njev, self._nhev,
                             self._func, self._func.func_and_grad)

        # really want the options as attributes
        for k, v in self.opt_options.items():
            setattr(self, k, v)

        self.x = None
        if self.opt_options['x0'] is not None:
            self.x = np.asarray(self.opt_options['x0']).copy()

        self.fun = None
        self.hess_inv = None

        self.status = 0
        self.return_all = False
        self.success = False
        self.message = None
        self.warn_flag = 0
        self.options = {}
        self._hyper = {}
        self.nit = 0

    @property
    def nfev(self):
        """The total number of function evaluations used by the solver."""
        return self._nfev[0]

    @nfev.setter
    def nfev(self, val):
        """Set the total number of function evaluations used by the solver."""
        self._nfev[0] = val

    @property
    def njev(self):
        """The number of jacobian evaluations"""
        return self._njev[0]

    @property
    def nhev(self):
        """The number of hessian evaluations."""
        return self._nhev[0]

    @property
    def hyper_parameters(self):
        """Dictionary containing solver hyperparameters."""
        return self._hyper

    @property
    def result(self):
        """
        The optimization result represented as a `OptimizeResult` object.
        Attributes
        ----------
        x : ndarray
            The solution of the optimization.
        success : bool
            Whether or not the status of the optimizer is error free AND the
            optimizer has converged.
        status : int
            Status of the optimizer.
            - 0 if no error,
            - 1 if too many function evaluations
            - 2 if too many iterations
            - other if stopped for another reason, given in `message`
        message : str
            Description of the cause of the termination.
        fun, jac, hess: ndarray
            Values of objective function, its Jacobian and its Hessian (if
            available). The Hessians may be approximations, see the documentation
            of the function in question.
        hess_inv : object
            Inverse of the objective function's Hessian; may be an approximation.
            Not available for all solvers. The type of this attribute may be
            either np.ndarray or scipy.sparse.linalg.LinearOperator.
        nfev, njev, nhev : int
            Number of evaluations of the objective functions and of its
            Jacobian and Hessian.
        nit : int
            Number of iterations performed by the optimizer.
        """
        # override to set other fields.

        # user may have set maxiter = 0, in which case the function may never
        # have been evaluated. Note: self.x may not have been set either. Most
        # optimizers that have x0 would set that in __init__, but some
        # optimizers (DifferentialEvolutionSolver) aren't given x0, and only
        # work out self.x in the first iteration. Therefore nfev will always
        # be at least one, even if maxiter ==  maxfun == 0.
        if self.fun is None:
            self.fun = self.func(self.x)

        result = OptimizeResult(
            x=self.x,
            fun=self.fun,
            nfev=self.nfev,
            nit=self.nit,
            message=self.message,
            success=(self.warning_flag == 0 and self.converged()))

        return result

    def __call__(self, iterations, maxfun=np.inf):
        """
        Advance the solver by a number of steps.

        Parameters
        ----------
        iterations : int
            The number of iterations to perform.
        maxfun : int, optional
            Limits the total number of function evaluations.

        Returns
        -------
        result: OptimizeResult
            The optimization result represented as a OptimizeResult object.

        Notes
        -----
        If the solver converges, if `Optimizer.nfev` reaches `maxfun`, if
        `Optimizer.nit` reaches `iterations`, or if `StopIteration` is raised,
        then iteration will stop before the requested number of iterations has
        been carried out. If you wish to do further iterations then reset
        `Optimizer.nit`, or `Optimizer.nfev`, to zero.
        Do not set `iterations` to `np.inf` without capping `maxfun`, as the
        solver may run indefinitely.
        """
        # arrange the users callback
        solver_callback = lambda: None
        start_time = time.time()
        if callback is not None:
            solver_callback = partial(self._callback,
                                      callback, start_time)

        # keep a record of all the iterations.
        _allvecs = []

        while self.nit < iterations and self.nfev < maxfun:
            try:
                x, f = next(self)
                if allvecs:
                    _allvecs.append(np.copy(self.x))

                solver_callback()
            except StopIteration:
                break

            if self.warn_flag:
                break

        if self.nfev >= maxfun:
            self.warn_flag = 1
            self.message = ("STOP: TOTAL NO. of FUNCTION EVALUATIONS EXCEEDS"
                           " LIMIT")
        elif self.nit >= iterations:
            self.warn_flag = 2
            self.message = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
        elif self.converged():
            self.message = _status_message['success']

        res = self.result
        res['walltime'] = time.time() - start_time
        if allvecs:
            res['allvecs'] = _allvecs

        return res

    def _callback(self, callback, start_time):
        """
        Function called after each iteration solver iteration.
        Parameters
        ----------
        callback : callable, optional
            Function called at the end of each iteration as ``callable(res)``,
            where `res` is an intermediate `OptimizeResult` object. The range
            of attributes provided in `res` depend on the specific Optimizer.
            The wall-time (in seconds) for this solver call is provided as the
            `walltime` attribute.
            The callback can halt optimization early by raising
            `StopIteration`, a final `OptimizeResult` is still returned. Other
            Exceptions are propgated upwards.
        """
        res = OptimizeResult(x=np.copy(self.x),
                             fun=self.fun,
                             nfev=self.nfev,
                             njev=self.njev,
                             nhev=self.nhev,
                             nit=self.nit,
                             walltime=time.time() - start_time)

        callback(res)

    @property
    def N(self):
        """
        The dimensionality of the problem, `np.size(x)`.
        """
        if self.x is not None:
            return self.x.size
        elif self.x0 is not None:
            return self.x0.size
        return RuntimeError("Cannot determine problem size at this time,"
                            " perform at least one iteration")

    def solve(self, maxiter=np.inf, maxfun=np.inf, callback=None, disp=False,
              allvecs=False):
        """
        Run the solver through to completion.
        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations to use during solving
        maxfun : int, optional
            Maximum number of function evaluations to use during solving
        callback : callable, optional
            Function called at the end of each iteration as ``callable(res)``,
            where `res` is an intermediate `OptimizeResult` object. The range
            of attributes provided in `res` depend on the specific Optimizer.
            The wall-time (in seconds) for this solver call is provided as the
            `time` attribute.
            The callback can halt optimization early by raising
            `StopIteration`, a final `OptimizeResult` is still returned. Other
            Exceptions are propgated upwards.
        disp : bool, optional
            Set to True to print convergence messages.
        allvecs : bool, optional
            Keep a record of `x` at each iteration step. This sets the
            `allvecs` attribute in `result`.
        Returns
        -------
        result: OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            See `Optimizer.result` for further details.
        Notes
        -----
        If both `maxiter` and `maxfun` are set to `np.inf`, then `maxiter` is
        capped at `len(x) * 200`
        If the solver converges, if `Optimizer.nfev` reaches `maxfun`, if
        `Optimizer.nit` reaches `iterations`, or if `StopIteration` is raised
        by `Function` or `callback`, then iteration will stop before the
        requested number of iterations has been carried out.
        """
        # If neither are set, then cap `maxiter`
        if maxiter == np.inf and maxfun == np.inf:
            maxiter = self.N * 200

        result = self(maxiter, callback=callback, maxfun=maxfun,
                      allvecs=allvecs)

        if disp and self.warn_flag:
            print('Warning: ' + self.message)
        elif disp and not self.warn_flag:
            print(self.message)
            print("         Current function value: %f" % self.fun)
            print("         Iterations: %d" % self.nit)
            print("         Function evaluations: %d" % self.nfev)

        self._finish_up()
        return result

    def converged(self):
        """
        The truth of whether the solver has converged.
        """
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        """
        Advance the solver by a single iteration.
        To indicate that further iteration is not possible you can raise
        StopIteration, or set warn_flag != 0.
        """
        # Should be over-ridden by each class based solver.
        raise NotImplementedError

    def next(self):
        """
        Advance the solver by a single iteration.
        """
        # next() is required for compatibility with Python2.7.
        return self.__next__()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._finish_up()

    def _finish_up(self):
        """
        Each class based solver should override this to define end-of-run
        actions. It gets called at the end of `solve`, and during
        `__exit__` of the context manager. It may be useful to close
        resources that need to be terminated.
        """
        pass

    def show_options(self):
        # each optimizer should return their own options text.
        return ""