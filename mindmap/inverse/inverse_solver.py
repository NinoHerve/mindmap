import mindmap as mm
import mne
import numpy as np

import pyxu.abc as pxa
import pyxu.operator as pxo
import pyxu.opt.stop as pxst
from pyxu.opt.solver import PGD

class InverseSolver():
    """
    Class to solve inverse problems. 

    Attributes
    ----------
    eeg: numpy array or mne.object
        EEG data.

    forward: numpy array or mne.object 
        Forward matrix.

    prior: numpy array or mne.object
        Prior for regularization.
        - 'mne': noise covariance matrix 
        - 'sloreta': noise covariance matrix 
        - 'eloreta': noise covariance matrix
        - 'cset': Connectome Fourier transform 

    lambda_: float
        Regularizer coefficient.
    """

    def __init__(self, method, eeg, forward, prior):
        """
        Initializer.
        """
        self.method  = self.check_method(method)
        self.eeg     = eeg
        self.forward = forward
        self.prior   = prior

    def solve(self, lambda_, loose=0, depth=0.8, rising_phase=0.5, initial_guess="adjoint", pyxu_kwargs={}):
        if self.method in ["MNE", "sLORETA", "eLORETA"]:
            return self.solve_mne(lambda_, loose, depth)   
        elif self.method == "CSET":
            return self.solve_pyxu(lambda_, rising_phase, initial_guess, kwargs=pyxu_kwargs)
        elif self.method == "CSET2":
            return self.solve_pyxu2()
        else:
            raise ValueError("'solve()' doesn't recognize method.")

    def solve_mne(self, lam, loose=0, depth=0.8):
        # configuration
        self.eeg.set_eeg_reference(projection=True)

        inverse = mne.minimum_norm.make_inverse_operator(
            self.eeg.info,
            self.forward,
            self.prior,
            loose=loose,
            depth=depth,
        )

        stc = mne.minimum_norm.apply_inverse(
            self.eeg, 
            inverse, 
            lambda2=lam,
            method=self.method,
        )

        return stc 

    def solve_pyxu(self, lam, rising_phase=0.5, initial_guess="adjoint", debug=False, kwargs={}):
        # ------ extract data ------
        fwd = self.forward.copy()
        fwd = mne.convert_forward_solution(
            fwd,
            surf_ori=True,
            force_fixed=True,
            use_cps=True,
            verbose="warning",
        )

        A = fwd["sol"]["data"].astype("float64")
        b, _ = mm.utils.eeg_rising_phase(self.eeg, rising_phase)

        # ------ normalize ------
        A_norms = np.linalg.norm(A, ord=2, axis=0)               
        A = A/A_norms[None,:]                                       # normalize rows of A, indexing [:,None] gives shape (M,1)

        b_norm = np.linalg.norm(b, ord=2)
        b = b/b_norm

        # ------ compute initial guess ------
        x0 = self.initial_guess(initial_guess, A, b)

        # ------ write optimization problem using pyxu ------
        A = pxa.LinOp.from_array(A@self.prior.T)
        F = pxo.SquaredL2Norm(dim_shape=A.codim_shape[0]).argshift(-b) * A     # fidelity
        F.diff_lipschitz = F.estimate_diff_lipschitz()              
        G = lam*pxo.L1Norm(dim_shape=x0.shape[0])                   # regularizer

        opt = PGD(f=F, g=G, **kwargs)

        # ------ define stopping criterion ------
        stop_crit = pxst.RelError(eps=1e-5, var="x", f=None, norm=2) | pxst.MaxIter(5000)

        # ------ solve optimization problem ------
        opt.fit(x0=x0, stop_crit=stop_crit, acceleration=True)
        
        if debug == True:
            return opt, A_norms, b_norm

        alpha = opt.solution().squeeze()
        x = self.prior.T@alpha
        x = x*A_norms*b_norm

        return x
    
    def solve_pyxu2(self, debug=False):
        #configuration
        config = mm.config.get()
        src_type = config["source space"]["type"]
        rising_phase = config["eeg"]["rising phase"]
        initial_guess = config["inverse"]["initial guess"]
        kwargs = config["inverse"]["pyxu"]["optkwargs"]

        # ------ extract data ------
        fwd = self.forward.copy()
        if src_type == "surf":
            fwd = mne.convert_forward_solution(
                fwd,
                surf_ori=True,
                force_fixed=True,
                use_cps=True,
                verbose="warning",
            )
        A = fwd["sol"]["data"]
        b, _ = mm.eeg.eeg_rising_phase(self.eeg, rising_phase)

        # ------ normalize ------
        A_norms = np.linalg.norm(A, ord=2, axis=1)
        A = A/A_norms[:,None]                                  # normalize rows of A, indexing [:,None] gives shape (M,1)  

        b_norm = np.linalg.norm(b)
        b = b/b_norm

        # ------ compute initial guess ------
        x0 = self.initial_guess(initial_guess, A, b)

        # ------ write optimization problem using pyxu ------
        A = pxa.LinOp.from_array(A)
        F = pxo.SquaredL2Norm(dim=A.shape[0]).asloss(b)*A     # fidelity
        F.diff_lipschitz = F.estimate_diff_lipschitz()
        GFT = pxa.LinOp.from_array(self.prior)              
        G = self.lamb*pxo.L1Norm(dim=x0.shape[0])*GFT  # regularizer

        opt = PGD(f=F, g=G, **kwargs)

        # ------ define stopping criterion ------
        stop_crit = pxst.RelError(eps=1e-5, var="x", f=F, norm=2, satisfy_all=True) | pxst.MaxIter(10000)

        # ------ solve optimization problem ------
        opt.fit(x0=x0, stop_crit=stop_crit)
        
        if debug == True:
            return opt, A_norms, b_norm

        x = opt.solution()
        x = x*A_norms*b_norm

        return x.squeeze()

    def check_method(self, method):
        """
        Check if method is implemented and rewrites the name with the right
        capital letters.

        Parameters:
        ----------
        method: str
            Name of inverse method. 'cset', 'mne', 'sloreta' or 'eloreta'.

        Returns:
        -------
        method: str
            Method written correctly for the functions that follow. 
        """
        m = method.lower()

        if m == "cset":
            return "CSET"
        elif m == "cset2":
            return "CSET2"
        elif m == "mne":
            return "MNE" 
        elif m == "sloreta":
            return "sLORETA" 
        elif m == "eloreta":
            return "eLORETA"
        else:
            raise ValueError(f"Method '{method}' not implemented.")
        
    def initial_guess(self, method, A, b):
        """
        Creates an initial guess x0.

        Parameters:
        ----------
        method: str
            Method for the initial guess.
            - zeros: array of zeros.
            - adjoint: computes inverse using the adjoint.

        Returns:
        -------
        x0: numpy 1D array
            Initial guess.
        """
        if method == "zeros":
            return np.zeros(A.shape[1])
        elif method == "adjoint":
            A_inv = np.linalg.pinv(A)
            return A_inv@b
        else: 
            raise ValueError("Unknown method {method} for initial guess.")