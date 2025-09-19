import math
import numpy
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final
from tvb.simulator.models.base import ModelNumbaDfun
from tvb.simulator.backend.ref import RefBase

class wFICJansenRit(ModelNumbaDfun):

    # Define traited attributes for this model, these represent possible kwargs.
    A = NArray(
        label=":math:`A`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    B = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].

        The usual value for this parameter is 6.0.""")

    nu_max = NArray(
        label=r":math:`\nu_{max}`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [s^-1].""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    J = NArray(
        label=":math:`J`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    a_1 = NArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.""")

    a_2 = NArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop.""")

    a_3 = NArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.""")

    a_4 = NArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.""")

    p_min = NArray(
        label=":math:`p_{min}`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.12, step=0.01),
        doc="""Minimum input firing rate.""")

    p_max = NArray(
        label=":math:`p_{max}`",
        default=numpy.array([0.32]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Maximum input firing rate.""")

    mu = NArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate.""")

    # FIC parameters:
    eta = NArray(
        label=":math:`eta `",
        default=numpy.array([0.0001]),
        domain=Range(lo=0.000001, hi=1, step=0.001),
        doc="""adaptation rate parameter for FIC""")

    y0_target = NArray(
        label=":math:`y0_target `",
        default=numpy.array([0.11]),
        domain=Range(lo=0.01, hi=1., step=0.001),
        doc="""Target y0: tunning target for FIC procedure""")

    tau_d = NArray(
        label=":math:`tau_d `",
        default=numpy.array([10.0]),
        domain=Range(lo=10, hi=100000., step=10),
        doc="""tau_d - corresponds to the time window over which y0 and y2 are averaged (10 = 10s)""")
    

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"y0": numpy.array([-1.0, 1.0]),
                 "y1": numpy.array([-500.0, 500.0]),
                 "y2": numpy.array([-50.0, 50.0]),
                 "y3": numpy.array([-6.0, 6.0]),
                 "y4": numpy.array([-20.0, 20.0]),
                 "y5": numpy.array([-500.0, 500.0]),
                 "y0_d": numpy.array([-1, 1]),
                 "y2_d": numpy.array([-50.0, 50.0]),
                 "wFIC": numpy.array([0.,  10.])},
        doc="""The values for each state-variable should be set to encompass
    the expected dynamic range of that state-variable for the current
    parameters, it is used as a mechanism for bounding random initial
    conditions when the simulation isn't started from an explicit history,
    it is also provides the default range of phase-plane plots.""")

    state_variables = tuple(
        'y0 y1 y2 y3 y4 y5 y0_d y2_d wFIC'.split())
    _nvar = 9
    cvar = numpy.array([1, 2], dtype=numpy.int32)

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=(['y0', 'y1', 'y2', 'y3',
                 'y4', 'y5', 'y0_d', 'y2_d', 'wFIC']),
        default=(['y0', 'y1', 'y2', 'wFIC']))

    def dfun(self, y, c, local_coupling=0.0):
        # state variables
        y_ = y.reshape(y.shape[:-1]).T

        # long couplings (connectome)
        c_ = c.reshape(c.shape[:-1]).T

        # short couplings (self loop)
        src = local_coupling*(y[1] - y[8]*y[2])[:,0]

        # model parameters
        params = [
            self.nu_max, self.r, self.v0, self.a,
            self.a_1, self.a_2, self.a_3, self.a_4, 
            self.A, self.b, self.B, self.J, 
            self.mu, 
            self.eta, self.y0_target, self.tau_d
        ]
        
        deriv = _numba_dfun_wFICjr(y_, c_, src, *params)
        deriv = deriv.T[..., numpy.newaxis]

        return deriv


@guvectorize([(float64[:],)*2 + (float64,)*17 + (float64[:],)], '(n),(m)' + ',()'*17 + '->(n)', nopython=True)
def _numba_dfun_wFICjr(
    y,
    c,
    src,
    nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu, eta, y0_target, tau_d,
    dx
):
    """
    Derivations of state variables.

    Parameters
    ----------
    y: list[]
        State variables
        - y[0] == y0
        - y[1] == y1
        - y[2] == y2
        - y[3] == y3
        - y[4] == y4
        - y[5] == y5
        - y[6] == y0_d
        - y[7] == y2_d
        - y[8] == wFIC
    c: ???
        Long range couplings (connectome).

    src: float
        Short range couplings (self loop).

    nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu: float
        Jansen Rit parameters.

    eta, y0_target, tau_d: float
        Feedback Inhibitory Constant (FIC) parameters.
    """
    PSP = y[1] - y[8]*y[2]  # Pyramidal entry with adapted inhibition

    sigm_y1_y2 = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - PSP)))
    sigm_y0_1  = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - (a_1 * J * y[0]))))
    sigm_y0_3  = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - (a_3 * J * y[0]))))

    dx[0] = y[3]
    dx[1] = y[4]
    dx[2] = y[5]
    dx[3] = A*a*sigm_y1_y2 - 2.0*a*y[3] - a**2*y[0]
    dx[4] = A*a*(mu + a_2 * J * sigm_y0_1 + c[0] + src) - 2.0*a*y[4] - a**2*y[1]
    dx[5] = B*b*a_4*J*sigm_y0_3 - 2.0*b*y[5] - b**2*y[2]
    dx[6] = (y[0]-y[6]) / tau_d
    dx[7] = (y[2]-y[7]) / tau_d
    dx[8] = eta*y[7] * (y[6]-y0_target)
    




class pFICJansenRit(ModelNumbaDfun):

    # Define traited attributes for this model, these represent possible kwargs.
    A = NArray(
        label=":math:`A`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    B = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].

        The usual value for this parameter is 6.0.""")

    nu_max = NArray(
        label=r":math:`\nu_{max}`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [s^-1].""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    J = NArray(
        label=":math:`J`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    a_1 = NArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.""")

    a_2 = NArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop.""")

    a_3 = NArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.""")

    a_4 = NArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.""")

    p_min = NArray(
        label=":math:`p_{min}`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.12, step=0.01),
        doc="""Minimum input firing rate.""")

    p_max = NArray(
        label=":math:`p_{max}`",
        default=numpy.array([0.32]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Maximum input firing rate.""")

    mu = NArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate.""")

    # FIC parameters:
    pFIC = NArray(
        label=":math:`pFIC`",
        default=numpy.array([1]),
        domain=Range(lo=0, hi=10, step=0.01),
        doc="""adaptation rate parameter for FIC""")
    

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"y0": numpy.array([-1.0, 1.0]),
                 "y1": numpy.array([-500.0, 500.0]),
                 "y2": numpy.array([-50.0, 50.0]),
                 "y3": numpy.array([-6.0, 6.0]),
                 "y4": numpy.array([-20.0, 20.0]),
                 "y5": numpy.array([-500.0, 500.0]),},
        doc="""The values for each state-variable should be set to encompass
    the expected dynamic range of that state-variable for the current
    parameters, it is used as a mechanism for bounding random initial
    conditions when the simulation isn't started from an explicit history,
    it is also provides the default range of phase-plane plots.""")

    state_variables = tuple(
        'y0 y1 y2 y3 y4 y5'.split())
    _nvar = 6
    cvar = numpy.array([1, 2], dtype=numpy.int32)

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=(['y0', 'y1', 'y2', 'y3', 'y4', 'y5',]),
        default=(['y0', 'y1', 'y2']))

    def dfun(self, y, c, local_coupling=0.0):
        # state variables
        y_ = y.reshape(y.shape[:-1]).T

        # long couplings (connectome)
        c_ = c.reshape(c.shape[:-1]).T

        # short couplings (self loop)
        src = local_coupling*(y[1] - self.pFIC*y[2])[:,0]

        # model parameters
        params = [
            self.nu_max, self.r, self.v0, self.a,
            self.a_1, self.a_2, self.a_3, self.a_4, 
            self.A, self.b, self.B, self.J, 
            self.mu, 
            self.pFIC,
        ]
        
        deriv = _numba_dfun_pFICjr(y_, c_, src, *params)
        deriv = deriv.T[..., numpy.newaxis]

        return deriv


@guvectorize([(float64[:],)*2 + (float64,)*15 + (float64[:],)], '(n),(m)' + ',()'*15 + '->(n)', nopython=True)
def _numba_dfun_pFICjr(
    y,
    c,
    src,
    nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu, pFIC,
    dx
):
    """
    Derivations of state variables.

    Parameters
    ----------
    y: list[]
        State variables
        - y[0] == y0
        - y[1] == y1
        - y[2] == y2
        - y[3] == y3
        - y[4] == y4
        - y[5] == y5
    c: ???
        Long range couplings (connectome).

    src: float
        Short range couplings (self loop).

    nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu: float
        Jansen Rit parameters.

    pFIC: float
        Feedback Inhibitory Constant (FIC) parameters.
    """
    PSP = y[1] - pFIC*y[2]  # Pyramidal entry with adapted inhibition

    sigm_y1_y2 = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - PSP)))
    sigm_y0_1  = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - (a_1 * J * y[0]))))
    sigm_y0_3  = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - (a_3 * J * y[0]))))

    dx[0] = y[3]
    dx[1] = y[4]
    dx[2] = y[5]
    dx[3] = A*a*sigm_y1_y2 - 2.0*a*y[3] - a**2*y[0]
    dx[4] = A*a*(mu + a_2 * J * sigm_y0_1 + c[0] + src) - 2.0*a*y[4] - a**2*y[1]
    dx[5] = B*b*a_4*J*sigm_y0_3 - 2.0*b*y[5] - b**2*y[2]






class habJansenRit(ModelNumbaDfun):

    # Define traited attributes for this model, these represent possible kwargs.
    A = NArray(
        label=":math:`A`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    B = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].

        The usual value for this parameter is 6.0.""")

    nu_max = NArray(
        label=r":math:`\nu_{max}`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [s^-1].""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    J = NArray(
        label=":math:`J`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    a_1 = NArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.""")

    a_2 = NArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop.""")

    a_3 = NArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.""")

    a_4 = NArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.""")

    # p_min and p_max are actually not used inside TVB. It is usually used outside to determine the stochastic noise of the integrator.
    p_min = NArray(
        label=":math:`p_{min}`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.12, step=0.01),
        doc="""Minimum input firing rate.""")

    p_max = NArray(
        label=":math:`p_{max}`",
        default=numpy.array([0.32]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Maximum input firing rate.""")

    mu = NArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate.""")

    eta = NArray(
        label=":math:`eta `",
        default=numpy.array([0.0001]),
        domain=Range(lo=0.000001, hi=1, step=0.001),
        doc="""adaptation rate parameter for FIC""")
    
    tau_d = NArray(
        label=":math:`tau_d `",
        default=numpy.array([10.0]),
        domain=Range(lo=10, hi=100000., step=10),
        doc="""tau_d - corresponds to the time window over which y0 and y2 are averaged (10 = 10s)""")
    

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"y0": numpy.array([-1.0, 1.0]),
                 "y1": numpy.array([-500.0, 500.0]),
                 "y2": numpy.array([-50.0, 50.0]),
                 "y3": numpy.array([-6.0, 6.0]),
                 "y4": numpy.array([-20.0, 20.0]),
                 "y5": numpy.array([-500.0, 500.0]),
                 "G0_d": numpy.array([0, 0]),
                 "C": numpy.array([0, 0]),},
        doc="""The values for each state-variable should be set to encompass
    the expected dynamic range of that state-variable for the current
    parameters, it is used as a mechanism for bounding random initial
    conditions when the simulation isn't started from an explicit history,
    it is also provides the default range of phase-plane plots.""")

    state_variables = tuple(
        'y0 y1 y2 y3 y4 y5 G0_d C'.split())
    _nvar = 8

    # variables used in coupling (Jansen Rit coupling - a sigmoid - needs two inputs)
    cvar  = numpy.array([1, 2], dtype=numpy.int32)

    # stimulated variables (Stimulation will affect only these states)
    stvar = numpy.array([1], dtype=numpy.int32)

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=(['y0', 'y1', 'y2', 'y3',
                 'y4', 'y5', 'G0_d', 'C']),
        default=(['y0', 'y1', 'y2', 'C']))

    def dfun(self, y, c, local_coupling=0.0):
        # state variables
        y_ = y.reshape(y.shape[:-1]).T

        # long couplings (connectome)
        c_ = c.reshape(c.shape[:-1]).T

        # short couplings (self loop)
        src = local_coupling*(y[1] - y[2])[:,0]

        # model parameters
        params = [
            self.nu_max, self.r, self.v0, self.a,
            self.a_1, self.a_2, self.a_3, self.a_4, 
            self.A, self.b, self.B, self.J, 
            self.mu, 
            self.eta, self.tau_d
        ]
        
        deriv = _numba_dfun_habjr(y_, c_, src, *params)
        deriv = deriv.T[..., numpy.newaxis]

        return deriv


@guvectorize([(float64[:],)*2 + (float64,)*16 + (float64[:],)], '(n),(m)' + ',()'*16 + '->(n)', nopython=True)
def _numba_dfun_habjr(
    y,
    c,
    src,
    nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu, eta, tau_d,
    dx
):
    """
    Derivations of state variables.

    Parameters
    ----------
    y: list[]
        State variables
        - y[0] == y0
        - y[1] == y1
        - y[2] == y2
        - y[3] == y3
        - y[4] == y4
        - y[5] == y5
        - y[6] == G0_d
        - y[7] == C
    c: ???
        Long range couplings (connectome).

    src: float
        Short range couplings (self loop).

    nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu: float
        Jansen Rit parameters.

    eta, y0_target, tau_d: float
        Feedback Inhibitory Constant (FIC) parameters.
    """
    PSP = y[1] - y[2]  # Pyramidal entry with adapted inhibition

    sigm_y1_y2 = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - PSP)))
    sigm_y0_1  = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - (a_1 * J * y[0]))))
    sigm_y0_3  = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - (a_3 * J * y[0]))))

    dx[0] = y[3]
    dx[1] = y[4]
    dx[2] = y[5]
    dx[3] = A*a*sigm_y1_y2 - 2.0*a*y[3] - a**2*y[0]
    dx[4] = A*a*(mu + a_2 * J * sigm_y0_1 + c[0] - y[7] + src) - 2.0*a*y[4] - a**2*y[1]
    dx[5] = B*b*a_4*J*sigm_y0_3 - 2.0*b*y[5] - b**2*y[2]
    dx[6] = (c[0]-y[7]) / tau_d
    dx[7] = eta*(y[6]-y[7])
    

class Olivier(ModelNumbaDfun):

    # Define traited attributes for this model, these represent possible kwargs.
    A = NArray(
        label=":math:`A`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    B = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].

        The usual value for this parameter is 6.0.""")

    nu_max = NArray(
        label=r":math:`\nu_{max}`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [s^-1].""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    J = NArray(
        label=":math:`J`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    a_1 = NArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.""")

    a_2 = NArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop.""")

    a_3 = NArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.""")

    a_4 = NArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.""")

    # p_min and p_max are actually not used inside TVB. It is usually used outside to determine the stochastic noise of the integrator.
    p_min = NArray(
        label=":math:`p_{min}`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.12, step=0.01),
        doc="""Minimum input firing rate.""")

    p_max = NArray(
        label=":math:`p_{max}`",
        default=numpy.array([0.32]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Maximum input firing rate.""")

    mu = NArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate.""")
    
    # backward connectivity
    gc_back = NArray(
        label=":math `a^B`",
        default = numpy.array([0.]),
        domain=Range(lo=0.0, hi=10, step=0.1),
        doc="""Backward connectivity."""
    )

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"y0": numpy.array([-1.0, 1.0]),
                 "y1": numpy.array([-500.0, 500.0]),
                 "y2": numpy.array([-50.0, 50.0]),
                 "y3": numpy.array([-6.0, 6.0]),
                 "y4": numpy.array([-20.0, 20.0]),
                 "y5": numpy.array([-500.0, 500.0]),},
        doc="""The values for each state-variable should be set to encompass
    the expected dynamic range of that state-variable for the current
    parameters, it is used as a mechanism for bounding random initial
    conditions when the simulation isn't started from an explicit history,
    it is also provides the default range of phase-plane plots.""")

    state_variables = tuple(
        'y0 y1 y2 y3 y4 y5'.split())
    _nvar = 6

    # variables used in coupling (Jansen Rit coupling - a sigmoid - needs two inputs)
    cvar  = numpy.array([1, 2], dtype=numpy.int32)

    # stimulated variables (Stimulation will affect only these states)
    stvar = numpy.array([1], dtype=numpy.int32)

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=(['y0', 'y1', 'y2', 'y3', 'y4', 'y5']),
        default=(['y0', 'y1', 'y2']))

    def dfun(self, y, c, local_coupling=0.0):
        # state variables
        y_ = y.reshape(y.shape[:-1]).T

        # long couplings (connectome)
        c_ = c.reshape(c.shape[:-1]).T

        # short couplings (self loop)
        src = local_coupling*(y[1] - y[2])[:,0]

        # model parameters
        params = [
            self.nu_max, self.r, self.v0, self.a,
            self.a_1, self.a_2, self.a_3, self.a_4, 
            self.A, self.b, self.B, self.J, 
            self.mu, self.gc_back,
        ]
        
        deriv = _numba_dfun_olive(y_, c_, src, *params)
        deriv = deriv.T[..., numpy.newaxis]

        return deriv


@guvectorize([(float64[:],)*2 + (float64,)*15 + (float64[:],)], '(n),(m)' + ',()'*15 + '->(n)', nopython=True)
def _numba_dfun_olive(
    y,
    c,
    src,
    nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu, gc_back,
    dx
):
    """
    Derivations of state variables.

    Parameters
    ----------
    y: list[]
        State variables
        - y[0] == y0
        - y[1] == y1
        - y[2] == y2
        - y[3] == y3
        - y[4] == y4
        - y[5] == y5
    c: ???
        Long range couplings (connectome).

    src: float
        Short range couplings (self loop).

    nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu: float
        Jansen Rit parameters.

    eta, y0_target, tau_d: float
        Feedback Inhibitory Constant (FIC) parameters.
    """
    PSP = y[1] - y[2]  # Pyramidal entry with adapted inhibition

    sigm_y1_y2 = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - PSP)))
    sigm_y0_1  = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - (a_1 * J * y[0]))))
    sigm_y0_3  = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - (a_3 * J * y[0]))))

    dx[0] = y[3]
    dx[1] = y[4]
    dx[2] = y[5]
    dx[3] = A*a*(sigm_y1_y2) - 2.0*a*y[3] - a**2*y[0]
    dx[4] = A*a*(mu + a_2 * J * sigm_y0_1 + c[0] + src) - 2.0*a*y[4] - a**2*y[1]
    dx[5] = B*b*(a_4*J*sigm_y0_3 + gc_back*c[0]) - 2.0*b*y[5] - b**2*y[2]
    



class jrTest(ModelNumbaDfun):

    # Define traited attributes for this model, these represent possible kwargs.
    A = NArray(
        label=":math:`A`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    B = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.05, hi=0.15, step=0.01),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")
    
    b2 = NArray(
        label=":math:`b`",
        default=numpy.array([0.05]),
        domain=Range(lo=0.025, hi=0.075, step=0.005),
        doc="""Reciprocal of the time constant of passive membrane and all
        other spatially distributed delays in the dendritic network [ms^-1].
        Also called average synaptic time constant.""")


    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV].

        The usual value for this parameter is 6.0.""")

    nu_max = NArray(
        label=r":math:`\nu_{max}`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [s^-1].""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    J = NArray(
        label=":math:`J`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    a_1 = NArray(
        label=r":math:`\alpha_1`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.""")

    a_2 = NArray(
        label=r":math:`\alpha_2`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop.""")

    a_3 = NArray(
        label=r":math:`\alpha_3`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.""")

    a_4 = NArray(
        label=r":math:`\alpha_4`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.""")

    # p_min and p_max are actually not used inside TVB. It is usually used outside to determine the stochastic noise of the integrator.
    p_min = NArray(
        label=":math:`p_{min}`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.12, step=0.01),
        doc="""Minimum input firing rate.""")

    p_max = NArray(
        label=":math:`p_{max}`",
        default=numpy.array([0.32]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Maximum input firing rate.""")

    mu = NArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate.""")
    
    # backward connectivity
    eta = NArray(
        label=":math:`\eta`",
        default = numpy.array([0]),
        domain=Range(lo=0.0, hi=1, step=0.001),
        doc="""Backward connectivity speed."""
    )

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "y0": numpy.array([-1.0, 1.0]),
            "y1": numpy.array([-500.0, 500.0]),
            "y2": numpy.array([-50.0, 50.0]),
            "y3": numpy.array([-6.0, 6.0]),
            "y4": numpy.array([-20.0, 20.0]),
            "y5": numpy.array([-500.0, 500.0]),
            "y6": numpy.array([-50.0, 50.0]),
            "y7": numpy.array([-500.0, 500.0]),
        },
        doc="""The values for each state-variable should be set to encompass
    the expected dynamic range of that state-variable for the current
    parameters, it is used as a mechanism for bounding random initial
    conditions when the simulation isn't started from an explicit history,
    it is also provides the default range of phase-plane plots.""")

    state_variables = tuple(
        'y0 y1 y2 y3 y4 y5 y6 y7'.split())
    _nvar = 8

    # variables used in coupling (Jansen Rit coupling - a sigmoid - needs two inputs)
    cvar  = numpy.array([1, 2], dtype=numpy.int32)

    # stimulated variables (Stimulation will affect only these states)
    stvar = numpy.array([1], dtype=numpy.int32)

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=(['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7']),
        default=(['y0', 'y1', 'y2', 'y6']))

    def dfun(self, y, c, local_coupling=0.0):
        # state variables
        y_ = y.reshape(y.shape[:-1]).T

        # long couplings (connectome)
        c_ = c.reshape(c.shape[:-1]).T

        # short couplings (self loop)
        src = local_coupling*(y[1] - y[2])[:,0]

        # model parameters
        params = [
            self.nu_max, self.r, self.v0, self.a,
            self.a_1, self.a_2, self.a_3, self.a_4, 
            self.A, self.b, self.b2, self.B, self.J, 
            self.mu,
        ]
        
        deriv = _numba_dfun_test(y_, c_, src, *params)
        deriv = deriv.T[..., numpy.newaxis]

        return deriv


@guvectorize([(float64[:],)*2 + (float64,)*15 + (float64[:],)], '(n),(m)' + ',()'*15 + '->(n)', nopython=True)
def _numba_dfun_test(
    y,
    c,
    src,
    nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, b2, B, J, mu,
    dx
):
    """
    Derivations of state variables.

    Parameters
    ----------
    y: list[]
        State variables
        - y[0] == y0
        - y[1] == y1
        - y[2] == y2
        - y[3] == y3
        - y[4] == y4
        - y[5] == y5
        - y[6] == y6
        - y[7] == y7
    c: ???
        Long range couplings (connectome).

    src: float
        Short range couplings (self loop).

    nu_max, r, v0, a, a_1, a_2, a_3, a_4, A, b, B, J, mu: float
        Jansen Rit parameters.

    eta, y0_target, tau_d: float
        Feedback Inhibitory Constant (FIC) parameters.
    """
    PSP = y[1] - y[2] - y[6]  # Pyramidal entry with adapted inhibition

    sigm_y1_y2 = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - PSP)))
    sigm_y0_1  = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - (a_1 * J * y[0]))))
    sigm_y0_3  = 2.0 * nu_max / (1.0 + math.exp(r * (v0 - (a_3 * J * y[0]))))

    dx[0] = y[3]
    dx[1] = y[4]
    dx[2] = y[5]
    dx[3] = A*a*(sigm_y1_y2) - 2.0*a*y[3] - a**2*y[0]
    dx[4] = A*a*(mu + a_2 * J * sigm_y0_1 + c[0] + src) - 2.0*a*y[4] - a**2*y[1]
    dx[5] = B*b*(a_4*J*sigm_y0_3 + y[6]*c[0]) - 2.0*b*y[5] - b**2*y[2]
    dx[6] = y[7]
    dx[7] = B*b2*(c[0]) - 2.0*b2*y[7] - b2**2*y[6]
    

class Generic2dOscillator(ModelNumbaDfun):
    r"""
    The Generic2dOscillator model is a generic dynamic system with two state
    variables. The dynamic equations of this model are composed of two ordinary
    differential equations comprising two nullclines. The first nullcline is a
    cubic function as it is found in most neuron and population models; the
    second nullcline is arbitrarily configurable as a polynomial function up to
    second order. The manipulation of the latter nullcline's parameters allows
    to generate a wide range of different behaviours.

    Equations:

    .. math::
            \dot{V} &= d \, \tau (-f V^3 + e V^2 + g V + \alpha W + \gamma I) \\
            \dot{W} &= \dfrac{d}{\tau}\,\,(c V^2 + b V - \beta W + a)

    See:


        .. [FH_1961] FitzHugh, R., *Impulses and physiological states in theoretical
            models of nerve membrane*, Biophysical Journal 1: 445, 1961.

        .. [Nagumo_1962] Nagumo et.al, *An Active Pulse Transmission Line Simulating
            Nerve Axon*, Proceedings of the IRE 50: 2061, 1962.

        .. [SJ_2011] Stefanescu, R., Jirsa, V.K. *Reduced representations of
            heterogeneous mixed neural networks with synaptic coupling*.
            Physical Review E, 83, 2011.

        .. [SJ_2010]	Jirsa VK, Stefanescu R.  *Neural population modes capture
            biologically realistic large-scale network dynamics*. Bulletin of
            Mathematical Biology, 2010.

        .. [SJ_2008_a] Stefanescu, R., Jirsa, V.K. *A low dimensional description
            of globally coupled heterogeneous neural networks of excitatory and
            inhibitory neurons*. PLoS Computational Biology, 4(11), 2008).


    The model's (:math:`V`, :math:`W`) time series and phase-plane its nullclines
    can be seen in the figure below.

    The model with its default parameters exhibits FitzHugh-Nagumo like dynamics.

    +---------------------------+
    |  Table 1                  |
    +--------------+------------+
    |  EXCITABLE CONFIGURATION  |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |     -2.0   |
    +--------------+------------+
    | b            |    -10.0   |
    +--------------+------------+
    | c            |      0.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    |  limit cycle if a is 2.0  |
    +---------------------------+


    +---------------------------+
    |   Table 2                 |
    +--------------+------------+
    |   BISTABLE CONFIGURATION  |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |      1.0   |
    +--------------+------------+
    | b            |      0.0   |
    +--------------+------------+
    | c            |     -5.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    | monostable regime:        |
    | fixed point if Iext=-2.0  |
    | limit cycle if Iext=-1.0  |
    +---------------------------+


    +---------------------------+
    |  Table 3                  |
    +--------------+------------+
    |  EXCITABLE CONFIGURATION  |
    +--------------+------------+
    |  (similar to Morris-Lecar)|
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |      0.5   |
    +--------------+------------+
    | b            |      0.6   |
    +--------------+------------+
    | c            |     -4.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    | excitable regime if b=0.6 |
    | oscillatory if b=0.4      |
    +---------------------------+


    +---------------------------+
    |  Table 4                  |
    +--------------+------------+
    |  GhoshetAl,  2008         |
    |  KnocketAl,  2009         |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |    1.05    |
    +--------------+------------+
    | b            |   -1.00    |
    +--------------+------------+
    | c            |    0.0     |
    +--------------+------------+
    | d            |    0.1     |
    +--------------+------------+
    | I            |    0.0     |
    +--------------+------------+
    | alpha        |    1.0     |
    +--------------+------------+
    | beta         |    0.2     |
    +--------------+------------+
    | gamma        |    -1.0    |
    +--------------+------------+
    | e            |    0.0     |
    +--------------+------------+
    | g            |    1.0     |
    +--------------+------------+
    | f            |    1/3     |
    +--------------+------------+
    | tau          |    1.25    |
    +--------------+------------+
    |                           |
    |  frequency peak at 10Hz   |
    |                           |
    +---------------------------+


    +---------------------------+
    |  Table 5                  |
    +--------------+------------+
    |  SanzLeonetAl  2013       |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | a            |    - 0.5   |
    +--------------+------------+
    | b            |    -10.0   |
    +--------------+------------+
    | c            |      0.0   |
    +--------------+------------+
    | d            |      0.02  |
    +--------------+------------+
    | I            |      0.0   |
    +--------------+------------+
    |                           |
    |  intrinsic frequency is   |
    |  approx 10 Hz             |
    |                           |
    +---------------------------+

    NOTE: This regime, if I = 2.1, is called subthreshold regime.
    Unstable oscillations appear through a subcritical Hopf bifurcation.


    .. figure :: img/Generic2dOscillator_01_mode_0_pplane.svg
    .. _phase-plane-Generic2D:
        :alt: Phase plane of the generic 2D population model with (V, W)

        The (:math:`V`, :math:`W`) phase-plane for the generic 2D population
        model for default parameters. The dynamical system has an equilibrium
        point.

    .. automethod:: Generic2dOscillator.dfun

    """

    # Define traited attributes for this model, these represent possible kwargs.
    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=5.0, step=0.01),
        doc="""A time-scale hierarchy can be introduced for the state
        variables :math:`V` and :math:`W`. Default parameter is 1, which means
        no time-scale hierarchy.""")

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Baseline shift of the cubic nullcline""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([-2.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Vertical shift of the configurable nullcline""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([-10.0]),
        domain=Range(lo=-20.0, hi=15.0, step=0.01),
        doc="""Linear slope of the configurable nullcline""")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""Parabolic term of the configurable nullcline""")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([0.02]),
        domain=Range(lo=0.0001, hi=1.0, step=0.0001),
        doc="""Temporal scale factor. Warning: do not use it unless
        you know what you are doing and know about time tides.""")

    e = NArray(
        label=":math:`e`",
        default=numpy.array([3.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the quadratic term of the cubic nullcline.""")

    f = NArray(
        label=":math:`f`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the cubic term of the cubic nullcline.""")

    g = NArray(
        label=":math:`g`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.5),
        doc="""Coefficient of the linear term of the cubic nullcline.""")

    alpha = NArray(
        label=r":math:`\alpha`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            slow variable to the fast variable.""")

    beta = NArray(
        label=r":math:`\beta`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
            slow variable to itself""")

    # This parameter is basically a hack to avoid having a negative lower boundary in the global coupling strength.
    gamma = NArray(
        label=r":math:`\gamma`",
        default=numpy.array([1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.1),
        doc="""Constant parameter to reproduce FHN dynamics where
               excitatory input currents are negative.
               It scales both I and the long range coupling term.""")
    
    eta = NArray(
        label=":math:`eta `",
        default=numpy.array([0.0001]),
        domain=Range(lo=0.000001, hi=1, step=0.001),
        doc="""adaptation rate parameter for FIC""")
    
    tau_d = NArray(
        label=":math:`tau_d `",
        default=numpy.array([10.0]),
        domain=Range(lo=10, hi=100000., step=10),
        doc="""tau_d - corresponds to the time window over which y0 and y2 are averaged (10 = 10s)""")

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"V": numpy.array([-2.0, 4.0]),
                 "W": numpy.array([-6.0, 6.0]),
                 "G0": numpy.array([0,0]),
                 "C": numpy.array([0,0]),
                 },
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random initial
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("V", "W", "V + W", "V - W", "G0", "C"),
        default=("V", "C"),
        doc="The quantities of interest for monitoring for the generic 2D oscillator.")

    state_variables = ('V', 'W', 'G0', 'C')
    _nvar = 4
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        r"""
        The two state variables :math:`V` and :math:`W` are typically considered
        to represent a function of the neuron's membrane potential, such as the
        firing rate or dendritic currents, and a recovery variable, respectively.
        If there is a time scale hierarchy, then typically :math:`V` is faster
        than :math:`W` corresponding to a value of :math:`\tau` greater than 1.

        The equations of the generic 2D population model read

        .. math::
                \dot{V} &= d \, \tau (-f V^3 + e V^2 + g V + \alpha W + \gamma I) \\
                \dot{W} &= \dfrac{d}{\tau}\,\,(c V^2 + b V - \beta W + a)

        where external currents :math:`I` provide the entry point for local,
        long-range connectivity and stimulation.

        """
        lc_0 = local_coupling * vw[0, :, 0]
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_g2d(vw_, c_, self.tau, self.I, self.a, self.b, self.c, self.d, self.e, self.f, self.g,
                                self.beta, self.alpha, self.gamma, lc_0, self.eta, self.tau_d)
        return deriv.T[..., numpy.newaxis]


@guvectorize([(float64[:],) * 18], '(n),(m)' + ',()' * 15 + '->(n)', nopython=True)
def _numba_dfun_g2d(vw, c_0, tau, I, a, b, c, d, e, f, g, beta, alpha, gamma, lc_0, eta, tau_d, dx):
    "Gufunc for Generic2dOscillator model equations."
    V = vw[0]
    V2 = V * V
    W = vw[1]
    dx[0] = d[0] * tau[0] * (
                alpha[0] * W - f[0] * V2 * V + e[0] * V2 + g[0] * V + gamma[0] * I[0] + c_0[0] - vw[3]  + lc_0[0])
    dx[1] = d[0] * (a[0] + b[0] * V + c[0] * V2 - beta[0] * W) / tau[0]
    dx[2] = (c_0[0]-vw[2]) / tau_d[0]
    dx[3] = eta[0]*(vw[2]-vw[3])
