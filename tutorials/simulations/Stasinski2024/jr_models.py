import math
import numpy
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final
from tvb.simulator.models.base import ModelNumbaDfun

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
    cvar = numpy.array([1, 2], dtype=numpy.int32)

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
    