"""Waloddi Weibull invented the weibull distribution.

The weibull distribution is commonly used to model duration-until-event scenarios.
Examples include amount of time until a widget breaks, customer churn rates, and
number of times one could bend a paperclip until it snaps.

The weibull distribution is particularly useful for this modeling as it can
reduce to the exponential and rayleigh distributions as well as approximate
a normal distribution. It can also model both early life failure (e.g. defects
in a semiconductor) as well as wear out failure (e.g. creep in metal structures).

This package is an aggregation of various simple 2-parameter weibull functions,
useful for modeling duration-until-event scenarios and learning about the
properties of the weibull distribution.
"""

import pandas as pd
import numpy as np

def pdf(beta, eta, x):
    '''Calculates the PDF of a weibull distribution
    with shape (beta) and scale (eta) parameters.
    
    Args:
        beta (float): shape parameter
        eta (float): scale parameter
        x (float): duration at which to calculate value (e.g. hours, damage, days, etc.)
        
    Returns
        float: weibull pdf value at x
    '''
    if x < 0: return 0
    pdf = beta / eta * (x / eta)**(beta - 1.0) * np.exp(-(x / eta)**beta)
    if isinstance(pdf, np.ndarray): pdf[pdf == np.inf] = 0
    elif np.isinf(pdf): pdf = 0
    return pdf

def cdf(beta, eta, x):
    '''Calculates the CDF of a weibull distribution
    with shape (beta) and scale (eta) parameters
    
    Args:
        beta (float): shape parameter
        eta (float): scale parameter
        x (float): duration at which to calculate value (e.g. hours, damage, days, etc.)
        
    Returns
        float: weibull cdf value at x
    '''
    if x < 0: return 0
    cdf = 1 - np.exp(-(x / eta)**beta)
    if isinstance(cdf, np.ndarray): cdf[cdf == np.inf] = 0
    elif np.isinf(cdf): cdf = 0
    return cdf

def unreliability(beta, eta, x):
    '''Calculate unreliability of a 2 parameter weibull distribution at duration x.
    Note that weibull unreliability and CDF are the same.
    
    Args:
        beta (float): shape parameter
        eta (float): scale parameter
        x (float): duration at which to calculate value (e.g. hours, damage, days, etc.)
        
    Returns
        float: unreliability at duration x
    '''
    import numpy as np
    return cdf(beta, eta, x)

def reliability(beta, eta, x):
    '''Calculate reliability of a 2 parameter weibull distribution at duration x
    Reliability R(x) is also often called the Survival Function.
    
    Args:
        beta (float): shape parameter
        eta (float): scale parameter
        x (float): duration at which to calculate value (e.g. hours, damage, days, etc.)
        
    Returns
        float: reliability
    '''
    return 1 - unreliability(beta, eta, x)

def failure_rate(beta, eta, x):
    '''Calculate instantaneous failure rate of a 2 parameter weibull distribution at duration x
    given that no failure has occurred up to duration x.
    
    Continuous failure rate is often called the Hazard Function.
    https://en.wikipedia.org/wiki/Failure_rate#Failure_rate_in_the_continuous_sense
    
    Units are [1 / units of x].
    
    Args:
        beta (float): shape parameter
        eta (float): scale parameter
        x (float): duration at which to calculate value (e.g. hours, damage, days, etc.)
        
    Returns
        float: failure rate
    '''
    return pdf(beta, eta, x) / reliability(beta, eta, x)

def median_rank(durations, events):
    '''Median rank is the method for estimating point estimates of unreliability for events -- the 
    y-values of the points on a weibull plot.
    
    The idea is to solve for the probability (Z-value) of the binomial CDF, summed for each failure,
    as outlined here: https://www.weibull.com/hotwire/issue187/hottopics187.htm
    
    In reality this is computationally difficult and we use Benard's approximation, which is
    a pretty good approximation as per the table at the bottom of the above link.
    
    Note that for Median Rank Regression, median ranks are manipulated into the linear CDF
    log-space before performing linear regression, but this function returns (x, y) rather
    than (ln(x), ln(-ln(1-y))).

    Args:
        events (iterable): series of booleans indicating affirmative event or not
        
    Returns:
        x_median_ranks (np.ndarray): durations of each event
        y_median_ranks (np.ndarray): point estimates of unreliability for each event
    '''
    events = np.array(events, dtype='bool')
    durations = np.array(durations)
    
    if not sum(events):
        return [], []

    # Sort events and durations by duration in ascending order
    events = events[durations.argsort()]
    durations = np.sort(durations)

    x_median_ranks = durations[events]

    # Get mean order numbers, important when suspensions exist (https://www.weibull.com/hotwire/issue187/hottopics187.htm)
    failure_indices = np.ravel(np.argwhere(events)) + 1 # index from 1
    N = len(durations) # number of samples
    mean_order_numbers = [1]
    mon_prev = mean_order_numbers[0]
    for i in failure_indices[1:]:
        mon = mon_prev + ((N + 1.0) - mon_prev) / (2.0 + N - i) # denom is (1 + # samples beyond present set)
        mon_prev = mon
        mean_order_numbers.append(mon)

    ## Benards approximation for Mean Rank
    # It is difficult to get the "true" Median Rank estimate using the Z-value
    # of the binomial CDF when P = 0.5 due to numerical difficulties (the binomial
    # coefficient is numerically undefined at large "N choose k" and solving for
    # Z can have convergence issues). Luckily the Benard approximation is extremely
    # simple and gives similar results to the "true" Median Rank as seen in the table
    # at the bottom here: https://www.weibull.com/hotwire/issue187/hottopics187.htm
    y_median_ranks = (np.array(mean_order_numbers) - 0.3) / (N + 0.4)
    
    return x_median_ranks, y_median_ranks

def median_rank_regression(durations, events):
    '''Median Rank Regression is one method of fitting a weibull plot to a set of failure data.
    Find the median ranks of failures, then fit a linear model to x = ln(-ln(1-unrels)), y=ln(durations).
    
    Waloddi recommended the "X vs Y" rather than "Y vs X" fitting because maximum variance is in the
    duration dimension rather than the unreliablity dimension.
    
    Args:
        durations (iterable): durations of events and suspensions
        events (iterable): associated booleans labels indicating event (1 or True) or suspension (0 or False)
    
    Returns:
        tuple: (shape/beta, scale/eta) parameters fit with MRR
    '''
    x_rank, y_rank = median_rank(events, durations)

    x = np.log(x_rank)
    y = np.log(-np.log(1 - y_rank))

    slope, intercept = _linear_regression(y, x)

    beta = 1 / slope
    eta = np.exp(intercept)
    return beta, eta

def _linear_regression(x, y):
    '''Helper function performing linear regression'''
    from scipy.stats import linregress
    fit = linregress(x, y)
    return fit.slope, fit.intercept

def mle_fit(durations, events):
    '''Maximum Likelihood Estimation is another method for fitting a weibull plot to a set of duration/event data,
    with and without suspensions.
    
    MLE fitting finds the set of parameters (in the 2-parameter weibull case, Beta and Eta) that maximize the likelihood
    function. The likelihood function is the product of the PDF values for each observation (modified for suspensions in
    the weibull case). Usually the log-likelihood is maximized as it's easier to solve a summation than a product.
    
    Methodology here is from Abernathy Weibull handbook, Appendix C-3.
    
    Note that MLE is preferable over median rank regression almost all the time, especially with suspension data.

    Args:
        durations (iterable): durations of events and suspensions
        events (iterable): associated booleans labels indicating event (1 or True) or suspension (0 or False)
    
    Returns:
        tuple: (shape/beta, scale/eta) parameters fit with MLE
    '''
    from scipy.optimize import brentq
    
    durations = np.array(durations)
    events = np.array(events, dtype='bool')
    
    beta = brentq(lambda x: _beta_likelihood(x, durations, events),
                  a=0,
                  b=15)
    eta = _eta_likelihood(beta, durations, events)
    return (beta, eta)

def _beta_likelihood(beta_estimate, durations, events):
    '''Helper function calculating beta likelihood
    From Abernathy Weibull handbook, Appendix C-3
    '''
    t = np.array(durations)
    e = np.array(events, dtype='bool')
    tf = t[e]
    
    tn = t / t.min()
    tfn = tf / t.min()

    A = np.sum(np.log(tfn) * 1.0) / tfn.size
    B = lambda k: np.sum(tn ** k)
    C = lambda k: np.sum(tn ** k * np.log(tn))
    D = lambda k: np.sum(np.log(tn) ** 2 * tn ** k)

    beta_likelihood = B(beta_estimate) / C(beta_estimate) - beta_estimate / (1 + beta_estimate * A)
    return beta_likelihood

def _eta_likelihood(beta, durations, events):
    '''Helper function calculating eta likelihood
    From Abernathy Weibull handbook, Appendix C-3
    '''
    num = np.sum(durations**beta)
    eta = (num / sum(events))**(1 / beta)
    return eta

def mle_vs_mrr_example():
    '''Plots a simple example where it is much better to use MLE fitting than MRR fitting
    due to the existance of suspensions.
    
    Args:
        None
        
    Returns:
        None
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    from seaborn import xkcd_rgb as xkcd
    sns.set_style('whitegrid')
    
    durations = [300, 400, 500] + [1100]*8
    events = [1, 1, 1] + [0]*8

    beta_mle, eta_mle = mle_fit(durations, events)
    beta_mrr, eta_mrr = median_rank_regression(durations, events)

    unreliability(beta_mle, eta_mle, 1100)

    x = np.logspace(2, 3.2)
    x_event, y_event = median_rank(events, durations)
    y = y_weibull(x, beta_mle, eta_mle)
    y_mrr = y_weibull(x, beta_mrr, eta_mrr)

    fig, ax = plt.subplots(figsize=(16,9))
    ax.plot(x, y, c=xkcd['denim blue'], label='MLE')
    ax.plot(x, y_mrr, c=xkcd['medium green'], label='MRR')
    ax.scatter(x_event, y_event, c=xkcd['pale red'], label='Failures')

    ax.axvline(1e3)
    leg = ax.legend(frameon=True, framealpha=1)
    ax.set_xscale('log')
    ax.set_yscale('log')

    print('True Unrel at 1100Hrs: {}'.format(3/11*100))
    print('MLE Unrel at 1100Hrs: {}'.format(unreliability(beta_mle, eta_mle, 1100)*100))
    print('MRR Unrel at 1100Hrs: {}'.format(unreliability(beta_mrr, eta_mrr, 1100)*100))

def mle_confidence(durations, events, confidence=0.95, verbose=False):
    '''MLE fit confidence bounds implemented by Ariel Shemtov.
    
    Performs some things that other CB determination methods don't do like correcting parameters
    by performing a transform should they fall out of rational bounds.
    
    There seems to be diverse literature on how to get MLE confidence bounds with suspensions.
    This is one implementation.

    Args:
        durations (iterable): durations of events and suspensions
        events (iterable): associated booleans labels indicating event (1 or True) or suspension (0 or False)
        confidence (float): default 0.95, two-sided confidence level from 0. to 1. e.g. 0.95 provides P(X>0.025 and X<0.975)
        verbose (bool): default False, print information about confidence fitting
        
    Returns:
        dict: keys 'beta'-->(beta_lower, beta_upper), 'eta'-->(eta_lower, eta_upper), 'beta_nom', 'eta_nom', 'confidence'
    '''

    beta, eta = mle_fit(durations, events)
    
    beta_std_dev, eta_std_dev = _weibull_std_devs(
        durations,
        events,
        beta,
        eta,
        verbose
    )

    bounds = _weibull_confidence_bounds(
        beta,
        eta,
        beta_std_dev,
        eta_std_dev,
        confidence=confidence
    )

    # Perform transform/correction on out of range bounds
    for param, se in zip(('beta', 'eta'), (beta_std_dev, eta_std_dev)):
        if np.any([_v < 0 for _v in bounds[param]]):
            if verbose: print('Bounds on {} was {}'.format(param, bounds[param]))
            bounds[param] = _transform_bounds_exp(se)
            if verbose: print('Bounds on {} now {}'.format(param, bounds[param]))
    return bounds
    
def _weibull_std_devs(durations, events, beta, eta, verbose=False):
    '''Helper function to get standard deviations of each parameter.
    '''
    hess_val = _weibull_hessian(durations, events, beta, eta)

    fisher_information_matrix = - hess_val

    if verbose:
        print('Inverse Fisher Information Matrix:')
        print(np.linalg.inv(fisher_information_matrix))
        
    std_dev = np.sqrt(np.diag(np.linalg.inv(fisher_information_matrix)))
    beta_std_dev, eta_std_dev = std_dev[0], std_dev[1]
    return beta_std_dev, eta_std_dev

def _weibull_hessian(durations, events, beta, eta):
    '''Helper function to get weibull hessian matrix (negative of fisher information matrix).
    '''
    r = np.sum(events)
    h = np.zeros((2,2))
    h[0,0] = (-r / beta**2
              - np.sum((durations / eta)**beta * (np.log(durations / eta))**2))
    h[1,1] = (r * beta / eta**2
              - (beta + 1) * beta / eta**(beta + 2) * np.sum(durations**beta))
    h[0,1] = h[1,0] = (-r / eta
                       + np.sum((durations / eta)**beta)
                       / eta
                       + (beta / eta) * np.sum(np.log(durations / eta) * (durations / eta)**beta))
    return h

def _weibull_confidence_bounds(beta, eta, beta_std_dev, eta_std_dev, confidence=0.95):
    '''Helper function to get confidence bounds on MLE.
    '''
    from scipy import stats
    
    K = stats.norm.ppf((1 + confidence) / 2.)

    beta_lower = beta - K * beta_std_dev
    eta_lower = eta - K * eta_std_dev
    beta_upper = beta + K * beta_std_dev
    eta_upper = eta + K * eta_std_dev
    return {
        'beta_nom':beta,
        'eta_nom':eta,
        'confidence':confidence,
        'beta':(beta_lower, beta_upper),
        'eta':(eta_lower, eta_upper),
    }

def _transform_bounds_exp(std_dev):
    '''Helper function to transform bounds if values are unreasonable (e.g. a negative eta).
    '''
    def _exp(p, a=0., sg=1.):
        fun_val  = sg*np.log(sg*(p-a))
        grad_val = float(sg)/(p-a)
        return fun_val, grad_val
    
    def _exp_inv(phi, a=0., sg=1.):
        return a + sg*np.exp(sg*phi)
    
    phi, phi_prime = _exp(std_dev)
    phi_inv = lambda x: _exp_inv(x)
    
    lb = phi_inv(phi - np.abs(phi_prime) * std_dev)
    ub = phi_inv(phi + np.abs(phi_prime) * std_dev)
    return lb, ub

def y_weibull(x, beta, eta, space='normal'):
    '''Calculate the y points for a Weibull Plot (linearized unreliability).
    Based on fact that ln(-ln(1 - F(x)) = beta * ln(x) - beta * ln(eta).
    
    By default returns in "normal" space so you can plot the data then set y-axis
    to be log to see linear plot.
    
    Args:
        x (iterable): points to evaluate y points
        beta (float): shape parameter
        eta (float): scale parameter
        space (str): default 'normal', 'normal' or 'weibull'.
            If 'normal', plot on log of y-axis.
            If 'weibull', plot on standard y-axis
    
    Returns:
        np.ndarray: y points in space specified
    '''
    x = np.array(x)
    y_points = beta * np.log(x) - beta * np.log(eta)
    if space == 'normal':
        y_points = 1 - np.exp(-np.exp(y_points))
    return y_points

def get_eta_from_point(y, x, beta):
    '''Solve for the eta/scale parameter given a specific
    unreliability (y), duration (x) and beta/shape parameter.
    
    Args:
        y (float): known unreliability point at duration
        x (float): known duration at unreliability
        beta (float): shape parameter of distribution
        
    Returns:
        float: eta/scale parameter according to inputs
    '''
    return x / ((-1*np.log(1-y))**(1/beta))

def get_eta_lower_chisq(beta, durations, events, confidence=0.95):
    '''With a very small number of events but a known (or assumed) beta exists, we can use the
    chi-square distribution to calculate a lower bound on eta with some confidence.
    
    When the Weibull shape parameter beta is known, the random variable "t^beta" follows
    an exponential distribution with mean of "eta^beta" whose confidence bound is 
    calculated by Chi-square distribution.
    
    Args:
        beta (float): shape parameter, known or assumed
        durations (iterable): durations of events and suspensions
        events (iterable): associated booleans labels indicating event (1 or True) or suspension (0 or False)
        confidence (float): In (0.0,1.0) default 0.95, chi-square confidence level
    
    Returns:
        float: lower bound for eta/scale parameter with input confidence
    '''
    from scipy import stats
    num_failures = sum(events)
    dof = 2 * num_failures + 2
    return (2 * np.sum(durations**beta) / stats.chi2.ppf(confidence, dof))**(1 / beta)

def get_eta_upper_chisq(beta, durations, events, confidence=0.95):
    '''With a very small number of events but a known (or assumed) beta exists, we can use the
    chi-square distribution to calculate an upper bound on eta with some confidence.
    
    When the Weibull shape parameter beta is known, the random variable "t^beta" follows
    an exponential distribution with mean of "eta^beta" whose confidence bound is 
    calculated by Chi-square distribution.
    
    Args:
        beta (float): shape parameter, known or assumed
        durations (iterable): durations of events and suspensions
        events (iterable): associated booleans labels indicating event (1 or True) or suspension (0 or False)
        confidence (float): In (0.0,1.0) default 0.95, chi-square confidence level
    
    Returns:
        float: upper bound for eta/scale parameter with input confidence
    '''
    from scipy import stats
    num_failures = sum(events)
    dof = 2 * num_failures
    return (2 * np.sum(durations**beta) / stats.chi2.ppf(1-confidence, dof))**(1 / beta)

def get_eta_two_sided_chisq(beta, durations, events, confidence=0.95):
    '''With a very small number of events but a known (or assumed) beta exists, we can use the
    chi-square distribution to calculate a bound on eta with some confidence.
    
    When the Weibull shape parameter beta is known, the random variable "t^beta" follows
    an exponential distribution with mean of "eta^beta" whose confidence bound is 
    calculated by Chi-square distribution.
    
    Args:
        beta (float): shape parameter, known or assumed
        durations (iterable): durations of events and suspensions
        events (iterable): associated booleans labels indicating event (1 or True) or suspension (0 or False)
        confidence (float): In (0.0,1.0) default 0.95, chi-square two-sided confidence level
    
    Returns:
        tuple: (lower, upper) bound for eta/scale parameter with input confidence
    '''
    from scipy import stats
    num_failures = sum(events)
    alpha = 1 - confidence
    
    dof_lower = 2 * num_failures + 2
    dof_upper = 2 * num_failures
    
    lower_bound = (2 * np.sum(durations**beta) / stats.chi2.ppf(alpha/2, dof_lower))**(1 / beta)
    upper_bound = (2 * np.sum(durations**beta) / stats.chi2.ppf(1-alpha/2, dof_upper))**(1 / beta)
    
    return (lower_bound, upper_bound)

def format_weibull(ax):
    '''Format matplotlib weibull plot axis
    
    Args:
        ax (matplotlib.axis): axis object to format
        
    Returns:
        matplotlib.axis: formatted axis
    '''
    import matplotlib.pyplot as plt

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='major', lw=3, alpha=0.7)
    ax.grid(True, which='minor', lw=1, alpha=0.5)
    
    yticks = np.array([0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 3])
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(x*100)+'%' if x < 0.99 else '' for x in yticks])
    ax.set_ylim(ymin=min(yticks)*0.1, ymax=max(yticks))

    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='x', which='major', labelsize=16)
    plt.tight_layout()
    
    return ax

def plot_weibull(durations,
                 events,
                 ax=None,
                 confidence=0.95,
                 show_events=True,
                 verbose=False,
                 **kwargs):
    '''Make a weibull plot or add a weibull plot to an existing plot.
    Uses MLE fitting for plot line and median rank for point estimates.
    
    Args:
        durations (iterable): durations of events and suspensions
        events (iterable): associated booleans labels indicating event (1 or True) or suspension (0 or False)
        ax (matplotlib.axis): default None, axis to add plot to
        confidence (float): default 0.95, number between 0 and 1
        show_events (bool): default True, flag to show median rank estimates of events
        verbose (bool): default False, print information about fit and confidence bounds
        **kwargs:
            figsize, min_x, max_x, color, label, events_label
            leg_ncols, yticks, low_bound_label, up_bound_label
    
    Returns:
        matplotlib.axis: axis object of plot
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    from seaborn import xkcd_rgb as xkcd
    sns.set_style('whitegrid')
    
    durations = np.array(durations)
    events = np.array(events, dtype='bool')
    
    if sum(events) < 5:
        if verbose:
            print('WARNING: less than 5 events! Results with MLE fit may have little confidence.')
            print('Consider assuming a beta value and using Chi-Squared upper bounds or Weibayes analysis instead')
        
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (16,9)))
    
    # Get median ranks for event point estimates
    x_events, y_events = median_rank(events, durations)
    
    # Fit using MLE
    # MLE Recommended for almost all situations here: 
    # https://pdfs.semanticscholar.org/da22/02a342b0ec6638ccbe78b992ed4371b549a8.pdf
    beta, eta = mle_fit(durations, events)

    # Compute nominal weibull fit for plotting
    min_x = kwargs.get('min_x', 0.1)
    max_x = kwargs.get('max_x', 1e5)
    x = np.linspace(min_x, max_x, 1000)
    y = y_weibull(x, beta, eta)

    # Plot
    default_label = ('\nBeta: '+ str(np.round(beta,2))
                     + '\nEta: ' + '{:.2e}'.format(eta)
                     + '\nF: ' + str(sum(events)) + ', S: ' + str(len(durations) - sum(events))
                     + '\n')
    ax.plot(x, y, lw=3, c=kwargs.get('color', 'k'), label=kwargs.get('label', '') + default_label)
    if show_events:
        ax.scatter(x_events, y_events, lw=3, c=kwargs.get('color', 'k'), label=kwargs.get('events_label', ''))

    if confidence:
        ax = _plot_mle_confidence_bounds(
            ax,
            kwargs,
            x,
            durations,
            events,
            confidence,
            confidence_type,
            verbose
        )
    

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=11, ncol=kwargs.get('leg_ncols',1))

    ax.set_xlim(xmin=min_x, xmax=max_x)
    
    ax = format_weibull(ax)
    return ax

def _plot_mle_confidence_bounds(ax,
                                kwargs,
                                x,
                                durations,
                                events,
                                confidence,
                                verbose):
    '''Helper function for getting and plotting confidence bounds of desired type.
    '''
    # Get bounds based on confidence_type
    bounds = mle_confidence(durations, events, confidence, verbose)
    
    y_lower = y_weibull(x, bounds['beta'][0], bounds['eta'][0])
    y_upper = y_weibull(x, bounds['beta'][1], bounds['eta'][1])

    ax.plot(x, y_lower, lw=1, alpha=0.5, c=kwargs.get('color', 'k'), label=kwargs.get('low_bound_label', ''))
    ax.plot(x, y_upper, lw=1, alpha=0.5, c=kwargs.get('color', 'k'), label=kwargs.get('up_bound_label', ''))
    ax.fill_between(x, y_lower, y_upper, alpha=0.05, color=kwargs.get('color', 'k'))
    return ax

def get_monte_carlo_bounds(beta,
                           eta,
                           n_events,
                           n_samples,
                           confidence=0.95,
                           n_sims=1000,
                           x=None,
                           plot=False):
    '''Monte Carlo bounds on MLE weibull fit given a specific number of events.
    Same implementation as in Reliasoft's simulation toolbox.
    
    Known to overestimate beta, as noted in appendix in Abernathy handbook. As such, not
    great for extrapolating out long distances and bounds are conservative.
    
    Args:
        beta (float): shape parameter, nominally true values
        eta (float): scale parameter, nominally true values
        n_events (int): number of events
        n_samples (int): total number of samples, including number of events
        confidence (float): default 0.95, confidence between 0. and 1.
        n_sims (int): default 1000, number of MC simulations to run
        x (iterable): default None, duration values to evaluate reliability bounds at
            If None, default values used.
        plot (bool): default False, flag for plotting simulations and bounds
    
    Returns:
        tuple: (lower bound on reliability, upper bound on reliability, axis object)
    '''
    from scipy.stats import scoreatpercentile

    betas, etas, ax = _get_monte_carlo_params(beta, eta, n_events, n_sims, n_samples, plot)
    
    lower_conf = (1 - confidence) / 2. * 100.
    upper_conf = (1 - (1 - confidence) / 2.) * 100.
    
    if x is None:
        x = np.logspace(np.floor(np.log10(1e-3)), np.ceil(np.log10(1e4)), 1000)
    else:
        x = np.array(x)
    
    lb, ub = [], []
    for _x in x:
        unrels = []
        for b, n in zip(betas, etas):
            unrels.append(unreliability(b, n, _x))
        lb.append(scoreatpercentile(unrels, lower_conf))
        ub.append(scoreatpercentile(unrels, upper_conf))
    
    if plot:
        ax.set_title('{}% Confidence Bounds'.format(int(confidence*100)), fontsize=24)
        ax.plot(x, lb, color='pink', lw=3, ls='--', zorder=1000)
        ax.plot(x, ub, color='pink', lw=3, ls='--', zorder=1000)
        ax.fill_between(x, lb, ub, color='pink', alpha=0.3, zorder=1000)
    return lb, ub, ax

def _get_monte_carlo_params(beta, eta, n_events, n_sims=1000, n_samples=10000, plot=False):
    '''Helper function that performs monte carlo sampling to get set of betas and etas.
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    from seaborn import xkcd_rgb as xkcd
    sns.set_style('whitegrid')
    from scipy.stats import exponweib

#     ax = None
    betas, etas = [], []
    for ii in range(n_sims):
        # Sample a number of events from the underlying input distribution
        durations = np.sort(exponweib.rvs(
            a=1,
            c=beta,
            loc=0,
            scale=eta,
            size=n_samples
        ))

        # Set smallest n_events durations to be events
        events = np.array([False] * len(durations))
        events[:n_events] = True

        # Set all suspension durations to the maximum event duration
        max_failure_duration = np.max(durations[events])
        durations[~events] = max_failure_duration

        # MLE fit to sampled data
        beta_sim, eta_sim = mle_fit(durations, events)
        betas.append(beta_sim)
        etas.append(eta_sim)
        
    if plot:
        fig, ax = plt.subplots(figsize=(16,9))
        ax.set_ylim(ymin=1e-4, ymax=1e-1)
        
        x = np.logspace(-3, np.ceil(np.log10(eta)), 10)
        y_true = y_weibull(x, beta, eta)
        ax.plot(x, y_true, c=xkcd['denim blue'], label='True', lw=5, zorder=100)
        
        for ii, (beta_sim, eta_sim) in enumerate(zip(betas, etas)):
            y_sim = y_weibull(x, beta_sim, eta_sim)
            ax.plot(x, y_sim, c=xkcd['medium green'],
                    label='Monte Carlo Simulation\nSimulations: {}\nSamples/Sim: {}\nEvents/Sim: {}'\
                    .format(n_sims, n_samples, n_events) if ii==1 else '')

        leg = ax.legend(frameon=True, framealpha=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax = None
    return betas, etas, ax