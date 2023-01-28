from scipy.special import gamma as Gamma
import numpy as np
from scipy.stats import poisson as Poisson_distribution
from scipy.special import kv,kn


def in_cone(env,r_ag,r0,tol=1e-3,n=1):
    return env.get_likelihood(r_ag[0]-r0[0],r_ag[1]-r0[1],1)>tol

def volume_ball(r, Ndim=2, norm='Euclidean'):
        if Ndim is None:
            Ndim = self.Ndim
        if norm == 'Manhattan':
            pm1 = 1
        elif norm == 'Euclidean':
            pm1 = 1 / 2
        elif norm == 'Chebyshev':
            pm1 = 0
        else:
            raise Exception("This norm is not implemented")
        return (2 * Gamma(pm1 + 1) * r) ** Ndim / Gamma(Ndim * pm1 + 1)

def initial_hit_aurore(env,hit=None):
    if hit is None:
        p_hit_table = np.zeros(env.numobs-1)
        r = np.arange(1, int(1000 * np.sqrt(env.D*env.tau)/env.dx))    
        shell_volume = volume_ball(r+0.5) - volume_ball(r-0.5)
        for h in range(1, env.numobs-1):
            p = Poisson(env,mean_number_of_hits(env,r), h)  # proba hit=h as a function of distance r to the source
            p_hit_table[h] = max(0, np.sum(p * shell_volume))  # not normalized
        p_hit_table /= np.sum(p_hit_table)
        hit = np.random.RandomState().choice(range(env.numobs-1), p=p_hit_table)
    return hit

def Poisson(env, mu, h):
        if h < env.numobs - 2:   # = Poisson(mu,hit=h)
            p = Poisson_unbounded(mu, h)
        elif h == env.numobs - 2:     # = Poisson(mu,hit>=h)
            sum = 0.0
            for k in range(h):
                sum += Poisson_unbounded(mu, k)
            p = 1 - sum
        else:
            raise Exception("h cannot be > Nhits - 1")
        return p


def Poisson_unbounded(mu, h):
        p = Poisson_distribution(mu).pmf(h)
        return p

def mean_number_of_hits(env, distance,Ndim=2):
        distance = np.array(distance)
        distance[distance == 0] = 1.0
        if Ndim == 1:
            mu = np.exp(-distance / np.sqrt(env.D*env.tau)*env.dx + 1)
        elif Ndim == 2:
            mu = kn(0, distance / np.sqrt(env.D*env.tau)*env.dx)/ kn(0, 1)
        elif Ndim == 3:
            mu = np.sqrt(env.D*env.tau)/env.dx / distance * np.exp(-distance / np.sqrt(env.D*env.tau)*env.dx + 1)
        elif Ndim > 3:
            mu = (np.sqrt(env.D*env.tau)/env.dx / distance) ** (Ndim / 2 - 1) \
                 * kv(Ndim / 2 - 1, distance*env.dx/ np.sqrt(env.D*env.tau)) \
                 / kv(Ndim / 2 - 1, 1)
        else:
            raise Exception("Problem with the number of dimensions")
        mu *= mu0_Poisson(env)
        return mu

def mu0_Poisson(env,Ndim=2):
        """Sets the value of mu0_Poisson (mean number of hits at distance = lambda), which is derived from the
         physical dimensionless parameters of the problem. It is required by _mean_number_of_hits().
        """
        dx_over_a = env.dx/env.agent_size  # agent step size / agent radius
        lambda_over_a = np.sqrt(env.D*env.tau)/env.agent_size
        a_over_lambda = 1.0 / lambda_over_a

        if Ndim == 1:
            mu0_Poisson = 1 / (1 - a_over_lambda) * np.exp(-1)
        elif Ndim == 2:
            mu0_Poisson = 1 / np.log(lambda_over_a) * kn(0, 1)
        elif Ndim == 3:
            mu0_Poisson = a_over_lambda * np.exp(-1)
        elif Ndim > 3:
            mu0_Poisson = (Ndim - 2) / Gamma(Ndim / 2) / (2 ** (Ndim / 2 - 1)) * \
                          a_over_lambda ** (Ndim - 2) * kv(Ndim / 2 - 1, 1)
        else:
            raise Exception("problem with Ndim")

        mu0_Poisson *= env.R*env.dt
        return mu0_Poisson


