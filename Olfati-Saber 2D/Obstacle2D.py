import numpy as np

class Obstacle:
    def __init__(self, kind, **kwargs):
        """
        kind = 'hyperplane' or 'sphere'
        for hyperplane:  normal=<2‑vector>, point=<2‑vector>
        for sphere:     center=<2‑vector>, radius=<scalar>
        """
        self.kind = kind
        if kind == 'hyperplane':
            self.a = np.asarray(kwargs['normal'], float)
            self.y = np.asarray(kwargs['point'],  float)
        elif kind == 'sphere':
            self.y = np.asarray(kwargs['center'], float)
            self.R = float(kwargs['radius'])
        else:
            raise ValueError("unknown obstacle kind")
        
def beta_agent_from_obstacle(qi, pi, obs):
    """
    qi, pi: 2‑vectors, the α‑agent’s position & velocity
    obs:    an Obstacle instance
    returns: (q_hat, p_hat) as in Lemma 4
    """
    I = np.eye(2)

    if obs.kind == 'hyperplane':
        # P = I - a a^T
        a = obs.a / np.linalg.norm(obs.a)
        P = I - np.outer(a, a)
        q_hat = P.dot(qi) + (I - P).dot(obs.y)
        p_hat = P.dot(pi)

    else:  # sphere
        delta = qi - obs.y
        dist  = np.linalg.norm(delta)
        a     = delta / dist
        mu    = obs.R / dist
        P     = I - np.outer(a, a)
        q_hat = mu * qi + (1 - mu) * obs.y
        p_hat = mu * P.dot(pi)

    return q_hat, p_hat