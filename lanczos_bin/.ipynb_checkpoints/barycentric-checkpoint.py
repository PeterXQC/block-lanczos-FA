import numpy as np

def compute_barycentric_weights(xt,dtype=np.longdouble):
    """
    compute weights for barycentric interpolation points xt[i]
    
    Input
    -----
    xt : interpolations points
    
    Returns
    -------
    w : weights
    """
    
    n = len(xt)
    
    # dists[i,j] = xt[i] - xt[j]
    # note that adding the idenity does not affect the numerator, and that log(1) = 0 so the diagonal terms are dropped from the sum in the denominator
    dists = (xt[:,None] - xt) + np.eye(n,dtype=dtype)

    num = np.prod(np.sign(dists),axis=1)
    denom = np.exp(n*np.log(2) + np.sum(np.log(np.abs(dists)),axis=1))
    
    return num/denom


def barycentric(x,xt,yt,w):
    """
    compute polynomial at x given weights w and interpolation points xt passing through points pt[i]
        
    Input
    -----
    x : values to evaluate function at
    xt : interpolations points
    pt : value of function at interpolation points
    w : barycentric weights
    
    Returns
    -------
    p : value of function at x
    
    Notes
    -----
    the approximation will fail if you are at an interpolation point. 
    We replace very small numbers with a very small number since this lets us recover yt
        - I'm not 100% sure if this is stable or not
    Right now when you run the code on a vector vs single values it returns different things..
        
    """

#    x_xj = x - xt[:,None]
    # make sure x is a numpy array
    x_xj = np.reshape(x,-1)[:,None] - xt
    
    # replace zeros with something very small
    bad_idx = np.where(x_xj == 0)
    x_xj[bad_idx] = 1e-100

    
#    num = np.sum(np.diag(w*yt)@(1/x_xj),axis=0)
#    denom = np.sum(np.diag(w)@(1/x_xj),axis=0)
    num = np.sum((w*yt)*(1/x_xj),axis=1)
    denom = np.sum(w*(1/x_xj),axis=1)
    
    p = num/denom
    
    return p
    