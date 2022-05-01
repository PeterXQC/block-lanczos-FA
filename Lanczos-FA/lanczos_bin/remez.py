import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from .barycentric import compute_barycentric_weights, barycentric

def check_reference(X,L):
    
    assert np.min(np.abs(X-X[:,None]+np.eye(len(X)))) != 0, f'reference must not contain duplicate points {X}'

    for x in X:
        assert np.any((x>=L[:,0])*x<=L[:,1]), f'reference point not contained in intervals {x} {x-L}'

def optimize_over_intervals(f,L,m0):
    """
    find global minimum of f over intervals L
    
    Input
    -----
    f : function to minimize
    L : union of intervals
    m0 : point at which function is small
    
    Returns
    -------
    x : location of minima
    """
    
    n_test_pts = 10
    
    num_int = len(L)
    local_min = np.zeros(num_int,dtype=np.longdouble)
    
    # loop over intervals of L
    for k in range(num_int):
        
        a,b = L[k]
        
        # if interval is a single point, return that point as candidate
        if a==b:
            local_min[k] = a
        
        # TODO: maybe case if |b-a| < tol

        # otherwise, apply general solver to find global minimum of f
        else:

            # define mesh to get approximate best solution
            # roughly, want mesh spacing to be able to resolve any features in f
            # i.e. want that f is monotonic (convex?) between x0 and true minima
            xx = np.linspace(a,b,300)

            # take initial guess as best point on mesh
            x0 =xx[np.argmin(f(xx))]
            
            # now optimize starting at this guess
            res = sp.optimize.minimize(f,x0,bounds=[(a,b)],tol=1e-15)
            local_min[k] = res.x

    # find best point from candidates in each interval of L
    opt_idx = np.argmin(f(local_min))
    x_opt = local_min[opt_idx]
    
    # verify that this point is inside L
    check_reference(np.array([x_opt]),L)

    return x_opt

def get_new_reference(pn,f,X,L):
    """
    find new reference for Remez algorithm
    
    Input
    -----
    X : current reference
    Pn : polynomial
    f : function to approximate
    L : union of intervals
    
    Returns
    -------
    Xn : updated reference
    pfx_opt : size of error of old polynomial on new reference
    """
    
    N = len(X) - 2
    assert N >= 0, N
    
    Xn = np.zeros_like(X)
    
    # define objective function
    obj = lambda x: -np.abs(pn(x) - f(x))

    # compute differences on old reference
    pfX = pn(X) - f(X)
    
    # find new global minima of objective
        # really should start at all the past points and go..
    x_opt = optimize_over_intervals(obj,L,X[N//2])
    
    # compute (signed) value of global extrema
    pfx_opt = (pn(x_opt) - f(x_opt))[0]

    
    # if the global extrema was at an old reference points, then we have converged.
    # return original reference and algorithm will terminate
    if np.any(x_opt == X):
        return X, np.abs(pfx_opt)
    
    # otherwise, we must update a single point in the referenceMatrix form of three term polynomial recurrence

    # there are three cases for the location of x_opt relative to the old reference
    if x_opt < X[0]:
        # if signs match, replace first point
        if np.sign(pfx_opt) == np.sign(pfX[0]):
            Xn[0] = x_opt
            Xn[1:] = X[1:]
        # otherwise, replace last point
        else:
            Xn[0] = x_opt
            Xn[1:] = X[:-1]

    elif x_opt > X[-1]:
        # if signs match, replace last point
        if np.sign(pfx_opt) == np.sign(pfX[-1]):
            Xn[-1] = x_opt
            Xn[:-1] = X[:-1]
        # otherwise replace first point
        else:
            Xn[-1] = x_opt
            Xn[:-1] = X[1:]

    else:
        Xn[:] = X
        
        # compute idenx of largset old reference point which is smaller than x_opt
        idx = np.sum(X < x_opt)
        
        # this point should not be the largest or smallest point in old reference
        assert idx != 0 and idx != len(X)
        
        # now, replace the nearest point in old reference wich has the same sign
        if np.sign(pfx_opt) == np.sign(pfX[idx-1]):
            Xn[idx-1] = x_opt
        else:
            Xn[idx] = x_opt
        
    check_reference(Xn,L)
    signs = np.sign(pn(Xn) - f(Xn))
    assert np.all(np.diff(signs)!=0), 'polynomial does not osscilate function on new reference'
    
    return Xn,np.abs(pfx_opt)

def remez(f,X,L,max_iter,rtol=1e-10,atol=1e-15,verbose=1):
    """
    finds minimax polynomial of degree N for f on L
    
    Input
    -----
    f : function to approximate
    X : initial reference
    L : union of intervals
    max_iter : number of iterations to use
    
    Returns
    -------
    pn : optimal polynomial
    X : final reference
    """
    
    sigma = (-1)**np.arange(len(X))

    for k in range(max_iter):
        
        check_reference(X,L)
            
        # find equiosillating polynomial pn on X
        f_X = f(X)
        w = compute_barycentric_weights(X)
        
        # compute osillation size
        h = (w@f_X)/(w@sigma)
        
        # compute values of p_X so that it equiosillatse f with distance h at points in X
        p_X = f_X - h * sigma
        
        # get function for p
        p = lambda x: barycentric(x,X,p_X,w)

        signs = np.sign(p(X)-f(X))
        assert np.all(np.diff(signs)!=0), f'polynomial does not osscilate function on starting reference {h}'

        # now get new reference from ossilating extrema of P
        X,pfx_opt = get_new_reference(p,f,X,L)
        
        # check that lower and upper bounds are sufficiently close, or that upper bound is small
        if ( k>0 and ((np.abs(pfx_opt-np.abs(h)) < atol) or (np.abs(pfx_opt-np.abs(h))/np.abs(h) < rtol) or (pfx_opt < atol)) ) and verbose>0:
            print(f'terminating at iteration {k} due to small decrease in error')
            break

    # h is lower bound, pfx_opt is upper bound
    return p,X,np.abs(h),pfx_opt

def get_initial_reference(L,N):

    X = np.zeros(N+2)
    k = 0
    j = -1
    while k<N+2:
        j += 1
        a,b = L[j%len(L)]

        # skip points after first pass
        if a==b and j//len(L)!=0:
            continue

        X[k] = np.random.rand(1)*(b-a)+a
        k += 1
    
    return np.sort(X)