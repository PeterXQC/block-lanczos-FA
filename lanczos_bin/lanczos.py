import numpy as np
import scipy as sp
from .barycentric import compute_barycentric_weights, barycentric

def exact_lanczos(A,q0,k,B=None,reorth=True):
    """
    run Lanczos with reorthogonalization
    
    Input
    -----
    A : entries of diagonal matrix A
    q0 : starting vector
    k : number of iterations
    B : entries of diagonal weights for orthogonalization
    """
    
    n = len(A)
    
    if B is None:
        B = np.ones(n,dtype=A.dtype)
    
    Q = np.zeros((n,k),dtype=A.dtype)
    a = np.zeros(k,dtype=A.dtype)
    b = np.zeros(k-1,dtype=A.dtype)
    
    Q[:,0] = q0 / np.sqrt(q0*B@q0)
    
    for i in range(1,k+1):
        # expand Krylov space

      #  if i>1:
       #     print(b[i-2],qi@Q[:,i-2])

        qi = A*Q[:,i-1] - b[i-2]*Q[:,i-2] if i>1 else A*Q[:,i-1]
        
        a[i-1] = (qi*B)@Q[:,i-1]
        qi -= a[i-1]*Q[:,i-1]
        
        if reorth:
            qi -= Q[:,:i-2]@(Q[:,:i-2].T@(B*qi))
            
        if i < k:
            b[i-1] = np.sqrt((qi*B)@qi)
            Q[:,i] = qi / b[i-1]
                
    return Q,(a,b)

def polyval_A_equil(d,a_,b_,s):
    """
    evaluate linear combination of Lanczos polynomails
    
    Input
    -----
    d : linear combination coefficients
    a_, b_ : main and off diagonal entires
    s : set of points to evaluate on
    
    Returns
    -------
    y : value of linear combination of Lanczos polynomials at s
    """
    
    M = 1 if np.isscalar(s) else len(s)
    n = len(d)-1
    W = np.zeros((M,n+1))
    W[:,0] = 1. # note that p_0(x) = c not necessarily 1, so everything must be rescaled to c
    
    for k in range(0,n):
        w = (s - a_[k])*W[:,k] - b_[k-1]*W[:,k-1] if k!=0 else (s - a_[k])*W[:,k]
        
        W[:,k+1] = w / b_[k]
    
    y = W@d
    return y

def lanczos_poly_approx(f,A,b,k):
    """
    compute degree k Lanczos approximation to f(A)b
4    """
    Q,(a,b) = exact_lanczos(A,b,k+1)
    theta = sp.linalg.eigvalsh_tridiagonal(a,b,tol=1e-30)
    
    w = compute_barycentric_weights(theta)
    
    return lambda x: barycentric(x,theta,f(theta),w)

def lanczos_FA(f,Q,a_,b_,normb=1):
    theta,S = sp.linalg.eigh_tridiagonal(a_,b_,tol=1e-30)
    
#    fT = S*f(theta)@S.T 
    
    # e0 = Q^Tb in exaxt arithmetic. IDK if this matters numerically..
    e0 = np.zeros_like(Q[0])
    e0[0] = normb
    
    return Q@(S@(f(theta)*(S.T@e0)))

def lanczos_CG_residual_coeff(z,a_,b_):
    theta,S = sp.linalg.eigh_tridiagonal(a_,b_[:-1],tol=1e-30)

    return np.abs(np.prod(b_)/np.prod(theta-z))

def lanczos_fAb(f,A,b,k,reorth=True):
    """
    get Lanczos degree k approximation to f(A)b : Q f(T) Q^Tb

    """
    Q,(a_,b_) = exact_lanczos(A,b,k+1,reorth=reorth)
    theta,S = sp.linalg.eigh_tridiagonal(a_,b_,tol=1e-18)
    
    fT = S*f(theta)@S.T 
    
    # e0 = Q^Tb in exaxt arithmetic. IDK if this matters numerically..
    e0 = np.zeros(k+1,dtype=np.longdouble)
    e0[0] = np.linalg.norm(b)
    
    return Q@fT@e0
    
def opt_poly_approx(f,A,b,k,B=None):
    n = len(A)

    unif_ones = np.ones(n,dtype=np.longdouble) / np.sqrt(np.ones(n)*B@np.ones(n))

    Q,(a_,b_) = exact_lanczos(A,unif_ones,k+1,B)

    p_opt = lambda x: polyval_A_equil(Q[:,:k+1].T@(B*f(A)),a_[:k+1],b_[:k+1],x) * Q[0,0]

    return p_opt

def opt_FA(fAb,Q,B=None,normb=1):
    
    if B is None:
        B = np.ones(len(A))
        
    return Q@Q.T@(B*fAb)


def opt_fAb(f,A,b,k,B=None):
    """
    get optimal p_k(A)b over K_k(A,b) in B norm (B has same eigenvalues as A)
    """
    
    if B is None:
        B = np.ones(len(A),dtype=np.longdouble)
        
    Q,(a_,b_) = exact_lanczos(A,b,k+1,B)

    return Q[:,:k+1]@(Q[:,:k+1].T@(B*f(A)*b))

def Q_wz(w,z,lmin,lmax):
    """
    max_{x\in[lmin,lmax]} |x-w|/|z-w|
    """
    
    if np.real(z) - w != 0:
        b_hat = ( np.abs(z)**2 - np.real(z)*w ) / (np.real(z) - w)
    else:
        b_hat = np.inf
        
    if lmin < b_hat <= lmax:
        return np.abs((z-w)/np.imag(z))
    else:
        return np.max([np.abs((lmax-w)/(lmax-z)), np.abs((lmin-w)/(lmin-z))])
    
def Q_z(z,lmin,lmax):
    """
    max_{x\in[lmin,lmax]} 1/|z-w|
    """
    
    b_hat = np.real(z)
        
    if lmin < b_hat <= lmax:
        return np.abs(1/np.imag(z))
    
    else:
        return np.max([np.abs(1/(lmax-z)), np.abs(1/(lmin-z))])
    
def get_a_priori_bound(f,gamma,endpts,k,w,lmin,lmax):
    """
    (1/2pi) \oint_{\Gamma} |f(z)| (Q_{w,z})^{k+1} |dz|
    """
    
    def F(t):
        z,dz = gamma(t)
        
        return (1/(2*np.pi)) * np.abs(f(z)) * Q_wz(w,z,lmin,lmax)**(k+1) * np.abs(dz)
    
    integral = sp.integrate.quad(F,endpts[0],endpts[1],epsabs=0,limit=200) 
    
    return integral


def get_a_posteriori_bound(f,gamma,endpts,a_,b_,w,lmin,lmax):
    """
    (1/2pi) \oint_{\Gamma} |f(z)| |D_{k,w,z}| Q_{w,z} |dz|
    """
    
    theta = sp.linalg.eigvalsh_tridiagonal(a_,b_,tol=1e-30)

    def F(t):
        z,dz = gamma(t)
        
        return (1/(2*np.pi)) * np.abs(f(z)) * np.abs(np.prod((theta-w)/(theta-z))) * Q_wz(w,z,lmin,lmax) * np.abs(dz)
    
    integral = sp.integrate.quad(F,endpts[0],endpts[1],epsabs=0,limit=200) 
    
    return integral

def get_exact_bound(f,gamma,endpts,a_,b_,w,lam):
    """
    (1/2pi) \oint_{\Gamma} |f(z)| (Q_{w,z})^{k+1} |dz|
    """
    
    theta = sp.linalg.eigvalsh_tridiagonal(a_,b_,tol=1e-30)

    def F(t):
        z,dz = gamma(t)
        
        return (1/(2*np.pi)) * np.abs(f(z)) * np.abs(np.prod((theta-w)/(theta-z))) * np.max(np.abs((lam-w)/(lam-z))) * np.abs(dz)
    
    integral = sp.integrate.quad(F,endpts[0],endpts[1],epsabs=0,limit=200) 
    
    return integral