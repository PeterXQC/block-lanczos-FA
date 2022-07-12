import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def block_lanczos(H,V,k,reorth = 0):
    """
    Input
    -----
    
    H    : d x d matrix
    V    : d x b starting block
    k    : number of iterations
    reorth : how many iterations to apply reorthogonalization on 
    
    Returns
    -------
    Q1k  : First k blocks of Lanczos vectors
    Qkp1 : final block of Lanczos vetors
    A    : diagonal blocks
    B    : off diagonal blocks (incuding block for starting with non-orthogonal block)
    """

    Z = np.copy(V)
    
    d = Z.shape[0]
    if np.shape(Z.shape)[0] == 1:
         b = 1
    else:
        b = Z.shape[1]
    
    A = [np.zeros((b,b),dtype=H.dtype)] * k
    B = [np.zeros((b,b),dtype=H.dtype)] * k
    
    Q = np.zeros((d,b*(k+1)),dtype=H.dtype)

    # B_0 accounts for non-orthogonal V and is not part of tridiagonal matrix
    Q[:,0:b],B_0 = np.linalg.qr(Z)
    for j in range(0,k):
        
#       Qj is the next column of blocks
        Qj = Q[:,j*b:(j+1)*b]

        if j == 0:
            Z = H@Qj
        else:
            Qjm1 = Q[:,(j-1)*b:j*b]
            Z = H @ Qj - Qjm1 @ (B[j-1].conj().T)
     
        A[j] = Qj.conj().T @ Z
        Z -= Qj @ A[j]
        
        # double reorthogonalization if needed
        if reorth > j:
            Z -= Q[:,:j*b]@(Q[:,:j*b].conj().T@Z)
            Z -= Q[:,:j*b]@(Q[:,:j*b].conj().T@Z)
        
        Q[:,(j+1)*b:(j+2)*b],B[j] = np.linalg.qr(Z)
    
    Q1k = Q[:,:b*k]
    Qkp1 = Q[:,b*k:]

    return Q1k, Qkp1, A, B, B_0


def get_block_tridiag(A,B):
    """
    Input
    -----
    
    A  : diagonal blocks
    B  : off diagonal blocks
        Without the first block B[0].
    
    Returns
    -------
    T  : block tridiagonal matrix
    """
    
    q = len(A)
    b = len(A[0])
    
    T = np.zeros((q*b,q*b),dtype=A[0].dtype)

    for k in range(q):
        T[k*b:(k+1)*b,k*b:(k+1)*b] = A[k]

    for k in range(q-1):
        T[(k+1)*b:(k+2)*b,k*b:(k+1)*b] = B[k]
        T[k*b:(k+1)*b,(k+1)*b:(k+2)*b] = B[k].conj().T
    
    return T

def Ei(n, b, i):
    """
    Input
    -----    
    n  : matrix size
    b  : block size
    i  : position of diagonal block (the first block is when i = 1)
    
    Returns
    -------
    Ei  : block zero vector with identity in i-th position
    """
    
    if (i == 0 or i > n/b):
        raise ValueError("Illegal Index: ", i, n, b, n/b)

    Ei = np.zeros((n,b))
    Ei[(i-1)*b:i*b,:] = np.identity(b)
    
    return Ei


def get_Cz(Eval,Evec,z,b,B_0):
    """
    Input
    -----
    Eval : eigevnalues of T
    Evec : eigenvectors of T
    z    : shift z
    b    : block size
    B_0  : first block
    
    Output
    ------
    Cz = -Ek^T(T-zI)^{-1}E_1B_0
    """
    
    K = len(Eval)//b

    # Cz = -Ei(b*K,b,K).T@Evec@np.diag(1/(Eval-z))@Evec.T@Ei(b*K,b,1)@B_0

    # avoid forming (T-zI) by computing (E_k Evec) (Eval-z I) (Evec^T E_1)
    # avoid forming diagonal matrix (Eval-z I) using broadcasting
    Cz = -(Evec[-b:]@((1/(Eval-z))[:,None]*Evec.T[:,:b]))@B_0

    return Cz

def get_CwinvCz(Eval,Evec,z,w,b,B_0):
    """
    Input
    -----
    Eval : eigevnalues of T
    Evec : eigenvectors of T
    z    : shift z
    w    : shift w
    b    : block size
    B_0  : first block
    
    Output
    ------
    CwinvCz = C(w)^{-1}C(z)
    """
        
    Cz = get_Cz(Eval,Evec,z,b,B_0)
    Cw = get_Cz(Eval,Evec,w,b,B_0)

    CwinvCz = np.linalg.solve(Cw,Cz)
    
    return CwinvCz

def oscTest(H, w):
    HwI = H-w*np.identity(np.shape(H)[0])
    EvalH, EvecH = np.linalg.eigh(H)
    
    for i in np.arange(len(EvalH)):
        if EvalH[i] <= 0:
            print("Oscilitory behavior is expected for this pair of H and w.")
            return False
    
    return True

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
    
def block_a_posteriori_bound(H, V, Q, T, f,gamma,endpts,w,lmin,lmax, k, B_0):
    """
    (1/2pi) \oint_{\Gamma} |f(z)| |D_{k,w,z}| Q_{w,z} |dz|
    """
    Eval, Evec = np.linalg.eigh(T)
    def F(t):
        z,dz = gamma(t)
        
        return (1/(2*np.pi)) * np.abs(f(z)) * np.linalg.norm(get_CwinvCz(Eval,Evec,z,w,np.shape(B_0)[0],B_0), ord = 2) * Q_wz(w,z,lmin,lmax) * np.abs(dz)
    
    integral = sp.integrate.quad(F,endpts[0],endpts[1],epsabs=0,limit=200) 
    
    return integral[0]*np.linalg.norm(exact_err(w, H, V, Q, T, B_0), ord = 2)

def block_a_posteriori_bound_mid(T, f, gamma, endpts, Q, H, V, B_0):
    """
    (1/2pi) \oint_{\Gamma} |f(z)| |D_{k,w,z}| Q_{w,z} |dz|
    """
    
    Eval, Evec = np.linalg.eigh(T)
    def F(t):
        z,dz = gamma(t)
        
        return (1/(2*np.pi)) * np.linalg.norm(f(z)* exact_err(z, H, V, Q, T, B_0), ord = 2) * np.abs(dz)
    
    integral = sp.integrate.quad(F,endpts[0],endpts[1],epsabs=0,limit=200)
    
    return integral[0]

def block_a_posteriori_bound_exact(T, f, H, V, Q, B_0, b):
    EvalH, EvecH = np.linalg.eigh(H)
    fEvalH = f(EvalH)
    fH = EvecH@np.diag(fEvalH)@EvecH.conj().T
    
    fHV = fH@V
    
    EvalT, EvecT = np.linalg.eigh(T)
#     EvalT may be negative, and np.sqrt only let imaginary output when input is complex
#     thus, we make first element of EvalT complex first. 
    EvalT = EvalT.astype(complex)
    EvalT[0] = EvalT[0]+0j
    
    fEvalT = f(EvalT)
    fT = EvecT@np.diag(fEvalT)@EvecT.conj().T
    
    return np.linalg.norm(fHV-(Q@fT)[:, :b]@B_0, ord = 2)

# use linear solver as oppose to get inverse
def exact_err(z, H, V, Q, T, B_0):
    Hinv = 1/(np.diag(H)-z)
#     HinvV2 = Hinv*V
    
#     NOTE: apparently this implementation further tanks performance as its nolonger "vectorized".
#     Below is the new way of computing HinvV for better compatibility with block size larger than 1.
#     Above commented out was the old code that only works with block size 1
#     this if statement checks if V is 1d, aka block size = 1. If true, reshape to 2D format for compatibility
    this_V = V
    if (np.shape(np.shape(this_V))[0] == 1): 
        this_V = np.reshape(this_V, (len(this_V), 1))
        
    HinvV = np.zeros(np.shape(this_V),dtype = 'complex_')
    for i in np.arange(np.shape(this_V)[1]):
        HinvV[:, i] = Hinv*this_V[:, i]
    
#     due to mismatched shape, we do transpose
    HinvV = HinvV.T

    E1 = Ei(np.shape(T)[0], np.shape(B_0)[0], 1)
    TinvE = np.linalg.solve((T-z*np.eye(T.shape[0])), E1)

    return HinvV.T - Q@TinvE@B_0