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