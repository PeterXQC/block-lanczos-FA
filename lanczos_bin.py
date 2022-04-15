import numpy as np
import scipy as sp

def block_lanczos(A,Z0,q,reorth=0):
        
    Z = np.copy(Z0)
    d,b = Z.shape
    
    M = [ np.zeros((b,b),dtype=A.dtype) ]*q
    R = [ np.zeros((b,b),dtype=A.dtype) ]*(q+1)
    
    Q = np.zeros((d,b*(q+1)),dtype=A.dtype)

    Q[:,0:b],R[0] = np.linalg.qr(Z)
    for k in range(0,q):
        
        Qk = Q[:,k*b:(k+1)*b]
        Qkm1 = Q[:,(k-1)*b:k*b]
        Z = A@Qk - Qkm1@(R[k].conj().T) if k>0 else A@Qk
        
        if reorth>k:
            Z -= Q[:,:k*b]@(Q[:,:k*b].conj().T@Z)
            Z -= Q[:,:k*b]@(Q[:,:k*b].conj().T@Z)

        M[k] = Qk.conj().T@Z
        Z -= Qk@M[k]
        
        Q[:,(k+1)*b:(k+2)*b],R[k+1] = np.linalg.qr(Z)
    
    return Q,M,R


def par_lanczos(A,Z0,q,reorth=0):
        
    Z = np.copy(Z0)
    d,b = Z.shape
    
    M = np.zeros((q,b),dtype=A.dtype)
    R = np.zeros((q+1,b),dtype=A.dtype)
    
    Q = np.zeros((d,q+1,b),dtype=A.dtype)

    for j in range(b):
        R[0,j] = np.linalg.norm(Z[:,j])
        Q[:,0,j] = Z[:,j] / R[0,j]

    for k in range(0,q):
        
        AQk = A@Q[:,k]
        for j in range(b):
            Qk = Q[:,k,j]
            Qkm1 = Q[:,k-1,j]
            Z = AQk[:,j] - Qkm1*(R[k,j]) if k>0 else AQk[:,j]

            if reorth>k:
                Z -= Q[:,:k,j]@(Q[:,:k,j].conj().T@Z)
                Z -= Q[:,:k,j]@(Q[:,:k,j].conj().T@Z)

            M[k,j] = Qk.conj().T@Z
            Z -= Qk*M[k,j]

            R[k+1,j] = np.linalg.norm(Z)
            Q[:,k+1,j] = Z / R[k+1,j]

    return Q,M,R


def get_block_tridiag(M,R):

    q = len(M)
    b = len(M[0])
    
    T = np.zeros((q*b,q*b),dtype=M[0].dtype)

    for k in range(q):
        T[k*b:(k+1)*b,k*b:(k+1)*b] = M[k]

    for k in range(q-1):
        T[(k+1)*b:(k+2)*b,k*b:(k+1)*b] = R[k]
        T[k*b:(k+1)*b,(k+1)*b:(k+2)*b] = R[k].conj().T
        
    return T


def krylov_trace_quadrature(A,b,q,n1,m,n2):

    d = A.shape[0]
   
    Ω = np.random.randn(d,b)
    Ψ = np.random.randn(d,m)

    QQ,M,R = block_lanczos(A,Ω,q+n1,reorth=q)
    T = get_block_tridiag(M[:q+1],R[1:q+1])
    Θ,S = np.linalg.eigh(T)
    Q = QQ[:,:q*b]
    
    Θ_defl = np.array([])
    W_defl = np.array([])
    if q*b>0:
        T = get_block_tridiag(M,R[1:-1])
        Θ,S = np.linalg.eigh(T)
        Sqb = S.conj().T[:,:q*b]

        Θ_defl = np.copy(Θ)
        W_defl = np.linalg.norm(Sqb,axis=1)**2

    
    Θ_rem1 = np.array([])
    W_rem1 = np.array([])
    W_rem2 = 0
    Y = Ψ - Q@(Q.conj().T@Ψ)

    Qt,Mt,Rt = par_lanczos(A,Y,n2,reorth=0)
    for i in range(m):

        try:
            Θt,St = sp.linalg.eigh_tridiagonal(Mt[:,i],Rt[1:-1,i])
        except:
            Tt = np.diag(Mt[:,i]) + np.diag(Rt[1:-1,i],-1) + np.diag(Rt[1:-1,i],1)
            Θt,St = np.linalg.eigh(Tt)

        Sm2Rt = St.conj().T[:,0]*Rt[0,i]

        Θ_rem1 = np.hstack([Θ_rem1,Θt])
        W_rem1 = np.hstack([W_rem1,Sm2Rt**2/m])
     
    return np.hstack([Θ_defl,Θ_rem1]),np.hstack([W_defl+W_rem2,W_rem1])





# old algs below

def krylov_trace(A,f,b,q,m1,m2,n1,n2,r):
    
    d = A.shape[0]
   
    Ω = np.random.randn(d,b)

    λmin = np.inf
    λmax = -np.inf
    
    for i in range(r-1):
        QQ,M,R = block_lanczos(A,Ω,q,reorth=q)
        
        T = get_block_tridiag(M[:q+1],R[1:q+1])
        Θ,S = np.linalg.eigh(T)
        
        # lazy approach: find good least squares polynomial on estimated spectrum interval
        λmin = np.min(np.append(Θ,λmin))
        λmax = np.max(np.append(Θ,λmax))
        xx = np.linspace(λmin-.01*(λmax-λmin),λmax+.01*(λmax-λmin),1000)
        yy = f(xx)
        p = np.polynomial.Chebyshev.fit(xx,yy,q)

        Y = S@np.diag(p(Θ))@S.conj().T[:,:b]
        Ω = QQ[:,:q*b]@Y@R[0]
    
    QQ,M,R = block_lanczos(A,Ω,q+n1,reorth=q)
    T = get_block_tridiag(M[:q+1],R[1:q+1])
    Θ,S = np.linalg.eigh(T)
    Q = QQ[:,:q*b]
    
    if m1>0:
        T = get_block_tridiag(M,R[1:-1])
        Θ,S = np.linalg.eigh(T)
        Sqb = S.conj().T[:,:q*b]
        F = Sqb.conj().T@np.diag(f(Θ))@Sqb
    else:
        F = np.zeros((q*b,q*b))
        
    t_defl = np.trace(F)
    

    t_rem = 0
    for i in range(m2):
        Ψ = np.random.randn(d,1)
        _,Mt,Rt = block_lanczos(A,Ψ,n2,reorth=True)

        Tt = get_block_tridiag(Mt,Rt[1:-1])
        Θt,St = np.linalg.eigh(Tt)
        Sm2Rt = St.conj().T[:,:1]@Rt[0]

        X = (Q.conj().T@Ψ)

        t_rem += (1/m2)*np.trace(Sm2Rt.conj().T@np.diag(f(Θt))@Sm2Rt)
        t_rem -= (1/m2)*np.trace(X.conj().T@F@X)
    return t_defl + t_rem

def hutchpp(A,f,b,q,m1,m2,n1,n2):
    
    d = A.shape[0]

    Ω = np.random.randn(d,b)
    
    QQ,M,R = block_lanczos(A,Ω,q,reorth=q)
    T = get_block_tridiag(M,R[1:-1])
    Θ,S = np.linalg.eigh(T)
    fAΩ = QQ[:,:q*b]@S@np.diag(f(Θ))@S.conj().T[:,:b]

    Q,_ = np.linalg.qr(fAΩ)
    
    QQ,M,R = block_lanczos(A,Q,n1,reorth=0)
    T = get_block_tridiag(M,R[1:-1])
    Θ,S = np.linalg.eigh(T)
    S0 = S.conj().T[:,:b]
    
    t_defl = np.trace(S0.conj().T@np.diag(f(Θ))@S0)

    if m2==0:
        t_rem = 0
    else:
        Ψ = np.random.randn(d,m2)
        _,Mt,Rt = block_lanczos(A,Ψ,n2,reorth=True)

        Tt = get_block_tridiag(Mt,Rt[1:-1])
        Θt,St = np.linalg.eigh(Tt)
        Sm2Rt = St.conj().T[:,:m2]@Rt[0]

        X = (Q.conj().T@Ψ)

        t_rem = (1/m2)*np.trace(Sm2Rt.conj().T@np.diag(f(Θt))@Sm2Rt)
        t_rem -= (1/m2)*np.trace(X.conj().T@F@X)
    
    return t_defl + t_rem

def krylov_trace_restart_quadrature(A,f,r,b,q,n1,m2,n2):
    
    d = A.shape[0]
   
    Ω = np.random.randn(d,b)

    λmin = np.inf
    λmax = -np.inf
    
    for i in range(r-1):
        QQ,M,R = block_lanczos(A,Ω,q,reorth=q)
        
        T = get_block_tridiag(M[:q+1],R[1:q+1])
        Θ,S = np.linalg.eigh(T)
        
        # lazy approach: find good least squares polynomial on estimated spectrum interval
        λmin = np.min(np.append(Θ,λmin))
        λmax = np.max(np.append(Θ,λmax))
        xx = np.linspace(λmin-.01*(λmax-λmin),λmax+.01*(λmax-λmin),1000)
        yy = f(xx)
        p = np.polynomial.Chebyshev.fit(xx,yy,q)

        Y = S@np.diag(p(Θ))@S.conj().T[:,:b]
        Ω = QQ[:,:q*b]@Y@R[0]
    
    QQ,M,R = block_lanczos(A,Ω,q+n1,reorth=q)
    T = get_block_tridiag(M[:q+1],R[1:q+1])
    Θ,S = np.linalg.eigh(T)
    Q = QQ[:,:q*b]
    
    Θ_defl = np.array([])
    W_defl = np.array([])
    if q*b>0:
        T = get_block_tridiag(M,R[1:-1])
        Θ,S = np.linalg.eigh(T)
        Sqb = S.conj().T[:,:q*b]
        F = Sqb.conj().T@np.diag(f(Θ))@Sqb
        Θ_defl = np.copy(Θ)
        W_defl = np.linalg.norm(Sqb,axis=1)**2

    
    Θ_rem1 = np.array([])
    W_rem1 = np.array([])
    W_rem2 = 0
    for i in range(m2):
        Ψ = np.random.randn(d,1)
        Ψ *= np.sqrt(d)/np.linalg.norm(Ψ)
#        Ψ = Ψ - Q@(Q.conj().T@Ψ)
        _,Mt,Rt = block_lanczos(A,Ψ,n2,reorth=0)

        Tt = get_block_tridiag(Mt,Rt[1:-1])
        Θt,St = np.linalg.eigh(Tt)
        Sm2Rt = St.conj().T[:,:1]@Rt[0]

        Θ_rem1 = np.hstack([Θ_rem1,Θt])
        W_rem1 = np.hstack([W_rem1,(St.conj().T[:,0]*Rt[0][0,0])**2/m2])
        
        if q*b>0:
            X = (Q.conj().T@Ψ)
            W_rem2 -= np.abs(Sqb@X)[:,0]**2/m2
        
#    print(f(Θ_defl)@W_defl,f(Θ_defl)@W_rem2,f(Θ_rem1)@W_rem1)
    return np.hstack([Θ_defl,Θ_rem1]),np.hstack([W_defl+W_rem2,W_rem1])