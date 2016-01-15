import numpy as np

class Eigenproblem():
    def __init__(self, EVP):
        """
        EVP is dedalus EVP object
        """
        self.EVP = EVP
        self.solver = EVP.build_solver()
        
    def solve(self, pencil=0):
        self.solver.solve(self.solver.pencils[pencil], rebuild_coeffs=True)
        self.process_evalues()
            
    def process_evalues(self):
        ev = self.solver.eigenvalues
        self.evalues = ev[np.isfinite(ev)]
    
    def growth_rate(self,params):
        for k,v in params.items():
            vv = self.EVP.namespace[k]
            vv.value = v
        self.solve()
        
        return np.max(self.evalues.real)
    
    def spectrum():
        pass
    
    def reject_spurious():
        """may be able to pull everything out of EVP to construct a new one with higher N..."""
        pass
