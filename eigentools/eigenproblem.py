from dedalus.tools.cache import CachedAttribute
import dedalus.public as de
import matplotlib.pyplot as plt
import numpy as np

class Eigenproblem():
    def __init__(self, EVP):
        """
        EVP is dedalus EVP object
        """
        self.EVP = EVP
        self.solver = EVP.build_solver()
        
    def solve(self, pencil=0):
        self.pencil = pencil
        self.solver.solve(self.solver.pencils[self.pencil], rebuild_coeffs=True)
        self.evalues = self.process_evalues(self.solver.eigenvalues)
            
    def process_evalues(self, ev):
        return ev[np.isfinite(ev)]
    
    def growth_rate(self,params,reject=True):
        for k,v in params.items():
            vv = self.EVP.namespace[k]
            vv.value = v
        self.solve()
        if reject:
            self.reject_spurious()
            return np.max(self.evalues_good.real)
        else:
            return np.max(self.evalues.real)
    
    def spectrum(self, title='spectrum',spectype='raw'):
        if spectype == 'raw':
            ev = self.evalues
        elif spectype == 'hires':
            ev = self.evalues_hires
        elif spectype == 'good':
            ev = self.evalues_good
        else:
            raise ValueError("Spectrum type is not one of {raw, hires, good}")

        # Color is sign of imaginary part
        colors = ["blue" for i in range(len(ev))]
        imagpos = np.where(ev.imag >= 0)
        for p in imagpos[0]:
            colors[p] = "red"

        # Symbol is sign of real part
        symbols = ["." for i in range(len(ev))]
        thickness = np.zeros(len(ev))
        realpos = np.where(ev.real >= 0)
        for p in realpos[0]:
            symbols[p] = "+"
            thickness[p] = 2

        print("Number of positive real parts", len(realpos[0]))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for x, y, c, s, t in zip(np.abs(ev.real), np.abs(ev.imag), colors, symbols, thickness):
            ax.plot(x, y, s, c=c, alpha = 0.5, ms = 8, mew = t)

        # Dummy plot for legend
        ax.plot(0, 0, '+', c = "red", alpha = 0.5, mew = 2, label = r"$\gamma \geq 0$, $\omega > 0$")
        ax.plot(0, 0, '+', c = "blue", alpha = 0.5, mew = 2, label = r"$\gamma \geq 0$, $\omega < 0$")
        ax.plot(0, 0, '.', c = "red", alpha = 0.5, label = r"$\gamma < 0$, $\omega > 0$")
        ax.plot(0, 0, '.', c = "blue", alpha = 0.5, label = r"$\gamma < 0$, $\omega < 0$")
        
        ax.legend(loc='lower right').draw_frame(False)
        ax.set_ylim(1e-18,1)
        ax.set_xlim(1e-5,1e15)
        ax.loglog()
        ax.set_xlabel(r"$\left|\gamma\right|$", size = 15)
        ax.set_ylabel(r"$\left|\omega\right|$", size = 15, rotation = 0)

        fig.savefig('{}.png'.format(title))

    def reject_spurious(self, factor=1.5):
        """may be able to pull everything out of EVP to construct a new one with higher N..."""
        self.factor = factor
        evg, indx = self.discard_spurious_eigenvalues()
        self.evalues_good = evg
        self.evalues_good_index = indx

        
    @CachedAttribute
    def evalues_hires(self):
        old_evp = self.EVP
        old_d = old_evp.domain
        old_x = old_d.bases[0]

        n_hi = int(old_x.coeff_size * self.factor)
        x = de.Chebyshev(old_x.name,n_hi,interval=old_x.interval)
        d = de.Domain([x],comm=old_d.dist.comm)
        self.EVP_hires = de.EVP(d,old_evp.variables,old_evp.eigenvalue)

        for k,v in old_evp.parameters.items():
            self.EVP_hires.parameters[k] = v

        for e in old_evp.equations:
            self.EVP_hires.add_equation(e['raw_equation'])

        for b in old_evp.boundary_conditions:
            self.EVP_hires.add_bc(b['raw_equation'])

        solver = self.EVP_hires.build_solver()
        solver.solve(solver.pencils[self.pencil], rebuild_coeffs=True)
        return self.process_evalues(solver.eigenvalues)

    def discard_spurious_eigenvalues(self):
        """
        Solves the linear eigenvalue problem for two different resolutions.
        Returns trustworthy eigenvalues using nearest delta, from Boyd chapter 7.
        """

        # Solve the linear eigenvalue problem at two different resolutions.
        #LEV1 = self.evalues
        #LEV2 = self.evalues_hires
        # Eigenvalues returned by dedalus must be multiplied by -1
        lambda1 = self.evalues #-LEV1.eigenvalues
        lambda2 = self.evalues_hires #-LEV2.eigenvalues

        # Reverse engineer correct indices to make unsorted list from sorted
        reverse_lambda1_indx = np.arange(len(lambda1)) 
        reverse_lambda2_indx = np.arange(len(lambda2))
    
        lambda1_and_indx = np.asarray(list(zip(lambda1, reverse_lambda1_indx)))
        lambda2_and_indx = np.asarray(list(zip(lambda2, reverse_lambda2_indx)))
        
        #print(lambda1_and_indx, lambda1_and_indx.shape, lambda1, len(lambda1))

        # remove nans
        lambda1_and_indx = lambda1_and_indx[np.isfinite(lambda1)]
        lambda2_and_indx = lambda2_and_indx[np.isfinite(lambda2)]
    
        # Sort lambda1 and lambda2 by real parts
        lambda1_and_indx = lambda1_and_indx[np.argsort(lambda1_and_indx[:, 0].real)]
        lambda2_and_indx = lambda2_and_indx[np.argsort(lambda2_and_indx[:, 0].real)]
        
        lambda1_sorted = lambda1_and_indx[:, 0]
        lambda2_sorted = lambda2_and_indx[:, 0]
    
        # Compute sigmas from lower resolution run (gridnum = N1)
        sigmas = np.zeros(len(lambda1_sorted))
        sigmas[0] = np.abs(lambda1_sorted[0] - lambda1_sorted[1])
        sigmas[1:-1] = [0.5*(np.abs(lambda1_sorted[j] - lambda1_sorted[j - 1]) + np.abs(lambda1_sorted[j + 1] - lambda1_sorted[j])) for j in range(1, len(lambda1_sorted) - 1)]
        sigmas[-1] = np.abs(lambda1_sorted[-2] - lambda1_sorted[-1])

        if not (np.isfinite(sigmas)).all():
            print("WARNING: at least one eigenvalue spacings (sigmas) is non-finite (np.inf or np.nan)!")
    
        # Nearest delta
        delta_near = np.array([np.nanmin(np.abs(lambda1_sorted[j] - lambda2_sorted)/sigmas[j]) for j in range(len(lambda1_sorted))])
    
        # Discard eigenvalues with 1/delta_near < 10^6
        lambda1_and_indx = lambda1_and_indx[np.where((1.0/delta_near) > 1E6)]
        #print(lambda1_and_indx)
        
        lambda1 = lambda1_and_indx[:, 0]
        indx = lambda1_and_indx[:, 1]
        
        #delta_near_unsorted = delta_near[reverse_lambda1_indx]
        #lambda1[np.where((1.0/delta_near_unsorted) < 1E6)] = None
        #lambda1[np.where(np.isnan(1.0/delta_near_unsorted) == True)] = None
    
        return lambda1, indx

    def discard_spurious_eigenvalues2(self, problem):
    
        """
        Solves the linear eigenvalue problem for two different resolutions.
        Returns trustworthy eigenvalues using nearest delta, from Boyd chapter 7.
        """

        # Solve the linear eigenvalue problem at two different resolutions.
        LEV1 = self.solve_LEV(problem)
        LEV2 = self.solve_LEV_secondgrid(problem)
    
        # Eigenvalues returned by dedalus must be multiplied by -1
        lambda1 = -LEV1.eigenvalues
        lambda2 = -LEV2.eigenvalues
    
        # Sorted indices for lambda1 and lambda2 by real parts
        lambda1_indx = np.argsort(lambda1.real)
        lambda2_indx = np.argsort(lambda2.real)
        
        # Reverse engineer correct indices to make unsorted list from sorted
        reverse_lambda1_indx = sorted(range(len(lambda1_indx)), key=lambda1_indx.__getitem__)
        reverse_lambda2_indx = sorted(range(len(lambda2_indx)), key=lambda2_indx.__getitem__)
        
        self.lambda1_indx = lambda1_indx
        self.reverse_lambda1_indx = reverse_lambda1_indx
        self.lambda1 = lambda1
        
        # remove nans
        lambda1_indx = lambda1_indx[np.isfinite(lambda1)]
        reverse_lambda1_indx = np.asarray(reverse_lambda1_indx)
        reverse_lambda1_indx = reverse_lambda1_indx[np.isfinite(lambda1) == True]
        #lambda1 = lambda1[np.isfinite(lambda1)]
        
        lambda2_indx = lambda2_indx[np.isfinite(lambda2)]
        reverse_lambda2_indx = np.asarray(reverse_lambda2_indx)
        reverse_lambda2_indx = reverse_lambda2_indx[np.isfinite(lambda2)]
        #lambda2 = lambda2[np.isfinite(lambda2)]
        
        # Actually sort the eigenvalues by their real parts
        lambda1_sorted = lambda1[lambda1_indx]
        lambda2_sorted = lambda2[lambda2_indx]
        
        self.lambda1_sorted = lambda1_sorted
        #print(lambda1_sorted)
        #print(len(lambda1_sorted), len(np.where(np.isfinite(lambda1) == True)))
    
        # Compute sigmas from lower resolution run (gridnum = N1)
        sigmas = np.zeros(len(lambda1_sorted))
        sigmas[0] = np.abs(lambda1_sorted[0] - lambda1_sorted[1])
        sigmas[1:-1] = [0.5*(np.abs(lambda1_sorted[j] - lambda1_sorted[j - 1]) + np.abs(lambda1_sorted[j + 1] - lambda1_sorted[j])) for j in range(1, len(lambda1_sorted) - 1)]
        sigmas[-1] = np.abs(lambda1_sorted[-2] - lambda1_sorted[-1])

        if not (np.isfinite(sigmas)).all():
            print("WARNING: at least one eigenvalue spacings (sigmas) is non-finite (np.inf or np.nan)!")
    
        # Nearest delta
        delta_near = np.array([np.nanmin(np.abs(lambda1_sorted[j] - lambda2_sorted)/sigmas[j]) for j in range(len(lambda1_sorted))])
    
        #print(len(delta_near), len(reverse_lambda1_indx), len(LEV1.eigenvalues))
        # Discard eigenvalues with 1/delta_near < 10^6
        delta_near_unsorted = np.zeros(len(LEV1.eigenvalues))
        for i in range(len(delta_near)):
            delta_near_unsorted[reverse_lambda1_indx[i]] = delta_near[i]
        #delta_near_unsorted[reverse_lambda1_indx] = delta_near#[reverse_lambda1_indx]
        #print(delta_near_unsorted)
        
        self.delta_near_unsorted = delta_near_unsorted
        self.delta_near = delta_near
        
        goodeigs = copy.copy(LEV1.eigenvalues)
        goodeigs[np.where((1.0/delta_near_unsorted) < 1E6)] = None
        goodeigs[np.where(np.isfinite(1.0/delta_near_unsorted) == False)] = None
    
        return goodeigs, LEV1
        
    def find_spurious_eigenvalues(self):
    
        """
        Solves the linear eigenvalue problem for two different resolutions.
        Returns drift ratios, from Boyd chapter 7.
        """
    
        lambda1 = self.evalues
        lambda2 = self.evalues_hires
        
        # Make sure argsort treats complex infs correctly
        for i in range(len(lambda1)):
            if (np.isnan(lambda1[i]) == True) or (np.isinf(lambda1[i]) == True):
                lambda1[i] = None
        for i in range(len(lambda2)):
            if (np.isnan(lambda2[i]) == True) or (np.isinf(lambda2[i]) == True):
                lambda2[i] = None        
        
        #lambda1[np.where(np.isnan(lambda1) == True)] = None
        #lambda2[np.where(np.isnan(lambda2) == True)] = None
                
        # Sort lambda1 and lambda2 by real parts
        lambda1_indx = np.argsort(lambda1.real)
        lambda1 = lambda1[lambda1_indx]
        lambda2_indx = np.argsort(lambda2.real)
        lambda2 = lambda2[lambda2_indx]
        
        # try using lower res (gridnum = N1) instead
        sigmas = np.zeros(len(lambda1))
        sigmas[0] = np.abs(lambda1[0] - lambda1[1])
        sigmas[1:-1] = [0.5*(np.abs(lambda1[j] - lambda1[j - 1]) + np.abs(lambda1[j + 1] - lambda1[j])) for j in range(1, len(lambda1) - 1)]
        sigmas[-1] = np.abs(lambda1[-2] - lambda1[-1])
        
        # Ordinal delta, calculated for the number of lambda1's.
        delta_ord = (lambda1 - lambda2[:len(lambda1)])/sigmas
        
        # Nearest delta
        delta_near = [np.nanmin(np.abs(lambda1[j] - lambda2)) for j in range(len(lambda1))]/sigmas
        
        # Discard eigenvalues with 1/delta_near < 10^6
        goodevals1 = lambda1[1/delta_near > 1E6]
        
        return delta_ord, delta_near, lambda1, lambda2, sigmas
        
