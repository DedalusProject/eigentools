from dedalus.tools.cache import CachedAttribute
from dedalus.core.field import Field
import dedalus.public as de
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.sparse.linalg

class Eigenproblem():
    def __init__(self, EVP, sparse=False):
        """
        EVP is dedalus EVP object
        """
        self.EVP = EVP
        self.solver = EVP.build_solver()
        self.sparse = sparse
        
    def solve(self, pencil=0, N=15, target=0):
        self.pencil = pencil
        self.N = N
        self.target = target

        if self.sparse:
            self.solver.solve_sparse(self.solver.pencils[self.pencil], N=self.N, target=self.target, rebuild_coeffs=True)
        else:
            self.solver.solve_dense(self.solver.pencils[self.pencil], rebuild_coeffs=True)
        self.evalues = self.solver.eigenvalues
            
    def process_evalues(self, ev):
        return ev[np.isfinite(ev)]
    
    def growth_rate(self,params,reject=True, tol=1e-10, **kwargs):
        """returns the growth rate, defined as the eigenvalue with the largest
        real part. May acually be a decay rate if there is no growing mode.
        
        also returns the index of the fastest growing mode.  If there are no
        good eigenvalue, returns nan, nan, nan.
        """
        for k,v in params.items():
            # Dedalus workaround: must change values in two places
            vv = self.EVP.namespace[k]
            vv.value = v
            self.EVP.parameters[k] = v
        try:
            self.solve(**kwargs)
            if reject:
                self.reject_spurious()
                gr_rate = np.max(self.evalues_good.real)
                gr_indx = self.evalues_good_index[self.evalues_good.real == gr_rate]
                freq = self.evalues_good[self.evalues_good.real == gr_rate].imag
            else:
                gr_rate = np.max(self.evalues.real)
                gr_indx = np.where(self.evalues.real == gr_rate)[0]
                freq = self.evalues[gr_indx[0]].imag

            return gr_rate, gr_indx[0], freq

        except np.linalg.linalg.LinAlgError:
            print("Eigenvalue solver failed to converge for parameters {}".format(params))
            return np.nan, np.nan, np.nan
        except (scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence, scipy.sparse.linalg.eigen.arpack.ArpackError):
            print("Sparse eigenvalue solver failed to converge for parameters {}".format(params))
            return np.nan, np.nan, np.nan
            

    def spectrum(self, title='eigenvalue',spectype='raw'):
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
            if x is not np.ma.masked:
                ax.plot(x, y, s, c=c, alpha = 0.5, ms = 8, mew = t)

        # Dummy plot for legend
        ax.plot(0, 0, '+', c = "red", alpha = 0.5, mew = 2, label = r"$\gamma \geq 0$, $\omega > 0$")
        ax.plot(0, 0, '+', c = "blue", alpha = 0.5, mew = 2, label = r"$\gamma \geq 0$, $\omega < 0$")
        ax.plot(0, 0, '.', c = "red", alpha = 0.5, label = r"$\gamma < 0$, $\omega > 0$")
        ax.plot(0, 0, '.', c = "blue", alpha = 0.5, label = r"$\gamma < 0$, $\omega < 0$")
        
        ax.legend(loc='lower right').draw_frame(False)
        ax.loglog()
        ax.set_xlabel(r"$\left|\gamma\right|$", size = 15)
        ax.set_ylabel(r"$\left|\omega\right|$", size = 15, rotation = 0)

        fig.savefig('{}_spectrum_{}.png'.format(title,spectype))

    def reject_spurious(self, factor=1.5, tol=1e-10):
        """may be able to pull everything out of EVP to construct a new one with higher N..."""
        self.factor = factor
        self.solve_hires(tol=tol)
        evg, indx = self.discard_spurious_eigenvalues()
        self.evalues_good = evg
        self.evalues_good_index = indx

        
    def solve_hires(self):
        old_evp = self.EVP
        old_d = old_evp.domain
        old_x = old_d.bases[0]
        old_x_grid = old_d.grid(0, scales=old_d.dealias)
        if type(old_x) == de.Compound:
            bases = []
            for basis in old_x.subbases:
                old_x_type = basis.__class__.__name__
                n_hi = int(basis.base_grid_size*self.factor)
                if old_x_type == "Chebyshev":
                    x = de.Chebyshev(basis.name, n_hi, interval=basis.interval)
                elif old_x_type == "Fourier":
                    x = de.Fourier(basis.name, n_hi, interval=basis.interval)
                else:
                    raise ValueError("Don't know how to make a basis of type {}".format(old_x_type))
                bases.append(x)
            x = de.Compound(old_x.name, tuple(bases))
        else:
            old_x_type = old_x.__class__.__name__
            n_hi = int(old_x.coeff_size * self.factor)
            if old_x_type == "Chebyshev":
                x = de.Chebyshev(old_x.name,n_hi,interval=old_x.interval)
            elif old_x_type == "Fourier":
                x = de.Fourier(old_x.name,n_hi,interval=old_x.interval)
            else:
                raise ValueError("Don't know how to make a basis of type {}".format(old_x_type))
        d = de.Domain([x],comm=old_d.dist.comm)
        self.EVP_hires = de.EVP(d,old_evp.variables,old_evp.eigenvalue, ncc_cutoff=old_evp.ncc_kw['cutoff'], max_ncc_terms=old_evp.ncc_kw['max_terms'], tolerance=old_evp.tol)

        x_grid = d.grid(0, scales=d.dealias)
        
        for k,v in old_evp.substitutions.items():
            self.EVP_hires.substitutions[k] = v

        for k,v in old_evp.parameters.items():
            if type(v) == Field: #NCCs
                new_field = d.new_field()
                v.set_scales(self.factor, keep_data=True)
                new_field['g'] = v['g']
                self.EVP_hires.parameters[k] = new_field
            else: #scalars
                self.EVP_hires.parameters[k] = v

        for e in old_evp.equations:
            self.EVP_hires.add_equation(e['raw_equation'])

        try:
            for b in old_evp.boundary_conditions:
                self.EVP_hires.add_bc(b['raw_equation'])
        except AttributeError:
            # after version befc23584fea, Dedalus no longer
            # distingishes BCs from other equations
            pass

        solver = self.EVP_hires.build_solver()
        if self.sparse:
            solver.solve_sparse(solver.pencils[self.pencil], N=10, target=0, rebuild_coeffs=True)
        else:
            solver.solve_dense(solver.pencils[self.pencil], rebuild_coeffs=True)
        self.evalues_hires = solver.eigenvalues

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
        indx = lambda1_and_indx[:, 1].astype(np.int)
        
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
        
