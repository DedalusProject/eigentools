from dedalus.tools.cache import CachedAttribute
from dedalus.core.field import Field
import dedalus.public as de
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.sparse.linalg
from . import tools

class Eigenproblem():
    def __init__(self, EVP, sparse=False, reject=True, factor=1.5):
        """
        EVP is dedalus EVP object
        """
        self.reject = reject
        self.sparse = sparse
        self.factor = factor
        self.EVP = EVP
        self.solver = EVP.build_solver()
        if self.reject:
            self.build_hires()

        self.evalues = None
        self.evalues_low = None
        self.evalues_high = None

    def set_parameters(self, parameters):
        """set the parameters in the underlying EVP object

        """
        for k,v in parameters.items():
            tools.update_EVP_params(self.EVP, k, v)
            if self.reject:
                tools.update_EVP_params(self.EVP_hires, k, v)

    def solve(self, parameters=None, pencil=0, N=15, target=0):
        if parameters:
            self.set_parameters(parameters)
        self.pencil = pencil
        self.N = N
        self.target = target

        self.run_solver(self.solver)
        self.evalues_low = self.solver.eigenvalues

        if self.reject:
            self.run_solver(self.hires_solver)
            self.evalues_high = self.hires_solver.eigenvalues
            self.reject_spurious()
        else:
            self.evalues = self.evalues_lowres

    def run_solver(self, solver):
        if self.sparse:
            solver.solve_sparse(solver.pencils[self.pencil], N=self.N, target=self.target, rebuild_coeffs=True)
        else:
            solver.solve_dense(solver.pencils[self.pencil], rebuild_coeffs=True)

    def process_evalues(self, ev):
        return ev[np.isfinite(ev)]
    
    def growth_rate(self, parameters=None, **kwargs):
        """returns the growth rate, defined as the eigenvalue with the largest
        real part. May acually be a decay rate if there is no growing mode.
        
        also returns the index of the fastest growing mode.  If there are no
        good eigenvalue, returns nan, nan, nan.
        """
        try:
            self.solve(parameters=parameters, **kwargs)
            gr_rate = np.max(self.evalues.real)
            gr_indx = np.where(self.evalues.real == gr_rate)[0]
            freq = self.evalues[gr_indx[0]].imag

            return gr_rate, gr_indx[0], freq

        except np.linalg.linalg.LinAlgError:
            print("Dense eigenvalue solver failed for parameters {}".format(params))
            return np.nan, np.nan, np.nan
        except (scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence, scipy.sparse.linalg.eigen.arpack.ArpackError):
            print("Sparse eigenvalue solver failed to converge for parameters {}".format(params))
            return np.nan, np.nan, np.nan

    def spectrum(self, title='eigenvalue',spectype='raw'):
        if spectype == 'raw':
            ev = self.evalues_low
        elif spectype == 'hires':
            ev = self.evalues_high
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

    def reject_spurious(self):
        """may be able to pull everything out of EVP to construct a new one with higher N..."""
        evg, indx = self.discard_spurious_eigenvalues()
        self.evalues_good = evg
        self.evalues_index = indx
        self.evalues = self.evalues_good

    def build_hires(self):
        old_evp = self.EVP
        old_x = old_evp.domain.bases[0]

        x = tools.basis_from_basis(old_x, self.factor)
        d = de.Domain([x],comm=old_evp.domain.dist.comm)
        self.EVP_hires = de.EVP(d,old_evp.variables,old_evp.eigenvalue, tolerance=self.EVP.tol)

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

        self.hires_solver = self.EVP_hires.build_solver()
        
    def discard_spurious_eigenvalues(self):
        """
        Solves the linear eigenvalue problem for two different resolutions.
        Returns trustworthy eigenvalues using nearest delta, from Boyd chapter 7.
        """

        # Solve the linear eigenvalue problem at two different resolutions.
        #LEV1 = self.evalues
        #LEV2 = self.evalues_hires
        # Eigenvalues returned by dedalus must be multiplied by -1
        lambda1 = self.evalues_low #-LEV1.eigenvalues
        lambda2 = self.evalues_high #-LEV2.eigenvalues

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
        
        lambda1 = lambda1_and_indx[:, 0]
        indx = lambda1_and_indx[:, 1].real.astype(np.int)
        
        #delta_near_unsorted = delta_near[reverse_lambda1_indx]
        #lambda1[np.where((1.0/delta_near_unsorted) < 1E6)] = None
        #lambda1[np.where(np.isnan(1.0/delta_near_unsorted) == True)] = None
    
        return lambda1, indx

