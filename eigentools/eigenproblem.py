from dedalus.tools.cache import CachedAttribute
import logging
from dedalus.core.field import Field
from dedalus.core.evaluator import Evaluator
from dedalus.core.system import FieldSystem
from dedalus.tools.post import merge_process_files
import dedalus.public as de
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.sparse.linalg
from . import tools

logger = logging.getLogger(__name__.split('.')[-1])

class Eigenproblem():
    def __init__(self, EVP, reject=True, factor=1.5, scales=1, drift_threshold=1e6, use_ordinal=False, grow_func=lambda x: x.real, freq_func=lambda x: x.imag):
        """An object for feature-rich eigenvalue analysis.

        Eigenproblem provides support for common tasks in eigenvalue
        analysis. Dedalus EVP objects compute raw eigenvalues and
        eigenvectors for a given problem; Eigenproblem provides support for
        numerous common tasks required for scientific use of those
        solutions. This includes rejection of inaccurate eigenvalues and
        analysis of those rejection criteria, plotting of eigenmodes and
        spectra, and projection of 1-D eigenvectors onto 2- or 3-D domains
        for use as initial conditions in subsequent initial value problems.

        Additionally, Eigenproblems can compute epsilon-pseudospectra for
        arbitrary Dedalus differential-algebraic equations.
        

        Parameters
        ----------
        EVP : dedalus.core.problems.EigenvalueProblem
            The Dedalus EVP object containing the equations to be solved
        reject : bool, optional
            whether or not to reject spurious eigenvalues (default: True)
        factor : float, optional
            The factor by which to multiply the resolution. 
            NB: this must be a rational number such that factor times the
            resolution of EVP is an integer. (default: 1.5)
        scales : float, optional
            A multiple for setting the grid resolution.  (default: 1)
        drift_threshold : float, optional
            Inverse drift ratio threshold for keeping eigenvalues during
            rejection (default: 1e6)
        use_ordinal : bool, optional
            If true, use ordinal method from Boyd (1989); otherwise use
            nearest (default: False)
        grow_func : func
            A function that takes a complex input and returns the growth
            rate as defined by the EVP (default: uses real part)
        freq_func : func
            A function that takes a complex input and returns the frequency
            as defined by the EVP (default: uses imaginary part)

        Attributes
        ----------
        evalues : ndarray
            Lists "good" eigenvalues
        evalues_low : ndarray
            Lists eigenvalues from low resolution solver (i.e. the
            resolution of the specified EVP)
        evalues_high : ndarray
            Lists eigenvalues from high resolution solver (i.e. factor
            times specified EVP resolution)
        pseudospectrum : ndarray
            epsilon-pseudospectrum computed at specified points in the
            complex plane
        ps_real : ndarray
            real coordinates for epsilon-pseudospectrum 
        ps_imag : ndarray
            imaginary coordinates for epsilon-pseudospectrum 

        Notes
        -----
        See references for algorithms in individual method docstrings.

        """
        self.reject = reject
        self.factor = factor
        self.EVP = EVP
        self.solver = EVP.build_solver()
        if self.reject:
            self._build_hires()

        self.grid_name = self.EVP.domain.bases[0].name
        self.evalues = None
        self.evalues_low = None
        self.evalues_high = None
        self.pseudospectrum = None
        self.ps_real = None
        self.ps_imag = None

        self.drift_threshold = drift_threshold
        self.use_ordinal = use_ordinal
        self.scales = scales
        self.grow_func = grow_func
        self.freq_func = freq_func

    def _set_parameters(self, parameters):
        """set the parameters in the underlying EVP object

        Parameters
	----------
        parameters : dict
            Dict of parameter names and values (keys and values
            respectively) to set in EVP


        """
        for k,v in parameters.items():
            tools.update_EVP_params(self.EVP, k, v)
            if self.reject:
                tools.update_EVP_params(self.EVP_hires, k, v)

    def grid(self):
        """get grid points for eigenvectors.

        """
        return self.EVP.domain.grids(scales=self.scales)[0]

    def solve(self, sparse=False, parameters=None, pencil=0, N=15, target=0, **kwargs):
        """solve underlying eigenvalue problem.

        Parameters
        ----------
        sparse : bool, optional
            If true, use sparse solver, otherwise use dense solver
            (default: False)
        parameters : dict, optional
            A dict giving parameter names and values to the EVP. If None,
            use values specified at EVP construction time.  (default: None)
        pencil : int, optional
            The EVP pencil to be solved. (default: 0)
        N : int, optional
            The number of eigenvalues to find if using a sparse solver
            (default: 15)
        target : complex, optional
            The target value to search for when using sparse solver
            (default: 0+0j)
        
        
        """
        if parameters:
            self._set_parameters(parameters)
        self.pencil = pencil
        self.N = N
        self.target = target
        self.solver_kwargs = kwargs

        self._run_solver(self.solver, sparse)
        self.evalues_low = self.solver.eigenvalues

        if self.reject:
            self._run_solver(self.hires_solver, sparse)
            self.evalues_high = self.hires_solver.eigenvalues
            self._reject_spurious()
        else:
            self.evalues = self.evalues_low
            self.evalues_index = np.arange(len(self.evalues),dtype=int)

    def _run_solver(self, solver, sparse):
        """wrapper method to run solver.

        Parameters
        ----------
        solver : dedalus.core.problems.EigenvalueProblem
            The Dedalus EVP object containing the equations to be solved
        sparse : bool
            If True, use sparse solver; otherwise use dense.
        """
        if sparse:
            solver.solve_sparse(solver.pencils[self.pencil], N=self.N, target=self.target, rebuild_coeffs=True, **self.solver_kwargs)
        else:
            solver.solve_dense(solver.pencils[self.pencil], rebuild_coeffs=True)

    def _set_eigenmode(self, index, all_modes=False):
        """use EVP solver's set_state to access eigenmode in grid or coefficient space
        
        The index parameter is either the index of the ordered good
        eigenvalues or the direct index of the low-resolution EVP depending
        on the all_modes option.

        Parameters
        ----------
        index : int
            index of eigenvalue corresponding to desired eigenvector
        all_modes : bool, optional
            If True, index specifies the unsorted index of the
            low-resolution EVP; otherwise it is the index corresponding to
            the self.evalues order (default: False)
        """
        if all_modes:
            good_index = index
        else:
            good_index = self.evalues_index[index]
        self.solver.set_state(good_index)

    def eigenmode(self, index, scales=None, all_modes=False):
        """Returns Dedalus FieldSystem object containing the eigenmode
        given by index.


        Parameters
        ----------
        index : int
            index of eigenvalue corresponding to desired eigenvector
        scales : float
            A multiple for setting the grid resolution. If not None, will
            overwrite self.scales.  (default: None)
        all_modes : bool, optional
            If True, index specifies the unsorted index of the
            low-resolution EVP; otherwise it is the index corresponding to
            the self.evalues order (default: False)
        """
        self._set_eigenmode(index, all_modes=all_modes)
        if scales is not None:
            self.scales = scales
        for f in self.solver.state.fields:
            f.set_scales(self.scales,keep_data=True)

        return self.solver.state
        
    def growth_rate(self, parameters=None, **kwargs):
        """returns the maximum growth rate, defined by self.grow_func(),
        the index of the maximal mode, and the frequency of that mode. If
        there is no growing mode, returns the slowest decay rate.
        
        also returns the index of the fastest growing mode.  If there are
        no good eigenvalues, returns np.nan for all three quantities.

        Returns
        -------
        growth_rate, index, freqency : tuple of ints
        
        """
        try:
            self.solve(parameters=parameters, **kwargs)
            gr_rate = np.max(self.grow_func(self.evalues))
            gr_indx = np.where(self.grow_func(self.evalues) == gr_rate)[0]
            freq = self.freq_func(self.evalues[gr_indx[0]])

            return gr_rate, gr_indx[0], freq

        except np.linalg.linalg.LinAlgError:
            logger.warning("Dense eigenvalue solver failed for parameters {}".format(params))
            return np.nan, np.nan, np.nan
        except (scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence, scipy.sparse.linalg.eigen.arpack.ArpackError):
            logger.warning("Sparse eigenvalue solver failed to converge for parameters {}".format(params))
            return np.nan, np.nan, np.nan

    def plot_mode(self, index, fig_height=8, norm_var=None, scales=None, all_modes=False):
        """plots eigenvector corresponding to specified index.

        By default, the plot will show the real and complex parts of the
        unnormalized components of the eigenmode. If a norm_var is
        specified, all components will be scaled such that variable chosen
        is purely real and has unit amplitude.

        Parameters
        ----------
        index : int
            index of eigenvalue corresponding to desired eigenvector
        fig_height : float, optional
            Height of constructed figure (default: 8)
        norm_var : str
            If not None, selects the field in the eigenmode with which to
            normalize. Otherwise, plots the unnormalized
            eigenmode. (default: None)
        scales : float
            A multiple for setting the grid resolution. If not None, will
            overwrite self.scales.  (default: None)
        all_modes : bool, optional
            If True, index specifies the unsorted index of the
            low-resolution EVP; otherwise it is the index corresponding to
            the self.evalues order (default: False)

        Returns
        -------
        matplotlib.figure.Figure

        """
        state = self.eigenmode(index, scales=scales, all_modes=all_modes)

        z = self.grid()
        nrow = 2
        nvars = len(self.EVP.variables)
        ncol = int(np.ceil(nvars/nrow))

        if norm_var:
            rotation = self.solver.state[norm_var]['g'].conj()
        else:
            rotation = 1.

        fig = plt.figure(figsize=[fig_height*ncol/nrow,fig_height])
        for i,v in enumerate(self.EVP.variables):
            ax  = fig.add_subplot(nrow,ncol,i+1)
            ax.plot(z, (rotation*state[v]['g']).real, label='real')
            ax.plot(z, (rotation*state[v]['g']).imag, label='imag')
            ax.set_xlabel(self.grid_name)
            ax.set_ylabel(v)
            if i == 0:
                ax.legend()
                
        fig.tight_layout()

        return fig

    def project_mode(self, index, domain, transverse_modes, all_modes=False):
        """projects a mode specified by index onto a domain of higher
        dimension.

        Parameters
        ----------
        index : 
            an integer giving the eigenmode to project
        domain : 
            a domain to project onto
        transverse_modes : 
            a tuple of mode numbers for the transverse directions

        Returns
        -------
           dedalus.core.system.FieldSystem
        """
        
        if len(transverse_modes) != (len(domain.bases) - 1):
            raise ValueError("Must specify {} transverse modes for a domain with {} bases; {} specified".format(len(domain.bases)-1, len(domain.bases), len(transverse_modes)))
        
        field_slice = tuple(transverse_modes) + (slice(None),)

        self._set_eigenmode(index, all_modes=all_modes)

        fields = []
        
        for v in self.EVP.variables:
            fields.append(domain.new_field(name=v))
            fields[-1]['c'][field_slice] = self.solver.state[v]['c']
        field_system = FieldSystem(fields)

        return field_system
    
    def write_global_domain(self, field_system, base_name="IVP_output"):
        """Given a field system, writes a Dedalus HDF5 file.

        Typically, one would use this to write a field system constructed by project_mode. 

        Parameters
        ----------
        field_system : dedalus.core.system.FieldSystem
            A field system containing the data to be written
        base_name : str, optional
            The base filename of the resulting HDF5 file. (default: IVP_output)

        """
        output_evaluator = Evaluator(field_system.domain, self.EVP.namespace)
        output_handler = output_evaluator.add_file_handler(base_name)
        output_handler.add_system(field_system)

        output_evaluator.evaluate_handlers(output_evaluator.handlers, timestep=0,sim_time=0, world_time=0, wall_time=0, iteration=0)

        merge_process_files(base_name, cleanup=True, comm=output_evaluator.domain.distributor.comm)

    def calc_ps(self, k, zgrid, mu=0., pencil=0, inner_product=None, norm=-2, maxiter=10, rtol=1e-3, parameters=None, **kw):
        """computes epsilon-pseudospectrum for the eigenproblem.

        Uses the algorithm described in section 5 of

        Embree & Keeler (2017). SIAM J. Matrix Anal. Appl. 38, 3:
        1028-1054.

        to enable the approximation of epsilon-pseudospectra for arbitrary
        differential-algebraic equation systems.


        Parameters:
        -----------
        k    : int
            number of eigenmodes in invariant subspace
        zgrid : tuple
            (real, imag) points
        mu : complex
            center point for pseudospectrum. 
        pencil : int
            pencil holding EVP
        inner_product : function
            a function that takes two field systems and computes their
            inner product
        parameters : dict, optional
            A dict giving parameter names and values to the EVP. If None,
            use values specified at EVP construction time.  (default: None)
        """

        self.solve(sparse=True, N=k, pencil=pencil, parameters=parameters, **kw) # O(N k)?
        pre_right = self.solver.pencils[pencil].pre_right
        pre_right_LU = scipy.sparse.linalg.splu(pre_right.tocsc()) # O(N)
        V = pre_right_LU.solve(self.solver.eigenvectors) # O(N k)

        # Orthogonalize invariant subspace
        Q, R = np.linalg.qr(V) # O(N k^2)

        # Compute approximate Schur factor
        E = -(self.solver.pencils[pencil].M_exp)
        A = (self.solver.pencils[pencil].L_exp)
        A_mu_E = A - mu*E
        A_mu_E_LU = scipy.sparse.linalg.splu(A_mu_E.tocsc()) # O(N)
        Ghat = Q.conj().T @ A_mu_E_LU.solve(E @ Q) # O(N k^2)

        # Invert-shift Schur factor
        I = np.identity(k)
        if inner_product is not None:
            M = self.compute_mass_matrix(pre_right@Q, inner_product)
            Z, S = np.linalg.qr(scipy.linalg.cholesky(M))
            Gmu = S@np.linalg.inv(S@Ghat) + mu*I
        else:
            logger.warning("No inner product given. Using 2-norm of state vector coefficients. This is probably not physically meaningful, especially if you are using Chebyshev polynomials.")
            Gmu = np.linalg.inv(Ghat) + mu*I # O(k^3)

        self.pseudospectrum = self._pseudo(Gmu, zgrid, maxiter=maxiter, rtol=rtol)
        self.ps_real = zgrid[0]
        self.ps_imag = zgrid[1]

    def compute_mass_matrix(self, Q, inner_product):
        """Compute the mass matrix M using a given inner product

        M must be hermitian, so we compute only half the inner products.

        Parameters
        ----------
        Q : ndarray
            Matrix of eigenvectors
        inner_product : function
            a function that takes two field systems and computes their
            inner product

        Returns
        -------
        ndarray
        
        """
        k = Q.shape[1]
        M = np.zeros((k,k), dtype=np.complex128)
        Xj = self._copy_system(self.solver.state)
        Xi = self._copy_system(self.solver.state)

        for j in range(k):
            self.set_state(Xj, Q[:,j])
            for i in range(j,k): # M must be hermitian
                self.set_state(Xi, Q[:,i])
                M[j,i] = inner_product(Xj, Xi)
                M[i,j] = M[j,i].conj()

        return M

    def set_state(self, system, evector):
        """
        Set system to given evector

        Parameters
        ----------
        system : FieldSystem
            system to fill in
        evector : ndarray
            eigenvector
        """
        system.data[:] = 0
        system.set_pencil(self.solver.eigenvalue_pencil, evector)
        system.scatter()

    def _copy_system(self, state):
        """copies a field system.
        
        Parameters
        ----------
        state : dedalus.core.system.FieldSystem
        
        Returns
        -------
        dedalus.core.system.FieldSystem
        """
        fields = []
        for f in state.fields:
            field = f.copy()
            field.name = f.name
            fields.append(field)
            
        return FieldSystem(fields)

    def _pseudo(self, L, zgrid, maxiter=10, rtol=1e-3):
        """computes epsilon-pseudospectrum for a regular eigenvalue
        problem.

        If maxiter is zero, uses a direct algorithm: at point z in the
        complex plane, the resolvant R is calculated

        R = ||z*I - L||_{-2}

        finding the maximum singular value.
        
        If maxiter is not zero, uses the iterative algorithm from figure
        39.3 (p.375) of

        Trefethen & Embree, "Spectra and Pseudospectra: The Behavior of
        Nonnormal Matrices and Operators" (2005, Princeton University
        Press)
        
        Parameters
        ----------
        L : square 2D ndarray
            the matrix to be analyzed
        zgrid : tuple
            (real, imag) points

        Returns
        -------
        ndarray
        """
        xx = zgrid[0]
        yy = zgrid[1]
        R = np.zeros((len(xx), len(yy)))
        matsize = L.shape[0]
        T, Z = scipy.linalg.schur(L, output='complex')
        if maxiter == 0:
            logger.debug("Using direct solver for calculating pseudospectrum")
        else:
            logger.debug("Using iterative solver for calculating pseudospectrum")
        for j, y in enumerate(yy):
            for i, x in enumerate(xx):
                z = (x + 1j*y)
                # if _maxiter is set to zero
                if maxiter == 0:
                    R[j,i] = np.linalg.norm((z*np.eye(matsize) - L), ord=-2)
                else:
                    T1 = z*np.eye(matsize) - T
                    T2 = T1.conj().T
                    sigold = 0
                    qold = np.zeros(matsize,dtype=np.complex128)
                    beta = 0

                    q = np.random.randn(matsize)+1j*np.random.randn(matsize)
                    q /= np.linalg.norm(q)
                    H = np.zeros((maxiter+1, maxiter+1), dtype=np.complex128)
                    for p in range(maxiter):
                        v = scipy.linalg.solve_triangular(T1, scipy.linalg.solve_triangular(T2,q,lower=True)) - beta*qold
                        alpha = np.dot(q.conj(), v)
                        v -= alpha*q
                        beta = np.linalg.norm(v)
                        qold = q
                        q = v/beta
                        H[p+1,p] = beta
                        H[p,p+1] = beta
                        H[p,p]   = alpha
                        sig = np.max(np.linalg.eigvalsh(H[:p+1,:p+1]))
                        if np.abs(sigold/sig - 1) < rtol:
                            break
                        sigold = sig
                    if p == (maxiter - 1):
                        logger.warning("Iterative solver did not converge for (x, y) = ({},{})".format(x,y))
                    R[j, i] = 1/np.sqrt(sig)
        return R

    def plot_spectrum(self, axes=None, spectype='good', xlog=True, ylog=True, real_label="real", imag_label="imag"):
        """Plots the spectrum.

        The spectrum plots real parts on the x axis and imaginary parts on
        the y axis.

        Parameters
        ----------
        spectype : {'good', 'low', 'high'}, optional
            specifies whether to use good, low, or high eigenvalues
        xlog : bool, optional
            Use symlog on x axis
        ylog : bool, optional
            Use symlog on y axis
        real_label : str, optional
            Label to be applied to the real axis
        imag_label : str, optional
            Label to be applied to the imaginary axis
        """
        if spectype == 'low':
            ev = self.evalues_low
        elif spectype == 'high':
            ev = self.evalues_high
        elif spectype == 'good':
            ev = self.evalues_good
        else:
            raise ValueError("Spectrum type is not one of {low, high, good}")

        if axes is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axes
            fig = axes.figure
                
        ax.scatter(ev.real, ev.imag)

        if xlog:
            ax.set_xscale('symlog')
        if ylog:
            ax.set_yscale('symlog')
        ax.set_xlabel(real_label)
        ax.set_ylabel(imag_label)
        if axes is None:
            fig.tight_layout()

        return ax

    def _reject_spurious(self):
        """perform eigenvalue rejection

        """
        evg, indx = self._discard_spurious_eigenvalues()
        self.evalues_good = evg
        self.evalues_index = indx
        self.evalues = self.evalues_good

    def _build_hires(self):
        """builds a high-resolution EVP from the EVP passed in at
        construction

        """
        old_evp = self.EVP
        old_x = old_evp.domain.bases[0]

        x = tools.basis_from_basis(old_x, self.factor)
        d = de.Domain([x],comm=old_evp.domain.dist.comm)
        self.EVP_hires = de.EVP(d,old_evp.variables,old_evp.eigenvalue, ncc_cutoff=old_evp.ncc_kw['cutoff'], max_ncc_terms=old_evp.ncc_kw['max_terms'], tolerance=self.EVP.tol)

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
        
    def _discard_spurious_eigenvalues(self):
        """ Solves the linear eigenvalue problem for two different
        resolutions.  Returns trustworthy eigenvalues using nearest delta,
        from Boyd chapter 7.
        """
        eval_low = self.evalues_low
        eval_hi = self.evalues_high

        # Reverse engineer correct indices to make unsorted list from sorted
        reverse_eval_low_indx = np.arange(len(eval_low)) 
        reverse_eval_hi_indx = np.arange(len(eval_hi))
    
        eval_low_and_indx = np.asarray(list(zip(eval_low, reverse_eval_low_indx)))
        eval_hi_and_indx = np.asarray(list(zip(eval_hi, reverse_eval_hi_indx)))
        
        # remove nans
        eval_low_and_indx = eval_low_and_indx[np.isfinite(eval_low)]
        eval_hi_and_indx = eval_hi_and_indx[np.isfinite(eval_hi)]
    
        # Sort eval_low and eval_hi by real parts
        eval_low_and_indx = eval_low_and_indx[np.argsort(eval_low_and_indx[:, 0].real)]
        eval_hi_and_indx = eval_hi_and_indx[np.argsort(eval_hi_and_indx[:, 0].real)]
        
        eval_low_sorted = eval_low_and_indx[:, 0]
        eval_hi_sorted = eval_hi_and_indx[:, 0]

        # Compute sigmas from lower resolution run (gridnum = N1)
        sigmas = np.zeros(len(eval_low_sorted))
        sigmas[0] = np.abs(eval_low_sorted[0] - eval_low_sorted[1])
        sigmas[1:-1] = [0.5*(np.abs(eval_low_sorted[j] - eval_low_sorted[j - 1]) + np.abs(eval_low_sorted[j + 1] - eval_low_sorted[j])) for j in range(1, len(eval_low_sorted) - 1)]
        sigmas[-1] = np.abs(eval_low_sorted[-2] - eval_low_sorted[-1])

        if not (np.isfinite(sigmas)).all():
            logger.warning("At least one eigenvalue spacings (sigmas) is non-finite (np.inf or np.nan)!")
    
        # Ordinal delta
        self.delta_ordinal = np.array([np.abs(eval_low_sorted[j] - eval_hi_sorted[j])/sigmas[j] for j in range(len(eval_low_sorted))])

        # Nearest delta
        self.delta_near = np.array([np.nanmin(np.abs(eval_low_sorted[j] - eval_hi_sorted)/sigmas[j]) for j in range(len(eval_low_sorted))])
    
        # Discard eigenvalues with 1/delta_near < drift_threshold
        if self.use_ordinal:
            inverse_drift = 1/self.delta_ordinal
        else:
            inverse_drift = 1/self.delta_near
        eval_low_and_indx = eval_low_and_indx[np.where(inverse_drift > self.drift_threshold)]
        
        eval_low = eval_low_and_indx[:, 0]
        indx = eval_low_and_indx[:, 1].real.astype(np.int)
    
        return eval_low, indx

    def plot_drift_ratios(self, axes=None):
        """Plot drift ratios (both ordinal and nearest) vs. mode number.

        The drift ratios give a measure of how good a given eigenmode is;
        this can help set thresholds.

        Returns
        -------
        matplotlib.figure.Figure        

        """
        if self.reject is False:
            raise NotImplementedError("Can't plot drift ratios unless eigenvalue rejection is True.")

        if axes is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = axes
            fig = axes.figure

        mode_numbers = np.arange(len(self.delta_near))
        ax.semilogy(mode_numbers,1/self.delta_near,'o',alpha=0.4)
        ax.semilogy(mode_numbers,1/self.delta_ordinal,'x',alpha=0.4)

        ax.set_prop_cycle(None)
        good_near = 1/self.delta_near > self.drift_threshold
        good_ordinal = 1/self.delta_ordinal > self.drift_threshold
        ax.semilogy(mode_numbers[good_near],1/self.delta_near[good_near],'o', label='nearest')
        ax.semilogy(mode_numbers[good_ordinal],1/self.delta_ordinal[good_ordinal],'x',label='ordinal')
        ax.axhline(self.drift_threshold,alpha=0.4, color='black')
        ax.set_xlabel("mode number")
        ax.set_ylabel(r"$1/\delta$")
        ax.legend()

        return ax
