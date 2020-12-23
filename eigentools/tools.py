import dedalus.public as de

# these are the currently supported dedalus eigenvalue bases
bases_register = {"Chebyshev": de.Chebyshev, "Fourier": de.Fourier, "Legendre": de.Legendre}

def update_EVP_params(EVP, key, value):
    # Dedalus workaround: must change values in two places
    vv = EVP.namespace[key]
    vv.value = value
    EVP.parameters[key] = value

def basis_from_basis(basis, factor):
    """duplicates input basis with number of modes multiplied by input factor.

    the new number of modes will be cast to an integer

    inputs
    ------
    basis : a dedalus basis
    factor : a float that will multiply the grid size by basis

    """
    basis_type = basis.__class__.__name__
    n_hi = int(basis.base_grid_size*factor)

    if type(basis) == de.Compound:
        sub_bases = []
        for sub_basis in basis.subbases:
            sub_basis_type = sub_basis.__class__.__name__
            try:
                nb = bases_register[sub_basis_type](basis.name, n_hi, interval=sub_basis.interval)
            except KeyError:
                raise KeyError("Don't know how to make a basis of type {}".format(basis_type))
            sub_bases.append(nb)
        new_basis = de.Compound(basis.name, tuple(sub_bases))
    else:
        try:
            new_basis = bases_register[basis_type](basis.name, n_hi, interval=basis.interval)
        except KeyError:
            raise KeyError("Don't know how to make a basis of type {}".format(basis_type))

    return new_basis
