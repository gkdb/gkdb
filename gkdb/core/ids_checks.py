from itertools import chain

import numpy as np
import scipy as sc
from scipy.interpolate import interp1d
from IPython import embed

allowed_codes = ['GKW', 'GENE', 'test']
error_msg = lambda errors: 'Entry does not meet GKDB definition: {!s}'.format(errors)

def check_wrapper(check_function, ids, errors, *args, on_disallowance=None, **kwargs):
    allow_entry = check_function(ids, errors, *args, **kwargs)
    if on_disallowance == 'raise_immediately' and allow_entry is not True:
        raise Exception(error_msg(errors))
    return allow_entry

def check_ids_entry(ids, on_disallowance='raise_at_end'):
    on_disallowance = 'raise_immediately'
    if on_disallowance not in ['raise_immediately', 'raise_at_end', 'print_at_end']:
        raise Exception
    allow_entry = True
    errors = []
    num_sp = len(ids['species'])
    electron = None
    allow_entry &= check_wrapper(check_code_allowed, ids, errors, on_disallowance=on_disallowance)
    allow_entry &= check_wrapper(check_electron_definition, ids, errors, on_disallowance=on_disallowance)
    allow_entry &= check_wrapper(check_quasineutrality, ids, errors, on_disallowance=on_disallowance)
    allow_entry &= check_wrapper(check_magnetic_flutter, ids, errors, on_disallowance=on_disallowance)
    allow_entry &= check_wrapper(check_magnetic_compression, ids, errors, on_disallowance=on_disallowance)
    allow_entry &= check_wrapper(check_number_of_modes, ids, errors, on_disallowance=on_disallowance)
    allow_entry &= check_wrapper(check_growth_rate_tolerance, ids, errors, on_disallowance=on_disallowance)
    allow_entry &= check_wrapper(check_poloidal_angle_grid_bounds, ids, errors, on_disallowance=on_disallowance)
    allow_entry &= check_wrapper(check_poloidal_angle_grid_lengths, ids, errors, on_disallowance=on_disallowance)
    allow_entry &= check_wrapper(check_phi_rotation, ids, errors, on_disallowance=on_disallowance)
    allow_entry &= check_wrapper(check_inconsistent_curvature_drift, ids, errors, on_disallowance=on_disallowance)
    if not allow_entry:
        if on_disallowance == 'raise_at_end':
            raise Exception(msg(errors))
        elif on_disallowance == 'print_at_end':
            print(msg(errors))
    print(allow_entry)
    return allow_entry

def check_code_allowed(ids, errors):
    allow_entry = True
    if not ids['code']['name'] in allowed_codes:
        allow_entry = False
        errors.append("Code '{!s}' not in allowed codes {!s}"
                      .format(ids['code']['name'], allowed_codes))
    return allow_entry

electron_mandatory = {'mass_norm': 2.7237e-4,
                      'temperature_norm': 1,
                      'density_norm': 1}
def check_electron_definition(ids, errors):
    allow_entry = True
    for spec in ids['species']:
        if spec['charge_norm'] == -1:
            electron = spec
    if electron is None:
        allow_entry = False
        errors.append('Electron species not found')
    else:
        for field, val in electron_mandatory.items():
            if not np.isclose(electron[field], val):
                allow_entry = False
                errors.append("Invalid value for electron species field '{!s}'".format(field))
    return allow_entry

def check_quasineutrality(ids, errors):
    allow_entry = True
    Zs = [spec['charge_norm'] for spec in ids['species']]
    ns = [spec['density_norm'] for spec in ids['species']]
    RLns = [spec['density_log_gradient_norm'] for spec in ids['species']]
    quasi = np.isclose(sum([Z * n for Z, n in zip(Zs, ns)]), 0)
    quasi_grad = np.isclose(sum([Z * n * RLn for Z, n, RLn in zip(Zs, ns, RLns)]), 0)
    if not quasi:
        allow_entry = False
        errors.append("Entry is not quasineutral! Zn = {!s} and ns = {!s}".format(Zs, ns))
    if not quasi_grad:
        allow_entry = False
        errors.append("Entry is not quasineutral for gradients! Zn = {!s}, ns = {!s} and Lns = {!s}".format(Zs, ns, Lns))

    return allow_entry

def check_magnetic_flutter(ids, errors):
    allow_entry = True
    if ids['species_all']['beta_reference'] > 0:
        if not ids['model']['include_a_field_parallel']:
            allow_entry = False
            errors.append('include_a_field_parallel should be true if beta_reference > 0')
    return allow_entry

def check_magnetic_compression(ids, errors):
    allow_entry = True
    if ids['species_all']['beta_reference'] > 0.5:
        if not ids['model']['include_b_field_parallel']:
            allow_entry = False
            errors.append('include_b_field_parallel should be true if beta_reference > 0')
    return allow_entry

def check_number_of_modes(ids, errors):
    allow_entry = True
    non_linear_run = ids['model']['non_linear_run']
    initial_value_run = ids['model']['initial_value_run']
    if not non_linear_run:
        for ii, wv in enumerate(ids['wavevector']):
            num_eigenmodes = len(wv['eigenmode'])
            if initial_value_run:
                if num_eigenmodes != 1:
                    allow_entry = False
                    errors.append('For an initial value run, the number of eigenmodes per wavevector should be 1, wavevector {!s} has {!s} eigenmodes'.format(ii, num_eigenmodes))
            else:
                if num_eigenmodes < 1:
                    allow_entry = False
                    errors.append('For an eigenvalue run, the number of eigenmodes per wavevector should be at least 1, wavevector {!s} has {!s} eigenmodes'.format(ii, num_eigenmodes))
    return allow_entry

def check_growth_rate_tolerance(ids, errors):
    growth_rate_tolerance_bound = 10
    allow_entry = True
    for ii, wv in enumerate(ids['wavevector']):
        for jj, eig in enumerate(wv['eigenmode']):
            if eig['growth_rate_tolerance'] > growth_rate_tolerance_bound:
                allow_entry = False
                errors.append('Growth rate tolerance has to be under {!s}%. Is {!s} for wavevector {!s} eigenmode {!s}'.format(growth_rate_tolerance_bound, eig['growth_rate_tolerance'], ii, jj))
    return allow_entry

def is_monotonic(array):
    return all(np.diff(array) > 0)

def check_moment_rotation(poloidal_grid, phi_potential_im, phi_theta_0_bound, check_visually=False):
    #xr = [poloidal_grid[ii] for ii in [176-1-20, 176-1, 176, 176+20]]
    #yr = [phi_potential_im[ii] for ii in [176-1-20, 176-1, 176, 176+20]]
    try:
        p_ind = poloidal_grid.index(0)
    except ValueError:
        f = interp1d(poloidal_grid, phi_potential_im, kind='cubic')
        phi_theta_0 = f(0)
        if check_visually:
            import matplotlib.pyplot as plt
            plt.scatter(poloidal_grid, phi_potential_im)
            x = np.linspace(poloidal_grid[0], poloidal_grid[-1], 100)
            plt.plot(x, f(x))
            plt.vlines(0, min(phi_potential_im), max(phi_potential_im), linestyles='--')
            plt.hlines(phi_theta_0, x[0], x[-1], linestyles='--')
            plt.title('({!s}, {!s})'.format(0, phi_theta_0))
            plt.show()
    else:
        phi_theta_0 = phi_potential_im[p_ind]

    if abs(phi_theta_0) < phi_theta_0_bound:
        rotation_okay = True
    else:
        rotation_okay = False
    return rotation_okay

def check_monoticity(ids, errors):
    allow_entry = True
    for ii, wv in enumerate(ids['wavevector']):
        for jj, eig in enumerate(wv['eigenmode']):
            grid = eig['poloidal_angle_grid']
            if not is_monotonic(grid):
                allow_entry = False
                errors.append('Poloidal angel grid should be monotonically increasing. For wavevector {!s} eigenmode {!s} it is not'.format(ii, jj))
    return allow_entry

def check_poloidal_angle_grid_bounds(ids, errors):
    allow_entry = True
    non_linear_run = ids['model']['non_linear_run']
    for ii, wv in enumerate(ids['wavevector']):
        for jj, eig in enumerate(wv['eigenmode']):
            if not non_linear_run:
                grid = eig['poloidal_angle_grid']
                poloidal_turns = wv['poloidal_turns']
                if not all([(el >= -poloidal_turns * np.pi) and el <= poloidal_turns * np.pi for el in grid]):
                    allow_entry = False
                    errors.append('Poloidal grid out of bounds! Should be between [-Np * pi, Np * pi]. For wavevector {!s} eigenmode {!s} it is not'.format(ii, jj))
    return allow_entry

def check_phi_rotation(ids, errors):
    allow_entry = True
    for ii, wv in enumerate(ids['wavevector']):
        for jj, eig in enumerate(wv['eigenmode']):
            grid = eig['poloidal_angle_grid']
            if 'phi_potential_perturbed_norm_imaginary' in eig:
                if not check_moment_rotation(grid, eig['phi_potential_perturbed_norm_imaginary'], 1e-3):
                    allow_entry = False
                    errors.append('Poloidal grid not rotated corretly! Im(phi(theta=0)) != 0 for wavevector {!s} eigenmode {!s}'.format(ii, jj))
    return allow_entry

def check_poloidal_angle_grid_lengths(ids, errors):
    allow_entry = True
    non_linear_run = ids['model']['non_linear_run']
    for ii, wv in enumerate(ids['wavevector']):
        for jj, eig in enumerate(wv['eigenmode']):
            grid = eig['poloidal_angle_grid']
            check_arrays_in = eig.items()
            if 'moments_norm_rotating_frame' in eig:
                check_arrays_in = chain(check_arrays_in, *[mom.items() for mom in eig['moments_norm_rotating_frame']])
            for field, val in check_arrays_in:
                if isinstance(val, list):
                    if len(val) != len(grid) and field not in ['moments_norm_rotating_frame', 'fluxes_norm']:
                        allow_entry = False
                        errors.append('Field {!s} for wavevector {!s} eigenmode {!s} same length as poloidal_grid'.format(field, ii, jj))
    return allow_entry

def check_inconsistent_curvature_drift(ids, errors):
    allow_entry = True
    ids['model']['include_b_field_parallel']
    if 'inconsistent_curvature_drift' in ids['model']:
        if 'include_b_field_parallel' in ids['model']:
            if ids['model']['include_b_field_parallel']:
                allow_entry = False
                errors.append('inconsistent_curvature_drift can only be defined if include_b_field_parallel is False.')
        else:
            allow_entry = False
            errors.append('inconsistent_curvature_drift can only be defined if include_b_field_parallel is defined and False.')
    return allow_entry
