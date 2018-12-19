import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

def calculate_a_N(theta, c_n, s_n):
    N_sh = len(s_n)
    n = np.arange(0, N_sh)
    a_N = sum([c_n[i] * np.cos(n[i] * theta) + s_n[i] * np.sin(n[i] * theta) for i in range(N_sh)])
    return a_N

def get_values_min_max_consistency_check(theta, a_N):
    cos = (a_N * np.cos(theta))
    sin = (a_N * np.sin(theta))
    max_check = np.amax(cos)
    i_max_check = np.argmax(cos)
    min_check = np.amin(cos)
    i_min_check = np.argmin(cos)
    return min_check, max_check, i_min_check, i_max_check

def check_r_minor_norm_shape_consistency(ids, errors):
    allow_entry = True
    s_n = ids['flux_surface']['shape_coefficients_s']
    c_n = ids['flux_surface']['shape_coefficients_c']
    dc_dr = ids['flux_surface']['dc_dr_minor_norm']
    ds_dr = ids['flux_surface']['ds_dr_minor_norm']

    if not isinstance(s_n[0], (float, int)):
        allow_entry = False
        errors.append('Shape parameters should be a 1D array!')
        return allow_entry

    N_sh = len(s_n)
    for param in [s_n, c_n, dc_dr, ds_dr]:
        if len(param) != N_sh:
            allow_entry = False
            errors.append('Plasma shape parameters have inconsistent lenghts')
            return allow_entry
    theta = get_theta_grid()
    n = np.arange(0, N_sh)
    a_N = calculate_a_N(theta, c_n, s_n)
    r_N = ids['flux_surface']['r_minor_norm']
    min_check, max_check, i_min_check, i_max_check = get_values_min_max_consistency_check(theta, a_N)
    if not np.isclose(r_N, (max_check - min_check) / 2, rtol=1e-3):
        allow_entry = False
        errors.append('Given r_minor_norm is not consistent with the given shape coefficients')

    return allow_entry

def plot_equilibrium(ids, plot_ids_check=True, plot_r_N=True):
    theta = get_theta_grid()
    s_n = ids['flux_surface']['shape_coefficients_s']
    c_n = ids['flux_surface']['shape_coefficients_c']
    r_N = ids['flux_surface']['r_minor_norm']
    r = a_N = calculate_a_N(theta, c_n, s_n)

    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, r)
    ax.set_rmax(1.2 * max(r))
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True) # Show grid
    ax.set_theta_direction(-1) # Theta is counter clockwise for GKDB
    if plot_r_N:
        circle3 = plt.Circle((0, 0), r_N, color='g', alpha=0.3, clip_on=False, transform=ax.transProjectionAffine + ax.transAxes)
        ax.add_artist(circle3)

    if plot_ids_check:
        # Plot the lines being used in IDS check
        min_check, max_check, i_min_check, i_max_check = get_values_min_max_consistency_check(theta, a_N)
        ax.plot([theta[i_min_check], theta[i_min_check]], [0, a_N[i_min_check]])
        ax.plot([theta[i_max_check], theta[i_max_check]], [0, a_N[i_max_check]])
        errors = []
        check_r_minor_norm_shape_consistency(ids, errors)
        if len(errors) == 0:
            print('IDS check success!')
        else:
            print('IDS check failed!')
            print(errors)
    plt.show()

def get_theta_grid(num=201):
    theta = np.linspace(0, 2 * np.pi, num=num)
    return theta

if __name__ == '__main__':
    import json
    with open('../../json_files/isomix1a_1.json') as file:
        ids = json.load(file)
    plot_equilibrium(ids)
