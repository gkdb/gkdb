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

if __name__ == '__main__':
    theta = np.linspace(0, 2*np.pi, 201)
    c_n = [0.19987104,-0.0071777779,-0.028320948,0.0092158969,0.0023028434,-0.0024344237,0.00031095894,0.0004311613,-0.00021866101]
    s_n = [0,1.3427971e-08,2.8093142e-08,4.5221303e-08,6.6012329e-08,9.1619627e-08,1.2312091e-07,1.6147836e-07,2.0748744e-07]
    r_N = 0.174
    #c_n = [0.16,-3.0339235e-18,-3.4382197e-18,-6.9793287e-18,-2.2601963e-17,-2.7555715e-18,4.1509537e-18,6.9185594e-19,-3.0438504e-18]
    #s_n = [0,2.4888265e-18,8.1877888e-18,3.9936703e-18,-1.1231339e-17,-3.234483e-18,-2.1642556e-18,2.1832123e-18,1.2146842e-18]
    r = a_N = calculate_a_N(theta, c_n, s_n)
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, r)
    ax.set_rmax(1.2 * max(r))
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    #ax.set_theta_direction(-1)
    min_check, max_check, i_min_check, i_max_check = get_values_min_max_consistency_check(theta, a_N)
    circle3 = plt.Circle((0, 0), r_N, color='g', alpha=0.3, clip_on=False, transform=ax.transProjectionAffine + ax.transAxes)
    ax.add_artist(circle3)
    ax.plot([theta[i_min_check], theta[i_min_check]], [0, a_N[i_min_check]])
    ax.plot([theta[i_max_check], theta[i_max_check]], [0, a_N[i_max_check]])

    #embed()
    plt.show()
