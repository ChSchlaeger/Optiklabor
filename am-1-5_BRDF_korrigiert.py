# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:57:58 2024

@author: janhen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

measurement = 'PV-Modul_gross'
path_data = measurement + '/Auswertung/' # create directories first
path_save = measurement + '/am1-5_refl/'
name_save = measurement

phi_diagram = 180
plot_type = 1 # Eingangswinkel fest: 0, Ausgangswinkel fest: 1
theta_list = [0,9,18,27,36,45,54,63,72]

"""
All BRDF values have to be divided by np.sqrt(2) and by np.cos(theta_out).
np.sqrt(2) because the calibration with spectralon is performed at 45°.
cos(45°)=1/sqrt(2)
BRDF = value * cos(45°) / cos(theta_out)
where value is obtained by running optiklabor with no correction factor.
adapt BRDF values by setting brdf_already_adapted = False
if already done or not wanted: brdf_already_adapted = True
"""
already_corrected_45 = False
# True if already multiplied with cos(45°)=1/sqrt(2)

already_corrected_theta_out = False
# True if already divided by cos(theta_out)

def goniometer_to_angle(alpha, beta, gamma, delta):
    """transform goniometer parameters to outgoing and incident direction"""
    alpha = -alpha
    theta_in = np.arccos(np.cos(alpha)*np.cos(beta))
    theta_out = np.arccos(np.cos(alpha)*np.cos(delta-beta))

    S1 = np.sin(alpha) * np.cos(gamma) * np.cos(beta-delta) + np.sin(gamma) * np.sin(beta-delta)
    S2 = np.sin(alpha) * np.cos(gamma) * np.cos(beta) + np.sin(gamma) * np.sin(beta)

    if abs(np.sqrt(1-np.cos(alpha)**2 * np.cos(beta-delta)**2)) < 1e-5:
        phi_out = np.pi
    else:
        phi_out_cos = (np.cos(beta - delta) * np.sin(alpha) * np.sin(gamma) - np.sin(beta - delta) * np.cos(gamma)) / \
                      np.sqrt(1 - np.cos(alpha) ** 2 * np.cos(beta - delta) ** 2)
        if phi_out_cos > 1:
            phi_out_cos = 1
        elif phi_out_cos < -1:
            phi_out_cos = -1
        phi_out = np.arccos(phi_out_cos)
        if S1 < 0:# and S2!=0:
            phi_out = 2*np.pi - phi_out
    
    if abs(np.sqrt(1-np.cos(alpha)**2 * np.cos(beta)**2)) < 1e-5:
        phi_in = 0.0
    else:
        phi_in_cos = (-np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.cos(beta) * np.sin(gamma)) / \
                 np.sqrt(1 - np.cos(alpha) ** 2 * np.cos(beta) ** 2)
        if phi_in_cos > 1:
            phi_in_cos = 1
        elif phi_in_cos < -1:
            phi_in_cos = -1
        phi_in = np.arccos(phi_in_cos)
        if S2 < 0:# and S1!=0:
            phi_in = 2*np.pi - phi_in
    
    return np.array([theta_in, phi_in, theta_out, phi_out])
"""
def read_file(file_name):
    brdf_data = pd.read_csv(file_name, sep='\t', header=1, index_col=4, decimal='.', encoding='latin-1')
    num_rows = len(brdf_data)
    brdf_matrix = np.zeros((num_rows,5))
    
    for i in range(num_rows):
        alpha_i = brdf_data['SampleTilt'][i] * np.pi / 180
        beta_i = brdf_data['SampleAngle'][i] * np.pi / 180
        gamma_i = brdf_data['SampleRotation'][i] * np.pi / 180
        delta_i = brdf_data['DetectorAngle'][i] * np.pi / 180
        angle_list_i = goniometer_to_angle(alpha_i, beta_i, gamma_i, delta_i)
        angle_list_i = 180 / np.pi * angle_list_i
        brdf_matrix[i][0:4] = angle_list_i
        if brdf_already_adapted == False:
            brdf_matrix[i][4] = brdf_data[' Reflexionsgrad [-]'][i] / np.sqrt(2) / np.cos(angle_list_i[2]*np.pi/180)
        elif brdf_already_adapted == True:
            brdf_matrix[i][4] = brdf_data[' Reflexionsgrad [-]'][i]
    
    return brdf_matrix
"""
def read_angles(file_name):
    brdf_data = pd.read_csv(file_name, sep='\t', header=1, index_col=4, decimal='.', encoding='latin-1')
    num_rows = len(brdf_data)
    brdf_matrix = np.zeros((num_rows,5))
    
    for i in range(num_rows):
        alpha_i = brdf_data['SampleTilt'][i] * np.pi / 180
        beta_i = brdf_data['SampleAngle'][i] * np.pi / 180
        gamma_i = brdf_data['SampleRotation'][i] * np.pi / 180
        delta_i = brdf_data['DetectorAngle'][i] * np.pi / 180
        angle_list_i = goniometer_to_angle(alpha_i, beta_i, gamma_i, delta_i)
        angle_list_i = 180 / np.pi * angle_list_i
        brdf_matrix[i][0:4] = angle_list_i
    
    return brdf_matrix

def read_brdf(file_name, brdf_matrix):
    brdf_data = pd.read_csv(file_name, sep='\t', header=1, index_col=4, decimal='.', encoding='latin-1')
    num_rows = len(brdf_data)
    brdf_values = np.zeros(num_rows)
    
    for i in range(num_rows):
        brdf_values[i] = brdf_data[' Reflexionsgrad [-]'][i]
        if already_corrected_45 == False:
            brdf_values[i] /= np.sqrt(2)
        if already_corrected_theta_out == False:
            brdf_values[i] /= np.cos(brdf_matrix[i][2]*np.pi/180)
        #brdf_values[i] = brdf_data[' Reflexionsgrad [-]'][i] / np.sqrt(2) / np.cos(brdf_matrix[i][2])
    
    print(brdf_values)
    return brdf_values

def angle_to_cartesian(theta, phi):
    """convert spherical coordinates with r=1 into cartesian coordinates"""
    return np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))

def find_x_y_brdf(brdf_matrix, a, th_plot, ph_plot):
    # Eingangswinkel fest: a = 0
    # Ausgangswinkel fest: a = 1
    
    dist_th_list = [abs(brdf_matrix[j][2*a] - th_plot) for j in range(len(brdf_matrix))]
    min_th = min(dist_th_list)
    print("min_th =", min_th)
    indices_th = [i for i, d in enumerate(dist_th_list) if d < min_th + 2]
    brdf_new = brdf_matrix[indices_th]
    
    dist_ph_list = [abs(brdf_new[j][2*a+1] - ph_plot) for j in range(len(brdf_new))]
    dist_ph_list = [min(d, 360-d) for d in dist_ph_list]
    min_ph = min(dist_ph_list)
    print("min_ph =", min_ph)
    indices = [i for i, d in enumerate(dist_ph_list) if d < min_ph + 2]
    brdf_final = brdf_new[indices]
    #print(brdf_final)
    
    brdf_color = np.array([values[4] for values in brdf_final])
    #print("BRDF:", brdf_color)
    
    if a == 0:
        th = np.array([values[2] for values in brdf_final])
        ph = np.array([values[3] for values in brdf_final])
        xyz = np.array([angle_to_cartesian(values[2]*np.pi/180, values[3]*np.pi/180) for values in brdf_final])
    elif a == 1:
        th = np.array([values[0] for values in brdf_final])
        ph = np.array([values[1] for values in brdf_final])
        xyz = np.array([angle_to_cartesian(values[0]*np.pi/180, values[1]*np.pi/180) for values in brdf_final])
    
    #xyz = np.array([angle_to_cartesian(values[0]*np.pi/180, values[1]*np.pi/180) for values in brdf_new])
    #xyz = np.array([angle_to_cartesian(values[0]*np.pi/180, values[1]*np.pi/180) for values in brdf_final])
    x = np.array([xyz[i][0] for i in range(len(xyz))])
    y = np.array([xyz[i][1] for i in range(len(xyz))])
    
    return th, ph, x, y, brdf_color

def interpolate_integrate(brdf_matrix, a, th_o, ph_o):
    th_in, ph_in, x, y, brdf_color = find_x_y_brdf(brdf_matrix, a, th_o, ph_o)
    #size = 401
    
    n_th = 100
    n_ph = 360
    th_ph_values = np.zeros((n_th*n_ph, 2))
    x_y_values = np.zeros((n_th*n_ph, 2))
    
    for i in range(n_th):
        theta = (i + 1/2) * np.pi / 2 / n_th
        for k in range(n_ph):
            phi = k * 2 * np.pi / n_ph
            th_ph_values[i*n_ph + k, 0] = theta
            th_ph_values[i*n_ph + k, 1] = phi
            xyz = angle_to_cartesian(theta, phi)
            x_y_values[i*n_ph + k, 0] = xyz[0]
            x_y_values[i*n_ph + k, 1] = xyz[1]
    
    interp = RBFInterpolator(list(zip(x, y)), brdf_color)
    
    use_values = interp(x_y_values)
    
    integral = 0
    for i in range(len(x_y_values)):
        th = th_ph_values[i,0]
        summand = use_values[i] * np.pi/2/n_th * 2*np.pi/n_ph * np.sin(th) * np.cos(th)
        integral += summand
    
    print(integral)
    return integral

def brdf_am_1_5():
    """compute the BRDF using AM 1.5 to remove dependency on wavelength"""
    file_name = "../../Desktop/am-1-5_10er-schritte.txt"
    am_1_5 = pd.read_csv(file_name, sep='\t', decimal='.', encoding='latin-1')
    wavelengths = np.array([*am_1_5["Wavelength"]])
    print("wavelengths", wavelengths)
    #d_c = np.array([*am_1_5["Direct+circumsolar"]])
    #print("Direct+circumsolar", d_c)
    g_t = np.array([*am_1_5["Global tilt"]])
    print("Global tilt", g_t)
    
    # for case that all wavelengths have been analysed (BRDF) in steps of 10 nm:
    sum_g_t = np.sum(g_t)
    print(sum_g_t)
    weights = g_t/sum_g_t
    print("weights", weights)
    
    brdf_matrix = read_angles(path_data + 'output_350.txt')
    len_brdf = len(brdf_matrix)
    
    for i in range(len(wavelengths)):
        txt_data = f'output_{int(wavelengths[i])}'
        file_l = path_data + txt_data + '.txt'
        brdf_values = read_brdf(file_l, brdf_matrix)
        for k in range(len_brdf):
            brdf_matrix[k][4] += brdf_values[k] * weights[i]
    
    print("brdf_matrix", brdf_matrix)
    np.savetxt(path_save + name_save + "_brdf_am_1_5.txt", brdf_matrix, header="theta_in phi_in theta_out phi_out BRDF")
    
    return brdf_matrix

def reflektivitaet_abh_v_wellenl(theta_refl):
    """determine the reflectivity for every wavelength (theta fixed)"""
    file_name = "../../Desktop/am-1-5_10er-schritte.txt"
    am_1_5 = pd.read_csv(file_name, sep='\t', decimal='.', encoding='latin-1')
    wavelengths = np.array([*am_1_5["Wavelength"]])
    print("wavelengths", wavelengths)
    
    brdf_m_refl = read_angles(path_data + 'output_350.txt')
    #len_brdf = len(brdf_m_refl)
    refl_list = []
    brdf_m_refl_t = brdf_m_refl.T
    
    for i in range(len(wavelengths)):
        txt_data = f'output_{int(wavelengths[i])}'
        file_l = path_data + txt_data + '.txt'
        brdf_values = read_brdf(file_l, brdf_m_refl)
        brdf_m_refl_t[4] = brdf_values
        brdf_m_refl = brdf_m_refl_t.T
        #for k in range(len_brdf):
            #brdf_k = brdf_values[k]
            #brdf_m_refl[k][4] = brdf_k
        integral = interpolate_integrate(brdf_m_refl, plot_type, theta_refl, phi_diagram)
        print(wavelengths[i], integral)
        refl_list.append(integral)
    
    refl = np.array(refl_list)
    print(refl)
    
    array1 = np.zeros((2,len(wavelengths)))
    array1[0] = wavelengths
    array1[1] = refl
    array2 = array1.T
    print(array2)
    np.savetxt(path_save + name_save + "_reflektivitaet_" + str(theta_refl) + "grad.txt", array2, header="wavelength reflectivity")
    
    return wavelengths, refl

def reflektivitaet_am1_5(brdf_matrix):
    """use the BRDF created with brdf_am_1_5() to determine the reflectivity"""
    refl_list = []
    for t in theta_list:
        integral = interpolate_integrate(brdf_matrix, plot_type, t, phi_diagram)
        refl_list.append(integral)
    array1 = np.zeros((2,len(theta_list)))
    array1[0] = np.array(theta_list)
    array1[1] = np.array(refl_list)
    array2 = array1.T
    print(array2)
    np.savetxt(path_save + name_save + "_reflektivitaet_am1_5.txt", array2, header="theta reflectivity")

def main():
    brdf_matrix_ = brdf_am_1_5()
    reflektivitaet_am1_5(brdf_matrix_)
    
    for t in theta_list:
        reflektivitaet_abh_v_wellenl(t)

if __name__ == "__main__":
    main()