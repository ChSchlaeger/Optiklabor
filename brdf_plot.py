# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:26:18 2024

@author: janhen
"""

import pandas as pd
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import RBFInterpolator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

##################################################

txt_data = "Putz_brdf_am_1_5"

path_data = "Measured_Data_AM-1-5_corrected/"
path_save = "BRDF_pictures_AM-1-5/"

file_name = path_data + txt_data + '.txt'

already_corrected_45 = True
# True if already multiplied with cos(45°)=1/sqrt(2)

already_corrected_theta_out = True
# True if already divided by cos(theta_out)

what_file = 1
# for Optiklabor output file: what_file = 0
# for files like with AM1.5 data: what_file = 1

plot_type = 1
# entrance angle fixed: plot_type = 0
# exit angle fixed: plot_type = 1

remove_max = False
# remove direct reflection to see more details, True or False

theta_diagram = 45
phi_diagram = 180

adapt_size = False

##################################################

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
    #return theta_out, phi_out, theta_in, phi_in
    #return theta_in, phi_in, theta_out, phi_out

def read_data():
    """for Optiklabor output file"""
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
        brdf_matrix[i][0:4] = angle_list_i#.round(2)
        brdf_matrix[i][4] = brdf_data[' Reflexionsgrad [-]'][i]
        #if <10^-7 ?
    
    print("brdf_matrix:", brdf_matrix)
    return brdf_matrix

def read_file():
    """for files like with AM1.5 data"""
    data1 = pd.read_csv(file_name, sep=' ', decimal='.', encoding='latin-1')
    theta_in = np.array([*data1["theta_in"]])
    phi_in = np.array([*data1["phi_in"]])
    theta_out = np.array([*data1["theta_out"]])
    phi_out = np.array([*data1["phi_out"]])
    brdf = np.array([*data1["BRDF"]])
    brdf_matrix_t = np.array([theta_in, phi_in, theta_out, phi_out, brdf])
    brdf_matrix = brdf_matrix_t.T
    return brdf_matrix

def angle_to_cartesian(theta, phi):
    """convert spherical coordinates with r=1 into cartesian coordinates"""
    return np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))

def find_x_y_brdf(a, th_plot, ph_plot):
    # Eingangswinkel fest: a = 0
    # Ausgangswinkel fest: a = 1
    
    if what_file == 0:
        brdf_matrix = read_data()
    elif what_file == 1:
        brdf_matrix = read_file()
    if already_corrected_45 == False:
        for i in range(len(brdf_matrix)):
            brdf_matrix[i][4] /= np.sqrt(2)
    if already_corrected_theta_out == False:
        for i in range(len(brdf_matrix)):
            brdf_matrix[i][4] /= np.cos(brdf_matrix[i][2]*np.pi/180)
    
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
    brdf_two = brdf_new[indices]
    
    if remove_max == True:
        max_brdf = max(brdf_two.T[4])
        max_index = list(brdf_two.T[4]).index(max_brdf)
        indices_not_max = [i for i, v in enumerate(brdf_two[:,4]) if i != max_index]
        brdf_final = brdf_two[indices_not_max]
        brdf_rest = brdf_two[max_index]
    else:
        brdf_final = brdf_two
        brdf_rest = False
    """
    indices_value = [i for i, v in enumerate(brdf_two[:,4]) if v < 1]
    brdf_final = brdf_two[indices_value]
    indices_rest = [i for i, v in enumerate(brdf_two[:,4]) if v >= 1]
    brdf_rest = brdf_two[indices_rest]
    """
    
    brdf_color = np.array([values[4] for values in brdf_final])
    print("BRDF:", brdf_color)
    
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
    
    return th, ph, x, y, brdf_color, brdf_rest

def plot_scatter_with_lines(x, y, brdf_color):
    circle = plt.Circle((0,0), 1, fill=False)#, alpha=0.3)
    circle15 = plt.Circle((0,0), np.sin(15*np.pi/180), alpha=0.3, fill=False)
    circle30 = plt.Circle((0,0), np.sin(30*np.pi/180), alpha=0.3, fill=False)
    circle45 = plt.Circle((0,0), np.sin(45*np.pi/180), alpha=0.3, fill=False)
    circle60 = plt.Circle((0,0), np.sin(60*np.pi/180), alpha=0.3, fill=False)
    circle75 = plt.Circle((0,0), np.sin(75*np.pi/180), alpha=0.3, fill=False)
    x0, y0 = [-1,1], [0,0]
    x1, y1 = [0,0], [-1,1]
    x2, y2 = [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2)]
    x3, y3 = [-1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]
    
    fig, ax = plt.subplots()
    ax.plot(x0,y0,x1,y1,x2,y2,x3,y3, c='gray', linewidth=0.5)
    abc = ax.scatter(x, y, c=brdf_color)
    ax.axis('equal')
    ax.add_patch(circle)
    ax.add_patch(circle15)
    ax.add_patch(circle30)
    ax.add_patch(circle45)
    ax.add_patch(circle60)
    ax.add_patch(circle75)
    if adapt_size == True:
        ax.set_xlim(min(x)-0.1, max(x)+0.1)
        ax.set_ylim(min(y)-0.1, max(y)+0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(abc, label='BRDF [1/sr]')
    if remove_max == True:
        plt.savefig(path_save + txt_data + '_type' + str(plot_type) + '_scatter_theta' + str(theta_diagram) + '_phi' + str(phi_diagram) + '_rem.png', dpi=200)
    else:
        plt.savefig(path_save + txt_data + '_type' + str(plot_type) + '_scatter_theta' + str(theta_diagram) + '_phi' + str(phi_diagram) + '.png', dpi=200)

def plot_interpolate_nearest(x, y, brdf_color):
    circle = plt.Circle((0,0), 1, fill=False)#, alpha=0.3)
    circle15 = plt.Circle((0,0), np.sin(15*np.pi/180), alpha=0.3, fill=False)
    circle30 = plt.Circle((0,0), np.sin(30*np.pi/180), alpha=0.3, fill=False)
    circle45 = plt.Circle((0,0), np.sin(45*np.pi/180), alpha=0.3, fill=False)
    circle60 = plt.Circle((0,0), np.sin(60*np.pi/180), alpha=0.3, fill=False)
    circle75 = plt.Circle((0,0), np.sin(75*np.pi/180), alpha=0.3, fill=False)
    
    xx = np.linspace(-1, 1, num=401)
    yy = np.linspace(-1, 1, num=401)
    
    xx, yy = np.meshgrid(xx, yy)  # 2D grid for interpolation
    interp = NearestNDInterpolator(list(zip(x, y)), brdf_color)
    zz = interp(xx, yy)
    
    for i in range(len(zz)):
        for k in range(len(zz[i])):
            if xx[i][k]**2 + yy[i][k]**2 > 1:
                zz[i][k] = "nan"
    
    fig, ax = plt.subplots(1, 1)
    cp = ax.pcolormesh(xx, yy, zz)
    ax.plot(x, y, "ok", label="measured angles", markersize=2)
    #ax.legend()
    fig.colorbar(cp, label='BRDF [1/sr]')  # Add a colorbar to a plot
    ax.axis('equal') #added this line
    ax.add_patch(circle)
    ax.add_patch(circle15)
    ax.add_patch(circle30)
    ax.add_patch(circle45)
    ax.add_patch(circle60)
    ax.add_patch(circle75)
    if adapt_size == True:
        ax.set_xlim(min(x)-0.1, max(x)+0.1)
        ax.set_ylim(min(y)-0.1, max(y)+0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

def plot_interpolate_rbf(x, y, brdf_color):
    size = 401
    
    circle = plt.Circle((0,0), 1, fill=False)#, alpha=0.3)
    circle15 = plt.Circle((0,0), np.sin(15*np.pi/180), alpha=0.3, fill=False)
    circle30 = plt.Circle((0,0), np.sin(30*np.pi/180), alpha=0.3, fill=False)
    circle45 = plt.Circle((0,0), np.sin(45*np.pi/180), alpha=0.3, fill=False)
    circle60 = plt.Circle((0,0), np.sin(60*np.pi/180), alpha=0.3, fill=False)
    circle75 = plt.Circle((0,0), np.sin(75*np.pi/180), alpha=0.3, fill=False)
    x0, y0 = [-1,1], [0,0]
    x1, y1 = [0,0], [-1,1]
    x2, y2 = [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2)]
    x3, y3 = [-1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]
    
    xx = np.linspace(-1, 1, num=size)
    yy = np.linspace(-1, 1, num=size)
    
    interp = RBFInterpolator(list(zip(x, y)), brdf_color)
    xx, yy = np.meshgrid(xx, yy)  # 2D grid for interpolation
    positions = np.vstack([xx.ravel(), yy.ravel()])
    positions = positions.T
    zz = interp(positions)
    zz = zz.reshape(size, size)
    
    for i in range(len(zz)):
        for k in range(len(zz[i])):
            if xx[i][k]**2 + yy[i][k]**2 > 1:
                zz[i][k] = "nan"
    
    fig, ax = plt.subplots(1, 1)
    cp = ax.pcolormesh(xx, yy, zz)
    ax.plot(x0,y0,x1,y1,x2,y2,x3,y3, c='black', linewidth=0.5)
    ax.plot(x, y, "ok", label="measured angles", markersize=2)
    #ax.legend()
    fig.colorbar(cp, label='BRDF [1/sr]')  # Add a colorbar to a plot
    ax.axis('equal') #added this line
    ax.add_patch(circle)
    ax.add_patch(circle15)
    ax.add_patch(circle30)
    ax.add_patch(circle45)
    ax.add_patch(circle60)
    ax.add_patch(circle75)
    if adapt_size == True:
        ax.set_xlim(min(x)-0.1, max(x)+0.1)
        ax.set_ylim(min(y)-0.1, max(y)+0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if remove_max == True:
        plt.savefig(path_save + txt_data + '_type' + str(plot_type) + '_interp_theta' + str(theta_diagram) + '_phi' + str(phi_diagram) + '_rem.png', dpi=200)
    else:
        plt.savefig(path_save + txt_data + '_type' + str(plot_type) + '_interp_theta' + str(theta_diagram) + '_phi' + str(phi_diagram) + '.png', dpi=200)

def polar_plot_scatter(th, ph, brdf_color):
    min_plot = min(0, min(brdf_color))
    print(min(ph), max(ph))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    c = ax.scatter(ph*np.pi/180, th, c=brdf_color, vmin=min_plot, vmax=max(brdf_color))
    #c = ax.scatter(ph*np.pi/180, th, c=brdf_color, vmin=min_plot, vmax=0.35)
    fig.colorbar(c, label='BRDF [1/sr]')
    #fig.colorbar(c, label=r'Rohdaten $\cdot$ cos(45°)/cos($\theta_r$)')
    if adapt_size == True:
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, max(th)+5)
    if remove_max == True:
        plt.savefig(path_save + txt_data + '_type' + str(plot_type) + '_polar_theta' + str(theta_diagram) + '_phi' + str(phi_diagram) + '_rem.png', dpi=200)
    else:
        plt.savefig(path_save + txt_data + '_type' + str(plot_type) + '_polar_theta' + str(theta_diagram) + '_phi' + str(phi_diagram) + '.png', dpi=200)

def polar_plot_scatter_more_details(th, ph, brdf_color):
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    c = ax.scatter(ph*np.pi/180, th, c=brdf_color)
    fig.colorbar(c, label='BRDF [1/sr]')
    #fig.colorbar(c, label=r'Rohdaten $\cdot$ cos(45°)/cos($\theta_r$)')
    if adapt_size == True:
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, max(th)+5)
    if remove_max == True:
        plt.savefig(path_save + txt_data + '_type' + str(plot_type) + '_polar_theta' + str(theta_diagram) + '_phi' + str(phi_diagram) + '_more_details_rem.png', dpi=200)
    else:
        plt.savefig(path_save + txt_data + '_type' + str(plot_type) + '_polar_theta' + str(theta_diagram) + '_phi' + str(phi_diagram) + '_more_details.png', dpi=200)

def main():
    th_, ph_, x_, y_, brdf_color_, brdf_rest_ = find_x_y_brdf(plot_type, theta_diagram, phi_diagram)
    plot_scatter_with_lines(x_, y_, brdf_color_)
    plot_interpolate_nearest(x_, y_, brdf_color_)
    plot_interpolate_rbf(x_, y_, brdf_color_)
    polar_plot_scatter(th_, ph_, brdf_color_)
    polar_plot_scatter_more_details(th_, ph_, brdf_color_)

if __name__ == "__main__":
    main()