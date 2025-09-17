import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colormaps
from scipy.fft import fft, ifft, fftfreq
import math as m

def gaussian(qTime, maxTime):
    imp0 = 377
    start = int(0.20 * maxTime) # to start after 20% of the main signal 
    return np.exp(-(qTime - start) * (qTime - start)/ 10000.) /imp0

def cw(qTime):
    imp0 = 377
    start = 30
    return np.exp(-(qTime - start) * (qTime - start) / 100000000.) /imp0

def cw2(qTime, del_t, f0):
    
    f = f0 #resonant frequency (only frequency in this case)
    t = qTime * del_t
    A = 1.0 #amplitude

    phi0 = 0 #phase if required
    # s = A*np.sin(2*np.pi*f*t*dt + phi0)
    s_complex = A*np.sin(2.0*np.pi*f0*t + phi0)
    return s_complex

def cw3(qTime, del_t, f0, A=1.0, phi0=0.0, complex_drive=True):
    """
    Continuous-wave source.
    qTime : integer timestep
    del_t : timestep duration
    f0    : drive frequency (Hz)
    A     : amplitude
    phi0  : initial phase (radians)
    """
    t = qTime * del_t
    if complex_drive:
        return A * np.exp(1j*(2*np.pi*f0*t + phi0))  # complex exponential
    else:
        return A * np.sin(2*np.pi*f0*t + phi0)       # real sinusoid

#Finding the resonant frequency of the CROW hoping for it to be around the resonant frequency of the single ring resonator (on-site frequency)
#We want to do this so that the system can be driven at the zero mode frequency and topological states can be accessed.

def zero_mode_freq(E, dt, f_ref, search_bw = 5e12): 
    #f_ref is the resonant frequency of the single ring around which the res freq of the CROW should land.
    
    N = len(E)
    t = np.arange(N) * dt
    
    start = int(0.30 * N) #using the last 70 percent of the signal so that steady state is reached.
    Eg = np.asarray(E[start:], dtype = np.complex128)
    Ng = len(Eg)
    w = np.hanning(Ng)
    
    #demodulating the baseband
    t_g = t[start:]
    Ebb = Eg * np.exp(-1j*2*np.pi*f_ref*t_g)
    
    #Taking the fft in detuned coordinates
    F = np.fft.fftshift(np.fft.fft(Ebb * w))
    freqs = np.fft.fftshift(np.fft.fftfreq(Ng, d=dt))
    detuning = freqs - f_ref
    mag = np.abs(F)
    
    
    #Searching for the frequency closest to f_ref i.e. close to zero detuning
    win = np.where(np.abs(detuning) <= search_bw)[0]
    k = win[np.argmax(max(win))]  #index of max in the window
    #quadratic interpolation around (k-1, k, k+1)
    if 0 < k < len(mag)-1:
        y1, y2, y3 = mag[k-1], mag[k], mag[k+1]
        denom = (y1 - 2*y2 + y3)
        delta = 0.5*(y1-y3)/denom if denom != 0 else 0.0
    else:
        delta = 0.0
    
    df = freqs[1] - freqs[0]
    detuning_peak = detuning[k] + delta*df
    f_zero = f_ref + detuning_peak
    df_fft = 1.0 /(Ng *dt)
    
    return f_zero, df_fft, detuning, mag
    

def cosMod(qTime, complex_signal, f0, del_t):
    dt=del_t
    f0 = f0
    sigma= 10e-15 # Width
    phase=0.0
    # f0 = 300
    # Time array
    t = qTime * dt
    t0 = 4 * sigma  # Center time
    
    # Gaussian envelope
    g = np.exp(-0.5 * ((t - t0) / sigma)**2)
    
    # Signal
    if complex_signal:
        signal = g * np.exp(1j * (2 * np.pi * f0 * t + phase))
    else:
        signal = g * np.cos(2 * np.pi * f0 * t + phase)
    
    return signal

def create_colored_arc(center, radius, values, angle_range, cmap='plasma'):
    """
    Creates a colored arc (full or partial ring).
    angle_range: tuple (start_angle, end_angle) in radians.
    """
    N = len(values)
    theta = np.linspace(angle_range[0], angle_range[1], N + 1)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    # Make line segments
    segments = [[[x[i], y[i]], [x[i+1], y[i+1]]] for i in range(N)]
    segments = np.array(segments)

    # Normalize and map to colormap
    norm_values = (values - np.min(values)) / (np.ptp(values) + 1e-8)
    colors = colormaps[cmap](norm_values)

    return LineCollection(segments, colors=colors, linewidths=14), x, y

def plot_field_ring(ez_tab_tp, N_rings):
    num_rings = N_rings - 2  # Number of full rings in the middle (will be 2 less than the total number of rings as they will act as i/p and o/p)
    N_seg = ez_tab_tp.shape[0]
    print(N_seg)
    
    radius = 1.0
    spacing = 2.3  # spacing between ring centers
    cmap = 'Reds'

    fig, ax = plt.subplots(figsize=(12, 4))
    all_x = []
    all_y = []

    # Left half-ring (start)
    center = (0, 0)
    values = np.real(np.hstack((ez_tab_tp[0,:], ez_tab_tp[1,:])) )

    arc, x, y = create_colored_arc(center, radius, values, (-np.pi/2, np.pi/2), cmap)
    ax.add_collection(arc)
    all_x.extend(x)
    all_y.extend(y)

    # Full rings
    for i in range(num_rings):
        center = ((i + 1) * spacing, 0)
        if i%2 == 0:
        # values = np.real(np.hstack((ez_tab_tp[2*(i+1),:], ez_tab_tp[2*(i+1)+1,:])))
        # arc, x, y = create_colored_arc(center, radius, values, (0, 2*np.pi), cmap)
        # arc, x, y = create_colored_arc(center, radius, values, (2*np.pi,0), cmap)
            arc, x, y = create_colored_arc(center, radius, values, (0*np.pi, 2*np.pi), cmap)
        else:
            arc, x, y = create_colored_arc(center, radius, values, (-1*np.pi, 1*np.pi), cmap)

        ax.add_collection(arc)
        all_x.extend(x)
        all_y.extend(y)

    # Right half-ring (end)
    center = ((num_rings + 1) * spacing, 0)
    if num_rings%2 == 0: #even no of rings in the middle
        values = np.real( np.hstack((ez_tab_tp[N_seg-2,:], ez_tab_tp[N_seg-1,:])) )
        arc, x, y = create_colored_arc(center, radius, values,(1*np.pi/2, 3*np.pi/2) , cmap)
    else:
        values = np.real( np.hstack((ez_tab_tp[N_seg-1,:], ez_tab_tp[N_seg-2,:])) )
        arc, x, y = create_colored_arc(center, radius, values,(3*np.pi/2, 1*np.pi/2) , cmap)
        
    ax.add_collection(arc)
    all_x.extend(x)
    all_y.extend(y)
    
    #need to fix the last (right) port for even number of rings in the middle.

    # Set axis limits based on all arc coordinates
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.title(f'{num_rings} Coupled Optical Rings with Half Arcs at Ends')
    plt.tight_layout()
    plt.show()
    
def create_colored_arc_1(center, radius, values, angle_range, cmap='plasma', linewidth=14):
    """
    Create a LineCollection for an arc. `values` should be length N (number of segments).
    angle_range is (start, end) in radians.
    Returns (LineCollection, x_coords, y_coords)
    """
    N = len(values)
    # We want N segments -> N+1 theta sample points
    theta = np.linspace(angle_range[0], angle_range[1], N + 1)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    segments = np.array([[[x[i], y[i]], [x[i+1], y[i+1]]] for i in range(N)])
    norm_values = (values - np.min(values)) / (np.ptp(values) + 1e-8)
    colors = colormaps.get_cmap(cmap)(norm_values)
    return LineCollection(segments, colors=colors, linewidths=linewidth), x, y

def plot_field_ring_1(ez_tab_tp, N_rings, flip_alternate=True):
    """
    ez_tab_tp : ndarray with shape (N_rows, M) where each "row" holds samples along a segment
                We assume 2 rows per ring and 2 rows for left/right ports as described above.
    N_rings : total rings including the two port-positions (so num_full_rings = N_rings - 2)
    flip_alternate : if True, reverse the ordering of values for odd-indexed rings to fix orientation
    """
    # number of full circular rings in the middle
    num_rings = N_rings - 2
    N_rows = ez_tab_tp.shape[0]

    radius = 1.0
    spacing = 2.3
    cmap = 'Reds'
    port_half_length = 2.0   # how far the vertical port extends above/below the ring
    samples_per_segment = 100  # how many color segments to draw on each straight port

    fig, ax = plt.subplots(figsize=(12, 4))
    all_x, all_y = [], []

    # --- Left vertical input port ---
    first_center = (spacing, 0)   # center of first full ring (x = spacing)
    x_port_pos = first_center[0] - radius  # x position tangent to left side of first ring
    y_port = np.linspace(-port_half_length, port_half_length, samples_per_segment)
    x_port = np.full_like(y_port, x_port_pos)

    # Grab left port values from ez_tab_tp rows 0 and 1 (concatenate along the spatial sample axis)
    left_values = np.real(np.hstack((ez_tab_tp[0, :], ez_tab_tp[1, :])))
    # create per-segment values equal in length to number of port segments (samples_per_segment-1)
    port_vals = np.linspace(left_values[0], left_values[-1], samples_per_segment-1)
    port_segments = [[[x_port[i], y_port[i]], [x_port[i+1], y_port[i+1]]] 
                     for i in range(len(y_port)-1)]
    norm_vals = (port_vals - np.min(port_vals)) / (np.ptp(port_vals) + 1e-8)
    port_colors = colormaps.get_cmap(cmap)(norm_vals)
    ax.add_collection(LineCollection(port_segments, colors=port_colors, linewidths=14))
    all_x.extend(x_port.tolist())
    all_y.extend(y_port.tolist())

    # --- Full rings in middle ---
    # We assume row indices for ring k (k from 0..num_rings-1) are: 2*(k+1), 2*(k+1)+1
    for k in range(num_rings):
        center = ((k + 1) * spacing, 0)
        idx0 = 2 * (k + 1)
        idx1 = idx0 + 1
        # Guard against indexing errors
        if idx1 >= N_rows:
            raise IndexError(f"ez_tab_tp does not have expected rows for ring {k}: idx1={idx1} >= N_rows={N_rows}")

        # Concatenate the two rows to form one circular sampling vector
        values = np.real(np.hstack((ez_tab_tp[idx0, :], ez_tab_tp[idx1, :])))
        # Optionally flip alternate rings to ensure consistent direction around ring
        if flip_alternate and (k % 2 == 1):
            values = values[::-1]

        # Choose an angle range of 0..2pi; ensure number of values == desired N segments
        arc, x, y = create_colored_arc_1(center, radius, values, (0.0, 2*np.pi), cmap=cmap)
        ax.add_collection(arc)
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())

    # --- Right vertical output port ---
    last_center = (num_rings * spacing, 0)   # center of last full ring
    x_port_pos = last_center[0] + radius
    y_port = np.linspace(-port_half_length, port_half_length, samples_per_segment)
    x_port = np.full_like(y_port, x_port_pos)

    # Grab right port values from last two rows
    right_values = np.real(np.hstack((ez_tab_tp[N_rows-2, :], ez_tab_tp[N_rows-1, :])))
    port_vals = np.linspace(right_values[0], right_values[-1], samples_per_segment-1)
    port_segments = [[[x_port[i], y_port[i]], [x_port[i+1], y_port[i+1]]] 
                     for i in range(len(y_port)-1)]
    norm_vals = (port_vals - np.min(port_vals)) / (np.ptp(port_vals) + 1e-8)
    port_colors = colormaps.get_cmap(cmap)(norm_vals)
    ax.add_collection(LineCollection(port_segments, colors=port_colors, linewidths=14))
    all_x.extend(x_port.tolist())
    all_y.extend(y_port.tolist())

    # Final view settings
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(f'{num_rings} Coupled Optical Rings with Vertical Tangential Ports')
    plt.tight_layout()
    plt.show()



def Sources(N_rings):
    seg_no = N_rings*2
    s = np.zeros((seg_no,2), dtype=int)
    s1 = np.zeros((N_rings,2), dtype=int) #forward transmission
    s2 = np.zeros((N_rings,2), dtype=int) #backward propagation
    s1[0][1] = -1
    s1[0][0] = -1
    for i in range (N_rings):
        if i > 0:
            s1[i][0] = (2*i + 1)  #odd indices belong to the s2 segments which are responsible for back propagation
            s1[i][1] = 2*i - 2 #because if it is 0, we will have -2 index which is not what we want.
        
        s2[i][0] = 2*i  #for the transmission of odd segments  
        s2[i][1] = 2*i  + 3
        if i==(N_rings-1):
            s2[i][1] =  2*i - 1
        
    for i in range(N_rings):
        s[2*i][0] = s1[i][0]
        s[2*i][1] = s1[i][1]
        s[2*i + 1][:] = s2[i][:]

    s[2*N_rings-1][0] = -2 
    s[2*N_rings-1][1] = -2 
    return s

def Sources1(N_rings): #it pnly works for 13 rings for some reason (2 exlcuded for i/p and o/p))
    seg_no = N_rings*2
    s = np.zeros((seg_no,2), dtype=int)
    s1 = np.zeros((N_rings,2), dtype=int) #forward transmission
    s2 = np.zeros((N_rings,2), dtype=int) #backward propagation
    s1[0][1] = -1
    s1[0][0] = -1
    for i in range (N_rings):
        if i > 0:
            s1[i][0] = (2*i + 1)  #odd indices belong to the s2 segments which are responsible for back propagation
            s1[i][1] = 2*i - 2 #because if it is 0, we will have -2 index which is not what we want.
        if i < N_rings - 1:
            s2[i][0] = 2*i  #for the transmission of odd segments  
            s2[i][1] = 2*i  + 3
        # if i==(N_rings-1):
        #     s2[i][1] =  2*i - 1
        elif i == (N_rings -1):
            s1[i][0] = -2 
            s1[i][1] = -2 
    for i in range(N_rings):
        s[2*i][0] = s1[i][0]
        s[2*i][1] = s1[i][1]
        s[2*i + 1][:] = s2[i][:]

  
    return s

def Couplings(N_rings, tau):
    t = []
    c = np.zeros((N_rings*2,2), dtype=complex)
    t = tau
    k = 1j* m.sqrt(1-t**2)
    for i in range(N_rings):
        if i == 0 or i == N_rings-1:
            c[i][:] = (t[0],k[0])
            c[i+1][:] = (t[0],k[0])
        else:
            c[i][:] = (t[1],k[1])
            c[i+1][:] = (t[1],k[1])
    return c

def Couplings_1(N_rings, kappa):
    c = np.zeros((N_rings*2,2), dtype=complex)
    k = kappa
    t = np.sqrt(1-np.abs(k)**2)
    for i in range(N_rings):
        if i == 0 or i == N_rings-1:
            c[2*i][:] = (t[0],k[0])
            c[2*i+1][:] = (t[0],k[0])
        else:
            c[2*i][:] = (t[1],k[1])
            c[2*i+1][:] = (t[1],k[1])
    return c

    
def SSH_Couplings(N_rings, tau_alt, kappa_alt, kappa_guide, tau_guide):
    c = np.zeros((N_rings*2,2), dtype=complex)
    #  k = np.zeros(len(t), dtype=complex)
    t = tau_alt
    k = kappa_alt
    k_g = kappa_guide
    t_g = tau_guide
   
    # for i in range(len(t)):
        # k[i] = 1j* m.sqrt(1-t[i]**2)
    for i in range(N_rings*2):
        if i%2 == 0:
            c[i][:] = (t[0],k[0])
        else:
            c[i][:] = (t[1],k[1])
    c[0,:] = (t_g,k_g) #coupling for the waveguide-resonator pair
    c[1,:] = (t_g,k_g)
    c[N_rings*2 -1,:] = (t_g,k_g)
    c[N_rings*2 -2,:] = (t_g,k_g)
    return c

def SSH_Couplings_1(N_rings, tau_alt, kappa_alt, kappa_guide, tau_guide):
    """
    Coupling setup for SSH-like chain with pairwise alternation (W-W, S-S,...).
    
    N_rings    : number of resonators
    tau_alt    : [tau_w, tau_s]   weak/strong ring-ring transmission
    kappa_alt  : [k_w, k_s]       weak/strong ring-ring coupling
    kappa_guide: waveguide-resonator coupling
    tau_guide  : waveguide-resonator transmission
    """
    N_seg = N_rings * 2
    c = np.zeros((N_seg, 2), dtype=complex)
    labels = [""] * N_seg  # for W/S labeling

    # waveguide-resonator coupling at input (two segs)
    c[0, :] = (tau_guide, kappa_guide)
    c[1, :] = (tau_guide, kappa_guide)
    labels[0:2] = ["G", "G"]

    # alternating pairs for resonator–resonator couplings
    alt = 0  # 0 = weak, 1 = strong
    pair_count = 0
    for i in range(2, N_seg-2):
        if alt == 0:  # weak
            c[i, :] = (tau_alt[0], kappa_alt[0])
            labels[i] = "W"
        else:         # strong
            c[i, :] = (tau_alt[1], kappa_alt[1])
            labels[i] = "S"

        pair_count += 1
        if pair_count == 2:  # switch every 2
            alt = 1 - alt
            pair_count = 0

    # waveguide-resonator coupling at output (two segs)
    c[N_seg-2, :] = (tau_guide, kappa_guide)
    c[N_seg-1, :] = (tau_guide, kappa_guide)
    c[N_seg-3, :] = (tau_guide, kappa_guide)
    c[N_seg-4, :] = (tau_guide, kappa_guide)
    labels[-2:] = ["G", "G"]

    return c
