#!/usr/bin/python3
# -*- coding: utf-8 -*-

import subprocess, os
import numpy as np
import lmfit

import matplotlib
matplotlib.use('agg')
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
import pylab as plt


# Helper functions
########################################
def parse_LAMMPS(infile, skiplines=9):

    with open(infile, 'r') as fin:
        
        atoms = []
        molecules = {}
        
        for i, line in enumerate(fin.readlines()):
            if i>skiplines:
                atom_id, molecule_id, atom_type, x, y, z, c_pe = line.split()
                
                atom = {
                    'atom_id' : int(atom_id) ,
                    'molecule_id' : int(molecule_id) ,
                    'atom_type' : int(atom_type) ,
                    'x' : float(x) ,
                    'y' : float(y) ,
                    'z' : float(z) ,
                    'c_pe' : float(c_pe) ,
                    }
                atoms.append(atom)
                
                if atom['molecule_id'] not in molecules:
                    molecules[ atom['molecule_id'] ] = { 'molecule_id': atom['molecule_id'], 'atoms' : [] }
                    
                molecules[atom['molecule_id']]['atoms'].append(atom)
                    
                
                
    return atoms, molecules

def parse_LAMMPS_box(infile):

    with open(infile, 'r') as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            if line[:17]=='ITEM: BOX BOUNDS ':
                break
            
        xi, xf = [ float(v) for v in lines[i+1].split() ]
        yi, yf = [ float(v) for v in lines[i+2].split() ]
        zi, zf = [ float(v) for v in lines[i+3].split() ]
                    
                
                
    return [ [xi, xf], [yi, yf], [zi, zf] ]
        
        
    
def generate_grid(atoms, atom_type=1, N=50, box=None):
    
    if box is None:
        xvals = [atom['x'] for atom in atoms]
        yvals = [atom['y'] for atom in atoms]
        zvals = [atom['z'] for atom in atoms]
        
        box = [ [np.min(xvals), np.max(xvals)], [np.min(yvals), np.max(yvals)], [np.min(zvals), np.max(zvals)] ] 
    
    [xi, xf], [yi, yf], [zi, zf] = box
    
    grid = np.zeros((N,N,N))
    
    for atom in atoms:
        if atom['atom_type']==atom_type:
            x = atom['x']
            y = atom['y']
            z = atom['z']
            
            ix = int( ( (x-xi)/(xf-xi) )*N )
            iy = int( ( (y-yi)/(yf-yi) )*N )
            iz = int( ( (z-zi)/(zf-zi) )*N )
            
            grid[ix,iy,iz] += 1
    
    
    return grid        
                
def pov_atoms(atoms, output_dir='./', atom_size=5.0, num_types=None, color_list=None):
    '''Generate a visualization of the atom positions. Note that this uses POV-Ray
    and makes many assumptions about how it's installed/configured on the system.'''
    
    
    
    # Generate a POV-Ray file for the atom coordinates
    if num_types is None:
        # Determine how many distinct types of atoms
        atom_types = np.unique([ atom['atom_type'] for atom in atoms ])
    else:
        atom_types = range(1, num_types+1, 1)
    
    
    if color_list is None:
        color_list = [
                    [1.0, 0.05, 0.05],
                    [0.05, 0.05, 1],
                    [0.1, 0.9, 0.1],
                    [0.2, 0.2, 0.2] ,
                    ]
    
    with open(output_dir+'atoms.pov', 'w') as fout:
        
        
        for i, atom_type in enumerate(atom_types):
            fout.write("#declare size{:d} = {:.1f};\n".format(atom_type, atom_size))
            
            r, g, b = color_list[i%len(color_list)]
            fout.write( "#declare texture{:d} = texture {{ pigment {{ rgb <{:.1f}, {:.1f}, {:.1f}> }} }}\n".format(atom_type, r, g, b) )
            
        fout.write("#declare atoms = union {\n")
        
        for atom in atoms:
            fout.write('    sphere {{ <{:.6f}, {:.6f}, {:.6f}>, size{:d} texture {{ texture{:d} }} }} // atom {:d}, molecule {:d}\n'.format(atom['x'], atom['y'], atom['z'], atom['atom_type'], atom['atom_type'], atom['atom_id'], atom['molecule_id']) )
            
            
        fout.write("} // End atoms\n")
        
    
    # Run POV-Ray
    p = subprocess.Popen(['povray', 'Library_Path=/usr/share/povray-3.7/include/ +W1600 +H1200 +A0.3 +UA +Irender_atoms.pov'], cwd=output_dir)
    p.wait()        


def pov_molecules(molecules, output_dir='./', atom_size=5.0, blobbiness=0.5, num_types=None, color_list=None):
    '''Generate a visualization of the atom positions. Note that this uses POV-Ray
    and makes many assumptions about how it's installed/configured on the system.'''
    
    
    
    # Generate a POV-Ray file for the atom coordinates
    if num_types is None:
        # Determine how many distinct types of atoms
        atom_types = []
        for molecule_id, molecule in molecules.items():
            cur_types = [ atom['atom_type'] for atom in molecule['atoms'] ]
            atom_types += cur_types
        atom_types = np.unique(atom_types)

    else:
        atom_types = range(1, num_types+1, 1)

    if color_list is None:
        color_list = [
                    [1.0, 0.05, 0.05],
                    [0.05, 0.05, 1],
                    [0.1, 0.9, 0.1],
                    [0.2, 0.2, 0.2] ,
                    ]
    
    
    threshold = 1.0
    blob_strength = (1+1/blobbiness)*threshold
    atom_size = atom_size/np.sqrt(1-np.sqrt(threshold/blob_strength))
    
    with open(output_dir+'molecules.pov', 'w') as fout:
        
        for i, atom_type in enumerate(atom_types):
            fout.write("#declare size{:d} = {:.1f};\n".format(atom_type, atom_size))
            
            r, g, b = color_list[i%len(color_list)]
            fout.write( "#declare texture{:d} = texture {{ pigment {{ rgb <{:.1f}, {:.1f}, {:.1f}> }} }}\n".format(atom_type, r, g, b) )
            
            
        fout.write('#declare blob_strength = {:.2f};\n\n'.format(blob_strength))
            
        for molecule_id, molecule in molecules.items():
            fout.write( "#declare molecule{:d} = blob {{\n".format(molecule_id) )
            fout.write( "    threshold {:.2f}\n".format(threshold) )
            for atom in molecule['atoms']:
                fout.write('    sphere {{ <{:.6f}, {:.6f}, {:.6f}>, size{:d}, blob_strength texture {{ texture{:d} }} }} // atom {:d}, molecule {:d}\n'.format(atom['x'], atom['y'], atom['z'], atom['atom_type'], atom['atom_type'], atom['atom_id'], atom['molecule_id']) )
                                                        
            fout.write("}} // End molecule {:d}\n".format(molecule_id))
            
            
        fout.write("\n\n#declare molecules = union {")
        for molecule_id, molecule in molecules.items():
            fout.write("    object {{ molecule{:d} }}\n".format(molecule_id))
        fout.write("} // End molecules\n")
                       
    # Run POV-Ray
    p = subprocess.Popen(['povray', 'Library_Path=/usr/share/povray-3.7/include/ +W1600 +H1200 +A0.3 +UA +Irender_molecules.pov'], cwd=output_dir)
    p.wait()        
                           
def load_result(infile, input_dir='./'):

    data = np.load(input_dir+infile)
    h, w, d = data.shape
    print('Loaded {:d}D matrix of size ({:d}×{:d}×{:d})'.format(len(data.shape), h, w, d))
    return data


def plot2D(data, outfile='output.png', scale=[1,1], size=10.0, plot_buffers=[0.1,0.05,0.1,0.05]):
    
    #print('  data ({} points) from {:.2g} to {:.2g} ({:.2g} ± {:.2g})'.format(data.shape, np.min(data), np.max(data), np.average(data), np.std(data)))
    
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15

    
    fig = plt.figure(figsize=(size,size), facecolor='white')
    left_buf, right_buf, bottom_buf, top_buf = plot_buffers
    fig_width = 1.0-right_buf-left_buf
    fig_height = 1.0-top_buf-bottom_buf
    ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height] )
    
    h, w = data.shape
    extent = [ 0, w*scale[0], 0, h*scale[1] ]
    
    im = plt.imshow(data, extent=extent, cmap='bone')
    ax.set_xlabel('$x \, (\mathrm{pixels})$', size=20)
    ax.set_ylabel('$y \, (\mathrm{pixels})$', size=20)
    
    
    plt.savefig(outfile, dpi=200)
    plt.close()

def plotFFT(data, outfile='output.png', scale=[1,1], size=10.0, ztrim=[0.02, 0.03], plot_buffers=[0.1,0.05,0.1,0.05]):
    
    print('  data ({} points) from {:.2g} to {:.2g} ({:.2g} ± {:.2g})'.format(data.shape, np.min(data), np.max(data), np.average(data), np.std(data)))
    
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15

    
    fig = plt.figure(figsize=(size,size), facecolor='white')
    left_buf, right_buf, bottom_buf, top_buf = plot_buffers
    fig_width = 1.0-right_buf-left_buf
    fig_height = 1.0-top_buf-bottom_buf
    ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height] )
    
    # Compute vscale
    values = np.sort(data.flatten())
    zmin = values[ +int( len(values)*ztrim[0] ) ]
    zmax = values[ -int( len(values)*ztrim[1] ) ]
    
    h, w = data.shape
    origin = [int(w/2), int(h/2)]
    extent = [ -(w/2)*scale[0], +(w/2)*scale[0], -(h/2)*scale[1], +(h/2)*scale[1] ]
    
    im = plt.imshow(data, extent=extent, cmap='jet', vmin=zmin, vmax=zmax)
    ax.set_xlabel('$q_x \, (\mathrm{pixels}^{-1})$', size=20)
    ax.set_ylabel('$q_y \, (\mathrm{pixels}^{-1})$', size=20)
    
    
    plt.savefig(outfile, dpi=200)
    plt.close()


def circular_average(data, scale=[1,1,1], origin=None, bins_relative=3.0):
    
    h, w, d = data.shape
    y_scale, x_scale, z_scale = scale
    if origin is None:
        x0, y0, z0 = int(w/2), int(h/2), int(d/2)
    else:
        x0, y0, z0 = origin
        
    # Compute map of distances to the origin
    x = (np.arange(w) - x0)*x_scale
    y = (np.arange(h) - y0)*y_scale
    z = (np.arange(d) - y0)*z_scale
    X,Y,Z = np.meshgrid(x,y,z)
    R = np.sqrt(X**2 + Y**2 + X**2)

    # Compute histogram
    data = data.ravel()
    R = R.ravel()
    
    scale = (x_scale + y_scale + z_scale)/2.0
    r_range = [0, np.max(R)]
    num_bins = int( bins_relative * abs(r_range[1]-r_range[0])/scale )
    num_per_bin, rbins = np.histogram(R, bins=num_bins, range=r_range)
    idx = np.where(num_per_bin!=0) # Bins that actually have data
    
    r_vals, rbins = np.histogram( R, bins=num_bins, range=r_range, weights=R )
    r_vals = r_vals[idx]/num_per_bin[idx]
    I_vals, rbins = np.histogram( R, bins=num_bins, range=r_range, weights=data )
    I_vals = I_vals[idx]/num_per_bin[idx]

    return r_vals, I_vals


def plot1DFFT(x, y, outfile='output.png', x_expected=0, range_rel=0, fit_line=None, fit_line_e=None, fit_result=None, size=10.0, plot_buffers=[0.1,0.05,0.1,0.05]):
    
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15

    
    fig = plt.figure(figsize=(size,size*3/4), facecolor='white')
    left_buf, right_buf, bottom_buf, top_buf = plot_buffers
    fig_width = 1.0-right_buf-left_buf
    fig_height = 1.0-top_buf-bottom_buf
    ax = fig.add_axes( [left_buf, bottom_buf, fig_width, fig_height] )
    
    ax.plot(x, y, '-', color='k', linewidth=2.0)
    ax.set_xlabel('$q \, (\mathrm{pixels}^{-1})$', size=20)
    ax.set_ylabel('$I_{\mathrm{FFT}}(q) \, (\mathrm{a.u.})$', size=20)
    xi, xf, yi, yf = ax.axis()
    xi, xf, yi, yf = 0, np.max(x), 0, yf
    
    
    ax.axvline(x_expected, color='b', linewidth=2.0, alpha=0.5, dashes=[5,5])
    ax.axvspan(x_expected*(1-range_rel), x_expected*(1+range_rel), color='b', alpha=0.05)
    s = '$q_{{0,\mathrm{{expected}} }} = {:.2g} \, \mathrm{{pix}}^{{-1}}$'.format(x_expected)
    ax.text(x_expected, yf, s, size=20, color='b', verticalalignment='top', horizontalalignment='left')
    
    if fit_line is not None:
        ax.plot(fit_line[0], fit_line[1], '-', color='purple', linewidth=2.0)
    if fit_line_e is not None:
        ax.plot(fit_line_e[0], fit_line_e[1], '-', color='purple', linewidth=0.5)
    if fit_result is not None:
        x = fit_result.params['x_center'].value
        s = '$q_{{0,\mathrm{{fit}} }} = {:.2g} \, \mathrm{{pix}}^{{-1}}$'.format(x)
        ax.text(x_expected, yi, s, size=20, color='purple', verticalalignment='bottom', horizontalalignment='left')
        ax.axvline(x, color='purple', linewidth=1.0, alpha=0.5)


        els = [
            '$p = {:.2g} \, \mathrm{{a.u.}}$'.format(fit_result.params['prefactor'].value) ,
            '$q_0 = {:.3g} \, \mathrm{{pix}}^{{-1}}$'.format(x) ,
            '$\sigma_0 = {:.2g} \, \mathrm{{pix}}^{{-1}}$'.format(fit_result.params['sigma'].value) ,
            '$m = {:.3g} \, \mathrm{{a.u./pix^{{-1}} }}$'.format(fit_result.params['m'].value) ,
            '$b = {:.3g} \, \mathrm{{a.u.}}$'.format(fit_result.params['b'].value) ,
            ]
        s = '\n'.join(els)
        ax.text(xf, yf, s, size=20, color='purple', verticalalignment='top', horizontalalignment='right')
        
    
    
    ax.axis([xi,xf,yi,yf])
    
    plt.savefig(outfile, dpi=200)
    plt.close()
    
    
# Define the fit model
def model(v, x):
    '''Gaussian with linear background.'''
    m = v['prefactor']*np.exp( -np.square(x-v['x_center'])/(2*(v['sigma']**2)) ) + v['m']*x + v['b']
    return m

def func2minimize(params, x, data):
    v = params.valuesdict()
    m = model(v, x)
    
    return m - data    

def peak_fit(xs, ys, x_expected, range_rel=0.5, vary=True, eps=1e-10):
    
    # Trim the curve to extract just the part we want
    xi, xf = x_expected*(1-range_rel), x_expected*(1+range_rel)
    idx_start, idx_end = np.where( xs>xi )[0][0], np.where( xs>xf )[0][0]
    xc = xs[idx_start:idx_end]
    yc = ys[idx_start:idx_end]
    span = np.max(xc)-np.min(xc)
    
    # Estimate linear background using the two endpoints
    m = (yc[-1]-yc[0])/(xc[-1]-xc[0])
    b = yc[0] - m*xc[0]
    
    # Estimate prefactor
    idx = np.where( xc>=x_expected )[0][0]
    xpeak, ypeak = xc[idx], yc[idx]
    p = ypeak - (m*xpeak + b)
    
    # Estimate standard deviation
    yc_peakonly = yc - (m*xc + b)
    mean = np.average(xc, weights=np.clip(yc_peakonly, eps, ypeak))
    variance = np.average(np.square(xc - mean), weights=np.clip(yc_peakonly, eps, ypeak))
    std = np.sqrt(variance)

    # Start the fit model using our best estimates; restrict the parameter ranges to be 'reasonable'
    params = lmfit.Parameters()
    params.add('prefactor', value=p, min=0, max=np.max(yc)+eps, vary=vary)
    params.add('x_center', value=x_expected, min=np.min(xc), max=np.max(xc)+eps, vary=vary)
    params.add('sigma', value=std, min=span*0.00001, max=span*0.75, vary=vary)
    params.add('m', value=m, min=abs(m)*-5, max=abs(m)*+5+eps, vary=vary)
    params.add('b', value=b, min=min(0, b*5), max=max(np.max(ys)*2, abs(b)*5)+eps, vary=vary)
    
    lm_result = lmfit.minimize(func2minimize, params, args=(xc, yc))
    
    fit_x = np.linspace(np.min(xc), np.max(xc), num=max(500, len(xc)))
    fit_y = model(lm_result.params.valuesdict(), fit_x)
    fit_line = fit_x, fit_y
    
    xe = 0.5
    xi = np.min(xc)-xe*span
    xf = np.max(xc)+xe*span
    fit_x = np.linspace(xi, xf, num=2000)
    fit_y = model(lm_result.params.valuesdict(), fit_x)
    fit_line_extended = fit_x, fit_y
        
    return lm_result, fit_line, fit_line_extended    
    

# Structure Vector
########################################
def structure_vector(result, expected, range_rel=1.0, scale=1, adjust=None, plot=False, output_dir='./', output_name='result', output_condition=''):
    
    
    if plot:
        #for islice in range(result.shape[0]):
            #plot2D(result[islice], outfile='{}{}2D_{}_slice{}.png'.format(output_dir, output_name, output_condition, islice), scale=[scale, scale])
        plot2D(result[0], outfile='{}{}2D_{}.png'.format(output_dir, output_name, output_condition), scale=[scale, scale])

    
    # Compute FFT
    result_fft = np.fft.fftn(result)
    
    # Recenter FFT (by default the origin is in the 'corners' but we want the origin in the center of the matrix)
    hq, wq, dq = result_fft.shape
    result_fft = np.concatenate( (result_fft[int(hq/2):,:,:], result_fft[0:int(hq/2),:,:]), axis=0 )
    result_fft = np.concatenate( (result_fft[:,int(wq/2):,:], result_fft[:,0:int(wq/2),:]), axis=1 )
    result_fft = np.concatenate( (result_fft[:,:,int(dq/2):], result_fft[:,:,0:int(dq/2)]), axis=2 )
    origin = [int(wq/2), int(hq/2), int(dq/2)]
    qy_scale, qx_scale, qz_scale = 2*np.pi/(scale*hq), 2*np.pi/(scale*wq), 2*np.pi/(scale*dq)

    data = np.absolute(result_fft)

    if plot:
        plotFFT(data[origin[0]], outfile='{}{}FFT_{}.png'.format(output_dir, output_name, output_condition), scale=[qx_scale, qz_scale])
    
    
    # Compute 1D curve by doing a circular average (about the origin)
    qs, data1D = circular_average(data, scale=[qy_scale, qz_scale, qz_scale], origin=origin)
    
    # Eliminate the first point (which is absurdly high due to FFT artifacts)
    qs = qs[1:]
    data1D = data1D[1:]
    
    # Optionally adjust the curve to improve data extraction
    if adjust is not None:
        data1D *= np.power(qs, adjust)
    
    ## Modify the code to use the max as the peak estimation
    idx = np.where( data1D==np.max(data1D) )[0][0]
    print(idx)
    print(qs[idx])
    expected = qs[idx] 
    
    # Fit the 1D curve to a Gaussian
    #lm_result, fit_line, fit_line_extended = peak_fit(qs, data1D, x_expected=expected, range_rel=range_rel)

    p = 0#lm_result.params['prefactor'].value # Peak height (prefactor)
    q = 0#lm_result.params['x_center'].value # Peak position (center) ==> use this
    sigma = 0#lm_result.params['sigma'].value # Peak width (stdev) ==> use this
    I = 0#p*sigma*np.sqrt(2*np.pi) # Integrated peak area
    m = 0#lm_result.params['m'].value # Baseline slope
    b = 0#lm_result.params['b'].value # Baseline intercept
    
    
    if plot:
        plot1DFFT(qs, data1D, outfile='{}{}1DFFT_{}.png'.format(output_dir, output_name, output_condition), x_expected=expected, range_rel=range_rel, fit_line=fit_line, fit_line_e=fit_line_extended, fit_result=lm_result)
    

    return p, q, sigma, I, m, b, qs, data1D
    
    
def getStrVector(infile,render=False):
    
    # Example: A single input file.
    ########################################
    
    #infile = '28_step_2m.2000000'     
    atoms, molecules = parse_LAMMPS(infile)
    box = parse_LAMMPS_box(infile)
    
    if render:
        # Plot the 3D configuration of the LAMMPS file using POV-Ray
        pov_atoms(atoms, output_dir='./POV/')
        pov_molecules(molecules, output_dir='./POV/')

    # Generate a coarse-grained grid representation
    N = 30
    grid = generate_grid(atoms, atom_type=1, N=N, box=box)

    #outfile = './results/{}_3d{:d}x{:d}x{:d}.npy'.format(infile, N, N, N)
    #outfile = os.path.abspath(outfile)
    #try:
    #    os.stat(os.path.dirname(outfile))
    #except:
    #    os.makedirs(os.path.dirname(outfile))   
    #np.save(outfile, grid)

    # Load result
    #infile = outfile
    #infile = './3d_results/{}_3d_100x100x100.npy'.format(infile)
    #result = load_result(infile)
    result = grid

    # Define expectations for simulations results    
    scale = 1 # Conversion of simulation units into realspace units
    d0_expected = 7.5
    q0_expected = 2*np.pi/d0_expected
    #print('Expect peak at q = {:.3g} pix^-1'.format(q0_expected))

    # Compute the structure vector
    vector = structure_vector(result, q0_expected, scale=scale)
    
    #newVector = vector[1,2]
                       
    return vector



