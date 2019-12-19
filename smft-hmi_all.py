import scipy.signal
import astropy.io.fits as fits
import astropy.time
import sunpy
import sunpy.coordinates
import matplotlib.pyplot as plt
import os
import numpy as np
import skimage.transform
import skimage.filters
import scipy.interpolate
import time

smft_directory = 'D:\\data\\smft-hmi\\smft\\'

hmi_directory = 'D:\\data\\smft-hmi\\hmi\\'

s_flist = os.listdir(smft_directory)
h_flist = os.listdir(hmi_directory)

s_files_list = list(smft_directory + s_flist[i] for i in range(len(s_flist)))
h_files_list = list(hmi_directory + h_flist[i] for i in range(len(h_flist)))


#list of regions to use 
#reg_list = [0, 1, 3, 6, 8]
reg_list = [2]

poly_coeffs = np.array([-1.66244474e-11,  -1.61810733e-09, 7.03341500e-05, 7.74491899e-04])

start = time.time()

nodes = 5

#threshold_list_s = np.linspace(0.35, 0.5, nodes)
#threshold_list_w = np.linspace(0.55, 0.7, nodes)
threshold_matrix = np.zeros((nodes, nodes))

threshold_list_s = [0.4]
threshold_list_w = [0.70]

scale_list = []
time_list = []


#arrays for images
smft_pic_arr = []
hmi_pic_arr = []
fixed_pic_arr = []


silent = []

thres_pics = []


for ja, k in enumerate(reg_list):
    
    print('pic', time.time() - start)
    #files open
    smft_file = fits.open(s_files_list[k])
    hmi_file = fits.open(h_files_list[k])
    hmi_file.verify('silentfix')        
    hmi_pic = hmi_file[1].data
    smft_pic = (smft_file[0].data[1] - smft_file[0].data[0])/((smft_file[0].data[1] + smft_file[0].data[0]))
    cont_pic = (smft_file[0].data[1] + smft_file[0].data[0])
    if (k > 5): smft_pic = smft_pic*(-1)
    
    #crop blur rotate
    smft_pic = smft_pic[100:800]
    cont_pic = cont_pic[100:800]
    
    hmi_pic = skimage.filters.gaussian(hmi_pic, sigma = 3)
    
    tms = astropy.time.Time(smft_file[0].header.get('T_START'), format='iso', scale='tai')
    p_angle = sunpy.coordinates.get_sun_P(tms)
    hmi_pic = skimage.transform.rotate(hmi_pic, 180 - p_angle.degree)
    hmi_pic = np.flip(hmi_pic, 0)
    
    #find must matching rescaling factor
    corr_list = []

    #!!!takes too much time
    for m in range(10):
        hmi_test_pic = skimage.transform.rescale(hmi_pic, 1/0.242*0.504372/(1.2 + m*0.001))
        corr = scipy.signal.correlate(smft_pic, hmi_test_pic)
        corr_coords = np.unravel_index(np.argmax(corr), corr.shape)
        corr_list.append(np.max(corr))


    #apply the best rescaling factor
    hmi_pic = skimage.transform.rescale(hmi_pic, 1/0.242*0.504372/(1.2 + np.argmax(corr_list)*0.001))
    
    
    #save information about rescale and time
    scale_list.append(0.242*(1.2 + np.argmax(corr_list)*0.001))
    time_list.append(astropy.time.Time(smft_file[0].header.get('T_START'), format='iso', scale='tai'))

    #find must matching postion and crop
    corr = scipy.signal.correlate(hmi_pic, smft_pic)
    corr_coords = np.unravel_index(np.argmax(corr), corr.shape)

    h_pic = hmi_pic[max(0, corr_coords[0] - smft_pic.shape[0]): min(hmi_pic.shape[0], corr_coords[0]), max(0, corr_coords[1] - smft_pic.shape[1]): min(hmi_pic.shape[1], corr_coords[1])]
    s_pic = smft_pic[max(0 + (smft_pic.shape[0] - corr_coords[0]), corr_coords[0] - smft_pic.shape[0] + (smft_pic.shape[0] - corr_coords[0])): min(hmi_pic.shape[0] + (smft_pic.shape[0] - corr_coords[0]) , corr_coords[0] + (smft_pic.shape[0] - corr_coords[0])), max(0 + (smft_pic.shape[1] - corr_coords[1]), corr_coords[1] - smft_pic.shape[1] + (smft_pic.shape[1] - corr_coords[1])): min(hmi_pic.shape[1] + (smft_pic.shape[1] - corr_coords[1]) , corr_coords[1] + (smft_pic.shape[1] - corr_coords[1]))]

    cont_pic = cont_pic[max(0 + (smft_pic.shape[0] - corr_coords[0]), corr_coords[0] - smft_pic.shape[0] + (smft_pic.shape[0] - corr_coords[0])): min(hmi_pic.shape[0] + (smft_pic.shape[0] - corr_coords[0]) , corr_coords[0] + (smft_pic.shape[0] - corr_coords[0])), max(0 + (smft_pic.shape[1] - corr_coords[1]), corr_coords[1] - smft_pic.shape[1] + (smft_pic.shape[1] - corr_coords[1])): min(hmi_pic.shape[1] + (smft_pic.shape[1] - corr_coords[1]) , corr_coords[1] + (smft_pic.shape[1] - corr_coords[1]))]
    
    print('pic ready', time.time() - start)
    
    #quiet sun intensity (mode of continuum intensity distribution)
    
    silent.append( int(scipy.stats.mode(cont_pic, axis = None)[0] ) )
    
    
    max_point = 0.06344
    t_point = 1150
    z_point = 1500
    
    #fixing image
    
    #scale coeff (calibration coeffitient for smft 
    #                   signal)
    aaa = np.linspace(-2000, 2000, 100)
    fit = np.polynomial.polynomial.polyval(aaa, np.flip(poly_coeffs, 0) )
    #using quantiles instead of max/min to make insensitive for outliners
    scale_coeff = 1.01*np.max( ( -1*np.quantile(s_pic, 0.01), (np.quantile(s_pic, 0.99)) ) )/np.max(fit)
    top = np.max(fit) #the polynomial is close to symmetric
    
    I_top = silent[-1]
    
    smft_fixed_pic = np.empty(s_pic.shape)
    over_map = np.zeros(s_pic.shape)
    masked = np.ones(s_pic.shape)
    
    #thresholds
    cont_thr_w = 0.65
    cont_thr_s = 0.4
    
    print('poly', time.time() - start)
    
    #weak and strong field division
    
    #find threshold values with smallest gaps
      
    find_coeffs = poly_coeffs*scale_coeff
               
    smft_fixed_pic = np.empty(s_pic.shape)
    masked = np.ones(s_pic.shape)
    fix_pic_h = np.zeros(s_pic.shape)  
    
    for i in range(s_pic.shape[0]):
        for j in range(s_pic.shape[1]):
            signal = s_pic[i][j]
            find_coeffs = poly_coeffs*scale_coeff
            #avoiding to signal run out of polynomial
            if np.abs(signal) > top*scale_coeff:
                signal = 0.99*top*scale_coeff*np.sign(signal)
            find_coeffs[3] -= signal
            roots = np.sort(np.roots(find_coeffs))
            
            if cont_pic[i][j] > cont_thr_w*I_top:
                smft_fixed_pic[i][j] = np.real(roots[1])
            elif cont_pic[i][j] < cont_thr_s*I_top:
                smft_fixed_pic[i][j] = np.real(roots[1 + int(np.sign(signal))])
            else: 
                smft_fixed_pic[i][j] = 0
                masked[i][j] = 0  
                
    print('divided', time.time() - start)
    
    
    #patching    
    
    #!!!changed - 2d-interpolate patching
    ddd = scipy.interpolate.griddata(np.nonzero(masked), 
                                     np.take(smft_fixed_pic.flatten(), np.flatnonzero(masked)), 
                                     np.nonzero(1 - masked), method = 'linear')
    fix_pic_p = np.zeros(masked.shape).flatten()
    np.put(fix_pic_p, np.flatnonzero(masked - 1), ddd)
    fix_pic_p = np.reshape(fix_pic_p, masked.shape)
    fixed_pic = smft_fixed_pic + fix_pic_p

    print('patched', time.time() - start)

    #bluring
    fixed_pic = skimage.filters.gaussian(fixed_pic, sigma = 3)
    
    
    #saving magnetograms
    smft_pic_arr.append(s_pic)
    hmi_pic_arr.append(h_pic)
    fixed_pic_arr.append(fixed_pic)
                
    print(k, time.time() - start)


#sets for scatter plots
points = np.empty((2, 0))
points_fixed = np.empty((2, 0))
for list_counter in range(len(smft_pic_arr)):
    aaa = hmi_pic_arr[list_counter].flatten()
    bbb = smft_pic_arr[list_counter].flatten()
    points = np.concatenate( (points, np.vstack((aaa, bbb))), axis = 1)
    bbb = fixed_pic_arr[list_counter].flatten()
    points_fixed = np.concatenate( (points_fixed, np.vstack((aaa, bbb))), axis = 1)
    
    


        
aaa = np.linspace(-2000, 2000, 100)

fit = np.polynomial.polynomial.polyval(aaa, np.flip(np.polyfit(points[0], points[1], 3), 0) )
#fit_spline = scipy.interpolate.LSQBivariateSpline(points[0], points[1], [-1100, 1100])

cmap = plt.get_cmap('gray')


#show common scatterplots for start and fixed magnetograms
def show_scatters():
    plt.plot(points[0], points[1], ',')
    plt.plot(aaa, fit)
    plt.show()
    
    plt.plot(points_fixed[0], points_fixed[1], ',')
    plt.plot([-2000, 2000], [-2000, 2000])
    plt.show()


def show_slice(y, x_start = 0, x_finish = -1):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 2000)
    plt.subplot(2,2,1)
    plt.plot(h_pic[y][x_start:x_finish])
    plt.subplot(2,2,2)
    plt.plot(s_pic[y][x_start:x_finish]/7.03341500e-05)
    plt.subplot(2,2,3)
    plt.plot(smft_fixed_pic[y][x_start:x_finish])
    plt.subplot(2,2,4)
    plt.plot(fixed_pic[y][x_start:x_finish])
    plt.show()



def save():
    hdul = fits.HDUList([fits.PrimaryHDU(h_pic)])
    hdul.writeto('hmi-pic.fits')
    
    hdul = fits.HDUList([fits.PrimaryHDU(s_pic)])
    hdul.writeto('smft-pic.fits')
    
    hdul = fits.HDUList([fits.PrimaryHDU(fixed_pic)])
    hdul.writeto('fixed-pic.fits')
    
    fig, (ax) = plt.subplots(nrows=1)
    im = ax.pcolormesh(h_pic, cmap=cmap, vmin = -2000, vmax = 2000)
    ax.tick_params(bottom = 0, left = 0, labelbottom = 0, labelleft = 0)
    fig.set_size_inches(20, 20/len(s_pic[0])*len(s_pic))
    fig.savefig('hmi-pic.png') 
    plt.close()
    
    fig, (ax) = plt.subplots(nrows=1)
    im = ax.pcolormesh(s_pic, cmap=cmap)
    ax.tick_params(bottom = 0, left = 0, labelbottom = 0, labelleft = 0)
    fig.set_size_inches(20, 20/len(s_pic[0])*len(s_pic))
    fig.savefig('smft-pic.png') 
    plt.close()
    
    fig, (ax) = plt.subplots(nrows=1)
    im = ax.pcolormesh(fixed_pic, cmap=cmap, vmin = -2000, vmax = 2000)
    ax.tick_params(bottom = 0, left = 0, labelbottom = 0, labelleft = 0)
    fig.set_size_inches(20, 20/len(s_pic[0])*len(s_pic))
    fig.savefig('fixed-pic.png') 
    plt.close()
