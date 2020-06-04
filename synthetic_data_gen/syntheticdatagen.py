import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 
import pymei
import random
import scipy as sp
import skimage
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.signal import convolve2d
from skimage import data
from skimage import draw, filters, morphology
from skimage.filters import gaussian
from skimage.transform import warp_coords


def generate_parameters(index=None, data_shape=(512,512), padding=100):
    # Geral
    parameters = {}
    index = index if index is not None else np.random.randint(1000)

    parameters['name'] = '{:04d}.tiff'.format(index+1)
    parameters['data_shape'] = data_shape
    parameters['padding'] = padding

    # Reflectivity
    parameters['reflectivity'] = list(np.random.uniform(-1, 1, size=data_shape[1]+2*padding))

    # Folding
    # frequência - [0.001 a 0.1]
    f_freq = 10**np.random.uniform(-3, -1)
    parameters['f_freq'] = f_freq
    # amplitude
    parameters['f_ampl'] = np.random.uniform(0, 1/f_freq)
    # distorção vertical
    parameters['f_vert'] = np.random.uniform(-0.0025 / f_freq, 0.0025 / f_freq)
    # aceleração
    parameters['f_acc'] = np.random.uniform(-f_freq / 1000, f_freq / 1000)
    # fase (0 a 2pi)
    parameters['f_phase'] = np.random.uniform(0, 2*np.pi)
    # inclinação
    parameters['f_slope'] = np.random.normal(scale=0.25)
    # Perturbação
    p_freq = 10**np.random.uniform(-3, -1)
    parameters['p_freq'] = p_freq
    parameters['p_ampl'] = np.random.uniform(0, 1/p_freq)
    parameters['p_vert'] = np.random.uniform(-0.0025 / p_freq, 0.0025 / p_freq)
    parameters['p_acc'] = np.random.uniform(-p_freq / 1000, p_freq / 1000)
    parameters['p_phase'] = np.random.uniform(0, 2*np.pi)
    # Textura
    # amplitude da textura (0 a 1)
    parameters['t_ampl'] = np.random.uniform(0, 1)
    # frequência da textura (0.1 a 1)
    parameters['t_freq'] = 10**np.random.uniform(-1, 0)
        
    # Fault
    pad_fault_center = 50

    x_low_limit = padding + pad_fault_center
    x_up_limit = data_shape[1] - x_low_limit
    y_low_limit = padding + pad_fault_center
    y_up_limit = data_shape[0] - x_low_limit

    num_faults = np.random.randint(0, 4)

    parameters['fault_centers'] = [(np.random.randint(y_low_limit, y_up_limit),
                    np.random.randint(x_low_limit, x_up_limit))
                    for _ in range(num_faults)]
    parameters['fault_angles'] = [(5 + np.random.rand()*80)*np.random.choice([-1,1])
                    for _ in range(num_faults)]
    parameters['fault_throws'] = [(5+ np.random.rand()*35)*np.random.choice([-1,1])
                    for _ in range(num_faults)]
            
    # Convolution
    from scipy.special import lambertw

    ricker_centralfreq = np.e**np.random.uniform(np.log(20), np.log(50))
    ricker_dt = 0.004
    parameters['ricker_centralfreq'] = ricker_centralfreq
    parameters['ricker_tapersize'] = np.random.randint(2, 8)
    parameters['ricker_dt'] = ricker_dt

    # computa tamanho da janela necessário para que a calda
    # da ricker chegue até valor 'eps'.
    eps = 1e-5
    inv_ricker = np.sqrt(np.real(lambertw((1+eps)/2)/np.pi))*4
    t = inv_ricker / ricker_centralfreq

    parameters['ricker_windowsize'] = int(2*t/ricker_dt)

    return parameters


def makeReflectivityPanel(params=None):
    """
    Generates a set of flat horizontal lines with random amplitudes
    
    Parameters
    ----------
    shape : tuple
        Data shape
    Returns
    -------
    output : array
        First step of data generation, translated (output.T is in the right orientation)
    """
    
    params = params if params is not None else generate_parameters()
    
    shape = params['data_shape']
    padding = params['padding']
    shape = [s+2*padding for s in shape]
    reflectivities = np.array(params['reflectivity'], dtype=np.float32)
    reflectivities = np.tile(reflectivities,(shape[0],1))
    reflectivities = np.expand_dims(reflectivities, axis=2)
    
    return reflectivities


def removePadding(seismic_data, params=None):
    params = params if params is not None else generate_parameters()

    shape = params['data_shape']
    padding = params['padding']
    x_begin = padding
    x_end = padding+shape[1]
    y_begin = padding
    y_end = padding+shape[0]

    output = seismic_data[y_begin:y_end,x_begin:x_end,...]

    return output


def applyTransform(seismic_data, transform_func, seismic_channel=0):
    def shift_coords(xy):
        x = xy[:,1]
        y = xy[:,0]
    
        eps = 1e-5

        #caso geral para fold
        z = y - transform_func(x,y)
        z[ np.abs(z)< eps ] = 0 
        
        xy[:,0] = z
        return xy

    coords = skimage.transform.warp_coords(shift_coords, seismic_data.shape)
    output = sp.ndimage.map_coordinates(input=seismic_data,
                                        coordinates=coords)

    return output
        




def fault( mat, fault_center, theta, throw, b_getMask = False, b_verbose=False):
    """
    Slices a 2D matrix over point fault_center with angle theta.
    Be advised, NAN is used here as a mark symbol, not as the usual way.
    If there are NANs in the mat, they will be "filtered out", and that 
    can mask other bugs inside the code. 
    
    Parameters
    ----------
    mat : 2D square array
        Matrix to be sliced
    fault_center : int, 2-tuple
        point that slice will go over 
    theta : float
        angle to slice, in degrees
    throw : float
        how much one side ("the faulted region") will slide over 
    b_verbose : boolean
        To verbose or not to verbose, that is the question
   
    Returns
    -------
    output : 2D array
        "faulted" panel. 
    """
    
    if( mat.shape[0] != mat.shape[1] ):
        print("[ERROR] Matrix should be square: {} x {}".format(mat.shape[0], mat.shape[1]))
        return None

    shape_max = mat.shape[0]
        
    def mat2D_shift(mat, int_dx, int_dy, fill_value = np.nan, b_verbose=False):
        """
        Just shift 2D matrices along its axis, filling with 'nan'
        """
        aux = np.full(mat.shape, fill_value = fill_value)
        

        if( int_dx > 0 ):
            if( b_verbose ):
                print("aux[{}:, :] = mat[:{}, :]".format(int_dx, shape_max-int_dx))
                print(" {} = {}".format(aux[int_dx:, :].shape, mat[:shape_max-int_dx, :].shape))
            aux[int_dx:, :] = mat[:shape_max-int_dx, :]

        else:
            if( b_verbose ):
                print("aux[:{}, :] = mat[{}:, :]".format(shape_max+int_dx, -int_dx))
                print(" {} = {}".format(aux[:shape_max+int_dx, :].shape, mat[-int_dx:, :].shape))
            aux[:shape_max+int_dx, :] = mat[-int_dx:, :]

        aux2 = np.full(mat.shape, fill_value = fill_value)

        if( int_dy > 0 ):
            if( b_verbose ):
                print("aux2[:, {}:] = aux[:, :{}]".format( int_dy, shape_max-int_dy ))
                print(" {} = {}".format(aux2[:, int_dy:].shape, aux[:, :shape_max-int_dy].shape))
            aux2[:, int_dy:] = aux[:, :shape_max-int_dy]

        else:
            if( b_verbose ):
                print("aux2[:, :{}] = aux[:, {}:]".format( shape_max+int_dy , -int_dy))
                print(" {} = {}".format(aux2[:, :shape_max+int_dy].shape, aux[:, -int_dy:].shape))
            aux2[:, :shape_max+int_dy] = aux[:, -int_dy:]

        return( aux2 )

    
    theta_rad = np.deg2rad(theta)
    
    a = np.tan(theta_rad)
    b = fault_center[1] - a*fault_center[0]

    function = lambda x: a*x +b

    if( theta_rad<0 ):
        dx = -throw*np.sin(theta_rad)
        dy = -throw*np.cos(theta_rad)

        x = np.arange(shape_max)
        y = np.arange(shape_max)
        xx, yy = np.meshgrid(x, y)

        faulted_region_mask = np.where(function(xx) > yy, 0, 1)
    else:
        dx = throw*np.sin(theta_rad)
        dy = throw*np.cos(theta_rad)

        x = np.arange(shape_max)
        y = np.arange(shape_max)
        xx, yy = np.meshgrid(x, y)

        faulted_region_mask = np.where(function(xx) > yy, 1, 0)
        

    #NaN-ing
    faulted_region_mask = np.where(faulted_region_mask==1, mat, np.nan)
    
    int_dx = int(np.round(dx))
    int_dy = int(np.round(dy))
    
    if(b_verbose):
        print("(dx,dy) = ({},{})".format(dx, dy))
        print("(int_dx, int_dy)= ({},{})".format(int_dx, int_dy))
        print("original throw= {}".format(throw))
        print("real throw= {}".format(np.sqrt(int_dx**2 + int_dy**2)))
    
    faulted_region_mask = mat2D_shift( faulted_region_mask, int_dx=int_dx, int_dy=int_dy, fill_value = np.nan, b_verbose=b_verbose)
    
    #de-NaN-ing
    output = np.where(np.isnan(faulted_region_mask), mat, faulted_region_mask)
    
    return output


def convolve_wavelet(panel, option='ricker', frequency=25.0, dt=0.004 , taper_size=4, window_size=50):
        
    if option == 'ricker':
        wavelet = get_ricker_wavelet_2D(frequency=frequency, dt=dt, taper_size=taper_size, window_size=window_size)
    if option == 'psf':
        wavelet = get_random_psf()

    new_panel = convolve2d(panel, wavelet, mode='same')
    
    return new_panel.astype('float32')

def load_pickle(pickle_file):
    """Carrega um conjunto de dados preservado em um pickle
    
    Args:
        pickle_file: nome do arquivo que contem os dados.

    Retorno:
        Dicionario com pares { nome do dado : conteudo }
    """
    with open(pickle_file, 'rb') as f:
        data_dic = pickle.load(f)
    #print('Dados carregados de', pickle_file)
    #for key in data_dic.keys():
        #print(" -", key)
    return data_dic

def get_random_psf(return_infos=False):
    # Seleciona um PSF para convolver com o dado de estruturas
    # A ultima PSF nao pode ser inserida pois esta com a polaridade invertida e a Ricker nao.
    psf_file = './janelas/jequitinhonha.npy'

#     data_dic = load_pickle(psf_file)
#     scatters = data_dic['scatters']
#     frequencies = data_dic['frequencies']
#     scatter_widths = data_dic['scatter_widths']
    scatters = np.load(psf_file)
    
    #psf_index = random.randint(0, len(scatters) - 2)
    psf_index = 3
    
    psf = scatters[psf_index]
#     psf_freq = frequencies[psf_index]
#     psf_width = scatter_widths[psf_index]

#     title = 'freq:%.2f, width:%d' % (psf_freq, psf_width)
    
    if return_infos:
        return psf, psf_freq, psf_width
    else:
        return psf
    
def get_ricker_wavelet_2D(frequency, dt, taper_size, window_size):
    """
    Computa a Ricker Wavelet para os tempos e frequencia da entrada.
    
    Args:
        frequency: frequencia da wavelet.
        size: tamanho do vetor
        dt: taxa de amostragem temporal (em segundos).
        
    Retorno:
        wavelet: vetor de amplitudes da wavelet para cada ponto
    """
    # Caso tamanho seja par, nao existe amostra central
    offset = (window_size-1)/2 + ((window_size-1)%2)/2.0
    offset *= dt
    times = np.arange(window_size, dtype=np.float32) * dt - offset
    cte = -(np.pi**2)*(frequency**2)
    exp = np.exp(cte*(times**2))
    wavelet = exp + 2*cte*(times**2)*exp
    
    window_size = int(window_size)
    wavelet = wavelet.reshape((1,-1)).T
    wavelet_2D = np.empty((window_size, taper_size*2+1), dtype=np.float32)
    wavelet_2D[:,:] = wavelet
    wavelet_2D = apply_taper(wavelet_2D, taper_size)
    
    return wavelet_2D

def apply_taper(data, taper_size):
    """Afina as bordas do dado (tapering) usando a funcao cosseno. As dimensoes que serao afinadas devem ter tamanho
    pelo menos taper_size
    
    Args:
        data: numpy.array de dimensoes 1D, 2D ou 3D. Se 1D ou 2D, afunila as bordas. Se 3D, supoe que eh uma sequencia
            de janelas 2D e afina as bordas de cada janela.
        taper_size: tamanho da borda a ser afinada. Deve ser um inteiro positivo.
        
    Retorno:
        data: dado com as bordas afinadas.
    """
    # Cria taper com tamanho taper_size no intervalo (1.0, 0.0]
    t = (np.pi / (taper_size)) * (np.arange(taper_size).astype(np.float32) + 1.0)
    taper = (np.cos(t) / 2.0) + 0.5

    # Caso 1D
    if data.ndim == 1:
        data_size = data.shape[0]
        # Sanity check
        if data_size < taper_size:
            print("Erro: dado menor que taper_size. Nenhum taper aplicado.")
            return data
        # Aplica taper na borda esquerda
        left_taper = np.flip(taper, axis=None)
        data[0:taper_size] *= left_taper
        # Aplica taper na borda direita
        right_taper = taper
        data[(data_size-taper_size):] *= right_taper
        
    # Caso 2D
    elif data.ndim == 2:
        data_height = data.shape[0]
        data_width = data.shape[1]
        # Sanity check
        if (data_height < taper_size) or (data_width < taper_size):
            print("Erro: dado menor que taper_size. Nenhum taper aplicado.")
            return data
        # Cria taper de cada borda 
        taper = taper.reshape((1,taper_size))
        right_taper = taper
        left_taper = np.flip(taper, axis=1)
        top_taper = left_taper.T
        bottom_taper = right_taper.T
        # Taper superior
        data[0:taper_size, :] *= top_taper
        # Taper inferior
        data[(data_height - taper_size):, :] *= bottom_taper
        # Taper esquerdo
        data[:, 0:taper_size] *= left_taper
        # Taper direito
        data[:, (data_width - taper_size):] *= right_taper

    # Caso 3D
    elif data.ndim == 3:
        data_depth = data.shape[0]
        data_height = data.shape[1]
        data_width = data.shape[2]
        # Sanity check
        if (data_height < taper_size) or (data_width < taper_size):
            print("Erro: dado menor que taper_size. Nenhum taper aplicado.")
            return data
        # Cria taper de cada borda 
        taper = taper.reshape((1,taper_size))
        right_taper = taper
        left_taper = np.flip(taper)
        top_taper = left_taper.T
        bottom_taper = right_taper.T
        # Aplica taper em cada janela do dado
        for i in range(data_depth):
            # Taper superior
            data[i,0:taper_size, :] *= top_taper
            # Taper inferior
            data[i,(data_height - taper_size):, :] *= bottom_taper
            # Taper esquerdo
            data[i,:, 0:taper_size] *= left_taper
            # Taper direito
            data[i,:, (data_width - taper_size):] *= right_taper
            
    # Caso >3D
    else:
        print("Erro: dado deve ter 1, 2 ou 3 dimensoes. Nenhum taper aplicado.")
        return data
    return data

def add_noise(diffractions, filter_sigma, signal_to_noise):
    noise_img = np.random.normal(loc=0.0, size=diffractions.shape)

    filtered_img = gaussian_filter(noise_img, sigma=filter_sigma) 
    max_noise = diffractions.max()*signal_to_noise
    filtered_img = (filtered_img/np.abs(filtered_img).max())*max_noise
    filtered_diff = diffractions + filtered_img
    
    return filtered_diff.astype('float32')

def pickle_data(pickle_file, data_list, name_list):
    """Salva (pickle) um conjunto de dados para uso futuro.
    Args:
        pickle_file: nome do arquivo onde o dado sera salvo
        data_list: lista com os dados a serem salvos
        name_list: lista com o nome de cada dado
    
    Retorno:
        Nenhum, somente salva o dado em disco.

    Obs: data_list e name_list devem ter o mesmo tamanho.
    """
    if (len(data_list) != len(name_list)):
        print("Erro: lista de dados e lista de nomes devem ter o mesmo tamanho. Nada foi salvo.")
        return
    # Cria diretorio para o arquivo, se necessarios
    pickle_dir = os.path.dirname(pickle_file)
    if not os.path.exists(pickle_dir): os.mkdir(pickle_dir)
    # Constroi dicionario que associa cada nome a um dado:
    save_dic = {}
    for data, name in zip(data_list, name_list):
        # Sanity check: verifica se nao tem nome repetido na lista de nomes
        if save_dic.has_key(name):
            print("Erro - nome duplicado:", name)
            print("Nada foi salvo.")
            return
        save_dic[name] = data
    # Salva o conjunto de dados
    try:
        f = open(pickle_file, 'wb')
        pickle.dump(save_dic, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Dado persistido em", pickle_file)
    except Exception as e:
        print('Erro: nao foi possivel salvar o dado em', pickle_file, ':', e)
        raise

def save_SU(save_file, model, dt=4000, scalco=1, dx=5):
    
    assert model.dtype == 'float32', 'Erro: Tipo do dado eh invalido'
    
    #with pymei.SU(save_file, mode='wb') as su_writer:
    su_writer = pymei.SU(save_file, mode='wb')
    ns = model.shape[0]

    for i in range(ns):
        sy = i*dx
        gy = i*dx
        cdp = i
               
        header = pymei.SUTraceHeader(ns=ns, cdp=cdp, dt=dt, sy=sy, gy=gy, scalco=scalco, counit=1, tracl=(i))
        data = model[:,i]
        su_writer.writeTrace(pymei.Trace(header, data, 0)) 
    su_writer.stream.close()

    print("Dados escritos em", save_file)


def normalize(data, eps=1e-10):
    absmax = np.abs(data).max()
    if absmax >= eps:
        data /= absmax
    return data


def applyFolding(seismic_data, params=None):
    """
    Folds a panel with a senoidal function (might be generalized with other functions in future versions)
    
    Parameters
    ----------
    seismic_data : array
        The Seismic Data
        
    Returns
    -------
    output : array
        Seismic Data with Folding
    """
    params = params if params is not None else generate_parameters()
    
    # Curva (folding)
    f_freq = params['f_freq'] 
    f_ampl = params['f_ampl'] 
    f_vert = params['f_vert'] 
    f_acc = params['f_acc'] 
    f_phase = params['f_phase'] 
    f_slope = params['f_slope'] 
    
    # Perturbação
    p_freq = params['p_freq'] 
    p_ampl = params['p_ampl'] 
    p_vert = params['p_vert'] 
    p_acc = params['p_acc'] 
    p_phase = params['p_phase'] 
    
    # Textura
    t_ampl = params['t_ampl'] 
    t_freq = params['t_freq'] 
    
    def _fold(x,y):
        folding = (f_ampl + f_vert*y)*np.sin((f_freq + f_acc*x)*x + f_phase) + f_slope*x
        perturb = (p_ampl + p_vert*y)*np.sin((p_freq+p_acc*x)*x + p_phase)
        texture = t_ampl * np.sin(t_freq*x) 
        
        return folding + perturb + texture
 
    output = applyTransform(seismic_data=seismic_data, transform_func=_fold)
    output = np.transpose(output, axes=(1,0,2))

    return output


# OBSOLETE
def applyShearing(seismic_data):
    """
    
    Parameters
    ----------
    seismic_data : array
        The Seismic Data
        
    Returns
    -------
    output : array
        Seismic Data with Shearing
    """
    
    #shear parameters
    a2 = np.random.uniform(0.05, 0.1)
    slope = 1e-1 # 1e-5 a 5e-4
    alpha = np.random.uniform(1e-5, 3e-4) # 1e-5 a 5e-4
    
    #f = lambda x,y: (a1 + a2 * x) + np.random.uniform(-abs(a1),abs(a1), size=x.size)
    def _shear(x,y):
        lin = alpha * x**2
        max_displace = lin.max() if slope >= 0 else lin.min()
        bias = -max_displace / 2
        return lin + bias

    output = applyTransform(seismic_data=seismic_data, transform_func=_shear)
    
    return output


def applyFault(seismic_data,
               params=None,
               seismic_channel=0,
               b_verbose=False):
    """
    Applies faulting to a data and puts fault mask in a channel
    
    Parameters
    ----------
    seismic_data : array
        The Seismic Data
    b_verbose : boolean
        To verbose or not to verbose, that is the question
        
    Returns
    -------
    output : array
        Seismic Data with Fault
    """
    params = params if params is not None else generate_parameters()
    
    centers = params['fault_centers'] 
    angles = params['fault_angles']
    throws = params['fault_throws']
    #thetas = (5+ np.random.rand()*80)*np.random.choice([-1,1])  #gives numbers in [-85,-5]U[5,85] interval 
    #throws = (5+ np.random.rand()*35)*np.random.choice([-1,1]) #gives numbers in [-40,-5]U[5,40] interval 

    output = seismic_data.copy()
        
    for c, ang, thrw in zip(centers, angles, throws):
        aux_seismic = fault(mat=output[...,seismic_channel],
                           fault_center=c,
                           theta=ang,
                           throw=thrw,
                           b_getMask=True,
                           b_verbose=b_verbose)
        output[..., seismic_channel] = aux_seismic

    return output


def applyConvolution(seismic_data,
                     params=None,
                     seismic_channel = 0,
                     b_verbose=False):
    """
    Applies convolution to a data 
    
    Parameters
    ----------
    seismic_data : array
        The Seismic Data
    seismic_channel : int, default value = 0 
        main channel
    fault_channel : int, default value = 3
        channel to put salt mask
    affected_channels : int, array, optional
        other channels to apply faulting operator
    output : ndarray, optional
        The results will be placed in this array. 
    b_verbose : boolean
        To verbose or not to verbose, that is the question
        
    Returns
    -------
    output : array
        Seismic Data with Convolution
    """
    params = params if params is not None else generate_parameters()
    
    #ricker_centralfreq = np.e**np.random.uniform(np.log(20), np.log(50)) # 20 a 50
    ricker_centralfreq = params['ricker_centralfreq']
    ricker_dt = params['ricker_dt']
    ricker_tapersize = params['ricker_tapersize']
    ricker_windowsize = params['ricker_windowsize']
    
    wavelet = get_ricker_wavelet_2D(frequency=ricker_centralfreq,
                                    dt=ricker_dt,
                                    taper_size=ricker_tapersize,
                                    window_size=ricker_windowsize)
    
    output = np.zeros_like(seismic_data)
    #aux_panel = sp.signal.convolve2d(seismic_data[...,seismic_channel], wavelet, mode='same', boundary='fill')
    aux_panel = sp.signal.fftconvolve(seismic_data[...,seismic_channel], wavelet, mode='same')

    output[...,seismic_channel] = aux_panel
    
    return output


def data_viewer(data, print_info=True):
    data = np.squeeze(data)
    vmax = np.abs(data).max()
    vmin = -vmax
    if print_info:
        print("Data range:", data.min(), data.max())
        print("Data type:", data.dtype)
        print("Data shape:", data.shape)

    plt.imshow(data, cmap=plt.cm.bone, vmin=vmin, vmax=vmax, interpolation='lanczos')
    plt.colorbar()
    plt.show()
