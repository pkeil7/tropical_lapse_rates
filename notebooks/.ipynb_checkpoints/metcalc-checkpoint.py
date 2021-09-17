# Python script for calculating some important atmospheric variables
import xarray as xr
import metpy
import numpy as np
import aes_thermo as mt
from scipy import interpolate,optimize
import scipy.integrate as integrate
import scipy.stats as stat
from metpy.interpolate import interpolate_1d
import metpy.calc as mpcalc
from metpy.units import units
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



# ------- SATISTICS

def covariance_gufunc(x, y):
    return ((x - x.mean(axis=-1, keepdims=True))
            * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)

def pearson_correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

def pearson_correlation(x, y, dim):
    return xr.apply_ufunc(
        pearson_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float])

def calc_slope(x,y) :
    regression = stat.linregress(x,y)
    return regression.slope

def xr_linear_regression(ds,var,dim) :
    ''' Calculates regressions from the scipy stats module.
    The input_core_dims correspond to the coordinate you want to use for the regression
    '''
    trend = xr.apply_ufunc(
    calc_slope, ds[dim], ds[var],
    input_core_dims=[[dim],[dim]],
    dask='parallelized',
    output_dtypes=[float],
    vectorize=True
    )
    return trend.to_dataset(name='trend')

# ----- USEFUL STUFF:

def get_z_from_p(P,T):
    '''P and T are single values or vectors'''
    z = -mt.Rd *T / mt.gravity*(np.log(P)-np.log(101000.))
    return z

def get_p_from_z(z,T):
    p = 101000. * np.exp(- ( z * mt.gravity) / (mt.Rd * T))


def xr_convert_vertical_velocity(wap,T):
    '''wap and T are xarray datasets or arrays with pressure as vertical corrdinate
    I use this formula from hydrostatic eq:
    dz = - R * T * dp / (p * g)
    '''

    w = - mt.Rd * T * wap  / ( mt.gravity * wap.plev )

    return w

def get_lat_name(ds):
    for lat_name in ['lat', 'latitude']:
        if lat_name in ds.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")

def get_lon_name(ds):
    for lon_name in ['lon', 'longitude']:
        if lon_name in ds.coords:
            return lon_name
    raise RuntimeError("Couldn't find a longitude coordinate")


def xr_fldmean_old(ds):
    ''' a basic fldmean that should work over latitude and longitude,
    or just latitude if a zonal mean has already been done.'''

    lon_name = None
    lat_name = get_lat_name(ds)
    lat = ds[lat_name]
    weight = np.cos(np.deg2rad(lat))
    weight /= weight.mean()
    for lon in ['lon', 'longitude']:
        if lon in ds.coords:
            lon_name = lon

    if lon_name == None :
        print('averigin over latitude')
        return (ds * weight).mean(lat_name)
    else :
        print('averigin over latitude (' + lat_name + ') and lonigtude (' + lon_name + ')')
        return (ds * weight).mean([lon_name,lat_name])


def xr_fldmean(ds):
    ''' fldmean that uses xarrays weighted function
    This function properly accounts for latitude weights and missing values

    '''

    lon_name = None
    lat_name = get_lat_name(ds)
    for lon in ['lon', 'longitude']:
        if lon in ds.coords:
            lon_name = lon

    lat = ds[lat_name]
    weights = np.cos(np.deg2rad(lat))
    ds_weighted = ds.weighted(weights)

    return ds_weighted.mean([lon_name,lat_name])


def subsampling_locations(ds,lons,lats,remove_doubles=True) :
    ''' Takes an xarray dataset or data array and a list of longitudes and latitudes.
    The nearest corresponding gridpoints on the dataset are chosen and double entries are removed.'''
    subsets = []
    lonlat_list =[]

    for i,lon in enumerate(lons) :
        lat = lats[i]
        subset = ds.sel(lon=lon,lat=lat,method="nearest")
        lonlat_list.append([float(subset.lon.values),float(subset.lat.values)])

    # check for doubles:
    if remove_doubles == True :
        lonlat_list = remove_doubles_list(lonlat_list)

    for lonlat in lonlat_list :
        subsets.append(ds.sel(lon=lonlat[0],lat=lonlat[1]))

    concat_dim = xr.DataArray(np.arange(1,len(subsets)+1,1), name='location',dims='location')
    print(str(len(lons)) + " coordinates were given, this function chose " + str(len(subsets)) + " corresponding gridpoints on the dataset.")
    return xr.concat(subsets, concat_dim)

def remove_doubles_list(a) :
    for i,element in enumerate(a) :
        if element in a[i+1:] :
            a.remove(element)
    return a

# ------ PLOTTING

def remove_axes(ax) :
    spines_to_keep = ['left', 'bottom']
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

def set_regional_cartopy(ax,projection=ccrs.PlateCarree(central_longitude=180),extent=[0, 359, -20, 20]) :
    ax.coastlines()
    ax.set_extent(extent, projection)
    

# ----- THERMODYNAMICS:

def potential_temp(T,p) :

    #exponent: R/cp
    exp = 0.286
    p0  = 100000

    theta = T * ( p0/p )**exp
    return theta

def xr_potential_temperature(T):
    '''calulates potential temperature and returns a dataset'''
    theta = xr.apply_ufunc(
        potential_temp, T, T.plev2*100,
        dask='parallelized',
        output_dtypes=[float])
    return theta.rename({'ta': 'theta'})


def virtual_temperature(T,q) :
    Rd = mt.Rd #287.
    Rv = mt.Rv #461.
    e1 = Rd / Rv
    e2 = ( 1. / e1 ) - 1.
    return T * (1. + e2 * q )

def calculate_qs(T,P):
    es = mt.es(T)
    rs = mt.Rd/mt.Rv*es/(P-es)
    qs = rs/(1+rs)
    return qs



def xr_es(T, state) :
    es = xr.apply_ufunc(
        mt.es2, T, state,
        dask='parallelized',
        output_dtypes=[float],
        #output_core_dims=[["lat"],["lon"]]
    )
    return es

def xr_mr2pp(mr, p, input_dims,output_dims) :
    pp = xr.apply_ufunc(
        mt.mr2pp, mr, p,
        dask='parallelized',
        output_dtypes=[float],
        input_core_dims=input_dims,
        output_core_dims=output_dims,
        vectorize=True  # loop over non-core dims
    )
    return pp


def xr_phase_change_enthalpy(ta, fusion=False) :
    '''Takes temperature data array and returns enthalpy in data array'''
    l = xr.apply_ufunc(
        mt.phase_change_enthalpy, ta, fusion,
        dask='parallelized',
        output_dtypes=[float])
    return l


def xr_get_theta_e(T, P, QT, formula='isentrope') :
    '''Takes temperature, humidity and pressure data array
    returns theta e'''
    theta_e = xr.apply_ufunc(
        mt.get_theta_e2, T, P, QT, formula,
        dask='parallelized',
        output_dtypes=[float],
        #output_core_dims=[["lat"],["lon"]]
    )
    return theta_e


def xr_get_theta_es(T, P) :
    '''Takes temperature and pressure data array
    returns theta es. This is always pseudoadiabatic'''

    Psat   = xr_es(T,state='liq')
    print('Psat:')
    print(Psat)
    print('P:')
    print(P)

    # not totally accurate but ok for now:
    qs     = (Psat/(P-Psat)) * (mt.Rd/mt.Rv)

    QT = qs

    print('QT:')
    print(QT)

    formula='isentrope'

    theta_es = xr.apply_ufunc(
        mt.get_theta_e2, T, P, QT, formula,
        dask='parallelized',
        output_dtypes=[float],
        #output_core_dims=[["lat"],["lon"]]
    )
    return theta_es


def xr_MSE(T,q):
    '''Takes temperature and specific humidtiy on pressure levels 
    and returns moist static energy'''
    
    assert T.plev2.values.all() == q.plev2.values.all()
    
    z = get_z_from_p(T.plev2 *100.,270.)
    
    hd = mt.cpd*(T-mt.T0) + mt.gravity*z
    hv = mt.cpv*(T-mt.T0) + mt.gravity*z + mt.lv0

    h = hd*(1-q) + q*hv

    return h

def deviations_from_moist_adiabat(ta_ds,plev_i=3,ensemble_name=None) :
    """This function uses xarray datasets of temperature to calculate the deviations from the moist adiabat
    The temperature dataset should have vertical pressure coordinates in hPa.
    The plev_i gives the index at which pressure level the calculations should start.
    The ensemble name refers to the variable name representing model ensemble memebers in the xarray dataset
    For cmip5 models plev_i=2 correspondes to 700hPa
    The result is a dataset"""

    multiple_dims = False
    # check if ensemble name exists:
    if not ensemble_name == None :
        ta_ds = ta_ds.rename({ ensemble_name : 'dimension'})
        multiple_dims = True
        dim_ids=ta_ds.dimension.values

    ta_ds = ta_ds.sel(plev2=slice(1000,100))
    p = ta_ds.plev2.values
    print(p)

    print('selected starting level: ' + str(p[plev_i]) + ' hPa')

    if multiple_dims == True :
        ta_dev_np = np.zeros(( len(dim_ids), len(p[plev_i:]) ))
        i=0
        for dim_id in dim_ids :
            T_model = ta_ds.sel(dimension = dim_id).ta.values
            T_,p_ = integrate_dTdP(T_model[plev_i], p[plev_i]*100.,
                                   10000., -100., qt=15.e-3, formula='pseudo')
            p_ = np.insert(p_, 0,p[plev_i]*100.)
            T_ = np.insert(T_, 0,T_model[plev_i])
            f = interpolate.interp1d(p_, T_)
            T_pseudo = f(p[plev_i:]*100.)
            ta_dev = T_model[plev_i:] - T_pseudo
            ta_dev_np[i,:] = ta_dev
            i=i+1

        ta_ds_dev = xr.Dataset({'ta': ([ensemble_name, 'plev2'],  ta_dev_np)},
                                  coords={ensemble_name: ([ensemble_name], dim_ids), 'plev2': (['plev2'], p[plev_i:])}
                                 )

    else :
        T_model = ta_ds.ta.values
        T_,p_ = integrate_dTdP(T_model[plev_i], p[plev_i]*100.,
                               10000., -100., qt=15.e-3, formula='pseudo')

        p_ = np.insert(p_, 0,p[plev_i]*100.)
        T_ = np.insert(T_, 0,T_model[plev_i])
        f = interpolate.interp1d(p_, T_)
        T_pseudo = f(p[plev_i:]*100.)
        ta_dev = T_model[plev_i:] - T_pseudo
        ta_ds_dev = xr.Dataset({'ta': (['plev2'],  np.array(ta_dev))}, coords={'plev2': (['plev2'], p[plev_i:])})

    return ta_ds_dev


def difference_ma(ta_profile, p_profile) :

    P0 = p_profile[0]
    T0 = ta_profile[0]
    P1 = p_profile[-1]
    dP = 1.
    qt = 15.

    ta, p = integrate_dTdP(T0,P0,P1,dP,qt,formula='pseudo')

    # get levels:
    ta_ = ta[ np.where(p == p_profile) ]
    print(ta_)
    diff = ta_profile - ta_
    return diff


# ---------- moist adiabat code created by Bjorn Stevens:
# source: https://owncloud.gwdg.de/index.php/s/tu8fvhM4klLW6rI
# and https://owncloud.gwdg.de/index.php/s/bMB1EtjXldxH9VT

# We start with a differential form, dX that allows us to construct how temperature changes with pressure following different processes.  The form we construct is similar to $\theta_\mathrm{l}$ allows for a saturated isentrope allowing only for a liquid phase, a saturated isentrope assuming only a solidphase, whereby the saturated vapor pressure over ice is set to that over water at temperatures above 273.15 K.  We also calculate the pseudo adiabats, one allowing only a liquid phase, and another that transitions to the ice-phase at temperatures below 273.15 K, and differs from the former by the additional fusion enthalpy.  The true reversible isentrope would follow the saturated liquid phase isentrope at temperatures above the triple point and the saturated ice-phase isentrope at temperatures below the triple point.

# Equations are absed on Stevens and Siebesma Chapter 2, section 2.2.2, page 13

# Here we calcualte dX_dT and dX_dP. X is not really anything physically usefull, but is used in usefull maths trick here. dX = dh + vdp = 0, dX is always zero in an adiabatic process. we want to find a way to express dX in terms of dT and dp, so that dX = dX_dT dT + dX_dP dP. That gives us expressions for dX_dT, dX_dP. This is done with dh = cp dT + qs lv. qs and lv then again depend on p and T. The formula gets complicated, but we can change the terms around to get the dX = dX_dT dT + dX_dP dP = 0 structure and thereby find an expression for dX_dT, dX_dP. Then we just say: dT_dP = dX_dP / dX_dT. To get T(p) we integrate dX_dP / dX_dT by p.

def dX(TK,PPa,qt,formula='isentrope',con_factor=0,ice_factor=0) :
    '''possibilities for formula:'''


    # determine saturations water vapour pressure Psat and from that calculate saturation mixing raio qs
    if ( ( formula == 'ice-isentrope' or formula == 'pseudo-ice' ) and TK  < 273.15 ) :
        Psat   = mt.es(TK,state='ice')
    elif ( formula == 'pseudo-mixed' and TK  < 273.15 ) :
        Psat   = (1- ice_factor) * mt.es(TK,state='ice') + ice_factor * mt.es(TK,state='ice')
    else:
        Psat   = mt.es(TK,state='liq')

    qs     = (Psat/(PPa-Psat)) * (mt.Rd/mt.Rv) * (1-qt)

    #Moist adiabatic ascent:

    if (qs < qt):
        ql = 0.

        # Pseudo means all water precipitates immedatly, therefore liquid water is zero and we only have WV and dry air.
        if (formula[:6] == 'pseudo'):
            qd = 1.- qs
        # mixed: part of the water precipitates and the rest stays as clouds
        elif formula == 'mixed'  :
            ql = con_factor * (qt - qs)
            qd = 1. - ( ql + qs )
        # Isentrope: All water is kept
        else:
            ql = qt-qs
            qd = 1.-qt

        # Calcualte heat capacity and phase change enthalpy:

        # Ice isenttrope: All water is kept and all of it is converted to ice
        if (formula == 'ice-isentrope' and TK < 273.15):
            cp     = qd * mt.cpd + qs * mt.cpv + ql * mt.cpi
            lv     = mt.phase_change_enthalpy(TK) + mt.phase_change_enthalpy(TK,fusion=True)
        # Pseudo-ice: all water is converted to ice (lv is the same), but ice falls out immediatly
        # ql is set to zero above, so last term does not mattter
        elif (formula == 'pseudo-ice' and TK < 273.15):
            cp     = qd * mt.cpd + qs * mt.cpv + ql * mt.cpl
            lv     = mt.phase_change_enthalpy(TK) + mt.phase_change_enthalpy(TK,fusion=True)
        elif (formula == 'pseudo-mixed' and TK < 273.15):
            cp     = qd * mt.cpd + qs * mt.cpv + ql * mt.cpl
            lv     = mt.phase_change_enthalpy(TK) + ice_factor * mt.phase_change_enthalpy(TK,fusion=True)
        elif (formula == 'isentrope-mixed' and TK < 273.15 ) :
            cp     = qd * mt.cpd + qs * mt.cpv + ql * mt.cpi # add som here still
            lv     = mt.phase_change_enthalpy(TK) + ice_factor * mt.phase_change_enthalpy(TK,fusion=True)

        # No ice: lv only includes the phase change from vapour to liquid.
            # - pseudo: ql is zero, so cp only depends on WV and dry air.
            # - reversible: ql is not zero, is used in cp
            # - mixed: ql not zero, but lower than isentropic type
        else:
            cp     = qd * mt.cpd + qs * mt.cpv + ql * mt.cpl
            lv     = mt.phase_change_enthalpy(TK)

        R      = qd * mt.Rd + qs * mt.Rv
        vol    = R * TK/ PPa

        beta_P = R/(qd*mt.Rd)
        if (formula[:6] == 'pseudo'):
            beta_P = R/mt.Rd

        # With cp, lv, beta_P and beta_t we can calculate DX_DT

        beta_T = beta_P * lv/(mt.Rv * TK)
        dX_dT  = cp + lv * qs * beta_T/TK
        dX_dP  = vol * ( 1.0 + lv * qs * beta_P/(R*TK))


    # Dry adiabatic ascent:

    else:
        cp     = mt.cpd + qt * (mt.cpv - mt.cpd)
        R      = mt.Rd  + qt * (mt.Rv  - mt.Rd)
        vol    = R * TK/ PPa
        dX_dT  = cp
        dX_dP  = vol

    return dX_dT, dX_dP;



def integrate_dTdP(T0,P0,P1,dP,qt,formula='isentrope',con_factor=0,ice_factor=0):
    def f(P, T, qt):
        dX_dT, dX_dP = dX(T,P,qt,formula,con_factor,ice_factor)
        return (dX_dP/dX_dT)

    r = integrate.ode(f).set_integrator('lsoda',atol=0.0001)
    r.set_initial_value(T0, P0).set_f_params(qt)
    t1 = P1
    dt = dP
    Te = []

    Tx =[T0]
    Px = [P0]


    while r.successful() and r.t > t1:
        r.integrate(r.t+dt)
        Tx.append(r.y[0])
        Px.append(r.t)

    return (np.array(Tx),np.array(Px))

# ------- Simple entraining plume:

# Created by Jiawei Bao (2021)

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')



def dT_entrainment_simple_Jiawei(T,P,RH,epsilon):
    qs = calculate_qs(T,P)
    Z = get_z_from_p(P,T)
    Delta_T = np.ones_like(P)*0.
    P_cb = 96000 # Level where entrainment effect starts (arbitrary)
    k_cb = 0
    for k in range(1,len(P)-1):
        if P[k]<=96000:
            Delta_T[k] = epsilon * (1-RH) / (1+mt.lv0/(mt.Rv*T[k]*T[k])*mt.lv0*qs[k]/mt.cpd) * np.sum(1.0/movingaverage(Z[k_cb:k+1],2)*mt.lv0/mt.cpd*movingaverage(qs[k_cb:k+1],2)*np.gradient(Z[k_cb:k+1]))
            #print(np.sum(1.0/movingaverage(Z[k_cb:k+1],2)*mt.lv0/mt.cpd*movingaverage(qs[k_cb:k+1],2)*np.gradient(Z[k_cb:k+1])))

    return Delta_T


# It does not make a difference in the end, but I define the function

def dT_entrainment(T,P,RH,epsilon):
    '''takes the temperature profile of a pseudoadiabat and calculates its devaition due to entrainment
    This calculation follows the that of Singh and O'Goreman 2013 and Zhou et al 2019.
    entrainment is included in the integral and assumed to be epsilon = epsilon_0/z

    Things that can be improved: the T used here is from the pseudoadiabat, but should really be from the entraining plume. Maybe try T = T_pseudo - Delta_T[k-1] (not perfectly correct, but better?) Or properly minimize like in Jiawei's complicated function...

    '''

    Delta_T = np.zeros_like(P)
    qs_env = calculate_qs(T,P)
    Z = get_z_from_p(P,T)

    P_cb = 96000 # Level where entrainment effect starts (arbitrary)

    def integrand(qs,z):
        return (mt.lv0 * qs)/(mt.cpd*z )

    for k in range(1,len(P)):
        if P[k]<=96000:
            Delta_T[k] = epsilon * (1-RH) / (1 + mt.lv0/(mt.Rv*T[k]*T[k]) * mt.lv0*qs_env[k]/mt.cpd) * \
            integrate.simps(integrand(qs_env[0:k], Z[0:k]), Z[0:k])


    return Delta_T



def dT_entrainment_700(T0,P0,RH,epsilon):
    '''This uses the entraining plume function to calcualte the deviations from the pseudoadiabat above 700hPa correctly.
    T0 and P0 should be at the cloud base and not at 700
    A good guess is T0 = 295., P0 = 96000.

    '''
    qt=30.e-3
    T_pseudo, Px = integrate_dTdP(T0,P0,15000.,-100., qt, formula='pseudo')

    Delta_T = dT_entrainment(T_pseudo,Px,RH,epsilon)

    T_entr = T_pseudo - Delta_T
    T_pseudo_700, Px_700 = integrate_dTdP(T_entr[np.where(Px == 70000.)],70000.,15000.,-100., qt, formula='pseudo')
    delta_T_entr_700 = T_pseudo_700 - T_entr[int(np.where(Px == 70000.)[0]):]
    T_entr_700 = T_entr[int(np.where(Px == 70000.)[0]):]

    return delta_T_entr_700, T_entr_700, Px_700
