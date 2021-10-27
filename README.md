# Code Repository for the Tropical Lapse Rates Paper

Short documentation on how to use the scripts that produce Figures for the article "Variations in Tropical Lapse Rates in Climate Models and their Implications for upper Tropospheric Warming".

- Link to paper: https://doi.org/10.1175/JCLI-D-21-0196.1
- Link to freely accessible preprint: http://hdl.handle.net/21.11116/0000-0009-6803-F

Please be aware that the scripts were all designed to run on a high performance computer (in this case "mistral" from the DKRZ) and therefore most of the scripts won't run out of the box on a laptop. The used data is large in size and processed with high memory usage. Some basic functions are stored in the `metcalc.py` and `aes_thermo.py` files, but the processing and visualising is done in jupyter notebooks (and one bash script with cdos for the ICON experiments). Python package versions used:

 - `xarray` : 0.16.2
 - `intake_esm` : 2020.12.18
 - `xesmf` : 0.3.0
 - `pandas` : 1.1.5
 - `numpy` : 1.17.2
 - `scipy` : 1.3.2

Data from two different simulations types are presented and analysed: CMIP6 simulations and parameter experiments with the ICON-A model. We will describe these two as well as the observations/Reanalysis separately:



## CMIP6:

CMIP6 data is processed with the python Jupyter Notebook `tropical_lapse_rates_get_cmip.ipynb`, which uses the intake python package to easily read the data on mistral. If you do not have access to mistral or other high performance computers, these packages also work well with the google cloud. For more info see here: https://gallery.pangeo.io/repos/pangeo-gallery/cmip6/
After the data is processed, the plots for the first part of the paper that only discusses CMIP6 data is produced with `tropical_lapse_rates_cmip.ipynb`. This script also includes ERA5 data IUK radiosondes, which are preprocessed with the `iuk.ipynb` and `tropical_lapse_rates_process_ERA5.ipynb`. The plots that analyse the warming trend in the last part of the paper are included in the `tropical_lapse_rates_icon.ipynb`.



## ICON-A:

General Information on how to download and run the ICON model can be found here: https://mpimet.mpg.de/en/science/modeling and here: https://wiki.mpimet.mpg.de/doku.php?id=models:icon:start
We ran AMIP experiments, which are described here: https://wiki.mpimet.mpg.de/doku.php?id=models:icon:running_the_model:generate_a_runscript
Once the model is correctly compiled and set up, there should be an automatically generated example AMIP run script. For reference, a run script that was used in the simulations is also included here, under the `icon-aes` folder. The parameter experiments were done by changing the parameter values in the runscript:

- Turbulent entrainment: `echam_cnv_config(1)%entrpen`
- Autoconversion: `echam_cnv_config(1)%cprcon`

To get an ensemble of simulations, individual simulations with slightly different rayleigh coefficients were run, for example `rayleigh_coeff = 0.1001` instead of `rayleigh_coeff = 0.1` was set in the runscript. If you have access to the MPI gitlab, you can download the exact model version used in the study with the git hash `7d005add0c8b705066e60774d811d61d6e79552f`, which you can get with `git fetch origin 7d005add0c8b705066e60774d811d61d6e79552f`. You can also contact me directly for access to model output data: `paul.keil AT mpimet.mpg.de`

Data from the ICON experiments are processed with the `process_icon-param_mm.sh`: horizontal and vertical interpolation, and tropical fieldmean.



## RADIOSONDES AND REANALYSIS:

Weblinks to data were checked on 7th September 2021

- ERA5 data can be downloaded here: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=form. ERA5.1 data download is more complicated, but not really necessary since differences between ERA5 and ERA5.1 are small in the troposphere. ERA5 data is then preprocessed in the `tropical_lapse_rates_process_ERA5.ipynb` script.

- IUK: IUK radiosondes can be downloaded here: https://www.ccrc.unsw.edu.au/professor-steven-sherwood/research-steven-sherwood/iuk-radiosonde-analysis-project-now-updated and are preprocessed with the `iuk.ipynb` script.

- Rich-Tau, Rich-Obs, Raobcore: https://www.univie.ac.at/theoret-met/research/raobcore/

- HadAT: https://hadleyserver.metoffice.gov.uk/hadat/

- RATPAC: https://www.ncei.noaa.gov/products/weather-balloon/radiosonde-atmospheric-temperature-products

- SUNY: ftp://aspen.atmos.albany.edu/data/UA-HRD
