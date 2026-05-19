import os
import argparse
import logging
import numpy as np
from pathlib import Path

from fermipy.gtanalysis import GTAnalysis
from fermipy.plotting import ROIPlotter
from astropy.io import fits
from pmfMaker import BgdModelAnalysis, PMFConfig, PlotSkyRegion
from dataclasses import dataclass
from configSetup import configure_input_files

from scipy.special import gammaln
from astropy import units as u

import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(message)s",
    filename="tweedleDEE.log",
    filemode="a",
    force=True,
)

logging.getLogger("astropy").setLevel(logging.ERROR)
logging.getLogger("h5py").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def main(args: argparse.Namespace) -> None:
    """Main entry point for TweedleDEE analysis pipeline.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing sky_location name, directory paths, and config files.
        Expected attributes: sky_location, input_dir, output_dir, targets_file, 
        td_config_file, configure.
    
    Returns
    -------
    None
    """
    # Setup input files (events.txt and config.yaml) for all sky_location galaxies 
    sky_location = args.sky_location # 'Reticulum_II'
    input_dir = args.input_dir # 'input/'
    targets_file = args.targets_file # 'targets.yaml'
    td_config_file = args.td_config_file # 'td_config.yaml'
    run_config = args.configure # False

    # Set the random seed for reproducibility (only needed if you want to reproduce the results)
    np.random.seed(34285972)

    td_config = PMFConfig.from_yaml(td_config_file)
    # CLI --output-dir is an override; otherwise use td_config.yaml paths.sky_location_files_dir
    output_dir = args.output_dir if args.output_dir is not None else td_config.sky_location_files_dir
    
    # Configure input files if specified (i.e. run_config=True); otherwise, assumes files are already configured and proceeds to analysis
    if run_config:
        logger.info(f"Configuring input files for {sky_location}...")
        configure_input_files(catalog=td_config.source_info_file, defaults=td_config.defaults, year=td_config.year, input_dir=input_dir, output_dir=output_dir, targets_file=targets_file)

    # Run GTAnalysis setup to create files for the PMF analysis
    gta = runGTA(sky_location, input_dir)

    # Core PMF analysis workflow
    pmf = BgdModelAnalysis(td_config, sky_location_files_dir=output_dir)
    pmf_results, like_results = pmf.generate_PMF(sky_location)

    # Option to Plot the PMF for the sky location
    PlotSkyRegion(pmf, sky_location, td_config.plot_output_dir).plot_PMF()
    
    # Optional: Run extended likelihood calculations and Fermi background analysis
    if td_config.run_likelihood:
        logger.info(f"Computing likelihood metrics for {sky_location}...")
        gta = runGTA(sky_location, input_dir)
        
        logL_likelihood_cov = like_results.log_likelihood_cov
        logL_likelihood_indep = like_results.log_likelihood_indep
        
        fermi_background_result = fermi_background(sky_location, output_dir, gta, td_config.sample_size)
        logL_fermi = fermi_background_result.logL_fermi
        fermi_bic = fermi_background_result.BIC
        fermi_aic = fermi_background_result.AIC
        fit_qual = fermi_background_result.fit_qual
        free_params = fermi_background_result.k

        Indep_BIC = calculate_BIC(logL_likelihood_indep, k=0, n=16)
        Indep_AIC = calculate_AIC(logL_likelihood_indep, k=0)
        Cov_BIC = calculate_BIC(logL_likelihood_cov, k=0, n=16)
        Cov_AIC = calculate_AIC(logL_likelihood_cov, k=0)

        logger.info(f"{sky_location}, {logL_likelihood_indep}, {Indep_BIC}, {Indep_AIC}, {logL_likelihood_cov}, {Cov_BIC}, {Cov_AIC}, {logL_fermi}, {fermi_bic}, {fermi_aic}, {fit_qual}, {free_params}")
        print(f"{sky_location}, {logL_likelihood_indep}, {Indep_BIC}, {Indep_AIC}, {logL_likelihood_cov}, {Cov_BIC}, {Cov_AIC}, {logL_fermi}, {fermi_bic}, {fermi_aic}, {fit_qual}, {free_params}")
    else:
        logger.info(f"Skipping likelihood calculations (run_likelihood=False in td_config.yaml)")


def count_free_parameters(gta: GTAnalysis) -> int:
    """Count the number of free parameters in the GTAnalysis ROI.
    
    Iterates through all sources in the ROI (excluding isodiff and galdiff)
    and counts spectral parameters marked as free.
    
    Parameters
    ----------
    gta : GTAnalysis
        The GTAnalysis object with the loaded ROI.
    
    Returns
    -------
    int
        The total number of free parameters in the ROI.
    """
    return sum(
        1 for src in gta.roi.sources
        if src.name not in ('isodiff', 'galdiff')
        for par_dict in src.spectral_pars.values()
        if par_dict.get('free', False)
    )

def roi_fit(sky_location: str, output_dir: str, gta: GTAnalysis) -> tuple[int, float]:
    """Fit the ROI for a sky location and generate residual maps.
    
    Loads the ROI FITS file, optimizes fit, finds new sources, frees parameters,
    and creates residual significance and excess maps.
    
    Parameters
    ----------
    sky_location : str
        Name of the sky location.
    output_dir : str
        Output directory path for saving results.
    gta : GTAnalysis
        The GTAnalysis object for Fermi-LAT analysis.
    
    Returns
    -------
    tuple[int, float]
        Number of free parameters (k) and fit quality metric.
    """
    roi_file = f'roi_{sky_location}.fits'
    gta.load_roi(roi_file)

    gta.optimize()

    gta.find_sources(sqrt_ts_threshold=5,min_separation=0.2,tsmap_fitter='tsmap')
    gta.write_roi(f'roi_{sky_location}_withNewSources.fits', make_plots=True)

    gta.free_sources(pars='norm', distance=3.0, minmax_ts=[20, None])
    gta.free_source('galdiff')
    gta.free_source('isodiff')

    gta.optimize()
    fit = gta.fit()
    fit_qual = fit['fit_quality']

    # Count the number of free parameters after fitting
    k = count_free_parameters(gta)

    gta.write_roi(f'roi_{sky_location}_postfit.fits', make_plots=True)
    gta.print_roi()

    resid = gta.residmap('_postfit',model={'SpatialModel' : 'PointSource', 'Index' : 2.0}, write_fits=True, write_npy=True, make_plots=True)

    plt.clf()
    fig = plt.figure(figsize=(14,6), dpi=150)
    ROIPlotter(resid['sigma'],roi=gta.roi).plot(vmin=-5,vmax=5,levels=[-5,-3,3,5],subplot=121,cmap='RdBu_r')
    plt.gca().set_title('Significance')
    ROIPlotter(resid['excess'],roi=gta.roi).plot(vmin=-100,vmax=100,subplot=122,cmap='RdBu_r')
    plt.gca().set_title('Excess')
    plt.savefig(f'{output_dir}/{sky_location}/resid_maps_{sky_location}.png')

    return k, fit_qual

def calculate_BIC(logL: float, k: int, n: int) -> float:
    """Calculate the Bayesian Information Criterion (BIC).
    
    Computes BIC = k * ln(n) - 2 * logL for model comparison.
    
    Parameters
    ----------
    logL : float
        The log-likelihood of the model.
    k : int
        The number of free parameters in the model.
    n : int
        The number of data points (energy bins).
    
    Returns
    -------
    float
        The BIC value.
    """
    return k * np.log(n) - 2 * logL 

def calculate_AIC(logL: float, k: int) -> float:
    """Calculate the Akaike Information Criterion (AIC).
    
    Computes AIC = 2 * k - 2 * logL for model comparison.
    
    Parameters
    ----------
    logL : float
        The log-likelihood of the model.
    k : int
        The number of free parameters in the model.
    
    Returns
    -------
    float
        The AIC value.
    """
    return 2 * k - 2 * logL

@dataclass
class FermiBackgroundResult:
    logL_fermi: float
    BIC: float
    AIC: float
    fit_qual: float
    k: int

def fermi_background(sky_location: str, output_dir: str, gta: GTAnalysis, sample_size: float) -> FermiBackgroundResult:
    """Calculate Fermi-LAT background likelihood for a sky location.
    
    Extracts counts and model predictions within a circular aperture, computes
    log-likelihoods, and calculates model selection criteria (BIC, AIC).
    
    Parameters
    ----------
    sky_location : str
        Name of the sky location.
    output_dir : str
        Output directory path for results.
    gta : GTAnalysis
        The GTAnalysis object with loaded ROI.
    sample_size : float
        Radius in degrees for the circular aperture region.
    
    Returns
    -------
    FermiBackgroundResult
        Dataclass containing: logL_fermi, BIC, AIC, fit_qual, and k (number of free parameters) metrics.
    """
    # If a postfit ROI exists, load it and skip re-fitting; otherwise run ROI fit
    postfit_path = Path(output_dir) / sky_location / f'roi_{sky_location}_postfit.fits'
    if postfit_path.exists():
        gta.load_roi(f'roi_{sky_location}_postfit.fits')
        k = count_free_parameters(gta)
        fit_qual = float('nan')
    else:
        k, fit_qual = roi_fit(sky_location, output_dir, gta)
        print(f"Baseline fit completed with {k} free parameters and fit quality {fit_qual}.")
        

    gta.print_roi()

    ccube = gta.counts_map()
    center_coordinate = ccube.geom.center_skydir.icrs
    ccube_cut = ccube.cutout(center_coordinate, (2*sample_size*u.deg, 2*sample_size*u.deg))

    model_map = gta.model_counts_map()
    model_cut = model_map.cutout(center_coordinate, width=(2*sample_size*u.deg, 2*sample_size*u.deg))

    geom = ccube_cut.geom
    sep = geom.separation(center_coordinate)
    sep_deg = sep.to_value(u.deg)
    mask = sep_deg < float(sample_size)

    n_E = np.sum(ccube_cut.data[:, mask], axis=1)   # shape (nE,)
    mu_E = np.sum(model_cut.data[:, mask], axis=1)   # shape (nE,)

    eps = 1e-8
    mu_E = np.clip(mu_E, eps, None)     # avoid log(0)
    logL_perE = n_E * np.log(mu_E) - mu_E - gammaln(n_E + 1.0) # logL per energy bin
    logL_fermi = np.sum(logL_perE)

    BIC = calculate_BIC(logL_fermi, k, n=len(n_E))
    AIC = calculate_AIC(logL_fermi, k)

    fermi_background_result = FermiBackgroundResult(
        logL_fermi=logL_fermi,
        BIC=BIC,
        AIC=AIC,
        fit_qual=fit_qual,
        k=k
    )

    return fermi_background_result

def runGTA(sky_location: str, input_dir: str) -> GTAnalysis:
    """Initialize and setup GTAnalysis for a sky location.
    
    Creates a GTAnalysis instance from the sky location's config YAML, runs setup,
    and writes the initial ROI file.
    
    Parameters
    ----------
    sky_location : str
        Name of the sky location.
    input_dir : str
        Input directory containing sky location configuration and data files.
    
    Returns
    -------
    GTAnalysis
        Initialized and setup GTAnalysis object ready for ROI analysis.
    """
    gta = GTAnalysis(f'{input_dir}/{sky_location}.yaml',logging={'verbosity': 3})
    gta.setup()
    gta.write_roi(f'roi_{sky_location}.fits')

    return gta

def calculate_exposure(sky_location: str) -> float:
    """
    Calculate the average exposure of the central region of interest (ROI) for a given sky location.

    This function loads the exposure map FITS file (bexpmap_roi_00.fits) for the specified sky_location, extracts a square
    region around the ROI center (default opening angle = 1°), and computes the average exposure
    value by first averaging over the spatial pixels in the region and then across all energy bins.

    Parameters
    ----------
    sky_location : str
        Name of the sky location. The function expects to find the exposure map file at
        ``output/{sky_location}/bexpmap_roi_00.fits``.

    Returns
    -------
    float or None
        The mean exposure across the ROI and energy bins. Returns ``None`` if the exposure
        map file is not found.

    Notes
    -----
    - The ROI center is defined as the midpoint of the exposure map dimensions.
    - The extraction region is determined by the `degree_opening` parameter (currently fixed at 1°).
    - Exposure is averaged over both spatial (x, y) dimensions and energy bins.
    """

    degree_opening = 1

    if not os.path.exists(f'output/{sky_location}/bexpmap_roi_00.fits'):
        print(f'Exposure map not found for {sky_location}.')
        return None

    with fits.open(f'output/{sky_location}/bexpmap_roi_00.fits') as bexpmap_roi:
        #print(bexpmap_roi[0].header)
        data = bexpmap_roi[0].data

        # Get the center coordinates 
        center_x = bexpmap_roi[0].header['NAXIS1'] / 2
        center_y = bexpmap_roi[0].header['NAXIS2'] / 2
        delta = abs(bexpmap_roi[0].header['CDELT1'])
        
        pixels_per_degree = 1 / delta
        half_width = int(degree_opening * pixels_per_degree / 2)  # Half-width in pixels

        # Create slice ranges for x and y dimensions (e.g. for 400x400 pixel image, the center is (199.5, 199.5)), so the -1 selects (199,199) as the center)
        x_slice = slice(int(center_x - half_width) - 1, int(center_x + half_width) - 1)
        y_slice = slice(int(center_y - half_width) - 1, int(center_y + half_width) - 1)

        # Extract the region using the calculated slices
        center_region = data[:, y_slice, x_slice]  # Shape: (energy_bin_edges, y_pixels, x_pixels)
        
        # Average over spatial dimensions, then over energy bins
        # Final value is the mean exposure across the central ROI
        spatial_avg = np.mean(center_region, axis=(1,2))  # shape (energy_bin_edges,)
        exposure_avg = np.mean(spatial_avg)

    return exposure_avg

def update_exposures(sky_location: str, pmf: BgdModelAnalysis) -> None:
    """Update the exposures tracking file with calculated values.
    
    Appends or creates PMFdata/Exposures_updated.tsv with sky_location exposure
    values, skipping sky_locations already in the PMF database.
    
    Parameters
    ----------
    sky_location : str
        Name of the sky location.
    pmf : BgdModelAnalysis
        The PMF analysis object containing known sky location database.
    
    Returns
    -------
    None
    """
    sky_locations, IDs = pmf.get_sky_locations()
    if sky_location not in sky_locations:
        if os.path.exists('PMFdata/Exposures_updated.tsv'):
            with open('PMFdata/Exposures_updated.tsv', 'a') as file:
                file.write(f'\n{sky_location}\t{calculate_exposure(sky_location)}\n')
                print(f'{sky_location} added to Exposures_updated.tsv.')
        else:
            with open('PMFdata/Exposures_updated.tsv', 'w') as file:
                file.write(f'Name\tExposure\n')
                file.write(f'{sky_location}\t{calculate_exposure(sky_location)}\n')

def get_new_ID(file_path: str) -> int:
    """Get the next sequential ID from the IDs tracking file.
    
    Reads the last line of the IDs file, extracts the ID, and returns
    the next sequential ID value.
    
    Parameters
    ----------
    file_path : str
        Path to the IDs tracking file (typically PMFdata/IDs_updated.tsv).
    
    Returns
    -------
    int
        The next available ID (last ID + 1).
    """
    with open(file_path, 'r') as file:
        data = file.readlines()
        new_ID = data[-1].split('\t')[0]
    return int(new_ID) + 1

def update_IDs(sky_location: str, pmf: BgdModelAnalysis) -> None:
    """Update the IDs tracking file with a new sky location.
    
    Appends or creates PMFdata/IDs_updated.tsv with a new sequential ID
    and sky location name, skipping sky locations already in the PMF database.
    
    Parameters
    ----------
    sky_location : str
        Name of the sky location to add.
    pmf : BgdModelAnalysis
        The PMF analysis object containing known sky location database.
    
    Returns
    -------
    None
    """
    sky_locations, IDs = pmf.get_sky_locations()
    if sky_location not in sky_locations:
        if os.path.exists('PMFdata/IDs_updated.tsv'):
            with open('PMFdata/IDs_updated.tsv', 'a') as file:
                new_ID = get_new_ID('PMFdata/IDs_updated.tsv')
                file.write(f'{new_ID}\t{sky_location}\n')
        else:
            with open('PMFdata/IDs_updated.tsv', 'w') as file:
                file.write(f'#ID\tDwarf Name\n')
                file.write(f'1\t{sky_location}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='TweedleDEE: a Tool for Determining the Background Emission Empirically',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Required arguments
    parser.add_argument(
        '--sky_location', '-d',
        type=str, 
        required=True,
        help='Name of the sky location to analyze'
    )
    # Configuration options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument(
        '--configure', '-c', 
        action='store_true',
        help='Run configuration setup to create events.txt and config YAML files'
    )
    config_group.add_argument(
        '--targets_file', '-tf',
        type=str, 
        default='targets.yaml',
        help='Targets YAML filename for sky locations (default: targets.yaml)'
    )
    config_group.add_argument(
        '--td_config_file', '-tcf',
        type=str, 
        default='td_config.yaml',
        help='Configuration YAML filename for PMF analysis (default: td_config.yaml)'
    )
    
    # Directory options
    dir_group = parser.add_argument_group('Directory Options')
    dir_group.add_argument(
        '--input-dir', '-i',
        type=str, 
        default='input/',
        help='Input directory containing sky location data (default: input/)'
    )
    dir_group.add_argument(
        '--output-dir', '-o',
        type=str, 
        default=None,
        help='Output directory override. If omitted, uses paths.sky_location_files_dir from td_config.yaml'
    )
    
    args = parser.parse_args()
    
    # Add file extensions if not present
    if not args.targets_file.endswith('.yaml'):
        args.targets_file = args.targets_file + '.yaml'
    if not args.td_config_file.endswith('.yaml'):
        args.td_config_file = args.td_config_file + '.yaml'
    
    main(args)