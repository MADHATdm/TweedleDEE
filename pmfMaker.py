import os
import logging
import numpy as np
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import linalg
from dataclasses import dataclass, field
from typing import Optional
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from configSetup import get_catalogs

@dataclass
class PMFConfig:
    """Configuration for PMF background analysis."""
    
    # Spatial parameters
    nsample: int = 1e5  # number of sample ROIs
    target_size: float = 10.0  # degrees
    sample_size: float = 0.5   # degrees
    source_size: float = 0.8   # degrees
    binning: int = 1
    
    # File paths
    ids_filepath: str = 'PMFdata/IDs_updated.tsv'
    source_info_file: str = 'gll_psc_v32.fit'
    sky_location_files_dir: str = 'output/'
    plot_output_dir: str = 'output/'

    # For using precomputed dwarf galaxies from Fermi
    defaults: bool = False
    year: int = 2023
    
    # Analysis options
    run_likelihood: bool = False  # If True, compute Fermi background likelihood and model comparison metrics
    
    # Derived (computed automatically)
    energy_bins: Optional[np.ndarray] = field(default=None, init=False)
    bpd: int = field(default=8, init=False)
    
    def __post_init__(self) -> None:
        """Compute derived values after initialization."""
        # Setup energy bins based on binning parameter
        if self.binning == 0:
            self.energy_bins = np.array([(1., 100.)]) * 10**3
            self.bpd = 1
        else:
            self.energy_bins = np.array([(1.,1.33352143), (1.33352143,1.77827941), (1.77827941,2.37137371), (2.37137371,3.16227766), (3.16227766,4.21696503), (4.21696503,5.62341325), (5.62341325,7.49894209), (7.49894209,10.), (10.,13.33521432), (13.33521432,17.7827941), (17.7827941,23.71373706), (23.71373706,31.6227766), (31.6227766,42.16965034), (42.16965034,56.23413252), (56.23413252,74.98942093), (74.98942093,100.)])*(10**3)
            self.bpd = 8
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'PMFConfig':
        """Load configuration from YAML file."""
        import yaml

        config_path = Path(filepath)
        if not config_path.exists():
            config_path = Path('config') / filepath

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with config_path.open('r') as f:
            data = yaml.safe_load(f)

        # Support a grouped 'paths' section while remaining backward compatible
        # with existing top-level keys.
        paths = data.pop('paths', None) or {}
        if isinstance(paths, dict):
            if 'ids' in paths and 'ids_filepath' not in data:
                data['ids_filepath'] = paths['ids']
            if 'source_catalog' in paths and 'source_info_file' not in data:
                data['source_info_file'] = paths['source_catalog']
            if 'sky_location_files_dir' in paths and 'sky_location_files_dir' not in data:
                data['sky_location_files_dir'] = paths['sky_location_files_dir']
            if 'plot_output_dir' in paths and 'plot_output_dir' not in data:
                data['plot_output_dir'] = paths['plot_output_dir']

        # Resolve relative paths from project root (repo root when config is in
        # ./config, otherwise from the config file's directory).
        config_parent = config_path.resolve().parent
        base_dir = config_parent.parent if config_parent.name == 'config' else config_parent

        for key in ('ids_filepath', 'source_info_file', 'sky_location_files_dir', 'plot_output_dir'):
            value = data.get(key)
            if isinstance(value, str) and value:
                value_path = Path(value)
                if not value_path.is_absolute():
                    data[key] = str((base_dir / value_path).resolve())

        return cls(**data)

@dataclass
class CoordinateSet:
    target: SkyCoord
    source: SkyCoord
    sample: SkyCoord | None
    pruned_sample: SkyCoord | None
    extended: np.ndarray

@dataclass
class PMFResults:
    pmf_data: np.ndarray
    pmf_list: list[np.ndarray]
    means: np.ndarray
    n_obs: np.ndarray
    photon_counts: np.ndarray
    max_counts: int

@dataclass
class LikelihoodResults:
    log_likelihood_indep: float
    log_likelihood_cov: float
    delta_log: float

@dataclass
class EnergyBinResult:
    pmf_hist: np.ndarray
    counts_per_roi: np.ndarray
    max_count: int


class FDError(Exception):
    """Custom exception for file and data errors in PMF analysis.
    
    Used to signal issues with file paths, data integrity, or analysis failures.
    """
    def __init__(self, message):
        super().__init__(message)

class BgdModelAnalysis:
    def __init__(self, config: PMFConfig, sky_location_files_dir: str | None = None) -> None:
        """Initialize background model analysis with configuration.
        
        Parameters
        ----------
        config : PMFConfig
            Configuration object with analysis parameters (sample sizes, energy bins, etc.).
        sky_location_files_dir : str, optional
            Directory containing sky_location galaxy output files (default: 'output/').
        
        Returns
        -------
        None
        """
        self.config = config
        self.nsample = config.nsample
        self.target_size = config.target_size
        self.sample_size = config.sample_size
        self.source_size = config.source_size
        self.energy_bins = config.energy_bins
        self.bpd = config.bpd
        self.sky_location_files_dir = sky_location_files_dir or config.sky_location_files_dir
        self.ids_filepath = config.ids_filepath
        self.source_info_filepath = config.source_info_file

    def get_sky_locations(self) -> tuple[list[str], list[int]]:
        """Load sky location names and IDs from the IDs tracking file.
        
        Reads PMFdata/IDs_updated.tsv and extracts sky location names and their
        sequential ID numbers.
        
        Returns
        -------
        tuple[list[str], list[int]]
            Tuple of (sky_location_names, sky_location_ids) lists.
        
        Raises
        ------
        FDError
            If the IDs file does not exist.
        """
        if not os.path.isfile(self.ids_filepath):
            raise FDError(f'No such file {self.ids_filepath}.')
        
        with open(self.ids_filepath, "r") as IDs_file:
            next(IDs_file) # Skip the header
            lines = [line.strip().split() for line in IDs_file if line.strip()]

            sky_locations = [line[1].upper() for line in lines]
            IDs = [int(line[0]) for line in lines]

        return sky_locations, IDs

    def set_coords_besides_events(self, target: str, nsample: int = None) -> tuple[SkyCoord, SkyCoord, SkyCoord | None, np.ndarray]:
        """Load target, source, and sample coordinates for analysis.
        
        Retrieves target coordinates from FITS header, source catalog coordinates,
        and optional random sample ROI coordinates.
        
        Parameters
        ----------
        target : str
            Name of the sky location.
        nsample : int, optional
            Number of sample ROIs (uses self.nsample if None).
        
        Returns
        -------
        tuple[SkyCoord, SkyCoord, SkyCoord | None, np.ndarray]
            Target coordinates, source coordinates, sample coordinates (or None),
            and extended source flags array.
        """
        target_coords = self.get_target_coords(target)
        source_coords, extended = self.get_source_coords(target_coords)
        sample_coords = None
        if self.nsample is not None:
            sample_coords = self.get_sample_coords(target_coords)
        return target_coords, source_coords, sample_coords, extended

    def get_target_coords(self, target: str, asArray: bool = False) -> SkyCoord:
        """Extract target coordinates from the counts cube FITS header.
        
        Reads CRVAL1 (RA) and CRVAL2 (Dec) from ccube_00.fits for the sky location.
        
        Parameters
        ----------
        target : str
            Name of the sky location.
        asArray : bool, optional
            If True, return coordinates as single-element arrays (default: False).
        
        Returns
        -------
        SkyCoord
            Target coordinates in ICRS frame.
        """
        target_RA_header = "CRVAL1"
        target_DEC_header = "CRVAL2"
        target_info_file = "ccube_00.fits"
        target_info_filepath = os.path.join(self.sky_location_files_dir, target, target_info_file)
        if not os.path.isfile(target_info_filepath):
            # Backward-compatible fallback for flat output layouts.
            target_info_filepath = os.path.join(self.sky_location_files_dir, target_info_file)
        if not os.path.isfile(target_info_filepath):
            raise FDError(f'No such target file {target_info_filepath}.')
        ra = fits.getval(target_info_filepath, target_RA_header)
        dec = fits.getval(target_info_filepath, target_DEC_header)
        print(f"Target {target} coordinates: RA={ra}, DEC={dec}")
        print(f"Target {target} coordinates in galactic: {SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs').galactic}")

        if asArray:
            ra = [ra]
            dec = [dec]

        return SkyCoord(ra=ra,dec=dec,unit='deg',frame='icrs')

    def get_source_coords(self, target_coords: SkyCoord) -> tuple[SkyCoord, np.ndarray]:
        """Load catalog point sources within ROI footprint.
        
        Retrieves Fermi-LAT 4FGL catalog sources within target_size+source_size+0.5
        degrees of target center.
        
        Parameters
        ----------
        target_coords : SkyCoord
            Target center coordinates.
        
        Returns
        -------
        tuple[SkyCoord, np.ndarray]
            Source coordinates within ROI and extended source flag array.
        """
        if not os.path.isfile(self.source_info_filepath):
            get_catalogs(self.source_info_filepath, False)
        with fits.open(self.source_info_filepath) as sfile:
            ra  = sfile[1].data['RAJ2000']
            dec = sfile[1].data['DEJ2000']
            extended = sfile[1].data['Extended_Source_Name']
        sources = SkyCoord(ra=ra,dec=dec,unit='deg',frame='icrs')
        d2d = target_coords.separation(sources)
        mask = d2d < (self.target_size + self.source_size + 0.5)* u.deg
        return sources[mask], extended[mask]

    def get_event_coords(self, target: str, energy_bin: np.ndarray, final_flag: bool) -> SkyCoord:
        """Extract Fermi-LAT events in specified energy bin.
        
        Loads events from ft1_00.fits, filters by energy range, and returns
        coordinates in ICRS frame.
        
        Parameters
        ----------
        target : str
            Name of the sky location.
        energy_bin : np.ndarray
            Energy range [E_min, E_max] in MeV.
        final_flag : bool
            If True, use closed interval [E_min, E_max]; else [E_min, E_max).
        
        Returns
        -------
        SkyCoord
            Coordinates of events in specified energy bin (ICRS frame).
        
        Raises
        ------
        FDError
            If the event file does not exist.
        """
        events_info_file = "ft1_00.fits"
        events_info_filepath = os.path.join(self.sky_location_files_dir, target, events_info_file)
        if not os.path.isfile(events_info_filepath):
            # Backward-compatible fallback for flat output layouts.
            events_info_filepath = os.path.join(self.sky_location_files_dir, events_info_file)
        if not os.path.isfile(events_info_filepath):
            raise FDError(f'No such event file {events_info_filepath}.')

        with fits.open(events_info_filepath) as efile:
            ra  = efile[1].data['RA']
            dec = efile[1].data['DEC']
            energy = efile[1].data['ENERGY']
        
        event_number = energy.size

        energy_mask = np.full(event_number, True)
        if final_flag:
            energy_mask = np.logical_and(energy_bin[0] <= energy, energy <= energy_bin[1]) #going with [ , ) intervals atm besides last bin which is a [ , ]
        else:
            energy_mask = np.logical_and(energy_bin[0] <= energy, energy < energy_bin[1]) #going with [ , ) intervals atm besides last bin which is a [ , ]
        ra = ra[energy_mask]
        dec = dec[energy_mask]
       
        return SkyCoord(ra=ra,dec=dec,unit='deg',frame='fk5').icrs

    def get_sample_coords(self, target_coords: SkyCoord) -> SkyCoord:
        """Generate random sample ROI centers within target region.
        
        Creates nsample random points distributed within target_size degrees
        of target center using isotropic sampling (Fibonacci sphere-like method).
        
        Parameters
        ----------
        target_coords : SkyCoord
            Center of target region.
        
        Returns
        -------
        SkyCoord
            nsample random sample ROI centers in ICRS frame.
        """
        az = 2.*np.pi * np.random.random(self.nsample)
        cos_pmin = np.cos(self.target_size * np.pi/180.)
        cos_pmax = 1.
        cos_polar = (cos_pmax - cos_pmin)*np.random.random(self.nsample) + cos_pmin
        polar = np.arccos(cos_polar)

        target_frame = target_coords.skyoffset_frame(rotation=az*u.rad)
        sample_wrt_target = SkyCoord(lon=polar,lat=0,unit='rad',
                                    frame=target_frame)

        origin_frame = SkyCoord(0,0,unit='deg',frame='icrs').skyoffset_frame()
        sample = sample_wrt_target.transform_to(origin_frame).icrs
        return sample

    def prune_samples(self, target: str, target_coords: SkyCoord, sample_coords: SkyCoord, source_coords: SkyCoord) -> SkyCoord:
        """Apply all masks to sample ROIs and return valid subset.
        
        Removes sample ROIs that: overlap target boundary, overlap target center,
        or are too close to catalog point sources. Prints rejection statistics.
        
        Parameters
        ----------
        target : str
            Name of the sky location.
        target_coords : SkyCoord
            Target center coordinates. 
        sample_coords : SkyCoord
            All sample ROI center coordinates.
        source_coords : SkyCoord
            Catalog point source coordinates.
        
        Returns
        -------
        SkyCoord
            Pruned sample ROI coordinates passing all mask criteria.
        """
        mask_tar_b  = self.prune_samples_target_boundary(target_coords, sample_coords)
        mask_tar_c  = self.prune_samples_target_center(target_coords, sample_coords)
        mask_source = self.prune_samples_source(sample_coords, source_coords)
        sample_mask = mask_tar_b * mask_tar_c * mask_source
        print(f'ROI sampling for {target}')
        print(f'  Attempted number: {np.size(sample_coords)}')
        print(f'  Overlapping target boundary: {np.sum(~mask_tar_b)}')
        print(f'  Overlapping target center:   {np.sum(~mask_tar_c)}')
        print(f'  Too close to point sources:  {np.sum(~mask_source)}')
        print(f'  Accepted number: {np.sum(sample_mask)}')
        pruned_sample = sample_coords[sample_mask]
        return pruned_sample

    def prune_samples_target_boundary(self, target_coords: SkyCoord, sample_coords: SkyCoord) -> np.ndarray:
        """Mask samples that overlap with target boundary region.
        
        Excludes sample ROIs within (target_size - sample_size) degrees of target.
        
        Parameters
        ----------
        target_coords : SkyCoord
            Target center coordinates.
        sample_coords : SkyCoord
            Sample ROI center coordinates.
        
        Returns
        -------
        np.ndarray
            Boolean mask (True = sample is far enough from target boundary).
        """
        d2d = target_coords.separation(sample_coords)
        mask = d2d < (self.target_size - self.sample_size)*u.deg
        return mask

    def prune_samples_target_center(self, target_coords: SkyCoord, sample_coords: SkyCoord) -> np.ndarray:
        """Mask samples that overlap with target center region.
        
        Excludes sample ROIs within 2*sample_size degrees of target center.
        
        Parameters
        ----------
        target_coords : SkyCoord
            Target center coordinates.
        sample_coords : SkyCoord
            Sample ROI center coordinates.
        
        Returns
        -------
        np.ndarray
            Boolean mask (True = sample is far enough from target center).
        """
        d2d = target_coords.separation(sample_coords)
        mask = d2d > 2.*self.sample_size*u.deg
        return mask

    def prune_samples_source(self, sample_coords: SkyCoord, source_coords: SkyCoord) -> np.ndarray:
        """Mask samples too close to known point sources.
        
        Excludes sample ROIs within (source_size + sample_size) degrees of
        any catalog point source.
        
        Parameters
        ----------
        sample_coords : SkyCoord
            Sample ROI center coordinates.
        source_coords : SkyCoord
            Catalog point source coordinates.
        
        Returns
        -------
        np.ndarray
            Boolean mask (True = sample is far enough from all sources).
        """
        idx_source, idx_sample, d2d, d3d = sample_coords.search_around_sky(
            source_coords, (self.source_size + self.sample_size)*u.deg)

        mask = np.full(np.size(sample_coords),True)
        if np.size(idx_sample) > 0:
            mask[np.unique(idx_sample)] = False
        return mask

    def create_PMF_values(self, pruned_sample: SkyCoord, event_coords: SkyCoord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate probability mass function from sample ROI photon counts.
        
        Counts events within each sample ROI, histograms ROI counts, then
        histograms the histogram to create PMF (arXiv:1108.2914).
        
        Parameters
        ----------
        pruned_sample : SkyCoord
            Pruned sample ROI center coordinates.
        event_coords : SkyCoord
            Event coordinates in energy bin.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Normalized PMF histogram, bin edges, and raw ROI photon counts.
        """

        # find events that lie within sample ROIs
        idx_sample, idx_event, d2d, d3d = event_coords.search_around_sky(
            pruned_sample, self.sample_size*u.deg)

        # count how many events each sample ROI contains
        nbins = np.size(pruned_sample)+1
        hist_event, bins_event = np.histogram(idx_sample, bins=range(nbins))

        # now histogram the sample ROI counts
        nbins = np.max(hist_event)+2
        hist_sample, bins_sample = np.histogram(hist_event, bins=range(nbins))

        print(f'  Largest number of counts in single ROI: {np.max(hist_event)}')
        pmf_hist = hist_sample
        pmf_hist = pmf_hist/np.sum(pmf_hist)
        pmf_bins = bins_sample

        return pmf_hist, pmf_bins, hist_event

    def create_PMF(self, target: str, energy_bin_number: int, make_plots: bool = False) -> tuple[PMFResults, LikelihoodResults]:
        """Generate PMF background models and compute likelihoods.
        
        Orchestrates full PMF analysis: coordinate setup, sample pruning,
        energy-bin PMF generation, and likelihood calculations. Optionally
        creates visualization plots.
        
        Parameters
        ----------
        target : str
            Name of the sky_location galaxy.
        energy_bin_number : int
            Number of energy bins (from config.energy_bins.shape[0]).
        make_plots : bool, optional
            If True, create sky region visualization plots (default: False).
        
        Returns
        -------
        tuple[PMFResults, LikelihoodResults]
            PMF data arrays and likelihood values (independent and covariance).
        """
        coords = self._setup_coordinates(target)

        # Generate PMF histograms across bins
        pmf_results = self._generate_pmf_all_bins(target, coords)

        # Likelihoods (compose into LikelihoodResults)
        log_l_cov = self.L_cov(pmf_results.n_obs, pmf_results.means, np.cov(pmf_results.photon_counts, ddof=0))
        log_l_indep = self.L_indep(pmf_results.n_obs, pmf_results.pmf_data)
        delta_log = log_l_indep - log_l_cov

        if make_plots:
            # Plot using full energy range for events
            all_events = self.get_event_coords(target, np.array([1., 100.]) * 10**3, 1)
            PlotSkyRegion.plot_sky_coords_comprehensive(
                target, coords.target, coords.source, coords.sample,
                coords.pruned_sample, all_events, coords.extended
            )

        # Compose PMFResults and LikelihoodResults inside AnalysisResult
        like_res = LikelihoodResults(
            log_likelihood_indep=log_l_indep,
            log_likelihood_cov=log_l_cov,
            delta_log=delta_log,
        )

        pmf_res = PMFResults(
            pmf_data=pmf_results.pmf_data,
            pmf_list=pmf_results.pmf_list,
            means=pmf_results.means,
            n_obs=pmf_results.n_obs,
            photon_counts=pmf_results.photon_counts,
            max_counts=pmf_results.max_counts,
        )

        return pmf_res, like_res

    def _setup_coordinates(self, target: str) -> CoordinateSet:
        """Prepare and prune coordinates for analysis."""
        target_coords, source_coords, sample_coords, extended = self.set_coords_besides_events(target)
        pruned_sample = self.prune_samples(target, target_coords, sample_coords, source_coords)
        return CoordinateSet(
            target=target_coords,
            source=source_coords,
            sample=sample_coords,
            pruned_sample=pruned_sample,
            extended=extended
        )

    def _process_energy_bin(self, target: str, energy_bin: np.ndarray, pruned_sample: SkyCoord, final_flag: bool) -> EnergyBinResult:
        """Process a single energy bin and return PMF info."""
        event_coords = self.get_event_coords(target, energy_bin, final_flag)
        pmf_hist, pmf_bins, hist_event = self.create_PMF_values(pruned_sample, event_coords)
        max_count = int(pmf_bins[-2])
        return EnergyBinResult(pmf_hist=pmf_hist, counts_per_roi=hist_event, max_count=max_count)

    def _generate_pmf_all_bins(self, target: str, coords: CoordinateSet) -> PMFResults:
        """Generate PMF data across all energy bins."""
        pmf_list: list[np.ndarray] = []
        max_counts = 0
        energy_bin_number = self.energy_bins.shape[0]
        photon_counts_per_energy_bin = np.zeros((energy_bin_number, len(coords.pruned_sample)), dtype=int)

        for j, energy_bin in enumerate(self.energy_bins):
            final_flag = int(j == energy_bin_number - 1)
            bin_res = self._process_energy_bin(target, energy_bin, coords.pruned_sample, final_flag)
            pmf_list.append(bin_res.pmf_hist)
            photon_counts_per_energy_bin[j] = bin_res.counts_per_roi
            max_counts = max(max_counts, bin_res.max_count)

        means = photon_counts_per_energy_bin.mean(axis=1)

        # Build rectangular PMF array
        pmf_data = self._build_pmf_array(pmf_list, max_counts)
        return PMFResults(
            pmf_data=pmf_data,
            means=means,
            photon_counts=photon_counts_per_energy_bin,
            n_obs=self._calculate_nobs(target, coords),
            pmf_list=pmf_list,
            max_counts=max_counts,
        )

    def _build_pmf_array(self, pmf_list: list[np.ndarray], max_counts: int) -> np.ndarray:
        """Create 2D PMF array from list of histograms with padding."""
        pmf_number = len(pmf_list)
        pmf_data = np.zeros((max_counts + 1, pmf_number))
        for i, pmf in enumerate(pmf_list):
            pmf_data[:pmf.size, i] = pmf
        return pmf_data

    def _calculate_nobs(self, target: str, coords: CoordinateSet) -> np.ndarray:
        """Compute observed counts in target region per energy bin."""
        energy_bin_number = self.energy_bins.shape[0]
        n_obs = np.zeros(energy_bin_number)
        for j, energy_bin in enumerate(self.energy_bins):
            final_flag = int(j == energy_bin_number - 1)
            event_coords = self.get_event_coords(target, energy_bin, final_flag)
            mask = coords.target.separation(event_coords) < self.sample_size * u.deg
            n_obs[j] = np.sum(mask)
        return n_obs
    
    def L_indep(self, N_obs: np.ndarray, pmf_data: np.ndarray) -> float:
        """Calculate log-likelihood assuming independent energy bins.
        
        Computes likelihood as product of PMF probabilities across energy bins,
        assuming no correlation between bins.
        
        Parameters
        ----------
        N_obs : np.ndarray
            Observed photon counts per energy bin.
        pmf_data : np.ndarray
            2D PMF array (photon_count x energy_bin).
        
        Returns
        -------
        float
            Log-likelihood value for independent model.
        """
        N_bin = len(self.energy_bins)
        N_obs = np.asarray(N_obs, dtype=int)

        probs = pmf_data[N_obs, np.arange(N_bin)]  

        eps = 1e-8 # Small value to avoid log(0)
        probs = np.maximum(probs, eps)  

        # Sum of log probabilities gives the total log-likelihood
        log_likelihood_indep = np.sum(np.log(probs))

        return log_likelihood_indep

    def L_cov(self, N_obs: np.ndarray, means: np.ndarray, cov: np.ndarray) -> float:
        """Calculate multivariate Gaussian log-likelihood with covariance.
        
        Computes likelihood treating observed counts as samples from multivariate
        normal distribution with given mean and covariance matrix.
        
        Parameters
        ----------
        N_obs : np.ndarray
            Observed photon counts per energy bin.
        means : np.ndarray
            Mean photon counts per energy bin.
        cov : np.ndarray
            Covariance matrix (energy_bins x energy_bins).
        
        Returns
        -------
        float
            Log-likelihood value for covariance model.
        """
        N_bin = len(self.energy_bins) # Number of energy bins
        residuals = N_obs - means

        temp_diag = np.diag(np.diag(cov))

        K_inv = linalg.inv(cov)
        #K_inv = linalg.inv(temp_diag)


        eigenvalues = linalg.eigvalsh(cov)

        if np.any(eigenvalues <= 0):
            print("Warning: Non-positive eigenvalues detected in covariance matrix. Adjusting to absolute values.")
            eigenvalues = np.abs(eigenvalues)

        quad_term = np.sum(residuals[:,None] * K_inv * residuals[None,:])
 
        log_det_term = -0.5 * np.sum(np.log(eigenvalues))
        log_norm_term = -N_bin / 2 * np.log(2 * np.pi)

        log_likelihood = log_norm_term + log_det_term - 0.5 * quad_term

        return log_likelihood

    def save_PMF(self, target: str, max_N_B_i: int, pmf_list: list[np.ndarray], gal_number: int, pmf_number: int) -> None:
        """Save PMF data to structured 2D NumPy array.
        
        Converts list of PMF histograms to 2D array with padding for uneven
        histogram lengths. Method implementation incomplete (save to file disabled).
        
        Parameters
        ----------
        target : str
            Name of the sky_location galaxy.
        max_N_B_i : int
            Maximum photon count in any ROI.
        pmf_list : list[np.ndarray]
            List of PMF histograms (one per energy bin).
        gal_number : int
            Number of galaxies in PMF set.
        pmf_number : int
            Number of energy bins.
        
        Returns
        -------
        None
        """
        #In the following chunk, the pmfs from the list are transferred to a simpler structure (a 2D NumPy array)
        pmf_data = np.zeros((max_N_B_i + 1, pmf_number + 1))
        pmf_data[:, 0] = np.arange(max_N_B_i + 1)
        for i in np.arange(pmf_number):
            pmf_data[:pmf_list[i].size, i + 1] = pmf_list[i]

#         #In the following chunk, the pmfs are output in a text file, with the first argument of "savetxt" dictating where the file is created and under what name
#         np.savetxt(f'PMFdata/pmf{self.bpd}bpd{target}.dat', pmf_data, fmt = '%.15g', delimiter = '\t', header = f"""############################################################
# # MADHAT (Model-Agnostic Dark Halo Analysis Tool) Fermi PMF
# # Ref: Atwood et al. [Fermi-LAT] [arXiv:0902.1089]; P. Bruel et al. [Fermi-LAT] [arXiv:1810.11394]; S. Abdollahi et al. [Fermi-LAT] [arXiv:2201.11184]; J. Ballet et al. [Fermi-LAT] [arXiv:2307.12546]
# #
# # Column 1: number of photons, N
# # Columns 2-{pmf_number+1}: PMF value for N photons for a given sky_location and energy bin (ascending ID# [1-{gal_number}] with energy bin oscillating, with 8bpd energy bins)
# ###########################################################""")

        #print(f'File pmf{self.bpd}bpd{target}.dat saved.')

    def generate_PMF(self, sky_location: str, make_plots: bool = False) -> tuple[PMFResults, LikelihoodResults]:
        """Public API for PMF background analysis.
        
        Wrapper around create_PMF that runs the full pipeline and returns
        results suitable for external use in likelihood calculations.
        
        Parameters
        ----------
        sky_location : str
            Name of the sky_location galaxy.
        make_plots : bool, optional
            If True, create visualization plots (default: False).
        
        Returns
        -------
        tuple[PMFResults, LikelihoodResults]
            PMF data arrays and likelihood values used for model comparison.
        """
        pmf_res, like_res = self.create_PMF(sky_location, self.energy_bins.shape[0], make_plots)
        return pmf_res, like_res


    def generate_NOBS(self, target: str) -> None:
        """Generate observed counts (NOBS) data file for background analysis.
        
        Counts photons in target region per energy bin, loads exposure values,
        and writes formatted TSV file with ID, energy bin, counts, and exposure.
        
        Parameters
        ----------
        target : str
            Name of the sky location.
        
        Returns
        -------
        None
        
        Notes
        -----
        Outputs file: PMFdata/nobs{bpd}bpd{target}.dat
        """
        sky_locations, IDs = self.get_sky_locations()

        #In the following chunk, some numbers are saved
        gal_number = 1
        energy_bin_number = self.energy_bins.shape[0]
        gal_energy_bin_pairs_number = energy_bin_number*gal_number

        #In the following chunk, the counts for each sky location x energy bin pair are determined and saved in an array
        counts = np.zeros(gal_energy_bin_pairs_number)
        i = 0

        target_coords, source_coords, sample_coords, extended = self.set_coords_besides_events(target, None)
        final_flag = 0
        for j in np.arange(energy_bin_number):         
            if j == energy_bin_number-1:
                final_flag = 1
            event_coords = self.get_event_coords(target, self.energy_bins[j], final_flag) 
            target_count_log = target_coords.separation(event_coords) < self.sample_size*u.deg # < or <= ?
            for k in target_count_log:
                if k:
                    counts[i] += 1
            print(f'Energy bin {j+1}, counts: {counts[i]}')
            i += 1

        #In the following line, the location and name of the file with exposure info is provided
        exposures_filepath = 'PMFdata/Exposures_updated.tsv'

        #In the following chunk, sky_location ID numbers and exposures are loaded into arrays
        with open(exposures_filepath, "r") as exposures_file:
            exposures_file_header_length = 1
            for l in np.arange(exposures_file_header_length):
                exposures_file.readline()
            exposures_file_lines_minus_header = [line for line in exposures_file.readlines() if line.strip()]
            exposure_number = np.size(exposures_file_lines_minus_header)
            exposures = np.zeros(exposure_number)
            for e_i in np.arange(exposure_number):
                exposures[e_i] = exposures_file_lines_minus_header[e_i].split()[1]
        IDs_array = np.repeat(IDs, energy_bin_number)
        exposures_array = np.repeat(exposures, energy_bin_number)

        #In the following pair of lines, an array to store a list of number labels for the energy bins as said list will appear in the nobs output file is created
        energy_bin_numbers = np.arange(1, energy_bin_number + 1)
        energy_bin_numbers_array = np.tile(energy_bin_numbers, gal_number)

        NOBS_data = None     

        # In the following line, an array of the "body" of the nobs output file is formed
        for i, sky_location in enumerate(sky_locations):
            if target == sky_location:
                if self.bpd == 1:
                    print("There")
                    NOBS_data = np.column_stack((IDs_array[i], energy_bin_numbers_array, counts, exposures_array[i]))
                else:
                    print("Here")
                    NOBS_data = np.column_stack((IDs_array[(16*i):16*(i+1)], energy_bin_numbers_array, counts, exposures_array[(16*i):16*(i+1)]))

        print(NOBS_data)
        # In the following chunk, an output file with the observed counts and exposures is created, with the first argument of np.savetxt being the name given to said output file
        np.savetxt(f'PMFdata/nobs{self.bpd}bpd{target}.dat', NOBS_data, fmt = '%.15g', delimiter = '\t', header = f"""############################################################
        # MADHAT (Model-Agnostic Dark Halo Analysis Tool) Fermi NOBS
        # Ref: Atwood et al. [Fermi-LAT] [arXiv:0902.1089]; P. Bruel et al. [Fermi-LAT] [arXiv:1810.11394]
        #
        # Column 1: ID# of the dwarf galaxy
        # Column 2: current energy bin# (increases with energy of bin, with 8bpd energy bins)
        # Column 3: number of observed photons, N_O_ij
        # Column 4: exposure = average effective area (A_eff) multiplied by observation time (T_obs)
        ###########################################################""")
    
        print(f'File nobs{self.bpd}bpd{target}.dat saved.')   


class PlotSkyRegion:
    """Plotting utilities for sky region visualization (i.e. which points are masked, and photon count colorbar)."""

    def __init__(self, pmf: 'BgdModelAnalysis', sky_location: str, output_dir: str) -> None:
        """Initialize PlotSkyRegion for visualization.
        
        Parameters
        ----------
        pmf : BgdModelAnalysis
            The PMF analysis object with configuration and coordinates.
        sky_location : str
            Name of the sky location being analyzed.
        output_dir : str
            Output directory for saving plots.
        """
        self.pmf = pmf
        self.sky_location = sky_location
        self.output_dir = output_dir
        self.sample_size = pmf.sample_size

    def plot_PMF(self) -> None:
        """Generate sky region visualization plots for PMF analysis.
        
        Retrieves coordinates from the PMF object and generates comprehensive
        sky region plots showing sample ROI locations, photon density, and
        source overlays.
        
        Returns
        -------
        None
        """
        # Setup coordinates for the sky location
        coords = self.pmf._setup_coordinates(self.sky_location)
        
        # Get all event coordinates across all energy bins for photon density visualization
        # Use full energy range from lowest to highest bin boundary
        full_energy_range = np.array([self.pmf.energy_bins[0, 0], self.pmf.energy_bins[-1, 1]])
        all_event_coords = self.pmf.get_event_coords(self.sky_location, full_energy_range, final_flag=True)
        
        # Generate the comprehensive sky region visualization
        self.plot_sky_coords_comprehensive(
            target=self.sky_location,
            coords=coords,
            all_event_coords=all_event_coords,
            outdir=self.output_dir
        )

    def plot_sky_coords_comprehensive(self, target: str, coords: CoordinateSet, all_event_coords: SkyCoord, outdir: str = 'PMFdata') -> None:
        """Create sky region visualization plots with photon density colormaps.
        
        Generates three separate plots (all samples, 95th percentile cap, pruned
        samples) plus combined panel showing photon counts per sample ROI.
        Color-codes sample ROIs by photon count, overlays sources and events.
        
        Parameters
        ----------
        target : str
            Name of the sky location.
        coords : CoordinateSet
            Dataclass containing all coordinate sets (target, sources, samples, etc.).
        all_event_coords : SkyCoord
            Event coordinates (typically all energies).
        outdir : str, optional
            Output directory for plots (default: 'PMFdata').
        
        Returns
        -------
        None
        """
        target_coords = coords.target
        source_coords = coords.source
        sample_coords = coords.sample
        pruned_coords = coords.pruned_sample
        extended = coords.extended

        plot_data = self._prepare_plot_data(target, target_coords, source_coords, sample_coords, pruned_coords, all_event_coords, extended)

        plot_configs = [
            ('sample', 'All Sample Regions', np.max(plot_data['sample']['counts'])),
            ('sample_95p', 'All Sample Regions (95th %ile)', np.percentile(plot_data['sample']['counts'], 95)),
            ('pruned', 'Pruned Sample Regions', np.max(plot_data['pruned']['counts']))
        ]

        for key, title, vmax in plot_configs:
            fig, ax = plt.subplots(figsize=(8,6), dpi=150)
            if target_coords.ra.degree < 50 or target_coords.ra.degree > 310:
                ra_coords = plot_data[key]['coords'].ra.wrap_at(180 * u.deg).degree
                target_ra = target_coords.ra.wrap_at(180 * u.deg).degree
                source_ra = source_coords.ra.wrap_at(180 * u.deg).degree
                extended_ra = [s.ra.wrap_at(180 * u.deg).degree for s in plot_data['extended']]
            else:
                ra_coords = plot_data[key]['coords'].ra.degree
                target_ra = target_coords.ra.degree
                source_ra = source_coords.ra.degree
                extended_ra = [s.ra.degree for s in plot_data['extended']]

            scatter = ax.scatter(ra_coords, plot_data[key]['coords'].dec.degree,
                                 c=plot_data[key]['counts'], s=1, cmap='viridis',
                                 alpha=0.7, vmax=vmax, label=plot_data[key]['label'])
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Photon Count', rotation=270, labelpad=15)
            ax.scatter(target_ra, target_coords.dec.degree, s=15, c='black', marker='*', alpha=0.8, label='Target')
            ax.scatter(source_ra, source_coords.dec.degree, s=10, c='r', alpha=0.8, label=f'Sources ({len(source_coords)})')
            n_extended = len(plot_data['extended'])
            ax.scatter(extended_ra, [s.dec.degree for s in plot_data['extended']], s=10, c='darkblue', alpha=0.8, label=f'Extended Sources ({n_extended})')
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            n_obs_val = plot_data['target_counts'][0]
            #ax.set_title(f'{target} - {title}')
            ax.legend()
            plt.grid()
            plt.tight_layout()
            plot_name = plot_data[key]['name']
            plt.savefig(f"{outdir}/{target}_{plot_name}_cmap_40.png", dpi=300)
            plt.close()

        fig_combined, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6), dpi=150)
        axes = [ax1, ax2, ax3]
        for i, (key, title, vmax) in enumerate(plot_configs):
            ax = axes[i]
            if target_coords.ra.degree < 50 or target_coords.ra.degree > 310:
                ra_coords = plot_data[key]['coords'].ra.wrap_at(180 * u.deg).degree
                target_ra = target_coords.ra.wrap_at(180 * u.deg).degree
                source_ra = source_coords.ra.wrap_at(180 * u.deg).degree
                extended_ra = [s.ra.wrap_at(180 * u.deg).degree for s in plot_data['extended']]
            else:
                ra_coords = plot_data[key]['coords'].ra.degree
                target_ra = target_coords.ra.degree
                source_ra = source_coords.ra.degree
                extended_ra = [s.ra.degree for s in plot_data['extended']]
            scatter = ax.scatter(ra_coords, plot_data[key]['coords'].dec.degree,
                                 c=plot_data[key]['counts'], s=1, cmap='viridis',
                                 alpha=0.7, vmax=vmax, label=plot_data[key]['label'])
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Photon Count', rotation=270, labelpad=15)
            ax.scatter(target_ra, target_coords.dec.degree, s=15, c='black', marker='*', alpha=0.8, label='Target')
            ax.scatter(source_ra, source_coords.dec.degree, s=10, c='r', alpha=0.8, label=f'Sources ({len(source_coords)})')
            n_extended = len(plot_data['extended'])
            ax.scatter(extended_ra, [s.dec.degree for s in plot_data['extended']], s=10, c='darkblue', alpha=0.8, label=f'Extended Sources ({n_extended})')
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            n_obs_val = plot_data['target_counts'][0]
            ax.set_title(f'{target} - {title}     N_O_ij: {n_obs_val}')
            ax.legend()
            ax.grid()
        plt.tight_layout()
        plt.savefig(f'{outdir}/{target}_combined_cmap_40.png', dpi=300)
        plt.close()

    def _prepare_plot_data(self, target: str, target_coords: SkyCoord, source_coords: SkyCoord, sample_coords: SkyCoord, pruned_coords: SkyCoord, event_coords: SkyCoord, extended: list[SkyCoord]) -> dict:
        """Prepare plot data structures from coordinate and photon count information.
        
        Counts photons in each sample ROI, identifies extended sources,
        and organizes data for visualization.
        
        Parameters
        ----------
        target : str
            Name of the sky_location galaxy.
        target_coords : SkyCoord
            Target center coordinates.
        source_coords : SkyCoord
            Catalog point source coordinates.
        sample_coords : SkyCoord
            Sample ROI center coordinates.
        pruned_coords : SkyCoord
            Pruned sample ROI center coordinates.
        event_coords : SkyCoord
            Event coordinates.
        extended : list[SkyCoord]
            Extended source coordinates.
        
        Returns
        -------
        dict
            Dictionary with photon count arrays, coordinates, and plot metadata
            keyed by sample type ('sample', 'sample_95p', 'pruned').
        """
        idx_sample, idx_event, d2d, d3d = event_coords.search_around_sky(sample_coords,  self.sample_size*u.deg) if sample_coords is not None else (np.array([]), np.array([]), None, None)
        nbins_sample = (np.size(sample_coords) + 1) if sample_coords is not None else 1
        photon_counts_sample, _ = np.histogram(idx_sample, bins=range(nbins_sample)) if sample_coords is not None else (np.array([0]), None)

        idx_pruned, idx_event, d2d, d3d = event_coords.search_around_sky(pruned_coords, self.sample_size*u.deg)
        nbins_pruned = np.size(pruned_coords) + 1
        photon_counts_pruned, _ = np.histogram(idx_pruned, bins=range(nbins_pruned))

        target_mask = target_coords.separation(event_coords) < self.sample_size * u.deg
        total_target_counts = np.sum(target_mask)

        extended_sources = []
        for i, source in enumerate(source_coords):
            coord = SkyCoord(ra=source.ra, dec=source.dec, frame='icrs')
            if extended[i] != '':
                extended_sources.append(coord)

        return {
            'sample': {
                'coords': sample_coords if sample_coords is not None else target_coords,
                'counts': photon_counts_sample,
                'name': 'all',
                'label': 'Sample Regions'
            },
            'sample_95p': {
                'coords': sample_coords if sample_coords is not None else target_coords,
                'counts': photon_counts_sample,
                'name': 'all_percent',
                'label': 'Sample Regions'
            },
            'pruned': {
                'coords': pruned_coords,
                'counts': photon_counts_pruned,
                'name': 'pruned',
                'label': 'Pruned Sample Regions'
            },
            'target_counts': [total_target_counts],
            'extended': extended_sources
        } 