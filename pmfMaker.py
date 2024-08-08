import os
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits

from configSetup import get_source_catalog

class FDError(Exception):
    '''Custom exception class for file and data errors.'''
    def __init__(self, message):
        super().__init__(message)

class createPMF:
    def __init__(self, Nsample, target_size, sample_size, source_size, binning, dwarf_files_dir, IDs_filepath, source_info_file):
        self.Nsample = Nsample
        self.target_size = target_size
        self.sample_size = sample_size
        self.source_size = source_size
        
        self.energy_bins = np.array([(1.,1.33352143), (1.33352143,1.77827941), (1.77827941,2.37137371), (2.37137371,3.16227766), (3.16227766,4.21696503), (4.21696503,5.62341325), (5.62341325,7.49894209), (7.49894209,10.), (10.,13.33521432), (13.33521432,17.7827941), (17.7827941,23.71373706), (23.71373706,31.6227766), (31.6227766,42.16965034), (42.16965034,56.23413252), (56.23413252,74.98942093), (74.98942093,100.)])*(10**3)
        self.bpd = 8
        if binning == 0:
            self.energy_bins = np.array([(1.,100)])*10**3
            self.bpd =  1
        
        self.dwarf_files_dir = dwarf_files_dir
        self.IDs_filepath = IDs_filepath 
        self.source_info_filepath = source_info_file

    def get_dwarfs(self):
        '''Get the dwarf names and IDs from the IDs file.'''
        if not os.path.isfile(self.IDs_filepath):
            raise FDError(f'No such file {self.IDs_filepath}.')
        
        with open(self.IDs_filepath, "r") as IDs_file:
            next(IDs_file) # Skip the header
            lines = [line.strip().split() for line in IDs_file if line.strip()]

            dwarfs = [line[1].upper() for line in lines]
            IDs = [int(line[0]) for line in lines]

        return dwarfs, IDs

    def set_coords_besides_events(self, target, Nsample=None):
        target_coords = self.get_target_coords(target)
        source_coords = self.get_source_coords(target_coords)
        sample_coords = None
        if self.Nsample is not None:
            sample_coords = self.get_sample_coords(target_coords)
        return target_coords, source_coords, sample_coords

    def get_target_coords(self, target, asArray=False):
        """ Return SkyCoord object for target. """
        target_RA_header = "CRVAL1"
        target_DEC_header = "CRVAL2"
        target_info_file = "ccube_00.fits"
        target_info_filepath = os.path.join(self.dwarf_files_dir, target, target_info_file)

        ra = fits.getval(target_info_filepath, target_RA_header)
        dec = fits.getval(target_info_filepath, target_DEC_header)

        if asArray:
            ra = [ra]
            dec = [dec]

        return SkyCoord(l=ra,b=dec,unit='deg',frame='galactic').icrs

    def get_source_coords(self, target_coords):
        """ Return SkyCoord object for catalog point sources. """
        if not os.path.isfile(self.source_info_filepath):
            get_source_catalog(self.source_info_filepath)
        with fits.open(self.source_info_filepath) as sfile:
            ra  = sfile[1].data['RAJ2000']
            dec = sfile[1].data['DEJ2000']
        sources = SkyCoord(ra=ra,dec=dec,unit='deg',frame='icrs')
        d2d = target_coords.separation(sources)
        mask = d2d < self.target_size * u.deg
        return sources[mask]

    def get_event_coords(self, target, energy_bin, final_flag):
        """ Return SkyCoord object for events selected by gtselect. """
        events_info_file = "ft1_00.fits"
        events_info_filepath = os.path.join(self.dwarf_files_dir, target, events_info_file)
        if not os.path.isfile(events_info_filepath):
            raise FDError(f'No such event file {events_info_filepath}.')

        with fits.open(events_info_filepath) as efile:
            ra  = efile[1].data['RA']
            dec = efile[1].data['DEC']
            energy = efile[1].data['ENERGY']
        
        event_number = energy.size

        energy_mask = np.full(event_number, True) #line inspired by https://stackoverflow.com/questions/21174961/how-do-i-create-a-numpy-array-of-all-true-or-all-false
        if final_flag:
            energy_mask = np.logical_and(energy_bin[0] <= energy, energy <= energy_bin[1]) #going with [ , ) intervals atm besides last bin which is a [ , ]
        else:
            energy_mask = np.logical_and(energy_bin[0] <= energy, energy < energy_bin[1]) #going with [ , ) intervals atm besides last bin which is a [ , ]
        ra = ra[energy_mask]
        dec = dec[energy_mask]
        return SkyCoord(ra=ra,dec=dec,unit='deg',frame='fk5')

    def get_sample_coords(self, target_coords):
        """
        Get 'Nsample' random sample ROIs, with angular size 'sample_size'
        degrees and whose centers lie within 'target_size'.
        Return SkyCoord object of length 'Nsample'.
        """
        az = 2.*np.pi * np.random.random(self.Nsample)
        cos_pmin = np.cos(self.target_size * np.pi/180.)
        cos_pmax = 1.
        cos_polar = (cos_pmax - cos_pmin)*np.random.random(self.Nsample) + cos_pmin
        polar = np.arccos(cos_polar)

        target_frame = target_coords.skyoffset_frame(rotation=az*u.rad)
        sample_wrt_target = SkyCoord(lon=polar,lat=0,unit='rad',
                                    frame=target_frame)

        origin_frame = SkyCoord(0,0,unit='deg',frame='icrs').skyoffset_frame()
        sample = sample_wrt_target.transform_to(origin_frame).icrs
        return sample

    def prune_samples(self, target, target_coords, sample_coords, source_coords):
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

    def prune_samples_target_boundary(self, target_coords, sample_coords):
        """ Create mask against samples that overlap with target boundary. """
        d2d = target_coords.separation(sample_coords)
        mask = d2d < (self.target_size - self.sample_size)*u.deg
        return mask

    def prune_samples_target_center(self, target_coords, sample_coords):
        """ Create mask against samples that overlap with target center. """
        d2d = target_coords.separation(sample_coords)
        mask = d2d > 2.*self.sample_size*u.deg
        return mask

    def prune_samples_source(self, sample_coords, source_coords):
        """ Create mask against samples too close to known point sources. """
        idx_source, idx_sample, d2d, d3d = sample_coords.search_around_sky(
            source_coords, (self.source_size + self.sample_size)*u.deg)

        mask = np.full(np.size(sample_coords),True)
        if np.size(idx_sample) > 0:
            mask[np.unique(idx_sample)] = False
        return mask

    def create_PMF_values(self, pruned_sample, event_coords):
        """ Create probability mass function, as per 1108.2914. """

        # find events that lie within sample ROIs
        idx_sample, idx_event, d2d, d3d = event_coords.search_around_sky(
            pruned_sample, self.sample_size*u.deg)

        # count how many events each sample ROI contains
        nbins = np.size(pruned_sample)+1
        hist_event,bins_event = np.histogram(idx_sample, bins=range(nbins))

        # now histogram the sample ROI counts
        nbins = np.max(hist_event)+2
        hist_sample, bins_sample = np.histogram(hist_event, bins=range(nbins))

        print(f'  Largest number of counts in single ROI: {np.max(hist_event)}')
        pmf_hist = hist_sample
        pmf_hist = pmf_hist/np.sum(pmf_hist)
        pmf_bins = bins_sample

        return pmf_hist, pmf_bins

    def create_PMF(self, target, energy_bin_number):
        max_N_B_i = 0
        pmf_list = [] #inspired by https://stackoverflow.com/questions/3386259/how-to-make-a-multidimension-numpy-array-with-a-varying-row-size

        target_coords, source_coords, sample_coords = self.set_coords_besides_events(target)
        pruned_sample = self.prune_samples(target, target_coords, sample_coords, source_coords)
        final_flag = 0

        # region_photon_counts = {region_idx: np.zeros(len(self.energy_bins)) for region_idx in range(len(pruned_sample))}

        for j in np.arange(energy_bin_number):        
            if j == energy_bin_number-1:
                final_flag = 1
            event_coords = self.get_event_coords(target, self.energy_bins[j], final_flag)
            pmf_hist, pmf_bins = self.create_PMF_values(pruned_sample, event_coords)

            pmf_list.append(pmf_hist)
            local_max_N_B_i = pmf_bins[-2]
            if local_max_N_B_i > max_N_B_i:
                max_N_B_i = local_max_N_B_i

        return max_N_B_i, pmf_list

    def save_PMF(self, target, max_N_B_i, pmf_list, gal_number, pmf_number):
        #In the following chunk, the pmfs from the list are transferred to a simpler structure (a 2D NumPy array)
        pmf_data = np.zeros((max_N_B_i + 1, pmf_number + 1))
        pmf_data[:, 0] = np.arange(max_N_B_i + 1)
        for i in np.arange(pmf_number):
            pmf_data[:pmf_list[i].size, i + 1] = pmf_list[i]

        #In the following chunk, the pmfs are output in a text file, with the first argument of "savetxt" dictating where the file is created and under what name
        np.savetxt(f'PMFdata/pmf{self.bpd}bpd{target}.dat', pmf_data, fmt = '%.15g', delimiter = '\t', header = f"""############################################################
        # MADHAT (Model-Agnostic Dark Halo Analysis Tool) Fermi PMF
        # Ref: Atwood et al. [Fermi-LAT] [arXiv:0902.1089]; P. Bruel et al. [Fermi-LAT] [arXiv:1810.11394]; S. Abdollahi et al. [Fermi-LAT] [arXiv:2201.11184]; J. Ballet et al. [Fermi-LAT] [arXiv:2307.12546]
        #
        # Column 1: number of photons, N
        # Columns 2-{pmf_number+1}: PMF value for N photons for a given dwarf and energy bin (ascending ID# [1-{gal_number}] with energy bin oscillating, with 8bpd energy bins)
        ###########################################################""")

        print(f'File pmf{self.bpd}bpd{target}.dat saved.')

    #commenting out lower lines cuz I just want the fn defintions and such from here for now
    #target_coords, source_coords, event_coords, sample_coords = set_coords(target, Nsample)
    #pmf_hist_34285972, pmf_bins_34285972 = create_PMF(target, target_coords, sample_coords, source_coords, event_coords, Nsample)
    #plt.plot(pmf_bins_34285972[:255], pmf_hist_34285972[:255])
        
    def generate_PMF(self, dwarf):
        #In the following chunk, some numbers are saved (with one being intialized) and the list in which the pmfs will be stored is initialized
        gal_number = 1
        energy_bin_number = self.energy_bins.shape[0]
        pmf_number = energy_bin_number*gal_number

        max_N_B_i, pmf_list = self.create_PMF(dwarf, energy_bin_number)
        self.save_PMF(dwarf, max_N_B_i, pmf_list, gal_number, pmf_number)


    def generate_NOBS(self, target):
        dwarfs, IDs = self.get_dwarfs()
        #In the following chunk, some numbers are saved
        gal_number = 1
        energy_bin_number = self.energy_bins.shape[0]
        gal_energy_bin_pairs_number = energy_bin_number*gal_number

        #In the following chunk, the counts for each dwarf x energy bin pair are determined and saved in an array
        counts = np.zeros(gal_energy_bin_pairs_number)
        i = 0

        target_coords, source_coords, sample_coords = self.set_coords_besides_events(target, None)
        final_flag = 0
        for j in np.arange(energy_bin_number):         
            if j == energy_bin_number-1:
                final_flag = 1
            event_coords = self.get_event_coords(target, self.energy_bins[j], final_flag) 
            target_count_log = target_coords.separation(event_coords) < self.sample_size*u.deg # < or <= ?
            for k in target_count_log:
                if k:
                    counts[i] += 1
            i += 1

        #In the following line, the location and name of the file with exposure info is provided
        exposures_filepath = 'PMFdata/Exposures_updated.tsv'

        #In the following chunk, dwarf ID numbers and exposures are loaded into arrays
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
        for i, dwarf in enumerate(dwarfs):
            if target == dwarf:
                if self.bpd == 1:
                    NOBS_data = np.column_stack((IDs_array[i], energy_bin_numbers_array, counts, exposures_array[i]))
                else:
                    NOBS_data = np.column_stack((IDs_array[(16*i):16*(i+1)], energy_bin_numbers_array, counts, exposures_array[(16*i):16*(i+1)]))

        
        # In the following chunk, an output file with the observed counts and exposures is created, with the first argument of np.savetxt being the name given to said output file
        np.savetxt(f'PMFdata/nobs{self.bpd}bpd{target}.dat', NOBS_data, fmt = '%.15g', delimiter = '\t', header = f"""############################################################
        # MADHAT (Model-Agnostic Dark Halo Analysis Tool) Fermi NOBS
        # Ref: Atwood et al. [Fermi-LAT] [arXiv:0902.1089]; P. Bruel et al. [Fermi-LAT] [arXiv:1810.11394]
        #
        # Column 1: ID# of the dwarf galaxy
        # Column 2: current energy bin# (increases with energy of bin, with 8bpd energy bins)
        # Column 3: number of observed photons, N_O_ij
        # Column 4: exposure = average effective area (A_eff) multiplied by observation time (T_obs)
        ###########################################################""") #a lot of the sites that were helpful for this code cell are saved in "Favorites Bar/Research Stuff/NOBs code (possibly) helpful sites"
    
        print(f'File nobs{self.bpd}bpd{target}.dat saved.')    