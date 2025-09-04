import os
import numpy as np

from fermipy.gtanalysis import GTAnalysis
from astropy.io import fits
from pmfMaker import createPMF
from configSetup import configure_input_files

# concurrent.futures is used for parallel processing of multiple dwarf galaxies
from concurrent.futures import ProcessPoolExecutor

def main():
    # Setup input files (events.txt and config.yaml) for all dwarf galaxies 
    source_info_file = 'gll_psc_v32.fit'
    defaults = True
    year = '2023'
    
    # Set the random seed for reproducibility (only needed if you want to reproduce the results)
    np.random.seed(34285972)

    # Initialize filepaths and variables
    binning = 1    
    dwarf_files_dir = 'output/'
    IDs_filepath = 'PMFdata/IDs_updated.tsv'

    Nsample = int(1e5) #This is the number of sample regions the program "attempts" to use for each dwarf's pmf (point sources generally cause some regions to be thrown out)
    target_size = 10 #This is the radius of each target region in degrees
    sample_size = 0.5 #This is the radius of each sample region in degrees
    source_size = 0.8 #This is the radius of each point source's exclusionary region in degrees

    # Configure the input files
    # configure_input_files(catalog=source_info_file, defaults=defaults, year=year) 

    # # Initialize the PMF object (used for calculating the PMF and NOBS)
    pmf = createPMF(Nsample, target_size, sample_size, source_size, binning, dwarf_files_dir, IDs_filepath, source_info_file)

    # # Get the list of dwarf galaxies and their IDs (From MADHAT GitHub: https://github.com/MADHATdm/MADHATv2/wiki/Dwarf-ID-Numbers)
    dwarfs, IDs = pmf.get_dwarfs() # Would use this to run the GTAnalysis on all dwarf galaxies

    for dwarf in dwarfs:
        calculate_exposure(dwarf)

    # # If using a single dwarf galaxy, set the dwarf galaxy name here
    # dwarf = 'LEO_VI'

    # # Running the GTAnalysis on the a single dwarf galaxy (Note: This can take a few hours to complete)
    # runGTA(dwarf)

    # # Note: The order of the following functions is important
    # update_exposures(dwarf, pmf) # Update the exposures file (to be used in the PMF creation)
    # update_IDs(dwarf, pmf) # Update the IDs file (to be used in the PMF creation)

    # # Example of how to generate the PMF for a single dwarf galaxy
    # pmf.generate_PMF(dwarf)
    # pmf.generate_NOBS(dwarf)
    
    # # Uncomment to run the GTAnalysis on the dwarf galaxies in parallel (max_workers sets the number of processes, i.e. cores to use)
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     executor.map(parallelize, dwarfs)

def runGTA(dwarf):
    '''Run the GTAnalysis on the dwarf galaxy.
    
    Parameters:
    dwarf (str): The name of the dwarf galaxy.

    Returns:
    None
    '''
    gta = GTAnalysis(f'input/{dwarf}/{dwarf}.yaml',logging={'verbosity': 3})
    gta.setup()

def calculate_exposure(dwarf):
    """
    Calculate the average exposure of the central region of interest (ROI) for a given dwarf galaxy.

    This function loads the exposure map FITS file (bexpmap_roi_00.fits) for the specified dwarf, extracts a square
    region around the ROI center (default opening angle = 1°), and computes the average exposure
    value by first averaging over the spatial pixels in the region and then across all energy bins.

    Parameters
    ----------
    dwarf : str
        Name of the dwarf galaxy. The function expects to find the exposure map file at
        ``output/{dwarf}/bexpmap_roi_00.fits``.

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

    if not os.path.exists(f'output/{dwarf}/bexpmap_roi_00.fits'):
        print(f'Exposure map not found for {dwarf}.')
        return None

    with fits.open(f'output/{dwarf}/bexpmap_roi_00.fits') as bexpmap_roi:
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

def update_exposures(dwarf, pmf):
    '''Update the exposures file with the new data.'''
    dwarfs, IDs = pmf.get_dwarfs()
    if dwarf not in dwarfs:
        if os.path.exists('PMFdata/Exposures_updated.tsv'):
            with open('PMFdata/Exposures_updated.tsv', 'a') as file:
                file.write(f'\n{dwarf}\t{calculate_exposure(dwarf)}\n')
                print(f'{dwarf} added to Exposures_updated.tsv.')
        else:
            with open('PMFdata/Exposures_updated.tsv', 'w') as file:
                file.write(f'Name\tExposure\n')
                file.write(f'{dwarf}\t{calculate_exposure(dwarf)}\n')

def get_new_ID(file_path):
    '''Get the new ID from the file.'''
    with open(file_path, 'r') as file:
        data = file.readlines()
        new_ID = data[-1].split('\t')[0]
    return int(new_ID) + 1

def update_IDs(dwarf, pmf):  
    dwarfs, IDs = pmf.get_dwarfs()
    if dwarf not in dwarfs:
        if os.path.exists('PMFdata/IDs_updated.tsv'):
            with open('PMFdata/IDs_updated.tsv', 'a') as file:
                new_ID = get_new_ID('PMFdata/IDs_updated.tsv')
                file.write(f'\n{new_ID}\t{dwarf}\n')
        else:
            with open('PMFdata/IDs_updated.tsv', 'w') as file:
                file.write(f'#ID\tDwarf Name\n')
                file.write(f'1\t{dwarf}\n')


def parallelize(dwarf):
    '''Parallelize the data analysis for the dwarf galaxies.'''
    runGTA(dwarf)

if __name__ == "__main__":
    main()