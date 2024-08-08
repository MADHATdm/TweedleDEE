import os
import numpy as np

from fermipy.gtanalysis import GTAnalysis
from astropy.io import fits
from pmfMaker import createPMF
from configSetup import configure_input_files

# concurrent.futures is used for parallel processing of multiple dwarf galaxies
from concurrent.futures import ProcessPoolExecutor


def runGTA(dwarf):
    '''Run the GTAnalysis on the dwarf galaxy.
    
    Parameters:
    dwarf (str): The name of the dwarf galaxy.

    Returns:
    None
    '''
    gta = GTAnalysis(f'input/{dwarf}/{dwarf}.yaml',logging={'verbosity': 3})
    gta.setup()

def analyze_data(dwarf):
    '''Analyze the data from the dwarf galaxy.'''
    # Open the FITS file
    with fits.open(f'output/{dwarf}/bexpmap_roi_00.fits') as bexpmap_roi:
        # print(bexpmap_roi.info())

        # Slice for the middle
        data = bexpmap_roi[0].data[:,189:209,189:209]

        # Calculate the exposure (NEED REFERENCE)
        numpy2=(1/16)*np.sum(data,0)
        numpy3=(1/20)*np.sum(numpy2,0)
        numpy4=(1/20)*np.sum(numpy3,0) #exposure

    return numpy4

def update_exposures(dwarf):
    Aeff = analyze_data(dwarf)

    if 'PMFdata/Exposures_updated.tsv' not in os.listdir():
        with open('PMFdata/Exposures_updated.tsv', 'w') as file:
            file.write(f'{dwarf}\t{Aeff}\n')
    else:
        # Open Exposures_updated.tsv
        with open('PMFdata/Exposures_updated.tsv', 'a') as file:
            file.write(f'{dwarf}\t{Aeff}\n')

def update_IDs(dwarf):
    def count_lines(filename):
        with open(filename) as file:
            return len(file.readlines())
    if 'PMFdata/IDs_updated.tsv' not in os.listdir():
        with open('PMFdata/IDs_updated.tsv', 'w') as file:
            file.write(f'{dwarf}\t{1}\n')
    else:
        # Open IDs_updated.tsv
        with open('PMFdata/IDs_updated.tsv', 'a') as file:
            file.write(f'{dwarf}\t{count_lines('PMFdata/IDs_updated') - 1}\n')

def parallelize(dwarf):
    '''Parallelize the data analysis for the dwarf galaxies.'''
    runGTA(dwarf)

def main():
    # Setup input files (events.txt and config.yaml) for all dwarf galaxies 
    # configure_input_files(catalog='gll_psc_v32.fit', True) # Set False to not retrieve the IDs and Exposures files from MADHAT
    
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

    # Initialize the PMF object (used for calculating the PMF and NOBS)
    pmf = createPMF(Nsample, target_size, sample_size, source_size, binning, dwarf_files_dir, IDs_filepath)

    # Get the list of dwarf galaxies and their IDs (From MADHAT GitHub: https://github.com/MADHATdm/MADHATv2/wiki/Dwarf-ID-Numbers)
    dwarfs, IDs = pmf.get_dwarfs() # Would use this to run the GTAnalysis on all dwarf galaxies

    # If using a single dwarf galaxy, set the dwarf galaxy name here
    dwarf = 'LEO_VI'

    # Running the GTAnalysis on the a single dwarf galaxy (Note: This can take a few hours to complete)
    runGTA(dwarf)

    update_exposures(dwarf) # Update the exposures file (to be used in the PMF creation)
    update_IDs(dwarf) # Update the IDs file (to be used in the PMF creation)

    # Example of how to generate the PMF for a single dwarf galaxy
    pmf.generate_PMF(dwarf)
    pmf.generate_NOBS(dwarf)
    
    # # Uncomment to run the GTAnalysis on the dwarf galaxies in parallel (max_workers sets the number of processes, i.e. cores to use)
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     executor.map(parallelize, dwarfs)

       
if __name__ == "__main__":
    main()