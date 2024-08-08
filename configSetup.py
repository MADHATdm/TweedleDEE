import os
import glob
import yaml
import requests

def loadYAML(filename):
    '''Load a YAML file and return the data.
    
    Parameters:
    filename (str): The name of the file to load.
    
    Returns:
    data (dict): The data from the file.
    '''
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

def saveYAML(filename, data):
    '''Save data to a YAML file.'''
    with open(filename, 'w') as file:
        yaml.safe_dump(data, file)

def upperDefaults(filename):
    '''Convert the keys (dwarf names) in a YAML file to uppercase.'''
    default = loadYAML(filename)
    default = {key.upper(): value for key, value in default.items()}
    filename = filename.split('/')[-1]
    saveYAML('Upper' + filename, default)

def setup_config_yaml(dwarf, catalog='gll_psc_v32.fit', year='2023'):
    '''Setup the config yaml file for the dwarf galaxy to be used with runGTA().

    This function loads the template config file (config.yaml) and the default
    values for the dwarf galaxy (Upperdefaults2023.yaml). It then sets the correct
    values for the dwarf galaxy (ra, dec, scfile, evfile, outdir) and saves them to
    a new yaml file named after the dwarf galaxy to be used in runGTA().
    
    Parameters:
    dwarf (str): The name of the dwarf galaxy.
    catalog (str): The name of the catalog to use.
    year (str): The year of the data.
    
    Returns:
    None
    '''  
    cwd = os.getcwd()
    os.chdir('../..')
    try:
        default = loadYAML(f'config/Upperdefaults{year}.yaml')
        config = loadYAML('config/config.yaml')
    except FileNotFoundError:
        try:
            upperDefaults(f'config/defaults{year}.yaml')
        except FileNotFoundError:
            raise FileNotFoundError(f'config/defaults{year}.yaml not found.')
        default = loadYAML(f'config/Upperdefaults{year}.yaml')
        config = loadYAML('config/config.yaml')

    ra = default[dwarf][f'default{year}']['ra']
    dec = default[dwarf][f'default{year}']['dec']

    # sets the scfile and evfile for the dwarf galaxy
    os.chdir(f'input/{dwarf}/')
    scfile = os.popen('echo -n $(ls *SC*.fits)').read()
    scfile = f'input/{dwarf}/' + scfile
        
    config['data']['scfile'] = scfile
    config['data']['evfile'] = f'input/{dwarf}/events.txt'
    config['selection']['ra'] = ra
    config['selection']['dec'] = dec
    config['fileio']['outdir'] = f'output/{dwarf}'

    # Allows the user to specify a different catalog
    if catalog != 'gll_psc_v32.fit':
        config['model']['catalogs'] = [catalog]

    saveYAML(f'{dwarf}.yaml', config)
    print(f'{dwarf}.yaml saved.')
    os.chdir(cwd)

def get_source_catalog(catalog='gll_psc_v32.fit'):
    '''Download the Fermi-LAT 4FGL catalog.'''
    # Send a GET request to the URL
    response = requests.get(f'https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/{catalog}', stream=True)
    # Check if the request was successful
    if response.status_code == 200:
        # Open a local file in binary write mode
        with open(catalog, 'wb') as f:
            # Write the content of the response to the file in chunks
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Source catalog: {catalog} downloaded.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def get_IDs_updated():
    '''Get the IDs_updated.tsv file.'''
    response = requests.get('https:...../PMFdata/IDs_updated.tsv')
    if response.status_code == 200:
        with open('IDs_updated.tsv', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("IDs_updated.tsv downloaded.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def get_Exposures_updated():
    '''Get the Exposures_updated.tsv file.'''
    response = requests.get('https:...../PMFdata/Exposures_updated.tsv')
    if response.status_code == 200:
        with open('Exposures_updated.tsv', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Exposures_updated.tsv downloaded.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def configure_input_files(catalog='gll_psc_v32.fit', get_IDs=False):
    """
    Iterate over all subdirectories in the base directory and create events.txt
    file for directories containing files matching the pattern *PH*.fits.
    """
    base_dir = os.getcwd()
    get_source_catalog(catalog)
    
    if get_IDs:
        get_IDs_updated()
        get_Exposures_updated()

    os.chdir('input/')
    for dir in os.listdir('.'):      
        if os.path.isdir(dir):
            dwarf = os.path.basename(dir)
            print(f"Processing {dwarf}...")
            os.chdir(dir)
            setup_config_yaml(dwarf, catalog)
            
            fits_files = glob.glob('*PH*.fits')
            if fits_files:
                with open('events.txt', 'w') as events_file:
                    for fits_file in fits_files:
                        events_file.write(f'input/{dwarf}/{fits_file}\n')
                print(f"Created events.txt for {dwarf}")
            else:
                print(f"No matching files in {dwarf}, skipping...")
            os.chdir('..')
    os.chdir(base_dir)