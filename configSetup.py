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
    saveYAML('config/Upper' + filename, default)

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
    if os.path.exists(f'config/Upperdefaults{year}.yaml'):
        default = loadYAML(f'config/Upperdefaults{year}.yaml')
        config = loadYAML('config/config.yaml')
    else:
        upperDefaults(f'config/defaults{year}.yaml')
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

def get_file(filename, url):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    # Check if the request was successful
    if response.status_code == 200:
        # Open a local file in binary write mode
        with open(filename, 'wb') as f:
            # Write the content of the response to the file in chunks
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{filename} downloaded.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def get_catalogs(catalog='gll_psc_v32.fit', defaults=True, year='2023'):
    '''Download the Fermi-LAT 4FGL catalog and defaults if True.'''
    cwd = os.getcwd()

    if not os.path.exists(catalog):
        get_file(catalog, f'https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/{catalog}')

    if not os.path.exists(f'config/defaults{year}.yaml') and defaults:
        os.chdir('config/')
        get_file(f'defaults{year}.yaml', f'https://github.com/fermiPy/dmsky/blob/master/dmsky/data/targets/dwarfs/defaults{year}.yaml')

    os.chdir(cwd)

def configure_input_files(catalog='gll_psc_v32.fit', defaults=True, year='2023'):
    """
    Iterate over all subdirectories in the base directory and create events.txt
    file for directories containing files matching the pattern *PH*.fits.
    """
    base_dir = os.getcwd()
    get_catalogs(catalog, defaults, year)

    if not os.path.exists('input/'):
        os.makedirs('input/')
    
    os.chdir('input/')
    for dir in os.listdir('.'):      
        if os.path.isdir(dir):
            dwarf = os.path.basename(dir)
            print(f"Processing {dwarf}...")
            os.chdir(dir)

            if defaults:
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