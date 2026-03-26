from pathlib import Path
import yaml
import requests
import logging

# This code is for setting up the config files, and should not need to be changed by the user

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def loadYAML(filename: str) -> dict:
    """Load and parse a YAML configuration file.
    
    Parameters
    ----------
    filename : str
        Path to the YAML file to load.
    
    Returns
    -------
    dict
        Parsed YAML data as a dictionary.
    
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    with path.open('r') as file:
        return yaml.safe_load(file)

def saveYAML(filename: str, data: dict) -> None:
    """Write data to a YAML configuration file.
    
    Creates parent directories if they don't exist and writes YAML with
    preserved key ordering (sort_keys=False).
    
    Parameters
    ----------
    filename : str
        Path where YAML file will be saved.
    data : dict
        Data dictionary to save.
    
    Returns
    -------
    None
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        yaml.safe_dump(data, file, sort_keys=False)

def upperDefaults(filename: str) -> None:
    """Convert dwarf galaxy names in YAML to uppercase.
    
    Loads a YAML file, converts all keys (dwarf names) to uppercase,
    and saves the result to config/Upper<filename>.
    
    Parameters
    ----------
    filename : str
        Path to the input YAML file with mixed-case dwarf names.
    
    Returns
    -------
    None
    """
    default = loadYAML(filename)
    default = {key.upper(): value for key, value in default.items()}
    filename = filename.split('/')[-1]
    saveYAML('config/Upper' + filename, default)

def get_file(filename: str, url: str) -> bool:
    """Download a file from a URL with streaming and timeout.
    
    Creates parent directories as needed and streams file content
    to handle large downloads efficiently.
    
    Parameters
    ----------
    filename : str
        Local path where file will be saved.
    url : str
        URL to download from.
    
    Returns
    -------
    bool
        True if download successful, False if download failed.
    """
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with destination.open('wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Downloaded {destination.name}")
        return True
    except requests.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        return False

def get_catalogs(catalog: str = 'gll_psc_v32.fit', defaults: bool = False, year: str = '2023') -> None:
    """Download Fermi-LAT catalog and optionally default dwarf configurations.
    
    Checks for existing files before downloading; skips if files already present.
    
    Parameters
    ----------
    catalog : str, optional
        Fermi catalog filename (default: 'gll_psc_v32.fit').
    defaults : bool, optional
        If True, download defaults file for the specified year (default: False).
    year : str, optional
        Year for defaults file (default: '2023').
    
    Returns
    -------
    None
    """
    catalog_path = Path(catalog)
    if not catalog_path.exists():
        logging.info(f"Downloading catalog {catalog}...")
        get_file(catalog, f'https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/{catalog}')

    if defaults:
        defaults_path = Path(f'config/defaults{year}.yaml')
        if not defaults_path.exists():
            logging.info(f"Downloading defaults{year}.yaml...")
            # Use raw.githubusercontent.com for direct file download
            get_file(str(defaults_path), f'https://raw.githubusercontent.com/fermiPy/dmsky/master/dmsky/data/targets/dwarfs/defaults{year}.yaml')

def setup_config_yaml(sky_location: str, catalog: str = 'gll_psc_v32.fit', year: str = '2023', defaults: bool = False, input_dir: str = 'input/', output_dir: str = 'output/', config_file: str = 'config.yaml', targets_file: str = 'targets.yaml') -> None:
    """Create fermipy configuration files for a specific sky location.
    
    Loads template config and defaults/targets YAML files, updates with location-specific
    values (RA, Dec, file paths), and saves to both input and output directories.
    Attempts to reuse existing ltcube FITS files if available.
    
    Parameters
    ----------
    sky_location : str
        Name of the sky location.
    catalog : str, optional
        Fermi catalog filename (default: 'gll_psc_v32.fit').
    year : str, optional
        Year for defaults configuration (default: '2023').
    defaults : bool, optional
        If True, use Upperdefaults{year}.yaml; else use targets file (default: False).
    input_dir : str, optional
        Input directory containing sky location data (default: 'input/').
    output_dir : str, optional
        Output directory for fermipy results (default: 'output/').
    config_file : str, optional
        Template config filename (default: 'config.yaml').
    targets_file : str, optional
        Targets YAML filename for custom sky locations (default: 'targets.yaml').
    
    Returns
    -------
    None
    """  
    root_dir = Path.cwd()
    config_dir = root_dir / 'config'
    input_sky_location_dir = Path(input_dir) / sky_location
    output_sky_location_dir = Path(output_dir) / sky_location

    # Load defaults/targets and validate sky_location exists
    if defaults:
        default_file = config_dir / f'Upperdefaults{year}.yaml'
        if not default_file.exists():
            upperDefaults(str(config_dir / f'defaults{year}.yaml'))
        
        default = loadYAML(str(default_file))
        if sky_location not in default:
            logging.warning(f"sky location {sky_location} not found in defaults{year}.yaml")
            return
        config = loadYAML(str(config_dir / config_file))
        ra = default[sky_location][f'default{year}']['ra']
        dec = default[sky_location][f'default{year}']['dec']
    else:
        targets_path = config_dir / targets_file
        if not targets_path.exists():
            logging.error(f"Targets file not found: {targets_path}")
            return
        default = loadYAML(str(targets_path))
        if sky_location not in default:
            logging.warning(f"sky location {sky_location} not found in {targets_file}")
            return
        config = loadYAML(str(config_dir / config_file))
        ra = default[sky_location]['ra']
        dec = default[sky_location]['dec']


    # sets the scfile and evfile for the sky location
    scfile = list(input_sky_location_dir.glob('*SC*.fits'))
    if scfile:
        scfile = str(scfile[0])
    else:
        logging.warning(f"No SC files found for {sky_location}")
        scfile = ""

    config['data']['scfile'] = scfile
    config['data']['evfile'] = str(input_sky_location_dir / 'events.txt')
    config['selection']['ra'] = ra
    config['selection']['dec'] = dec
    config['fileio']['outdir'] = str(output_sky_location_dir)
    config['model']['catalogs'] = [catalog]

    # Check for existing ltcube using absolute path
    ltcube_path = output_sky_location_dir / 'ltcube_00.fits'
    if ltcube_path.exists():
        logging.info(f'Found existing ltcube for {sky_location}, using it.')
        config['data']['ltcube'] = str(ltcube_path)
    
    input_save_path = input_sky_location_dir / f'{sky_location}.yaml'
    output_save_path = output_sky_location_dir / f'{sky_location}.yaml'
    saveYAML(str(input_save_path), config)
    saveYAML(str(output_save_path), config)
    logging.info(f'Saved {sky_location}.yaml to input and output directories')

def configure_input_files(catalog: str = 'gll_psc_v32.fit', input_dir: str = 'input/', output_dir: str = 'output/', config_file: str = 'config.yaml', defaults: bool = True, year: str = '2023', targets_file: str = 'targets.yaml') -> None:
    """Batch configure all sky locations in input directory.
    
    Iterates through location subdirectories, creates events.txt from *PH*.fits files,
    and generates fermipy config YAMLs. Downloads catalogs and defaults if needed.
    
    Parameters
    ----------
    catalog : str, optional
        Fermi catalog filename (default: 'gll_psc_v32.fit').
    input_dir : str, optional
        Root directory containing the sky location subdirectories (default: 'input/').
    output_dir : str, optional
        Root output directory for fermipy results (default: 'output/').
    config_file : str, optional
        Template config YAML filename (default: 'config.yaml').
    defaults : bool, optional
        If True, use defaults file; else use targets file (default: True).
    year : str, optional
        Year for default configurations (default: '2023').
    targets_file : str, optional
        Targets YAML filename for custom sky locations (default: 'targets.yaml').
    
    Returns
    -------
    None
    
    Notes
    -----
    Logs: Number of successfully processed dwarfs and any skipped directories.
    """
    get_catalogs(catalog, defaults, year)

    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"Input directory not found: {input_dir}")
        return

    logging.info(f"Configuring input files in {input_dir}...")
    processed = 0
    for sky_location_dir in input_path.iterdir():
        if not sky_location_dir.is_dir():
            continue

        sky_location = sky_location_dir.name
        logging.info(f"Processing {sky_location}...")

        setup_config_yaml(sky_location, catalog, year, defaults, input_dir, output_dir, config_file, targets_file)

        fits_files = list(sky_location_dir.glob('*PH*.fits'))
        if fits_files:
            events_path = sky_location_dir / 'events.txt'
            with events_path.open('w') as events_file:
                for fits_file in fits_files:
                    events_file.write(f'{fits_file.resolve()}\n')
            logging.info(f"Created {events_path} with {len(fits_files)} files")
            processed += 1
        else:
            logging.warning(f"No *PH*.fits files in {sky_location}, skipping...")
    
    logging.info(f"Configuration complete: processed {processed} sky locations.")

# If running as a script outside of TweedleDEE:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Configure input files for sky locations.')
    parser.add_argument('--catalog', '-c', type=str, default='gll_psc_v32.fit', help='Catalog file name')
    parser.add_argument('--input_dir', '-i', type=str, default='input/', help='Input directory containing sky location subdirectories')
    parser.add_argument('--output_dir', '-o', type=str, default='output/', help='Output directory for configured files')
    parser.add_argument('--config_file', '-f', type=str, default='config.yaml', help='Template config file name')
    parser.add_argument('--targets_file', '-tf', type=str, default='targets.yaml', help='Targets YAML file name')
    parser.add_argument('--defaults', action='store_true', help='Use default configurations')
    parser.add_argument('--year', type=str, default='2023', help='Year for default configurations')
    args = parser.parse_args()
    configure_input_files(catalog=args.catalog, input_dir=args.input_dir, output_dir=args.output_dir, config_file=args.config_file, defaults=args.defaults, year=args.year, targets_file=args.targets_file)