import stoch_gpmp
from pathlib import Path
import yaml


# get paths
def get_root_path():
    path = Path(stoch_gpmp.__path__[0]).resolve() / '..'
    return path


def get_assets_path():
    path = get_root_path() / 'assets'
    return path

