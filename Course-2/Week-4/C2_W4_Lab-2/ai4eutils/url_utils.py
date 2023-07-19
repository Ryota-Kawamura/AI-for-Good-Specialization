#
# url_utils.py
#
# Frequently-used functions for downloading or manipulating URLs
#

#%% Imports

import re
import urllib
import os
import tempfile

from urllib.parse import urlparse

# pip install progressbar2
import progressbar

ai4e_utils_temp_dir = None
max_path_len = 255


#%% Download functions


class DownloadProgressBar():
    """
    https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    """
    
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(max_value=total_size)
            self.pbar.start()
            
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
            

def get_temp_folder(preferred_name='ai4eutils'):

    global ai4e_utils_temp_dir
    
    if ai4e_utils_temp_dir is None:
        ai4e_utils_temp_dir = os.path.join(tempfile.gettempdir(),preferred_name)
        os.makedirs(ai4e_utils_temp_dir,exist_ok=True)
        
    return ai4e_utils_temp_dir
    
           
def download_url(url, destination_filename=None, progress_updater=None, 
                 force_download=False, verbose=True):
    """
    Download a URL to a file.  If no file is specified, creates a temporary file, 
    with a semi-best-effort to avoid filename collisions.
    
    Prints some diagnostic information and makes sure to omit SAS tokens from printouts.
    
    progress_updater can be "None", "True", or a specific callback.
    """
    
    if progress_updater is not None and isinstance(progress_updater,bool):
        if not progress_updater:
            progress_updater = None
        else:
            progress_updater = DownloadProgressBar()
            
    url_no_sas = url.split('?')[0]
        
    if destination_filename is None:
        target_folder = get_temp_folder()
        url_without_sas = url.split('?', 1)[0]
        
        # This does not guarantee uniqueness, hence "semi-best-effort"
        url_as_filename = re.sub(r'\W+', '', url_without_sas)
        n_folder_chars = len(ai4e_utils_temp_dir)
        if len(url_as_filename) + n_folder_chars > max_path_len:
            print('Warning: truncating filename target to {} characters'.format(max_path_len))
            url_as_filename = url_as_filename[-1*(max_path_len-n_folder_chars):]
        destination_filename = \
            os.path.join(target_folder,url_as_filename)
        
    if (not force_download) and (os.path.isfile(destination_filename)):
        if verbose:
            print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url_no_sas)))
    else:
        if verbose:
            print('Downloading file {} to {}'.format(os.path.basename(url_no_sas),destination_filename),end='')
        urllib.request.urlretrieve(url, destination_filename, progress_updater)  
        assert(os.path.isfile(destination_filename))
        nBytes = os.path.getsize(destination_filename)
        if verbose:
            print('...done, {} bytes.'.format(nBytes))
        
    return destination_filename


def download_relative_filename(url, output_base, verbose=False):
    """
    Download a URL to output_base, preserving relative path
    """
    
    p = urlparse(url)
    # remove the leading '/'
    assert p.path.startswith('/'); relative_filename = p.path[1:]
    destination_filename = os.path.join(output_base,relative_filename)
    download_url(url, destination_filename, verbose=verbose)
    

