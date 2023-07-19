"""
Miscellaneous useful utils for path manipulation, things that could *almost*
be in os.path, but aren't.

See unit tests in tests/test_path_utils.py.
"""

#%% Imports and constants

import zipfile
import glob
import ntpath
import os
import posixpath
import string
import unicodedata

from datetime import datetime
from typing import Container, Iterable, List, Optional, Tuple

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.gif', '.png', '.tif', '.tiff')

VALID_FILENAME_CHARS = f"~-_.() {string.ascii_letters}{string.digits}"
SEPARATOR_CHARS = r":\/"
VALID_PATH_CHARS = VALID_FILENAME_CHARS + SEPARATOR_CHARS
CHAR_LIMIT = 255


#%% General path functions

def recursive_file_list(base_dir, convert_slashes=True):
    r"""
    Enumerate files (not directories) in [base_dir], optionally converting
    \ to /
    """
    
    all_files = []

    for root, _, filenames in os.walk(base_dir):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            if convert_slashes:
                full_path = full_path.replace('\\', '/')
            all_files.append(full_path)

    return all_files


def split_path(path: str) -> List[str]:
    r"""
    Splits [path] into all its constituent tokens.

    Non-recursive version of:
    http://nicks-liquid-soapbox.blogspot.com/2011/03/splitting-path-to-list-in-python.html

    Examples
    >>> split_path(r'c:\dir\subdir\file.txt')
    ['c:\\', 'dir', 'subdir', 'file.txt']
    >>> split_path('/dir/subdir/file.jpg')
    ['/', 'dir', 'subdir', 'file.jpg']
    >>> split_path('c:\\')
    ['c:\\']
    >>> split_path('/')
    ['/']
    """
    
    parts = []
    while True:
        # ntpath seems to do the right thing for both Windows and Unix paths
        head, tail = ntpath.split(path)
        if head == '' or head == path:
            break
        parts.append(tail)
        path = head
    parts.append(head or tail)
    return parts[::-1]  # reverse


def fileparts(path: str) -> Tuple[str, str, str]:
    r"""
    Breaks down a path into the directory path, filename, and extension.

    Note that the '.' lives with the extension, and separators are removed.

    Examples
    >>> fileparts('file')
    ('', 'file', '')
    >>> fileparts(r'c:\dir\file.jpg')
    ('c:\\dir', 'file', '.jpg')
    >>> fileparts('/dir/subdir/file.jpg')
    ('/dir/subdir', 'file', '.jpg')

    Returns:
        p: str, directory path
        n: str, filename without extension
        e: str, extension including the '.'
    """
    
    # ntpath seems to do the right thing for both Windows and Unix paths
    p = ntpath.dirname(path)
    basename = ntpath.basename(path)
    n, e = ntpath.splitext(basename)
    return p, n, e


def insert_before_extension(filename: str, s: str = '') -> str:
    """
    Insert string [s] before the extension in [filename], separated with '.'.

    If [s] is empty, generates a date/timestamp. If [filename] has no extension,
    appends [s].

    Examples
    >>> insert_before_extension('/dir/subdir/file.ext', 'insert')
    '/dir/subdir/file.insert.ext'
    >>> insert_before_extension('/dir/subdir/file', 'insert')
    '/dir/subdir/file.insert'
    >>> insert_before_extension('/dir/subdir/file')
    '/dir/subdir/file.2020.07.20.10.54.38'
    """
    
    assert len(filename) > 0
    if len(s) == 0:
        s = datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
    name, ext = os.path.splitext(filename)
    return f'{name}.{s}{ext}'


def top_level_folder(p: str, windows: Optional[bool] = None) -> str:
    r"""
    Gets the top-level folder from path [p].

    This function behaves differently for Windows vs. Unix paths. Set
    windows=True if [p] is a Windows path. Set windows=None (default) to treat
    [p] as a native system path.

    On Windows, will use the top-level folder that isn't the drive.
    >>> top_level_folder(r'c:\blah\foo')
    'c:\blah'

    On Unix, does not include the leaf node.
    >>> top_level_folder('/blah/foo')
    '/blah'
    """
    
    if p == '':
        return ''

    default_lib = os.path  # save default os.path
    if windows is not None:
        os.path = ntpath if windows else posixpath

    # Path('/blah').parts is ('/', 'blah')
    parts = split_path(p)

    drive = os.path.splitdrive(p)[0]
    if len(parts) > 1 and (
            parts[0] == drive
            or parts[0] == drive + '/'
            or parts[0] == drive + '\\'
            or parts[0] in ['\\', '/']):
        result = os.path.join(parts[0], parts[1])
    else:
        result = parts[0]

    os.path = default_lib  # restore default os.path
    return result


#%% Image-related path functions

def is_image_file(s: str, img_extensions: Container[str] = IMG_EXTENSIONS
                  ) -> bool:
    """Checks a file's extension against a hard-coded set of image file
    extensions.
    """
    ext = os.path.splitext(s)[1]
    return ext.lower() in img_extensions


def find_image_strings(strings: Iterable[str]) -> List[str]:
    """Given a list of strings that are potentially image file names, looks for
    strings that actually look like image file names (based on extension).
    """
    return [s for s in strings if is_image_file(s)]


def find_images(dirname: str, recursive: bool = False) -> List[str]:
    """Finds all files in a directory that look like image file names. Returns
    absolute paths.
    """
    if recursive:
        strings = glob.glob(os.path.join(dirname, '**', '*.*'), recursive=True)
    else:
        strings = glob.glob(os.path.join(dirname, '*.*'))
    return find_image_strings(strings)


#%% Filename cleaning functions

def clean_filename(filename: str, whitelist: str = VALID_FILENAME_CHARS,
                   char_limit: int = CHAR_LIMIT) -> str:
    r"""
    Removes non-ASCII and other invalid filename characters (on any
    reasonable OS) from a filename, then trims to a maximum length.

    Does not allow :\/, use clean_path if you want to preserve those.

    Adapted from
    https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
    """
    
    # keep only valid ascii chars
    cleaned_filename = (unicodedata.normalize('NFKD', filename)
                        .encode('ASCII', 'ignore').decode())

    # keep only whitelisted chars
    cleaned_filename = ''.join([c for c in cleaned_filename if c in whitelist])
    return cleaned_filename[:char_limit]


def clean_path(pathname: str, whitelist: str = VALID_PATH_CHARS,
               char_limit: int = CHAR_LIMIT) -> str:
    """
    Removes non-ASCII and other invalid path characters (on any reasonable
    OS) from a path, then trims to a maximum length.
    """
    
    return clean_filename(pathname, whitelist=whitelist, char_limit=char_limit)


def flatten_path(pathname: str, separator_chars: str = SEPARATOR_CHARS) -> str:
    """
    Removes non-ASCII and other invalid path characters (on any reasonable
    OS) from a path, then trims to a maximum length. Replaces all valid
    separators with '~'.
    """
    
    s = clean_path(pathname)
    for c in separator_chars:
        s = s.replace(c, '~')
    return s


#%% Platform-independent way to open files in their associated application

import sys,subprocess

def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


#%% zipfile management functions

def unzip_file(input_file, output_folder=None):
    """
    Unzip a zipfile to the specified output folder, defaulting to the same location as
    the input file    
    """
    
    if output_folder is None:
        output_folder = os.path.dirname(input_file)
        
    with zipfile.ZipFile(input_file, 'r') as zf:
        zf.extractall(output_folder)
