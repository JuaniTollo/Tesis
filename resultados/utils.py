
def is_leaf_directory(path, ignore_dirs=None):
    """ Verifica si un directorio es un directorio hoja, ignorando ciertos subdirectorios. """
    if ignore_dirs is None:
        ignore_dirs = []
    if os.path.isdir(path):
        # Lista todos los elementos en el directorio que no están en 'ignore_dirs'
        entries = [entry for entry in os.listdir(path) if entry not in ignore_dirs]
        for entry in entries:
            entry_path = os.path.join(path, entry)
            # Si algún elemento es un directorio, entonces 'path' no es hoja
            if os.path.isdir(entry_path):
                return False
        return True
    return False

def list_leaf_directories(root_dir, ignore_dirs=None):
    """ Lista todos los directorios hoja en el directorio raíz dado, ignorando ciertos subdirectorios. """
    leaf_directories = []
    for root, dirs, files in os.walk(root_dir):
        # Modificar 'dirs' in-place para ignorar directorios específicos
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        # Comprobar si 'root' es un directorio hoja considerando directorios a ignorar
        if is_leaf_directory(root, ignore_dirs):
            leaf_directories.append(root)
    return leaf_directories
