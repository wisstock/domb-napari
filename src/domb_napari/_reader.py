import napari
import numpy as np
import oiffile


PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]


def get_reader(path: PathOrPaths) -> Optional[ReaderFunction]:
    if isinstance(path, str) and path.endswith(".oib"):
        print(path)
        return oib_file_reader
    return None


def oib_file_reader(path: PathOrPaths) -> List[LayerData]:
    data = oiffile.OibImread(path)
    data_name = data_name.split('/')[-1]
    data_name = data_name.split('.')[0]
    layer_attributes = {'name': 'My Image', 'colormap': 'turbo'}
    return [(data, layer_attributes, , 'image')]
