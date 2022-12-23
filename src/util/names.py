import os

def from_vsi_path(name: str):
    no_path = os.path.basename(name)
    return from_vsi(no_path)

def from_vsi(name: str):
    return Name(name[:-4])

def is_vsi(name: str):
    return name[-4:] == '.vsi'

def from_qupath(name: str):
    no_mag = name.split(' - ')[0]
    return from_vsi(no_mag)

def is_qupath(name: str):
    return name.count('- 20x_BF_01') != 0

def is_base(name: str):
    return not is_vsi(name) and not is_qupath(name)

class Name:
    def __init__(self, base_name: str):
        self.base = base_name

    def to_vsi(self):
        return f'{self.base}.vsi'

    def to_qupath(self):
        return f'{self.to_vsi()} - 20x_BF_01'
