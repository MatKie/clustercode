import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NeighborSearch
import warnings
import matplotlib as mlp
import matplotlib.pyplot as plt
import sys
sys.path.append("../../baseuniverse/")
from BaseUniverse import BaseUniverse
#from MDAnalysis.core.groups import ResidueGroup
"""
ToDo:
    Make sure PBC do what we want 
    Ensure behaviour for gro files
    Make paths to BaseUniverse universal
"""

class OrderParameterEnsemble(BaseUniverse):
    """A class used to perform analysis on Clusters of molecules

    Attributes
    ----------
    universe : MDAnalysis universe object
        Universe of the simulated system 
    cluster_objects : list of str
        Strings used for the definition of species which form 
        clusters. Can be atom names or molecule names.
    cluster_list : list of list of MDAnalysis ResidueGroups
        a list of ResidueGroups forms one cluster at a given time,
        for multiple times a list of these lists is produced.

    Methods
    -------
    cluster_analysis(cut_off=7.5, style="atom", measure="b2b", 
                     algorithm="static")
        Calculates which molecules are clustering together for all
        timesteps. 
    """
    
    def __init__(self, coord, traj, cluster_objects):
        """
        Parameters
        ---------- 
        coord : string 
            Path to a coordinate-like file. E.g. a gromacs tpr or 
            gro file
        traj : string
            Path to a trajectory like file. E.g. a xtc or trr file. 
            Needs to fit the coord file
        cluster_objects : list of string
            Strings used for the definition of species which form 
            clusters. Can be atom names or molecule names.
        """
        super().__init__(coord, traj, cluster_objects)