import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NeighborSearch
import MDAnalysis.lib.mdamath as mdamath
import warnings
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../../baseuniverse/")
from BaseUniverse import BaseUniverse
#from MDAnalysis.core.groups import ResidueGroup
"""
ToDo:
    Make sure PBC do what we want 
    Ensure behaviour for gro files
    Make paths to BaseUniverse universal784803
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
                     principal_axis="static")
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

    def calc_nematic_op(self, times=None, style="molecule", principal_axis="inertial"):
        """High level function clustering molecules together
        
        Example
        -------
        No Example yet

        Note
        ----
        Not all of the functionality is actually coded yet

        Parameters
        ----------
        cut_off : float, optional
            Minimal distance for two particles to be in the
            same cluster, in Angstroem. Results still depend 
            on the measure parameter.
        time : list of floats, optional
            If None, do for whole trajectory. If an interval
            is given like this (t_start, t_end) only do from start
            to end.
        style : string, optional
            "atom" or "molecule". Dependent on this, the 
            cluster_objects attribute is interpreted as molecule
            or atoms within a molecule. 
        principal_axis : string, optional
            "inertial" or "end-to-end". Defines the principal molecular axis as either the end to end vector of the molecule or the dominant axis of the inertial tensor.

        Raises
        ------ 
        NotImplementedError
            If an unspecified principal axis is choosen
        
        ToDo
        ----
        
        """
        self.universe = self._get_universe(self._coord, traj=self._traj)

        self.selected_species = self._select_species(self.universe,
                                                            style=style)
        if principal_axis == "inertial":
            self.principal_axis = self._get_inertial_axis
        elif principal_axis == "end-to-end":
            self.principal_axis = self._get_end_to_end
        else:
            raise NotImplementedError("{:s} is unspecified principal axis".format(principal_axis))
        
        # Loop over all trajectory times
        for time in self.universe.trajectory:
            if times is not None:
                if time.time > max(times) or time.time < min(times):
                    continue
            principal_axis_list = self.principal_axis(self.selected_species)
            print("****TIME: {:8.2f}".format(time.time))

        # Rewind Trajectory to beginning for other analysis
        self.universe.trajectory.rewind()

    def _get_inertial_axis(self, selected_species):
        """ Get list of principal intertial axis of selected_species
        
        """
        print(selected_species.principal_axes(pbc=True))
        return 0

    def _calc_inertia_tensor():
        return 0