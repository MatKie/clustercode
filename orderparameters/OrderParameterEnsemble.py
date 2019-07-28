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
    nematic_op_analysis(self, times=None, style="molecule", principal_axis="inertial", custom_list=None)
        Calculates nematic order parameter and system director for all
        timesteps. 
    """
    
    def __init__(self, coord, traj, selection):
        """
        Parameters
        ---------- 
        coord : string 
            Path to a coordinate-like file. E.g. a gromacs tpr or 
            gro file
        traj : string
            Path to a trajectory like file. E.g. a xtc or trr file. 
            Needs to fit the coord file
        selection : list of string
            Strings used for the definition of species to be studied. Can be atom names or molecule names.
        """
        super().__init__(coord, traj, selection)

    def nematic_op_analysis(self, times=None, style="molecule", principal_axis="inertial", custom_list=None, pbc_style=None):
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
        times : list of floats, optional
            If None, do for whole trajectory. If an interval
            is given like this (t_start, t_end) only do from start
            to end.
        style : string, optional
            "atom" or "molecule". Dependent on this, the 
            cluster_objects attribute is interpreted as molecule
            or atoms within a molecule. 
        principal_axis : string, optional
            "inertial" or "end-to-end". Defines the principal axis as either the end to end vector of the molecule or the dominant axis of the inertial tensor.
        custom_list : list of lists of AtomGroups, optional
            To be specified if the analysis is to be applied to clusters or other custom AtomGroups (i.e. if you want to consider different parts of the same molecule separately). The list should be the same length as the trajectory, each list of AtomGroups representing a trajectory timestep.
        pbc_style : string, optional
            Gromacs pbc definitions: mol, atom, nojump

        Raises
        ------ 
        NotImplementedError
            If an unspecified principal axis is choosen
        
        ToDo
        ----
        Test custom_list feature
        """

        self._set_pbc_style(pbc_style)

        self.universe = self._get_universe(self._coord, traj=self._traj)

        self.selected_species = self._select_species(self.universe,
                                                            style=style)
        self._custom_list_check(times, custom_list)

        if principal_axis == "inertial":
            self.principal_axis = self._get_inertial_axis
        elif principal_axis == "end-to-end":
            self.principal_axis = self._get_end_to_end_vector
        else:
            raise NotImplementedError("{:s} is unspecified molecular axis".format(principal_axis))

        # Convert selected species into a list of atom groups
        selected_species_list = [self._select_species(residue.atoms, style=style) for residue in self.selected_species.residues]

        # Initialise outputs
        self.time_list = []
        self.nematic_op_list = []
        self.system_director_list = []
        sum_saupe_tensor = np.zeros((3,3))

        # Loop over all trajectory times
        for time in self.universe.trajectory:
            if times is not None:
                if time.time > max(times) or time.time < min(times):
                    continue
            
            if custom_list is not None:
                atom_group_list = custom_list[self.custom_list_idx]
                self.custom_list_idx += 1
            else:
                atom_group_list = selected_species_list

            principal_axis_list = self.principal_axis(atom_group_list)
            saupe_tensor = self._get_saupe_tensor(principal_axis_list)
            nematic_op, system_director = self._get_dominant_eig(saupe_tensor)

            # Check how matthias gets times.
            self.time_list.append(time)
            self.nematic_op_list.append(nematic_op)
            self.system_director_list.append(system_director)
            sum_saupe_tensor += saupe_tensor

            print("Nematic order parameter: {:.3f}".format(nematic_op))
            print("****TIME: {:8.2f}".format(time.time))

        # Obtain the ensemble average saupe_tensor
        self.ensemble_saupe_tensor = sum_saupe_tensor/len(self.nematic_op_list)

        # Calculate the mean nematic order parameter and system director from the ensemble average saupe tensor
        self.mean_nematic_op, self.mean_system_director = self._get_dominant_eig(self.ensemble_saupe_tensor)

        print("Mean nematic order parameter: {:.3f}".format(self.mean_nematic_op))
        print("Mean system director: {:s}".format(np.array2string(self.mean_system_director)))

        # Rewind Trajectory to beginning for other analysis
        self.universe.trajectory.rewind()

    def translational_op_analysis():
        return 0

    def _custom_list_check(self, times, custom_list):
        """ Check if custom_list is the correct length relative to the trajectory and times specified. And initialise variable self.custom_list_idx
        
        Parameters
        ----------
        custom_list : list of lists of AtomGroups
            To be specified if the analysis is to be applied to clusters or other custom AtomGroups (i.e. if you want to consider different parts of the same molecule separately). The list should be the same length as the trajectory, each list of AtomGroups representing a trajectory timestep.
        """
        if custom_list is not None:
            if times is not None:
                no_timesteps = int((max(times)-min(times))/self.universe.trajectory.dt+1)
                if len(custom_list) != no_timesteps:
                    raise IndexError("custom_list (len: {:d}) supplied is not the same length as the times in trajectory specified (len: {:d})".format(len(custom_list),no_timesteps))
            else:
                if len(custom_list) != len(self.universe.trajectory):
                    raise IndexError("custom_list (len: {:d}) supplied is not the same length as the trajectory (len: {:d})".format(len(custom_list),len(self.universe.trajectory)))
            self.custom_list_idx = 0


    def _get_inertial_axis(self, atom_group_list):
        """ Get list of principal molecular axis based on the intertia tensor
        
        Parameters
        ----------
        atom_group_list : list of AtomGroups
        
        Returns
        -------
        principal_axis_list : list of numpy arrays
            list of numpy arrays of the principal axis vector
        """
        principal_axis_list = []
        for atom_group in atom_group_list:
            principal_axis_list.append(atom_group.principal_axes()[2])
        
        return principal_axis_list

    def _get_end_to_end_vector(self, atom_group_list):
        """ Get the end-to-end vector of atom group. Note it finds the vector between the first and last value.
        
        Parameters
        ----------
        atom_group_list : list of AtomGroups
        
        Returns
        -------
        end_to_end_list : list of numpy arrays
            list of numpy arrays of the principal axis vector
        """
        end_to_end_list = []
        for atom_group in atom_group_list:
            end_to_end_list.append(atom_group[0]-atom_group[-1])

        return end_to_end_list

    def _get_saupe_tensor(self, principal_axis_list):
        """ Calculate saupe tensor from principal axes
        
        Parameters
        ----------
        principal_axis_list : list of numpy arrays
            list of numpy arrays of the principal axis vector
        
        Returns
        -------
        saupe_tensor : numpy array
        """
        saupe_tensor = np.zeros((3,3))
        half_identity_matrix = np.identity(3)/2.0
        for axis in principal_axis_list:
            saupe_tensor += 1.5 * np.outer(axis,axis) - half_identity_matrix
        saupe_tensor /= len(principal_axis_list)

        return saupe_tensor

    def _get_dominant_eig(self, matrix):
        """ Calculate dominant eigen value and vector
        
        Parameters
        ----------
        matrix : numpy array
        
        Returns
        -------
        eig_val1 : float
            Dominant eigen value (the one with highest magnitude)
        eig_vec1 : numpy array
            Eigen vector corresponding to dominant eigen value
        """
        eig_val, eig_vec = np.linalg.eig(matrix)

        idxs = np.argsort(abs(eig_val))[::-1]  
        eig_val1 = eig_val[idxs][0]
        eig_vec1 = eig_vec[:,idxs][:,0]

        return eig_val1, eig_vec1