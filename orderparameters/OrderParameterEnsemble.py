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
    nematic_op_analysis(self, times=None, style="molecule", principal_axis="inertial", custom_traj=None)
        Calculates nematic order parameter and system director for all
        timesteps. 
    translational_op_analysis(self, director, times=None, style="molecule", pbc_style=None, trans_op_style="com", search_param=None, custom_traj=None)
        Calculates translational order parameter and translational spacing for input director or list of directors.
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

    def nematic_op_analysis(self, times=None, style="molecule", principal_axis="inertial", custom_traj=None, pbc_style=None):
        """High level function for calculating the nematic order parameter
        
        Example
        -------
        No Example yet

        Parameters
        ----------
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
        custom_traj : list of list of AtomGroup, optional
            To be specified if the analysis is to be applied to clusters or other custom AtomGroups (i.e. if you want to consider different parts of the same molecule separately). The list should be the same length as the trajectory, each list of AtomGroups representing a trajectory timestep.
        pbc_style : string, optional
            Gromacs pbc definitions: mol, atom, nojump

        Raises
        ------ 
        NotImplementedError
            If an unspecified principal axis is choosen
        IndexError
            If custom_traj is different length from trajetory or times
        
        ToDo
        ----
        Test custom_traj feature
        """

        self._set_pbc_style(pbc_style)

        self.universe = self._get_universe(self._coord, traj=self._traj)

        self.selected_species = self._select_species(self.universe,
                                                            style=style)
        self._custom_traj_check(times, custom_traj)

        # Select which principal axis in the AtomGroup to use
        if principal_axis == "inertial":
            self.principal_axis = self._get_inertial_axis
        elif principal_axis == "end-to-end":
            self.principal_axis = self._get_end_to_end_vector
        else:
            raise NotImplementedError("{:s} is unspecified molecular axis".format(principal_axis))

        # If custom_traj is not specified initialise select_species as a list of AtomGroups (one for each residue)
        if custom_traj is None:
            selected_species_list = [self._select_species(residue.atoms, style=style) for residue in self.selected_species.residues]

        # Initialise outputs
        self.nematic_op_list = []
        self.system_director_list = []
        sum_saupe_tensor = np.zeros((3,3))

        # Loop over all trajectory times
        for time in self.universe.trajectory:
            if times is not None:
                if time.time > max(times) or time.time < min(times):
                    continue
            
            # Either use custrom_traj or the selected species
            if custom_traj is not None:
                atom_group_list = custom_traj[self.custom_traj_idx]
                self.custom_traj_idx+= 1
            else:
                atom_group_list = selected_species_list

            principal_axis_list = self.principal_axis(atom_group_list)
            saupe_tensor = self._get_saupe_tensor(principal_axis_list)
            nematic_op, system_director = self._get_dominant_eig(saupe_tensor)

            self.nematic_op_list.append(nematic_op)
            self.system_director_list.append(system_director)
            sum_saupe_tensor += saupe_tensor

            print("****TIME: {:8.2f}".format(time.time))
            print("Nematic order parameter: {:.3f}".format(nematic_op))
            
        # Obtain the ensemble average saupe_tensor
        self.ensemble_saupe_tensor = sum_saupe_tensor/len(self.nematic_op_list)

        # Calculate the mean nematic order parameter and system director from the ensemble average saupe tensor
        self.mean_nematic_op, self.mean_system_director = self._get_dominant_eig(self.ensemble_saupe_tensor)

        self.stdev_nematic_op = np.std(self.nematic_op_list)

        print("****MEAN:")
        print("Mean nematic order parameter: {:.3f} +/- {:.3f}".format(self.mean_nematic_op,self.stdev_nematic_op))
        print("Mean system director: {:s}".format(np.array2string(self.mean_system_director)))

        # Rewind Trajectory to beginning for other analysis
        self.universe.trajectory.rewind()

    def translational_op_analysis(self, director, times=None, style="molecule", pbc_style=None, trans_op_style="com", search_param=[0.1, 50, 500], custom_traj=None, plot=False):
        """High level function for calculating the translational order parameter
        
        Example
        -------
        No Example yet

        Parameters
        ----------
        director : numpy array(3) or list of numpy array(3)
            Specify one if the same director is applied to all timesteps in the trajectory or a list if a different director is used each timestep. Note it should be a unit vector or list of unit vectors
        times : list of floats, optional
            If None, do for whole trajectory. If an interval
            is given like this (t_start, t_end) only do from start
            to end.
        style : string, optional
            "atom" or "molecule". Dependent on this, the 
            cluster_objects attribute is interpreted as molecule
            or atoms within a molecule. 
        pbc_style : string, optional
            Gromacs pbc definitions: mol, atom, nojump
        trans_op_style : string, optional
            Center of mass ("com") or "atom"
        search_space : [float, float, int], optional
            Specify [min, max, n_points], where min and max are the minimum and maximum translational spacings considered in Angstrom and n_points is the number of points between these values.
        custom_traj : list of list of AtomGroup
            To be specified if the analysis is to be applied to clusters or other custom AtomGroups (i.e. if you want to consider different parts of the same molecule separately). The list should be the same length as the trajectory, each list of AtomGroups representing a trajectory timestep.
        plot :boolean, optional
            If True the translational order parameter is plotted as a function of the spacing for the first time in the trajectory or specified in times.

        Raises
        ------ 
        TypeError
            If director has the wrong type / form
        IndexError
            If custom_traj is different length from trajetory or times
            If director is different length from trajetory or times
            If search_param has index other than 3.
        NotImplementedError
            If an unspecified trans_op_style is choosen
        
        ToDo
        ----
        Test custom_traj feature
        Format/cleanup plot (possibly with external function)
        """
        self._set_pbc_style(pbc_style)

        self.universe = self._get_universe(self._coord, traj=self._traj)

        self.selected_species = self._select_species(self.universe,
                                                            style=style)
        self._custom_traj_check(times, custom_traj)

        director = self._director_check(times, director)
        
        # Set search_param if it is not specified by user. If it is specified check its length and make sure the minimum value is not zero.
        if search_param is None:
            search_param = [0.1, 50, 500]
        else:
            if len(search_param) != 3:
                IndexError("len(search_param) is not 3")
            elif search_param[0] == 0:
                search_param[0] += 0.01

        spacing_array = np.linspace(*search_param)

        # Convert selected species into a list of atom groups
        if custom_traj is None:
            selected_species_list = [self._select_species(residue.atoms, style=style) for residue in self.selected_species.residues]

        # Initialise outputs
        self.trans_op_list = []
        self.trans_spacing_list = []
        director_idx = 0

        # Loop over all trajectory times
        for time in self.universe.trajectory:
            if times is not None:
                if time.time > max(times) or time.time < min(times):
                    continue

            # Get center of mass array for custom_traj or selected_species (in either center of mass of the molecule or just the atom positions)
            if trans_op_style is "com":
                if custom_traj is not None:
                    atom_group_list = custom_traj[self.custom_traj_idx]
                    self.custom_traj_idx += 1
                else:
                    atom_group_list = selected_species_list
                center_of_mass_array = self._get_center_of_mass(atom_group_list)
            elif trans_op_style is "atom":
                if custom_traj is not None:
                    center_of_mass_array = custom_traj[self.custom_traj_idx][0].positions
                    for atom_group in custom_traj[self.custom_traj_idx][1:]:
                        center_of_mass_array = np.vstack(atom_group.positions)
                    self.custom_traj_idx += 1
                else:
                    center_of_mass_array = self.selected_species.positions
            else:
                raise NotImplementedError("{:s} is unspecified trans_op_style".format(trans_op_style))
            
            # Optimise the translational order parameter and determine the spacing
            trans_op_k = []
            spacing_list = []
            for spacing in spacing_array:
                k_vector = 2*np.pi/spacing * director[director_idx]
                trans_op_k.append(self._get_system_fourier_transform_mod(center_of_mass_array, k_vector)/float(len(center_of_mass_array)))
            
            idx_max = np.argmax(trans_op_k)
            trans_op = trans_op_k[idx_max]
            trans_spacing = spacing_array[idx_max]

            print("****TIME: {:8.2f}".format(time.time))
            print("Translational order parameter: {:.3f}".format(trans_op))
            print("Translational spacing: {:.3f} Angstrom".format(trans_spacing))
            
            self.trans_op_list.append(trans_op)
            self.trans_spacing_list.append(trans_spacing)

            director_idx += 1

            if plot:
                plt.plot(spacing_array,trans_op_k)
                plt.show()
                plot=False

        # Calculate mean and standard deviations
        self.mean_trans_op = np.mean(self.trans_op_list)
        self.stdev_trans_op = np.std(self.trans_op_list)
        self.mean_trans_spacing = np.mean(self.trans_spacing_list)
        self.stdev_trans_spacing = np.std(self.trans_spacing_list)
        
        print("****MEAN:")
        print("Mean translational order parameter: {:.3f} +/- {:.3f}".format(self.mean_trans_op,self.stdev_trans_op))
        print("Mean translational spacing: {:.3f} +/- {:.3f} Angstrom".format(self.mean_trans_spacing,self.stdev_trans_spacing))

        # Rewind Trajectory to beginning for other analysis
        self.universe.trajectory.rewind()

    def structure_factor_analysis(self, directors=None, times=None, style="molecule", pbc_style=None, Sq_style="com", gen_q_param=["strict",0,1,1], custom_traj=None, normalise=False, plot=False, ):
        """High level function for calculating the structure factor
        
        Example
        -------
        No Example yet

        Parameters
        ----------
        directors: numpy array(1 to 3, 3) or list of numpy array(1 to 3, 3)
            list of directors (or custom axes) along which to generate the q values. One numpy array for each time step with maximum dimensions of 3x3 with the rows as the directors and minimum dimensions of 1x3
        times : list of floats, optional
            If None, do for whole trajectory. If an interval
            is given like this (t_start, t_end) only do from start
            to end.
        style : string, optional
            "atom" or "molecule". Dependent on this, the 
            cluster_objects attribute is interpreted as molecule
            or atoms within a molecule. 
        pbc_style : string, optional
            Gromacs pbc definitions: mol, atom, nojump
        Sq_style : string, optional
            Center of mass ("com") or "atom"
        gen_q_param : list [string, float, float, float], optional
            Specify [style, q_min, q_max, q_step], where q_min and q_max are the minimum and maximum values of q in each direction and q_step is the steps in q in between these values in each direction. style can be either "strict" or "grid". For "strict" q_step is automatically overridden by q_step that correspond to integer values times (2*pi)/|reciprocal vector|
        custom_traj : list of list of AtomGroup, optional
            To be specified if the analysis is to be applied to clusters or other custom AtomGroups (i.e. if you want to consider different parts of the same molecule separately). The list should be the same length as the trajectory, each list of AtomGroups representing a trajectory timestep.
        normalise : boolean, optional
            Whether the structure factor should be normalised (like the translational order parameter) or not
        plot : boolean, optional
            If True the translational order parameter is plotted as a function of the spacing for the first time in the trajectory or specified in times.

        Raises
        ------ 
        TypeError
            If director has the wrong type / form
        IndexError
            If custom_traj is different length from trajetory or times
            If curstom_director different length than 3
            If search_param has index other than 3.
        NotImplementedError
            If an unspecified trans_op_style is choosen
        
        ToDo
        ----
        Test custom_traj feature
        Format/cleanup plot (possibly with external function)
        Play around with real spacing
        """
        self._set_pbc_style(pbc_style)

        self.universe = self._get_universe(self._coord, traj=self._traj)

        self.selected_species = self._select_species(self.universe,
                                                            style=style)
        self._custom_traj_check(times, custom_traj)

        # gen_q_param checks

        # director_list checks and q_array generation
        # use director_check (could split it up into three and then merge them so that we do not need a new function)
        if directors is not None:
            if directors.ndim == 1:
                directors = np.reshape(directors,(1,3))
                print(directors)
                print(directors.ndim)
                exit()

            print(np.size(directors,axis=0))
            print(np.size(directors,axis=1))
            directors = self._director_check(times,directors)

            print(directors)

        return 0

    def _gen_qArray_grid(director_list,qMin,qMax,qStep):
        qRange = np.arange(-qMax,qMax+qStep,qStep)
        qArray = np.asarray([p for p in itertools.product(qRange,repeat=3)])
        normq = np.linalg.norm(qArray,axis=1)
        Noq = len(normq)

        tol = 1e-5
        tol = max(tol,qMin)

        IndexList = []
        for i in xrange(Noq):
            if normq[i] < tol or normq[i] > qMax:
                IndexList.append(i)

        qArray = np.delete(qArray,np.asarray(IndexList),axis=0)
        normq = np.delete(normq,np.asarray(IndexList),axis=0)

        return normq, qArray

    def _gen_qArray_tric(qMin,qMax,FileName):
        BoxDim = box_edge_len(FileName)
        box_vec_tric = box_dim_tric(FileName)

        v2xv3 = np.cross(box_vec_tric[1],box_vec_tric[2])
        v3xv1 = np.cross(box_vec_tric[2],box_vec_tric[0])
        v1xv2 = np.cross(box_vec_tric[0],box_vec_tric[1])

        recip_lat_vec =np.asarray([2*np.pi*v2xv3/(np.dot(box_vec_tric[0],v2xv3)),\
                                   2*np.pi*v3xv1/(np.dot(box_vec_tric[1],v3xv1)),\
                                   2*np.pi*v1xv2/(np.dot(box_vec_tric[2],v1xv2))])

        prefac_qx = 2.0*np.pi/float(BoxDim[0])
        prefac_qy = 2.0*np.pi/float(BoxDim[1])
        prefac_qz = 2.0*np.pi/float(BoxDim[2])

        nx_max = int(qMax/prefac_qx)
        ny_max = int(qMax/prefac_qy)
        nz_max = int(qMax/prefac_qz)

        nxRange = np.arange(-nx_max,nx_max+1,1)
        nyRange = np.arange(-ny_max,ny_max+1,1)
        nzRange = np.arange(-ny_max,ny_max+1,1)
        nArray = np.asarray([p for p in itertools.product(nxRange,nyRange,nzRange)])
        qArray = np.matmul(nArray,recip_lat_vec)

        normq = np.linalg.norm(qArray,axis=1)
        Noq = len(normq)

        tol = 1e-5
        tol = max(tol,qMin)

        IndexList = []
        for i in xrange(Noq):
            if normq[i] < tol or normq[i] > qMax:
                IndexList.append(i)

        qArray = np.delete(qArray,np.asarray(IndexList),axis=0)
        normq = np.delete(normq,np.asarray(IndexList),axis=0)

        return normq, qArray

    def _custom_traj_check(self, times, custom_traj):
        """ Check if custom_traj is the correct length relative to the trajectory and times specified. And initialise variable self.custom_traj_idx
        
        Parameters
        ----------
        times : list of floats, optional
            If None, do for whole trajectory. If an interval
            is given like this (t_start, t_end) only do from start
            to end.
        custom_traj : list of list of AtomGroup
            To be specified if the analysis is to be applied to clusters or other custom AtomGroups (i.e. if you want to consider different parts of the same molecule separately). The list should be the same length as the trajectory, each list of AtomGroups representing a trajectory timestep.

        Raises
        ------ 
        IndexError
            If list is different length from trajetory or times
        """
        if custom_traj is not None:
            status, n_timesteps = _custom_list_v_traj_check(self, times, custom_traj)
            if not status:
                raise IndexError("custom_traj (len: {:d}) supplied is not the same length as the times in trajectory/times specified (len: {:d})".format(len(custom_traj),n_timesteps))
            self.custom_traj_idx = 0

    def _director_check(self, times, director):
        """ Check if director is the correct length and form. If it is a numpy array(3) convert it into a list of correct length relative to times and trajectory. If it is a list check size relative to the trajectory and times specified.
        
        Parameters
        ----------
        times : list of floats, optional
            If None, do for whole trajectory. If an interval
            is given like this (t_start, t_end) only do from start
            to end.
        custom_traj : list of list of AtomGroup
            To be specified if the analysis is to be applied to clusters or other custom AtomGroups (i.e. if you want to consider different parts of the same molecule separately). The list should be the same length as the trajectory, each list of AtomGroups representing a trajectory timestep.

        Raises
        ------ 
        IndexError
            If list is different length from trajetory or times
        TypeError
            If specified director is not a numpy array or has the wrong form
        """
        if type(director) is np.ndarray and len(director) == 3:
            if times is None:
                n_timesteps = len(self.universe.trajectory)
                director = [director for idx in range(n_timesteps)]
            else:
                n_timesteps = int((max(times)-min(times))/self.universe.trajectory.dt+1)
                director = [director for idx in range(n_timesteps)]
        elif type(director) is list and all([type(director[i]) is np.ndarray for i in range(len(director))]) and all([len(director[i]) == 3 for i in range(len(director))]):
            
            status, n_timesteps = self._custom_list_v_traj_check(times, director)
            if not status:
                raise IndexError("director (len: {:d}) supplied is not the same length as the times in trajectory/times specified (len: {:d})".format(len(director),n_timesteps))
        else:
            raise TypeError("The specified director has the wrong type and/or form. It must be a numpy.array(3) or a list of numpy.array(3)")
        return director

    def _custom_list_v_traj_check(self, times, custom_list):
        """ Check if a list is the correct size relative to the trajectory and times specified.

        times : list of floats, optional
            If None, do for whole trajectory. If an interval
            is given like this (t_start, t_end) only do from start
            to end.
        custom_list : list 

        """
        if times is not None:
            n_timesteps = int((max(times)-min(times))/self.universe.trajectory.dt+1)
            if len(custom_list) != n_timesteps:
                return False, n_timesteps
            return True, n_timesteps
        else:
            n_timesteps = len(self.universe.trajectory)
            if len(custom_list) != n_timesteps:
                return False, n_timesteps
            return True, n_timesteps

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

    def _get_center_of_mass(self, atom_group_list):
        """ Get list of the center of mass based on the intertia tensor
        
        Parameters
        ----------
        atom_group_list : list of AtomGroups
        
        Returns
        -------
        center_of_mass_list : list of numpy array(3)
            list of numpy arrays of the center of mass of each atom_group
        """
        center_of_mass_list = []
        for atom_group in atom_group_list:
            center_of_mass_list.append(atom_group.center_of_mass())
        center_of_mass_array = np.asarray(center_of_mass_list)
        return center_of_mass_array

    def _get_system_fourier_transform_mod(self, positions, k_vector):
        """ Get the normal of the system fourier transform at specfied k_vector
        
        Parameters
        ----------
        positions : numpy array
            numpy array of system positions
        k_vector : numpy array(3)
            k-space vector
        
        Returns
        -------
        norm_fourier_transform : float
            The magnitude of the fourier transform at the specified value of the k_vector
        """
        pos_dot_k = np.dot(positions,k_vector)
        sum_cos = (np.sum(np.cos(pos_dot_k)))**2
        sum_sin = (np.sum(np.sin(pos_dot_k)))**2
        mod_fourier_transform = np.sqrt((sum_cos + sum_sin))

        return mod_fourier_transform