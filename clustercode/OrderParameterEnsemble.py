import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NeighborSearch
import MDAnalysis.lib.mdamath as mdamath
import warnings
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import sys
import itertools
import scipy
from clustercode.BaseUniverse import BaseUniverse

"""
ToDo:
    Make sure PBC do what we want 
    Ensure behaviour for gro files
"""

class OrderParameterEnsemble(BaseUniverse):
    """A class used to perform analysis of the structuring of the 
    molecules or clusters of molecules

    Attributes
    ----------
    selection : list of str
        Strings used for the definition of species which form clusters. 
        Can be atom names or molecule names.
    universe : MDAnalysis universe object
        Universe of the simulated system 
    selected_species : MDAanalysis Atom(s)Group object
        The atoms that have been selected for analysis
    nematic_op_list : list of float
        Nematic order parameter value at each time
    system_director_list : list of numpy array(3)
        System director at each time
    mean_nematic_op : float
        Mean nematic order parameter over all times
    mean_system_director : : numpy array(3)
        Mean system director over all times
    trans_op_list : list of float
        Translational order parameter values at each time
    trans_spacing_list : list of float
        Translational spacing values at each time in Angstrom
    mean_trans_op : float
        Mean translational order parameter over all times
    stdev_trans_op : foat
        Standard deviation of translational order parameter over all 
        times
    mean_trans_spacing : float
        Mean translational spacing in Angstrom over all times
    stdev_trans_spacing : float
        Standard deviation of translational spacing in Angstrom over all 
        times
    q_norm_array : numpy array



    Methods
    -------
    nematic_op_analysis(times=None, style="molecule", 
                        principal_axis="inertial", custom_traj=None)
        Calculates nematic order parameter and system director for all
        timesteps. 
    translational_op_analysis(director, times=None, 
                              style="molecule", pbc_style=None, 
                              pos_style="com", search_param=None, 
                              custom_traj=None)
        Calculates translational order parameter and translational 
        spacing for input director or list of directors.
    structure_factor_analysis(directors=None, times=None, 
                              style="molecule", pbc_style=None, 
                              pos_style="com", q_style="strict", 
                              q_min=0, q_max=1, q_step = 0.01, 
                              active_dim=[1,1,1], custom_traj=None, 
                              plot_style="scatter", chunk_size=10000, 
                              n_bins = 500)
        Calculates structure factor as a function of the wave vector.

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
            Strings used for the definition of species to be studied. 
            Can be atom names or molecule names.
        """
        super().__init__(coord, traj, selection)

    def nematic_op_analysis(self, times=None, style="molecule", 
                            principal_axis="inertial", custom_traj=None, 
                            pbc_style=None):
        """High level function for calculating the nematic order 
        parameter
        
        Example
        -------
        No Example yet

        Parameters
        ----------
        times : list of floats, optional
            If None, do for whole trajectory. If an interval is given 
            like this (t_start, t_end) only do from start to end, by 
            default None.
        style : string, optional
            "atom" or "molecule". Dependent on this, the 
            cluster_objects attribute is interpreted as molecule
            or atoms within a molecule. 
        principal_axis : string, optional
            "inertial" or "end-to-end". Defines the principal axis as 
            either the end to end vector of the molecule or the dominant 
            axis of the inertial tensor.
        custom_traj : list of list of AtomGroup, optional
            To be specified if the analysis is to be applied to clusters 
            or other custom AtomGroups (i.e. if you want to consider 
            different parts of the same molecule separately). The list 
            should be the same length as the trajectory, each list of 
            AtomGroups representing a trajectory timestep.
        pbc_style : string, optional
            Gromacs pbc definitions: mol or atom, by default
            None

        Raises
        ------ 
        NotImplementedError
            If an unspecified principal axis is choosen
        
        ToDo
        ----
        """

        self._set_pbc_style(pbc_style)

        self.universe = self._get_universe(self._coord, traj=self._traj)

        self.selected_species = self._select_species(self.universe,
                                                            style=style)
        self._custom_traj_check(times, custom_traj)

        # Select which principal axis in the AtomGroup to use
        if principal_axis == "inertial":
            principal_axis = self._get_inertial_axis
        elif principal_axis == "end-to-end":
            principal_axis = self._get_end_to_end_vector
        else:
            raise NotImplementedError("{:s} is unspecified molecular axis"\
                                                    .format(principal_axis))

        # If custom_traj is not specified initialise select_species as a
        # list of AtomGroups (one for each residue)
        if custom_traj is None:
            selected_species_list = [
                self._select_species(residue.atoms, style=style)
                for residue in self.selected_species.residues]

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
                atom_group_list = custom_traj[self._custom_traj_idx]
                self._custom_traj_idx+= 1
            else:
                atom_group_list = selected_species_list

            principal_axis_list = principal_axis(atom_group_list)
            saupe_tensor = self._get_saupe_tensor(principal_axis_list)
            nematic_op, system_director = self._get_dominant_eig(saupe_tensor)

            self.nematic_op_list.append(nematic_op)
            self.system_director_list.append(system_director)
            sum_saupe_tensor += saupe_tensor

            print("****TIME: {:8.2f}".format(time.time))
            print("Nematic order parameter: {:.3f}".format(nematic_op))
            
        # Obtain the ensemble average saupe_tensor
        ensemble_saupe_tensor = sum_saupe_tensor/len(self.nematic_op_list)

        # Calculate the mean nematic order parameter and system director 
        # from the ensemble average saupe tensor
        self.mean_nematic_op, self.mean_system_director = (
            self._get_dominant_eig(ensemble_saupe_tensor))

        self.stdev_nematic_op = np.std(self.nematic_op_list)

        print("****MEAN:")
        print("Mean nematic order parameter: {:.3f} +/- {:.3f}".format(
            self.mean_nematic_op,self.stdev_nematic_op))
        print("Mean system director: {:s}".format(np.array2string(
            self.mean_system_director)))

        # Rewind Trajectory to beginning for other analysis
        self.universe.trajectory.rewind()

    def translational_op_analysis(self, director, times=None, style="molecule",
                                  pbc_style=None, pos_style="com", 
                                  search_param=[0.1, 50, 500], 
                                  custom_traj=None, plot=False):
        """High level function for calculating the translational order 
        parameter
        
        Example
        -------
        No Example yet

        Parameters
        ----------
        director : numpy array(3) or list of numpy array(3)
            Specify one if the same director is applied to all timesteps 
            in the trajectory or a list if a different director is used 
            each timestep. Note it should be a unit vector or list of 
            unit vectors
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
        pos_style : string, optional
            Center of mass ("com") or "atom"
        search_param : [float, float, int], optional
            Specify [min, max, n_points], where min and max are the 
            minimum and maximum translational spacings considered in 
            Angstrom and n_points is the number of points between these 
            values.
        custom_traj : list of list of AtomGroup
            To be specified if the analysis is to be applied to clusters
            or other custom AtomGroups (i.e. if you want to consider 
            different parts of the same molecule separately). The list 
            should be the same length as the trajectory, each list of 
            AtomGroups representing a trajectory timestep.
        plot : boolean, optional
            If True the translational order parameter is plotted as a 
            function of the spacing for the first time in the trajectory 
            or specified in times.
        
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
        
        # Set search_param if it is not specified by user. If it is 
        # specified check its length and make sure the minimum value is 
        # not zero.
        if search_param is None:
            search_param = [0.1, 50, 500]
        else:
            if len(search_param) != 3:
                IndexError("len(search_param) is not 3")
            elif search_param[0] == 0:
                search_param[0] += 0.01

        spacing_array = np.linspace(*search_param)

        # Initialise outputs
        self.trans_op_list = []
        self.trans_spacing_list = []
        director_idx = 0

        # Loop over all trajectory times
        for time in self.universe.trajectory:
            if times is not None:
                if time.time > max(times) or time.time < min(times):
                    continue

            position_array = self._get_position_array(style, pos_style, 
                                                      custom_traj)
            
            # Optimise the translational order parameter and determine the spacing
            trans_op_k = []
            spacing_list = []
            for spacing in spacing_array:
                k_vector = 2*np.pi/spacing * director[director_idx]
                trans_op_k.append(np.sqrt(
                    self._get_system_fourier_transform_mod2(position_array, 
                                                            k_vector, 1)
                    )/float(len(position_array)))
            
            idx_max = np.argmax(trans_op_k)
            trans_op = trans_op_k[idx_max]
            trans_spacing = spacing_array[idx_max]

            print("****TIME: {:8.2f}".format(time.time))
            print("Translational order parameter: {:.3f}".format(trans_op))
            print("Translational spacing: {:.3f} Angstrom".format(
                                                                trans_spacing))
            
            self.trans_op_list.append(trans_op)
            self.trans_spacing_list.append(trans_spacing)

            director_idx += 1

            if plot:
                plt.plot(spacing_array, trans_op_k)
                plt.xlim([search_param[0], search_param[1]])
                plt.ylim([0, 1])
                plt.ylabel('Translational order parameter')
                plt.xlabel('Translational spacing (Angstrom)')
                plt.show()
                plot=False

        # Calculate mean and standard deviations
        self.mean_trans_op = np.mean(self.trans_op_list)
        self.stdev_trans_op = np.std(self.trans_op_list)
        self.mean_trans_spacing = np.mean(self.trans_spacing_list)
        self.stdev_trans_spacing = np.std(self.trans_spacing_list)
        
        print("****MEAN:")
        print("Mean translational order parameter: {:.3f} +/- {:.3f}"\
            .format(self.mean_trans_op,self.stdev_trans_op))
        print("Mean translational spacing: {:.3f} +/- {:.3f} Angstrom"\
            .format(self.mean_trans_spacing,self.stdev_trans_spacing))

        # Rewind Trajectory to beginning for other analysis
        self.universe.trajectory.rewind()

    def structure_factor_analysis(self, directors=None, times=None, 
                                  style="molecule", pbc_style=None, 
                                  pos_style="com", q_style="strict", q_min=0.0, 
                                  q_max=1.0, q_step=0.01, active_dim=[1, 1, 1], 
                                  custom_traj=None, chunk_size=10000, 
                                  plot_style="smooth", n_bins=1000):
        """High level function for calculating the structure factor as a 
        function of the wave vector q.
        
        Example
        -------
        No Example yet

        Parameters
        ----------
        directors: numpy array(=<3, 3) or list of numpy array(=<3, 3)
            List of directors along which to generate the wave vector 
            (q) values. Either one set of directors as a numpy 
            array(=<3,3) each row corresponding to a director for all 
            timesteps or a list of arrays one for each timestep
        times : list of floats, optional
            If None, do for whole trajectory. If an interval
            is given like this (t_start, t_end) only do from start
            to end.
        style : string, optional
            "atom" or "molecule". Dependent on this, the 
            cluster_objects attribute is interpreted as molecule
            or atoms within a molecule. 
        pbc_style : string, optional
            Gromacs pbc definitions: mol or atom
        pos_style : string, optional
            Center of mass ("com") or "atom"
        q_style : string, optional
            Style of wave vector q can be either "strict" or "grid". For 
            style "strict" the variable q_step is ignored. If directors 
            is not None q_style defaults to grid
        q_min : float, optional
            Minimum modulus of wave vector q considered
        q_max : float, optional
            Maximum modulus of wave vector q considered
        q_step : float, optional
            Used only for q_style "grid" or if directors are specified
        active_dim : list of int
            Used only when directors is None. List of length 3 each 
            entry being 1 for an active dimension and 0 for inactive 
            dimension. 
        custom_traj : list of list of AtomGroup, optional
            To be specified if the analysis is to be applied to clusters 
            or other custom AtomGroups (i.e. if you want to consider 
            different parts of the same molecule separately). The list 
            should be the same length as the trajectory, each list of 
            AtomGroup objects representing a trajectory timestep.
        chunk_size : integer, optional
            The array of wave vectors is split into chunks of this size 
            for the square modulus of the fourier transform calculation.
            A high number means more ram usage, a lower number means 
            lower ram usage. Overall it does not have a major impact on 
            performance.
        plot_style : string, optional
            If None no plot is generated. Other options are "smooth" and 
            "scatter".
        n_bins : integer, optional
            In case of data smoothing, the number of bins used.

        Raises
        ------ 
        NotImplementedError
            If unspecified q_style is supplied by user
            If plot_style is not "scatter" or "smooth"
        
        ToDo
        ----
        Test custom_traj feature
        """
        self._set_pbc_style(pbc_style)

        self.universe = self._get_universe(self._coord, traj=self._traj)

        self.selected_species = self._select_species(self.universe,
                                                            style=style)
        self._custom_traj_check(times, custom_traj)

        if directors is not None:
            # Check form of directors and initialise the director_idx variable
            directors_list = self._director_check(times,directors)
            directors_idx = 0

            print("****NOTE: As directors are specified, the wave vector q "\
                  "generation method defaults to grid and the active_dim list"\
                  " is not used")
            
            q_style = "grid"

        if q_style is "strict":
            self._gen_q = self._gen_q_array_strict
            print("****NOTE: As q_style strict is selected, the variable "\
                  "q_step is not used")
        elif q_style is "grid":
            self._gen_q = self._gen_q_array_grid
        else:
            raise NotImplementedError("q_style {:s} is not implemented"\
                                        .format(q_style))

        # generate q at each timestep flag
        gen_q_flag = True
        # Note if directors is of type numpy array then the director is 
        # the same for all timesteps and the q_array can be generated in 
        # advance
        if type(directors) == np.ndarray:
            # Use first entry in directors_list as this has been 
            # converted into the right format: numpy array(1,3)
            q_array = self._gen_q(directors_list[0], q_min, q_max, q_step)
            q_norm = np.linalg.norm(q_array, axis=1)
            gen_q_flag = False

        # Flag used to initialise the output numpy arrays
        initialise_flag = True
        
        # Loop over all trajectory times
        for time in self.universe.trajectory:
            if times is not None:
                if time.time > max(times) or time.time < min(times):
                    continue

            # Check if q needs to be generated
            if gen_q_flag:
                if directors == None:
                    timestep_directors = self._calc_directors(active_dim)
                else:
                    timestep_directors = directors_list[directors_idx]
                    directors_idx += 1

                q_norm, q_array = self._gen_q(timestep_directors, q_min, q_max, 
                                             q_step)

            position_array = self._get_position_array(style, pos_style, 
                                                      custom_traj)

            Sq = self._get_system_fourier_transform_mod2(position_array,
                                                         q_array,
                                                         chunk_size
                                                         )/len(position_array)

            if initialise_flag:
                q_array_all = q_array
                self.q_norm_array = q_norm
                self.Sq_array = Sq
                initialise_flag = False
            else:
                q_array_all = np.vstack((q_array_all, q_array))
                self.q_norm_array = np.append(self.q_norm_array, q_norm)
                self.Sq_array = np.append(self.Sq_array, Sq)

            print("****TIME: {:8.2f}".format(time.time))

        # Rewind Trajectory to beginning for other analysis
        self.universe.trajectory.rewind()

        # Plot structure factor
        if plot_style is not None:
            if plot_style is "smooth":
                self.smooth_q_norm, self.smooth_Sq = (
                    self._smooth_structure_factor(q_min, q_max, n_bins))
                plt.plot(self.smooth_q_norm, self.smooth_Sq)
                plt.xlabel('$q$ / ${\AA}^{-1}$')
                plt.ylabel('$S(q)$')
                plt.xlim([min(self.smooth_q_norm), max(self.smooth_q_norm)])
                plt.show()
            elif plot_style == "scatter":
                plt.scatter(self.q_norm_array, self.Sq_array)
                plt.xlabel('$q$ / ${\AA}^{-1}$')
                plt.ylabel('$S(q)$')
                plt.xlim([min(self.q_norm_array), max(self.q_norm_array)])
                plt.show()
            else:
                NotImplementedError("plot_style {:s} has not been implemented"\
                                        .format(plot_style))

    def _custom_traj_check(self, times, custom_traj):
        """ Check if custom_traj is the correct length relative to the 
        trajectory and times specified. And initialises variable 
        self._custom_traj_idx
        
        Parameters
        ----------
        times : list of floats
            If None, do for whole trajectory. If an interval is given 
            like this (t_start, t_end) only do from start to end.
        custom_traj : list of list of AtomGroup
            To be specified if the analysis is to be applied to clusters 
            or other custom AtomGroup object (i.e. if you want to 
            consider different parts of the same molecule separately). 
            The list should be the same length as the trajectory, each 
            list of AtomGroup objects representing a trajectory timestep

        Raises
        ------ 
        IndexError
            If list is different length from trajetory or times
        """
        if custom_traj is not None:
            status, n_timesteps = self._custom_list_v_traj_check(times, 
                                                                 custom_traj)
            if not status:
                raise IndexError("custom_traj (len: {:d}) supplied is not the"\
                                 " same length as the times in trajectory"\
                                 "/times specified (len: {:d})".format(
                                     len(custom_traj), n_timesteps))
            self._custom_traj_idx = 0

    def _custom_list_v_traj_check(self, times, custom_list):
        """ Check if a list is the correct size relative to the 
        trajectory and times specified.

        times : list of floats
            If None, do for whole trajectory. If an interval
            is given like this (t_start, t_end) only do from start
            to end.
        custom_list : list
            List of for example AtomGroup objects

        """
        if times is not None:
            n_timesteps = int((max(times) - min(times)
                                        )/self.universe.trajectory.dt + 1)
            if len(custom_list) != n_timesteps:
                return False, n_timesteps
            return True, n_timesteps
        else:
            n_timesteps = len(self.universe.trajectory)
            if len(custom_list) != n_timesteps:
                return False, n_timesteps
            return True, n_timesteps

    def _get_inertial_axis(self, atom_group_list):
        """ Get list of principal molecular axis based on the intertia 
        tensor
        
        Parameters
        ----------
        atom_group_list : list of AtomGroup objects
            List of AtomGroup objects for which to calculate the 
            intertia tensor
        
        Returns
        -------
        principal_axis_list : list of numpy array(3)
            List of numpy array(3) of the principal axis vector for each
            AtomGroup object
        """
        principal_axis_list = []
        for atom_group in atom_group_list:
            principal_axis_list.append(atom_group.principal_axes()[2])
        
        return principal_axis_list

    def _get_end_to_end_vector(self, atom_group_list):
        """ Get the end-to-end vector of AtomGroup. Note it finds the 
        vector between the first and last atom.
        
        Parameters
        ----------
        atom_group_list : list of AtomGroup objects
            For each AtomGroup the end-to-end vector is calculated
        
        Returns
        -------
        end_to_end_list : list of numpy array(3)
            List of numpy arrays of the principal axis vector
        """
        end_to_end_list = []
        for atom_group in atom_group_list:
            end_to_end_vec = atom_group[0].position- atom_group[-1].position
            end_to_end_list.append(end_to_end_vec
                                  /np.linalg.norm(end_to_end_vec))

        return end_to_end_list

    def _get_saupe_tensor(self, principal_axis_list):
        """ Calculate saupe tensor from principal axes
        
        Parameters
        ----------
        principal_axis_list : list of numpy array(3)
            List of numpy arrays of the principal axis vector
        
        Returns
        -------
        saupe_tensor : numpy array(3,3)
        """
        saupe_tensor = np.zeros((3, 3))
        half_identity_matrix = np.identity(3) / 2.0
        for axis in principal_axis_list:
            saupe_tensor += 1.5 * np.outer(axis, axis) - half_identity_matrix
        saupe_tensor /= len(principal_axis_list)

        return saupe_tensor

    def _get_dominant_eig(self, matrix):
        """ Calculate dominant eigen value and vector
        
        Parameters
        ----------
        matrix : numpy array(3,3)
            Matrix for eigen value and vector analysis
        
        Returns
        -------
        eig_val1 : float
            Dominant eigen value (the one with highest magnitude)
        eig_vec1 : numpy array(3)
            Eigen vector corresponding to dominant eigen value
        """
        eig_val, eig_vec = np.linalg.eig(matrix)

        # Find index of eigenvalue with highest absolute value
        idxs = np.argsort(abs(eig_val))[::-1]  
        eig_val1 = eig_val[idxs][0]
        eig_vec1 = eig_vec[:,idxs][:,0]

        return eig_val1, eig_vec1

    def _director_check(self, times, director):
        """ Check if director is the correct length and form. If it is a 
        numpy array: convert it into a list of correct length relative 
        to times and trajectory. If it is a list: check size relative to
        the trajectory and times specified. If the dimension of the 
        director numpy array is 1, convert it into a numpy array(1,3).
        
        Parameters
        ----------
        times : list of floats
            If None, do for whole trajectory. If an interval
            is given like this (t_start, t_end) only do from start
            to end.
        director : numpy array(3), numpy array(=<3,3), 
                   list of numpy array(=<3,3), or list of numpy array(3)
            The directors along which to calculate the wave vectors.

        Raises
        ------ 
        IndexError
            If numpy arrays in director are not (=<3,3)
            If list is different length from trajetory or times
        TypeError
            If specified director is not a numpy array or list of numpy 
            arrays
        """
        if type(director) is np.ndarray:
            director = self._director_dim_check(director)

            # Check if director has 3 columns and not more than 3 rows
            if (np.size(director, axis=1) == 3 
                    and np.size(director, axis=0) < 3.5):
                if times is None:
                    n_timesteps = len(self.universe.trajectory)
                else:
                    n_timesteps = int((max(times) - min(times)
                                            )/self.universe.trajectory.dt + 1)
                # Convert director into list of numpy arrays (one for 
                # each timestep)
                director = [director for idx in range(n_timesteps)]
            else:
                raise IndexError("director dimensions ({:d},{:d}) not (=<3,3)"\
                                    .format(np.size(director, axis=0),
                                            np.size(director, axis=1)))
        
        elif type(director) is list and all(
                [type(director_i) is np.ndarray for director_i in director]):
            status, n_timesteps = self._custom_list_v_traj_check(times, director)
            if not status:
                raise IndexError("director (len: {:d}) supplied is not the "\
                                 "same length as the times in trajectory/"\
                                 "times specified (len: {:d})".format(
                                     len(director), n_timesteps))

            director = [self._director_dim_check(director_i) 
                        for director_i in director]

            # Check if director at each timestep has 3 columns and not 
            # more than 3 rows
            if not all([np.size(director_i, axis=1) == 3 and 
                        np.size(director_i, axis=0) < 3.5 
                        for director_i in director]):
                raise IndexError("numpy arrays in director not if dimensions "\
                                 "(=<3,3)")
            
        else:
            raise TypeError("The specified director has the wrong type.")
        return director

    def _director_dim_check(self, director):
        """ Checks director dimensions and if its has dimension 1 
        reshapes it into a (1,:) array.

        Parameters
        ----------
        director : numpy array(3) or numpy array(=<3,3)
            Vector along which to calculate the wave vector.

        Returns
        -------
        director : numpy array(1,3) or numpy array(=<3,3)
            Vector along which to calculate the wave vector.

        """
        if director.ndim == 1:
            director = np.reshape(director, (1, -1))
        elif director.ndim > 2.5:
            raise IndexError("director (ndim: {:d}) more than 2 dimensionss"\
                                .format(director.ndim))
        return director

    def _get_position_array(self, style, pos_style, custom_traj):
        """  Get positions array for custom_traj or selected_species as 
        either the positions of the atoms or the centers of mass of the 
        provided AtomGroups.

        Parameters
        ----------
        style : string
            "atom" or "molecule". Dependent on this, the 
            cluster_objects attribute is interpreted as molecule
            or atoms within a molecule. 
        pos_style : string
            Center of mass ("com") or "atom"
        custom_traj : list of list of AtomGroup
            To be specified if the analysis is to be applied to clusters
            or other custom AtomGroup objects (i.e. if you want to 
            consider different parts of the same molecule separately). 
            The list should be the same length as the trajectory, each 
            list of AtomGroups representing a trajectory timestep.

        Returns
        -------
        position_array : numpy array(n,3)
            Array of positions corresponding to atoms or center of 
            masses of AtomGroups supplied

        Raises
        ------
        NotImplementedError
            If unspecified pos_style is given
        """
        if pos_style is "com":
            if custom_traj is not None:
                atom_group_list = custom_traj[self._custom_traj_idx]
                self._custom_traj_idx += 1
            else:
                atom_group_list = [self._select_species(residue.atoms, 
                    style=style) for residue in self.selected_species.residues]
            position_array = self._get_center_of_mass(atom_group_list)
        elif pos_style is "atom":
            if custom_traj is not None:
                position_array = custom_traj[
                                    self._custom_traj_idx][0].positions
                for atom_group in custom_traj[self._custom_traj_idx][1:]:
                    position_array = np.vstack(atom_group.positions)
                self._custom_traj_idx += 1
            else:
                position_array = self.selected_species.positions
        else:
            raise NotImplementedError("{:s} is unspecified style".format(
                                                                    pos_style))
        return position_array

    def _get_center_of_mass(self, atom_group_list):
        """ Get list of the center of mass of AtomGroup objects
        
        Parameters
        ----------
        atom_group_list : list of AtomGroup objects
            List of AtomGroup objects for each of which to calculate the
            center of mass.
        
        Returns
        -------
        center_of_mass_list : list of numpy array(3)
            list of numpy arrays of the center of mass of each AtomGroup
        """
        center_of_mass_list = []
        for atom_group in atom_group_list:
            center_of_mass_list.append(atom_group.center_of_mass())
        position_array = np.asarray(center_of_mass_list)
        return position_array

    def _get_system_fourier_transform_mod2(self, positions, k_vectors, 
                                           chunk_size):
        """ Get the square modulus of the system fourier transform at 
        specfied k_vector

        Note
        ----
        - scipy.linalg.blas reduces computation time by 25% relative to 
          numpy.matmul
        - Chunking does not seem to negatively impact computation time, 
          but reduces ram usage significantly
        
        Parameters
        ----------
        positions : numpy array(n,3)
            numpy array of system positions
        k_vectors : numpy array(m,3)
            k-space vectors
        chunk_size : integer
            size of chunks of matrix multiplictions to carry out at the
            same time.
        
        Returns
        -------
        mod2_fourier_transform : float (if m=1) or numpy array(m)
            The square modulus of the fourier transform at the specified
            value of the k_vectors
        """
        # Convert array into fortran form (neccesary for 
        # scipy.linalg.blas)
        positions = np.array(positions, order='F')

        # If only one k_vector do not chunk select sgemv, otherwise use 
        # sgemm
        if np.size(k_vectors,axis=0) == 1:
            k_vectorsT = np.array(k_vectors, order='F')
            blas_algorithm = scipy.linalg.blas.sgemv
        else:
            k_vectorsT = np.array(k_vectors.T, order='F')
            blas_algorithm = scipy.linalg.blas.sgemm

        if chunk_size == 1:
            k_vectorsT_chunks = k_vectorsT
        else:
            # Find number of chunks and get list of chunks
            len_k_vectorsT = np.size(k_vectorsT, axis=1)
            n_chunks = max(1,int(len_k_vectorsT/chunk_size))
            
            k_vectorsT_remainder = k_vectorsT[:,(n_chunks*chunk_size):]

            k_vectorsT = k_vectorsT[:,0:(n_chunks*chunk_size)]

            k_vectorsT_chunks = np.split(k_vectorsT, n_chunks, axis=1)

            if k_vectorsT_remainder.size > 0.5:
                k_vectorsT_chunks.append(k_vectorsT_remainder)

        # Loop over chunks of wave vectors
        for idx, k_vectorsT_i in enumerate(k_vectorsT_chunks):
            pos_dot_k = blas_algorithm(1.0, positions,k_vectorsT_i)
            sum_cos = np.square(np.sum(np.cos(pos_dot_k),axis=0))
            sum_sin = np.square(np.sum(np.sin(pos_dot_k),axis=0))
            if idx == 0:
                mod2_fourier_transform = (sum_cos + sum_sin)
            else:
                mod2_fourier_transform = np.append(mod2_fourier_transform,
                                                   (sum_cos + sum_sin))

        return mod2_fourier_transform

    def _gen_q_array_strict(self, directors, q_min, q_max, *args):
        """ Generate wave vector (q) array strictly as integer 
        combinations of the directors, which should correspond to the 
        reciprocal lattice vectors.
        
        Note
        ----
        *args is added so that it can accept q_step as an argument 
        although it is not used, but this avoids additional 
        if statements when looping over the trajectory.

        Parameters
        ----------
        directors : numpy array(=<3,3)
            Numpy array of directors
        q_min : float
            Minimum modulus of q
        q_max : float
            Maximum modulus of q

        Returns
        -------
        q_norm : numpy array(n)
            All the moduli of q
        q_array : numpy array(n,3)
            All the vectors of q
        """
        # periodic_len = 2*pi / (distance between opposite simulation 
        # box faces)
        periodic_len = np.linalg.norm(directors, axis=0)

        # Get maximum integer index
        n_max_vec = (q_max / periodic_len).astype(int)
        n_min_vec = (q_min / periodic_len).astype(int)

        # Generate combinations of the integer values
        n_range = []
        for n_max, n_min in zip(n_max_vec,n_min_vec):
            if n_min == 0:
                n_range.append(np.arange(-n_max, n_max+1, 1))
            else:
                n_range.append(np.append(np.arange(-n_max, n_min + 1, 1),
                                         np.arange( n_min, n_max + 1, 1)))

        n_array = np.asarray([p for p in itertools.product(*[
                                            range_i for range_i in n_range])])

        # Get all linear combinations of directors
        q_array = np.matmul(n_array,directors)

        q_norm = np.linalg.norm(q_array,axis=1)

        # Remove values that violate the limits
        q_norm, q_array = self._check_lim_q_array(q_norm, q_array, q_min, 
                                                  q_max)

        return q_norm, q_array

    def _check_lim_q_array(self, q_norm, q_array, q_min, q_max):
        """ Check if q_norm is within limits and remove values that 
        violate the limits from q_norm and q_array

        Parameters
        ----------
        q_norm : numpy array(n)
            All the moduli of q
        q_array : numpy array(n,3)
            All the vectors of q
        q_min : float
            minimum modulus of q
        q_max : float
            maximum modulus of q

        Returns
        -------
        q_norm : numpy array(m)
            All the moduli of q
        q_array : numpy array(m,3)
            All the vectors of q

        """
        q_min = max(1e-5, q_min)

        check_q_max = (q_norm > q_max)
        check_q_min = (q_norm < q_min)

        # Get indices of values in q_norm that violate the limits
        del_idx_max = [i for i, x in enumerate(check_q_max) if x]
        del_idx_min = [i for i, x in enumerate(check_q_min) if x]

        del_idx = np.append(del_idx_max,del_idx_min)

        q_array = np.delete(q_array,del_idx,axis=0)
        q_norm = np.delete(q_norm,del_idx)

        return q_norm, q_array

    def _gen_q_array_grid(self, directors, q_min, q_max, q_step):
        """ Generate wave vector (q) array in a grid as linear 
        combinations of the directors

        Parameters
        ----------
        directors : numpy array(=<3,3)
        q_min : float
            minimum modulus of q
        q_max : float
            maximum modulus of q
        q_step : float
            step size of q in each direction

        Returns
        -------
        q_norm : numpy array(n)
            All the moduli of q
        q_array : numpy array(n,3)
            All the vectors of q
        """
        # Generate q_range from q_max to q_min in steps of q_step (both 
        # for positive and negative integers)
        if q_min == 0:
            q_range = np.arange(-q_max, q_max + q_step, q_step)
        else:
            q_range = np.append(np.arange(-q_max, -q_min + q_step, q_step),
                                np.arange(q_min, q_max + q_step, q_step))

        # Convert directors to unit vectors
        for idx, director in enumerate(directors):
            directors[idx, :] = director / np.linalg.norm(director)

        q_range_comb = np.asarray(
            [p for p in itertools.product(q_range, 
                                          repeat=np.size(directors, axis=0))])

        # Get linear combination of q_range and directors
        q_array = np.matmul(q_range_comb, directors)

        q_norm = np.linalg.norm(q_array, axis=1)

        # Remove values that violate limits
        q_norm, q_array = self._check_lim_q_array(q_norm, q_array, q_min, 
                                                  q_max)

        return q_norm, q_array
    
    def _calc_directors(self, active_dim):
        """Calculate directors as the reciprocal lattice vectors. For 
        orthorombic and triclinic simulation boxes the reciprocal 
        lattice vectors are vectors perpendicular to each face of the
        simulation cell.
        
        Parameters
        ----------
        active_dim : list of int
            Active dimensions each intereger must be either 0 for off or
            1 for on.

        Returns
        -------
        directors : numpy array(=<3,3)
            Directors along which to calculate the wave vector.

        """
        # Get triclinic box vectors
        box_edge_vectors = mdamath.triclinic_vectors(self.universe.dimensions)

        # Calculate reciprocal lattice vectors
        recip_lat_vecs = self._calc_reciprocal_lattice_vectors(
                                                            box_edge_vectors)

        # Remove inactive dimensions
        check_active_dim = [active_dim_i is 0 for active_dim_i in active_dim]
        del_idx_active_dim = [i for i, x in enumerate(check_active_dim) if x]
        directors = np.delete(recip_lat_vecs,del_idx_active_dim,axis=0)

        return directors

    def _calc_reciprocal_lattice_vectors(self, edge_vectors):
        """ Calculate reciprocal lattice vectors from box edge vectors
        
        Parameters
        ----------
        edge_vectors : numpy array(3,3)

        Returns
        -------
        recip_lat_vec : numpy array(3,3)

        """
        v2xv3 = np.cross(edge_vectors[1],edge_vectors[2])
        v3xv1 = np.cross(edge_vectors[2],edge_vectors[0])
        v1xv2 = np.cross(edge_vectors[0],edge_vectors[1])

        recip_lat_vecs =np.asarray(
            [2.0*np.pi*v2xv3/np.dot(edge_vectors[0],v2xv3),
             2.0*np.pi*v3xv1/np.dot(edge_vectors[1],v3xv1),
             2.0*np.pi*v1xv2/np.dot(edge_vectors[2],v1xv2)])
        return recip_lat_vecs

    def _smooth_structure_factor(self, q_min, q_max, n_bins):
        """ Smooth structure factor

        Parameters
        ----------
        q_min : float
        q_max : float
        n_bins : integer

        Returns
        -------
        norm_q : numpy array(n_bins)
        smooth_q : numpy array(n_bins)

        """
        norm_q = np.linspace(q_min, q_max, n_bins+1)
        norm_q_step = (norm_q[1] - norm_q[0]) / 2.0
        norm_q = norm_q[0:-1] + norm_q_step

        norm_q_count = np.zeros((n_bins))
        smooth_Sq = np.zeros((n_bins))

        bin_number = ((self.q_norm_array - q_min) 
                      / (q_max - q_min) * n_bins).astype(int)

        for idx, Sq in zip(bin_number, self.Sq_array):
            norm_q_count[idx] += 1
            smooth_Sq[idx] += Sq

        # Adjust empty bins by adding one to avoid divide by zero errors
        norm_q_count_add = (norm_q_count == 0).astype(int)

        norm_q_count = norm_q_count + norm_q_count_add

        smooth_Sq = smooth_Sq / norm_q_count
        return norm_q, smooth_Sq
