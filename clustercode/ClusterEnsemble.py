import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NeighborSearch
import warnings
import matplotlib.pyplot as plt
from clustercode.BaseUniverse import BaseUniverse
import numpy as np

# from MDAnalysis.core.groups import ResidueGroup
"""
ToDo:
    Make sure PBC do what we want
    Ensure behaviour for gro files
    Make paths to BaseUniverse universal
"""


class ClusterEnsemble(BaseUniverse):
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

    def cluster_analysis(
        self,
        cut_off=7.5,
        times=None,
        style="atom",
        measure="b2b",
        algorithm="dynamic",
        work_in="Residue",
        traj_pbc_style=None,
        pbc=True,
        verbosity=0,
    ):
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
        measure : string, optional
            "b2b (bead to bead), COM or COG(center of geometry)
        algorithm : string, optional
            "dynamic" or "static". The static one is slower. I
            loops over all atoms and then merges cluster, whereas
            the dynamic algorithm grows clusters dynamically.
        work_in : string, optional
            "Residue" or "Atom". Either work in (and output)
            ResidueGroups or AtomGroups.
            The former may be faster for systems in which all
            parts of the same molecule are always in the same cluster,
            whereas the latter is useful for systems in which different
            parts of the same molecule can be in different clusters
            (i.e. block copolymers).
        traj_pbc_style : string, optional
            Gromacs pbc definitions: mol or atom, by default
            None
        pbc : bool, optional
            Whether to consider periodic boundary conditions in the 
            neighbour search (for determining whether atoms belong to
            the same cluster), by default True. Note that if work_in is 
            set to "Residue" periodic boundary conditions are taken into
            account implicitly for atoms in molecules passing across the 
            boundaries.
        verbosity: int, optional
            Controls how much the code talks.

        Raises
        ------
        NotImplementedError
            If an unspecified algorithm or work_in is choosen
        ValueError
            If pbc is not boolean
        
        ToDo
        ----
        -Make static and dynamic algorithms store the same arguments
        -Add plotting capabilities
        -Add capabilities to only look at certain time windows
        -Get rid of traj and coord attributes
        """

        self._set_pbc_style(traj_pbc_style)

        self.universe = self._get_universe(self._coord, traj=self._traj)

        self.style = style

        self.aggregate_species = self._select_species(self.universe, style=self.style)

        self.cluster_list = []
        self.cluster_sizes = []
        self.times = times

        # Initialise the neighboursearch object

        if pbc == True:
            self.neighbour_search = NeighborSearch.AtomNeighborSearch(
                self.aggregate_species, box=self.universe.dimensions, bucket_size=10
            )
        elif pbc == False:
            if work_in == "Residue" and traj_pbc_style != "mol":
                warnings.warn(
                    'work_in = "Residue" implicitly enforces pbc '
                    "for atoms in the same molecule if pbc_style "
                    '= "atom"',
                    UserWarning,
                )
                print("Warning")
            self.neighbour_search = NeighborSearch.AtomNeighborSearch(
                self.aggregate_species, box=None, bucket_size=10
            )
        else:
            raise ValueError("pbc has to be boolean")

        if work_in == "Residue":
            self.search_level = "R"
        elif work_in == "Atom":
            self.search_level = "A"
        else:
            raise NotImplementedError(
                "{:s} is unspecified work_in variable".format(work_in)
            )

        if algorithm == "static":
            cluster_algorithm = self._get_cluster_list_static
        elif algorithm == "dynamic":
            cluster_algorithm = self._get_cluster_list_dynamic
        else:
            raise NotImplementedError("{:s} is unspecified algorithm".format(algorithm))
        # Loop over all trajectory times
        for time in self.universe.trajectory:
            if self.times is not None:
                if time.time > max(self.times) or time.time < min(self.times):
                    continue
            self.cluster_list.append(cluster_algorithm())
            self.cluster_sizes.append(
                [len(cluster) for cluster in self.cluster_list[-1]]
            )
            if verbosity > 0:
                print("****TIME: {:8.2f}".format(time.time))
                print("---->Number of clusters {:d}".format(len(self.cluster_list[-1])))

        # Rewind Trajectory to beginning for other analysis
        self.universe.trajectory.rewind()
        self.cluster_list = self._create_generator(self.cluster_list)

    def condensed_ions(
        self,
        cluster,
        headgroup,
        ion,
        distances,
        valency=1,
        traj_pbc_style=None,
        method="pkdtree",
        pbc=True,
        wrap=False,
        verbosity=0,
    ):
        """
        Calculate number of species ion around each distance specified
        in distances around each cluster a cluster. 
        MDAnalsys.lib.distances.capped_distances() is used for this, 
        there is an issue with this code see this PR:
            https://github.com/MDAnalysis/mdanalysis/pull/2937
        as long as this is not fixed, I put pkdtree as standard method.

        Parameters
        ----------
        cluster: MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        headgroup : str
            atom identifier of the headgroup, can also be a specific
            part of the headgroup or even a tailgroup.
        ion : str
            atom identifier of the species whose degree of condensation
            around the headgroups is to be determined. 
        distances : float, list of floats
            Distance(s) up to which to determine the degree of 
            condenstation. Can be multiple.
        valency : int, optional
            How many ions are there per headgroup, by default 1.
        traj_pbc_style : string, optional
            Gromacs pbc definitions: mol or atom, by default
            None
        method : {'bruteforce', 'nsgrid', 'pkdtree'}, optional
            Method to be passed to mda.lib.distances.capped_distance(). 
        pbc : bool, optional
            Wether or not to take pbc into account, by default True
        verbosity : int, optional
            Determines how much the code talks, by default 0

        Returns:
        --------
        condensed_ions: list of ints
            the number of ions around headgroup for each distance.
        """
        condensed_ions = []
        # Handle pbc
        # self._set_pbc_style(traj_pbc_style)
        if pbc:
            box = self.universe.dimensions
        else:
            box = None
        if wrap:
            self.unwrap_cluster(cluster)

        # Define configuration set
        # This could be done with _select_species if refactored correctly.
        # Improvement: When looping over multiple distances do it first
        # for the largest distance, then adapt selection for shorter one.
        if isinstance(ion, str):
            ion = [ion]
        configset = self.universe.select_atoms("name {:s}".format(" ".join(ion)))
        configset = configset.atoms.positions
        # Define reference set
        if isinstance(headgroup, str):
            headgroup = [headgroup]
        refset = cluster.atoms.select_atoms("name {:s}".format(" ".join(headgroup)))
        refset = refset.atoms.positions

        condensed_ions = []
        for distance in distances:
            unique_idx = []
            # Call capped_distance for pairs
            pairs = MDAnalysis.lib.distances.capped_distance(
                refset,
                configset,
                distance,
                box=box,
                method=method,
                return_distances=False,
            )
            # Make unique
            if pairs.size > 0:
                unique_idx = MDAnalysis.lib.util.unique_int_1d(
                    np.asarray(pairs[:, 1], dtype=np.int64)
                )
            condensed_ions.append(len(unique_idx))

        return condensed_ions

    def unwrap_cluster(self, resgroup, box=None, unwrap=True, verbosity=0):
        """
        Make cluster which crosses pbc not cross pbc. Algorithm inspired
        by GROMACS but optimised so that it runs somewhat fast in
        python.

        Parameters
        ----------
        resgroup : MDAnalysis.ResidueGroup
            Cluster residues
        box : boxvector, optional
            boxvector. If None is given taken from trajectory, 
            by default None
        unwrap : bool, optional
            Wether or not to make molecules whole before treatment 
            (only necessary if pbc = atom in trjconv) but doesn't hurt.
            By default True
        verbosity : int, optional
            Chattiness, by default 0
        """
        # cluster is passed as resgroup and boxdimensions as xyz
        # will only support triclinic as of now..
        if box is None:
            box = self.universe.dimensions
        # Unwrap position if needed:
        for residue in resgroup:
            residue.atoms.unwrap(reference="cog", inplace=True)
        # Find initial molecule (closest to Centre of box)
        COB = box[:3] / 2
        weights = None  # This selects COG, if we mass weight its COM
        rmin = np.sum(box[:3]) * np.sum(box[:3])
        imin = -1
        for i, residue in enumerate(resgroup):
            tmin = self._pbc(residue.atoms.center(weights=None), COB, box)
            tmin = np.dot(tmin, tmin)
            if tmin < rmin:
                rmin = tmin
                imin = i

        # added and to_be_added store indices of residues
        # in resgroup
        nr_mol = resgroup.n_residues
        to_be_added = [i for i in range(nr_mol)]
        added = [to_be_added.pop(imin)]
        nr_added = 1

        # Setup the COG matrices
        refset_cog = resgroup[added[0]].atoms.center(None)

        configset_cog = np.zeros((nr_mol - 1, 3))
        for i, index in enumerate(to_be_added):
            configset_cog[i, :] = resgroup[index].atoms.center(None)

        while nr_added < nr_mol:
            # Find indices of nearest neighbours of a) res in micelle
            # (imin) and b) res not yet in micelle (jmin). Indices
            # w.r.t. resgroup
            imin, jmin = self._unwrap_ns(
                refset_cog, configset_cog, added, to_be_added, box
            )
            # Translate jmin by an appropriate vector if separated by a
            # pbc from rest of micelle.
            cog_i = resgroup[imin].atoms.center(weights=weights)
            cog_j = resgroup[jmin].atoms.center(weights=weights)

            dx = self._pbc(cog_j, cog_i, box)
            xtest = cog_i + dx
            shift = xtest - cog_j
            if np.dot(shift, shift) > 1e-8:
                if verbosity > 0.5:
                    print(
                        "Shifting molecule {:d} by {:.2f}, {:.2f}, {:.2f}".format(
                            jmin, shift[0], shift[1], shift[2]
                        )
                    )
                for atom in resgroup[jmin].atoms:
                    atom.position += shift
                cog_j += shift

            # Add added res COG to res already in micelle
            refset_cog = np.vstack((refset_cog, cog_j))
            nr_added += 1
            added.append(jmin)

            # Remove added res from res not already in micelle.
            _index = to_be_added.index(jmin)
            configset_cog = np.delete(configset_cog, _index, 0)
            del to_be_added[_index]

    def _unwrap_ns(
        self, refset_cog, configset_cog, added, to_be_added, box, method="pkdtree"
    ):
        """
        Find NN in refset_cog and configset_cog and pass back
        the indices stored in added and to_be added. 
        """
        distances = []
        dist = 8.0
        while len(distances) < 1:
            pairs, distances = MDAnalysis.lib.distances.capped_distance(
                refset_cog,
                configset_cog,
                dist,
                box=box,
                method=method,
                return_distances=True,
            )
            dist += 0.5

        minpair = np.where(distances == np.amin(distances))[0][0]

        imin = added[pairs[minpair][0]]
        jmin = to_be_added[pairs[minpair][1]]
        return imin, jmin

    @staticmethod
    def _pbc(r1, r2, box):
        # Rectangular boxes only
        # Calculate fdiag, hdiag, mhdiag
        fdiag = box[:3]
        hdiag = fdiag / 2

        dx = r1 - r2
        # Loop over dims
        for i in range(3):
            # while loop upper limit: if dx > hdiag shift by - fdiag
            while dx[i] > hdiag[i]:
                dx[i] -= fdiag[i]
            # while loop lower limit: if dx > hdiag shift by + fdiag
            while dx[i] < -hdiag[i]:
                dx[i] += fdiag[i]

        return dx

    @staticmethod
    def _sort_eig(eig_val, eig_vec, test=False, tensor=False):
        for i in range(2, 0, -1):
            index = np.where(eig_val == np.max(eig_val[: i + 1]))[0][0]
            # Switch columns
            eig_vec[:, [i, index]] = eig_vec[:, [index, i]]
            eig_val[i], eig_val[index] = eig_val[index], eig_val[i]

        if test:
            if isinstance(tensor, np.ndarray):
                for i in range(3):
                    t1 = np.matmul(tensor, eig_vec[:, i])
                    t2 = eig_val[i] * eig_vec[:, i]
                    if not np.allclose(t1, t2):
                        print(i, t1, t2)
                        raise RuntimeError("Eigenvector sorting gone wrong!")

            assert eig_val[2] >= eig_val[1]
            assert eig_val[1] >= eig_val[0]

        return eig_val, eig_vec

    @staticmethod
    def _gyration_tensor(cluster, weights, test=False):
        r = np.subtract(cluster.atoms.positions, cluster.atoms.center(weights))

        if weights is None:
            weights = np.ones(r.shape)
        else:
            weights = np.broadcast_to(weights, (3, weights.shape[0])).transpose()

        if test:
            assert np.abs(np.sum(r * weights)) < 1e-7

        gyration_tensor = np.matmul(r.transpose(), r * weights)

        return gyration_tensor

    def gyration(self, cluster, unwrap=False, test=False):
        """
        Calculte the gyration tensor defined as:

        Rg_ab = 1/N sum_i a_i*b_i ; a,b = {x,y,z}

        The eigenvalues of these vector are helpful
        to determine the shape of clusters. See:

        J. Phys. Chem. B 2014, 118, 3864−3880, and:
        MOLECULAR SIMULATION 2020, VOL. 46, NO. 4, 308–322.

        Parameters:
        -----------
        cluster: MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        unwrap: bool, optional
            Wether or not to unwrap cluster around pbc. Default False.
        """
        if unwrap:
            self.unwrap_cluster(cluster)

        gyration_tensor = self._gyration_tensor(cluster, None, test=test)

        gyration_tensor /= cluster.n_residues

        eig_val, eig_vec = np.linalg.eig(gyration_tensor)

        # Sort eig_vals and vector
        eig_val, eig_vec = self._sort_eig(
            eig_val, eig_vec, test=test, tensor=gyration_tensor
        )

        # Return in nm^2
        return eig_val / 100.0

    def inertia_tensor(self, cluster, unwrap=False, test=True):
        """
        Calculte the gyration tensor defined as:

        Ig_ab = 1/M sum_i m_i*(r^2 d_ab - r_a*r_b) 
        with a,b = {x,y,z} and r = (x,y,z) a d_ab is the
        kronecker delta. 

        Parameters:
        -----------
        cluster: MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        unwrap: bool, optional
            Wether or not to unwrap cluster around pbc. Default False.
        test: bool, optional
            Useful to compare some raw data with mdanalysis functions
            on the fly for when you're not sure if you fucke something 
            up.
        """
        if unwrap:
            self.unwrap_cluster(cluster)

        masses = cluster.atoms.masses
        inertia_tensor = self._gyration_tensor(cluster, masses, test=test)
        trace = np.trace(inertia_tensor)
        trace_array = trace * np.eye(3)
        inertia_tensor = trace_array - inertia_tensor
        if test:
            assert np.sum(inertia_tensor - cluster.moment_of_inertia() < 1e-6)

        inertia_tensor /= np.sum(cluster.masses)

        eig_val, eig_vec = np.linalg.eig(inertia_tensor)

        # Sort eig_vals and vector
        eig_val, eig_vec = self._sort_eig(
            eig_val, eig_vec, test=test, tensor=inertia_tensor
        )

        return eig_val / 100.0

    def rgyr(self, cluster):
        pass

    def _create_generator(self, cluster_list):
        """
        Make cluster_list a generator expression.

        This works a bit weirdly. when looping over 
        it, all transformations are only valid as long
        as the trajectory is not loaded again. For example
        when you run the loop do the cluster unwrapping
        and then run the loop again, the unwrapping will
        be 'lost'.
        """
        i = 0
        for j, time in enumerate(self.universe.trajectory):
            if self.times is not None:
                if time.time < min(self.times):
                    i = i + 1
                    continue
                elif time.time > max(self.times):
                    break
            yield cluster_list[j - i]

        self.cluster_list = self._create_generator(cluster_list)

    def _get_cluster_list_static(self, cut_off=7.5):
        """Get Cluster from single frame with the static method
        
        This code simply loops over all atoms in the aggregate
        species and finds a cluster for each atom (all neighbours).
        This cluster is merged into the already found clusters.

        Parameters
        ----------
        cut_off : float, optional
            Radius around which to search for neighbours 

        Returns
        -------
        cluster_list : list of ResGroups or AtomGroups
        """
        cluster_list = []
        if self.search_level == "R":
            aggregate_species_atoms = self.aggregate_species.groupby("resids").values()
        elif self.search_level == "A":
            aggregate_species_atoms = self.aggregate_species

        for atoms in aggregate_species_atoms:
            cluster_temp = set(
                self.neighbour_search.search(
                    atoms=atoms, radius=cut_off, level=self.search_level
                )
            )

            cluster_list = self._merge_cluster(cluster_list, cluster_temp)

        return cluster_list

    def _merge_cluster(self, cluster_list, cluster_temp):
        """Code to merge a cluster into a cluster list
        
        This code merges a cluster into a clusterlist by reverse
        looping over the cluster list. Whenever an overlap is detected
        the new, temporary, cluster is merged into the old one and
        the new one is deleted. At the same time the temporary cluster
        is updated to be the old one merged with the new one to
        further look for overlap.

        Parameters
        ----------
        cluster_list : list of sets of clusterids
            All the sets of clusters already present.
        cluster_temp : set of clusterids
            Last found cluster, to be merged into the cluster_list

        Returns
        -------
        cluster_list : list of sets of clusterids
            Updated cluster_list

        """
        cluster_list.append(cluster_temp)
        merged_index = []

        # Loop through cluster_list in reverse with the actual also
        # the revesed indices
        for i, cluster in reversed(list(enumerate(cluster_list))):
            # If there is an intersection start merging. There is
            # always an intersection with the first cluster (identical)
            if bool(cluster.intersection(cluster_temp)):
                cluster_temp = cluster_temp | cluster  # Updating cluster_temp
                cluster_list[i] = cluster_temp
                merged_index.append(i)
                # If the cluster is not completely new there will be a
                # second intersection and we delete the previous cluster
                # Note that len(merged_index) goes back to 1 afterwards
                # and cluster_temp is the new merged cluster
                if len(merged_index) > 1.5:
                    del cluster_list[merged_index[0]]
                    del merged_index[0]
                elif len(merged_index) > 1.5:
                    warnings.warn("Wrong behaviour in cluster merge", UserWarning)

        return cluster_list

    def _get_cluster_list_dynamic(self, cut_off=7.5):
        """Get Cluster from single frame with dynamic algorithm

        Faster algorithm to find clusters. We single out one
        molecule or atom and find all its neighbours, then we call the same
        function again and find the neighbours neighbours,
        while excluding already found neighbours. We do this until
        we cant find any more neighbours. Then we choose another
        molecule or atom until we do not have any molecules/atoms left.

        Parameters
        ----------
        cut_off : float, optional
            Radius around which to search for neighbours

        Returns
        -------
        cluster_list : list of ResGroups or AtomGroups

        """
        cluster_list = []

        if self.search_level == "R":
            aggregate_species = self.aggregate_species.residues
        elif self.search_level == "A":
            aggregate_species = self.aggregate_species

        # Loop until all molecules have been in clusters
        while len(aggregate_species) > 0:
            # Initialize the search_set and temporary cluster with
            # one molecule or atom
            if self.search_level == "R":
                species_temp = aggregate_species[0].atoms.residues
            elif self.search_level == "A":
                species_temp = MDAnalysis.core.groups.AtomGroup([aggregate_species[0]])
            search_set = species_temp
            cluster_temp = species_temp
            # In this loop search set gets updated to only have new
            # neighours and cluster_temp grows
            while len(search_set) > 0:
                search_set, cluster_temp = self._grow_cluster(
                    cut_off, search_set, cluster_temp
                )
            # Once no more neighbours are found add the cluster to
            # the list and subtract the cluster from the aggregate
            # species.
            # Possible Improvement: Update NeighbourSearch object
            cluster_list.append(cluster_temp)
            aggregate_species = aggregate_species.difference(cluster_temp)

        return cluster_list

    def _grow_cluster(self, cut_off, search_set, cluster_temp):
        """Code to grow a cluster (cluster_temp) and obtain new search set
        (search_set)

        This algorithm looks for neighbours of atoms in search_set
        and adds them to the temporary cluster. The search_set
        is updated to only include newly found neighbours not
        already present in search_set. The neighbours searching
        still looks in all the atoms present in the aggregate_species.

        Parameters
        ----------
        cut_off : float, optional
            Radius around which to search for neighbours
        search_set : MDAnalysis ResidueGroup
            Atoms for which to look for neighbours
        cluster_temp : MDAanalysis ResidueGroup
            All the atoms/residues currently in the cluster

        Returns
        ------- 
        search_set : MDAnalysis ResidueGroup
            Atoms for which to look for neighbours updated to be
            the latest found ones
        cluster_temp : MDAanalysis ResidueGroup
            All the atoms/residues currently in the cluster updated
            to include the latest found ones

        """

        if self.search_level == "R":
            # Find neighbours and cast into ResidueGroup
            new_cluster_species = MDAnalysis.core.groups.ResidueGroup(
                self.neighbour_search.search(
                    atoms=self._select_species(search_set.atoms, style=self.style),
                    radius=cut_off,
                    level=self.search_level,
                )
            )
        elif self.search_level == "A":
            new_cluster_species = self.neighbour_search.search(
                atoms=search_set.atoms, radius=cut_off, level=self.search_level
            )

        # The new search_set should only have atoms not already in the cluster
        search_set = new_cluster_species.difference(cluster_temp)
        # The new temporary cluster is updated
        cluster_temp = cluster_temp.union(new_cluster_species)

        return search_set, cluster_temp

    def plot_histogram(
        self,
        ax,
        frames=(0, 1, 1),
        maxbins=False,
        density=True,
        filename=None,
        *args,
        **kwargs
    ):
        """Method to plot histograms for different timeframes

        Examples
        --------
        None yet

        Note
        ----
        By passing a axis to this function, loads of things can actually
        be externally overriden to fit every need.

        Parameters
        ----------
        ax : matplotlib axis object
        frames : list of tuples, optional
            for each tuple, the corresponding frames are averaged to
            give one cluster size distribution. Give tuple as:
            (first frame, last frame, stepsize to go throug frames)
        maxbins : bool, optional
            Set to true if you want as many bins as there are monomers
            in the largest cluster
        density : bool, optional
            Whether or not to plot for absolute occurences or
            probabilities
        filename : string, optional
            If string is given, save the plot under that name. Specify
            if you want pdf, png etc..

        Returns
        -------
        ax : matplotlib axis object
        """
        # Check if the frames desired are available
        if not isinstance(frames, list):
            frames = [frames]
        cluster_list_length = len(self.cluster_sizes)
        maxframe = max([index[1] for index in frames])
        if maxframe > cluster_list_length:
            raise ValueError(
                "Upper Frame limit out of range, maximal frame \
                              Number is {:d}".format(
                    cluster_list_length
                )
            )

        # Get the size distribution of all frames for all frames
        masterlist = []
        for frames_i in frames:
            masterlist.append(self._get_cluster_distribution(frames_i))

        # By making as many bins as molecules in the largest cluster
        # there is a bar for each clustersize
        if maxbins is True:
            bins = max(
                [max(cluster_distribution) for cluster_distribution in masterlist]
            )
        else:
            bins = None

        ax.hist(masterlist, bins=bins, density=density, *args, **kwargs)

        ax.set_xlabel("Number of Monomers")
        if not density is True:
            ystring = "Number of Occurrences"
        else:
            ystring = "Probability"
        ax.set_ylabel(ystring)

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def _get_cluster_distribution(self, frames):
        """Helper for plot_histogram to get a cluster distribution

        Parameters
        ----------
        frames : tuple of int
            first frame, last frame and stepsize

        Returns
        -------
        cluster_distribution: list of int
            All the clusterssizes in all the frames specified
        """
        cluster_distribution = []
        for frame in self.cluster_sizes[slice(*frames)]:
            for size in frame:
                cluster_distribution.append(size)

        return cluster_distribution
