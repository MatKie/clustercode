import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NeighborSearch
import warnings
import matplotlib.pyplot as plt
from clustercode.BaseUniverse import BaseUniverse
from clustercode.ClusterSearch import ClusterSearch
from clustercode.CondensedIons import CondensedIons
from clustercode.UnwrapCluster import UnwrapCluster
from clustercode.Gyration import Gyration
import numpy as np

from clustercode.UnwrapCluster import UnwrapCluster

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
        self.universe = self._get_universe(self._coord, self._traj)

        self.ClusterSearch = ClusterSearch(self.universe, self.selection)
        self.ClusterSearch.cluster_analysis(
            cut_off=cut_off,
            times=times,
            style=style,
            measure=measure,
            algorithm=algorithm,
            work_in=work_in,
            traj_pbc_style=traj_pbc_style,
            pbc=pbc,
            verbosity=verbosity,
        )
        self.cluster_sizes = self.ClusterSearch.cluster_sizes

    @property
    def cluster_list(self):
        if self.ClusterSearch is None:
            raise RuntimeError("cluster_list must be generated by cluster_analysis")
        return self.ClusterSearch.cluster_list

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
        UnwrapCluster().unwrap_cluster(
            resgroup, box=box, unwrap=unwrap, verbosity=verbosity
        )

    def condensed_ions(
        self,
        cluster,
        headgroup,
        ion,
        distances,
        method="pkdtree",
        pbc=True,
        wrap=False,
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
        method : {'bruteforce', 'nsgrid', 'pkdtree'}, optional
            Method to be passed to mda.lib.distances.capped_distance().
        pbc : bool, optional
            Wether or not to take pbc into account, by default True

        Returns:
        --------
        condensed_ions: list of ints
            the number of ions around headgroup for each distance.
        """
        return CondensedIons().condensed_ions(
            cluster, headgroup, ion, distances, method=method, pbc=pbc, wrap=wrap
        )

    def calc_f_factors(self, cluster, unwrap=False, test=False):
        """
        Calculate eigenvalues of gryation tensor (see self.gyration())
        and calculate f_32 and f_21 from their square roots:

        f_32 = (Rg_33 - Rg_22) / Rg_33
        f_21 = (Rg_22 - Rg_11) / Rg_33

        Rg_33 is the eigenvalue belonging to the principal axis -- largest
        value.

        J. Phys. Chem. B 2014, 118, 3864−3880, and:
        MOLECULAR SIMULATION 2020, VOL. 46, NO. 4, 308–322.

        Parameters:
        -----------
        cluster: MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        unwrap: bool, optional
            Wether or not to unwrap cluster around pbc. Default False.

        Returns:
        --------
        f-factors : tuple of float
            f_32 and f_21, as defined above.
        """
        return Gyration().calc_f_factors(cluster, unwrap, test)

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

        Returns:
        --------
        eigenvalues : tuple of float
            eigenvalues (Rg_11^2, Rg_22^2, Rg_33^2) of the gyration
            tensor in nm, starting with the largest one corresponding to the
            major axis (different than for inertia per gyration definiton).
        """
        return Gyration().gyration(cluster, unwrap, test)

    def inertia_tensor(self, cluster, unwrap=False, test=True):
        """
        Calculte the inertia tensor defined as:

        Ig_ab = 1/M sum_i m_i*(r^2 d_ab - r_a*r_b)
        with a,b = {x,y,z} and r = (x,y,z) a d_ab is the
        kronecker delta. Basically mass weightes distance of a particle
        from an axis.
        The matrix is diagonalised and the eigenvalues are the
        moment of inertia along the principal axis, where the smallest
        value accompanies the major axis (the most mass is
        close to this axis). The largest value accompanies the minor
        axis.

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
        Returns:
        --------
        eigenvalue : tuple of float
        Starting with the lowest value corresponding to the major axis.
        """
        return Gyration().inertia_tensor(cluster, unwrap, test)

    def rgyr(
        self, cluster, mass=False, components=True, pca=True, unwrap=False, test=False
    ):
        """
        Calculate the radius of gyration with mass weightes or non
        mass weighted units (along prinicipal components)

        Rg = sqrt(sum_i mi(xi^2+yi^2+zi^2)/sum_i mi) if mass weighted
        Rg = sqrt(sum_i (xi^2+yi^2+zi^2)/sum_i i) if not mass weighted

        component rg defined like this:

        rg_x = sqrt(sum_i mi(yi^2+zi^2)/sum_i mi),

        Parameters
        ----------
        cluster: MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        mass : boolean, optional
            wether or not to mass weight radii, by default False
        components : boolean, optional
            wether or not to calculate rgyr components, by default False
        pca : booelan, optional
            wether or not to calculate rgyr components w.r.t. principal
            component vectors or not, by default True
        unwrap : boolean, optional
            wether or not to unwrap cluster around pbc, by default False
        test : boolean, optional
            wether or not to perform some sanity checks, by default False

        Returns:
        rg : float
           radius of gyration
        rg_i : floats, optional
            If components is True, the components along x, y, z direction
            and if pca is also true along the threeprincipal axis, starting
            with the principal axis with the largest eigenvalue.
        """
        return Gyration().rgyr(cluster, mass, components, pca, unwrap, test)

    def angle_distribution(self, cluster, ref1, ref2, ref3, unwrap=False):
        """
        Calculate all the angles between the three atoms specified
        (ref1-3) for each molecule in cluster.

        Parameters
        ----------
        cluster : MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        ref1 : string
            atomname within the molecules clustered in 'cluster'. One
            of the edge atoms of the angle: ref1 - ref2 - ref3
        ref2 : string
            atomname within the molecules clustered in 'cluster'.
            Central atom of the angle: ref1 - ref2 - ref3
        ref3 : string
            atomname within the molecules clustered in 'cluster'. One
            of the edge atoms of the angle: ref1 - ref2 - ref3
        unwrap : boolean, optional
            wether or not to unwrap cluster around pbc, by default False

        Returns
        -------
        angles (list)
            List of the angles between atoms ref1 - ref2 - ref3

        Raises
        ------
        ValueError
            In case the name of any of the references is ambiguous
            (more than one atom with this name -- should be impossible).

        @Todo: Make it possible to pass atoms instead of atom strings
               to reduce the number of atom selection when doing bond
               and angle distributions
        """
        if unwrap:
            self.unwrap_cluster(cluster)
        angles = []
        for molecule in cluster:
            refs = []
            for item in [ref1, ref2, ref3]:
                if item != "COM":
                    ref = molecule.atoms.select_atoms("name {:s}".format(item))
                    ref_pos = ref.positions
                    if len(ref_pos) > 1.2:
                        raise ValueError(
                            "Ambiguous reference choosen ({:s})".format(item)
                        )
                    refs.append(ref_pos[0])
                else:
                    refs.append(cluster.atoms.center_of_mass())
            a, b, c = refs
            r1 = a - b
            r2 = c - b
            r1_norm = np.linalg.norm(r1)
            r2_norm = np.linalg.norm(r2)
            cos_a = np.matmul(r1, r2.transpose()) / (r1_norm * r2_norm)
            alpha = np.arccos(cos_a) * 180.0 / np.pi
            angles.append(alpha)
        return angles

    def distance_distribution(self, cluster, ref1, ref2, unwrap=False):
        """
        Calculate all the distances between the two atoms specified
        (ref1, ref2) for each molecule in cluster.

        Parameters
        ----------
        cluster : MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        ref1 : string
            atomname within the molecules clustered in 'cluster'.
        ref2 : string
            atomname within the molecules clustered in 'cluster'.
        unwrap : boolean, optional
            wether or not to unwrap cluster around pbc, by default False

        Returns
        -------
        distances (list)
            List of the distances between atoms ref1 - ref2

        Raises
        ------
        ValueError
            In case the name of any of the references is ambiguous
            (more than one atom with this name -- should be impossible).

        @Todo: Make it possible to pass atoms instead of atom strings
               to reduce the number of atom selection when doing bond
               and angle distributions
        """
        if unwrap:
            self.unwrap_cluster(cluster)
        distances = []
        for molecule in cluster:
            refs = []
            for item in [ref1, ref2]:
                if item != "COM":
                    ref = molecule.atoms.select_atoms("name {:s}".format(item))
                    ref_pos = ref.positions
                    if len(ref_pos) > 1.2:
                        raise ValueError(
                            "Ambiguous reference choosen ({:s})".format(item)
                        )
                    refs.append(ref_pos[0])
                else:
                    refs.append(cluster.atoms.center_of_mass())
            a, b = refs
            r1_norm = np.linalg.norm(a - b)
            distances.append(r1_norm)
        return distances

    def plot_histogram(
        self,
        ax,
        frames=(0, 1, 1),
        maxbins=False,
        density=True,
        filename=None,
        sizeweight=True,
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

        if sizeweight:
            weights = masterlist
        else:
            weights = None
        # By making as many bins as molecules in the largest cluster
        # there is a bar for each clustersize
        if maxbins is True:
            bins = max(
                [max(cluster_distribution) for cluster_distribution in masterlist]
            )
        else:
            bins = None

        ax.hist(
            masterlist, weights=weights, bins=bins, density=density, *args, **kwargs
        )

        ax.set_xlabel("Number of Monomers")
        if not density is True:
            ystring = "Number of Occurrences"
        else:
            ystring = "Probability"
        ax.set_ylabel(ystring)

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
