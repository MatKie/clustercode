from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.lib.util import unique_int_1d
from clustercode.UnwrapCluster import UnwrapCluster
import numpy as np


class CondensedIons(object):
    def __init__(self, universe):
        self.universe = universe

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
        condensed_ions = []
        # Handle pbc
        # self._set_pbc_style(traj_pbc_style)
        if pbc:
            box = self.universe.dimensions
        else:
            box = None
        if wrap:
            UnwrapCluster(self.universe).unwrap_cluster(cluster)

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
            pairs = capped_distance(
                refset,
                configset,
                distance,
                box=box,
                method=method,
                return_distances=False,
            )
            # Make unique
            if pairs.size > 0:
                unique_idx = unique_int_1d(np.asarray(pairs[:, 1], dtype=np.int64))
            condensed_ions.append(len(unique_idx))

        return condensed_ions