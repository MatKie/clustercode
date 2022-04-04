from clustercode.UnwrapCluster import UnwrapCluster
import numpy as np


class Distributions(object):
    def __init__(self):
        pass

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
            UnwrapCluster().unwrap_cluster(cluster)
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
            UnwrapCluster().unwrap_cluster(cluster)
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
