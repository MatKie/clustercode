from pytest import approx
import MDAnalysis as mda
from clustercode.ClusterEnsemble import ClusterEnsemble
import os
import copy

# The trajectory includes a large single micelles split across mutliple PBCs
# It was pretreated with gmx trjconv -pbc atom
traj = "clustercode/tests/cluster/files/traj_pbc_problematic_atom.xtc"
tpr = "clustercode/tests/cluster/files/topol_no_solv.tpr"

uni = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"])

uni.cluster_analysis()


class TestCondensation:
    def test_condensation_wrap_insensitive(self):
        """
        This tests if the result from ClusterEnsemble.condensed_ions() is
        the same wether or not the cluster got wrapped.
        """
        for clusters in uni.cluster_list:
            for cluster in clusters:
                a, b = uni.condensed_ions(cluster, "SU", "NA", [4.4, 7.6])
                ai, bi = uni.condensed_ions(cluster, "SU", "NA", [4.4, 7.6], wrap=True)
                assert a == approx(ai)
                assert b == approx(bi)
