import sys
import os, copy
from pytest import approx
from clustercode.ClusterEnsemble import ClusterEnsemble
from clustercode.clustering import cluster_analysis

# This includes a single micelle (atomistic forcefield) of 100 surfactants
traj = "clustercode/tests/cluster/files/traj_small.xtc"
tpr = "clustercode/tests/cluster/files/topol_small.tpr"

ClstrEnsStatic = ClusterEnsemble(tpr, traj, ["C{:d}".format(i) for i in range(1, 13)])
ClstrEnsDynamic = ClusterEnsemble(tpr, traj, ["C{:d}".format(i) for i in range(1, 13)])

ClstrEnsStatic.cluster_analysis(algorithm="static")
ClstrEnsDynamic.cluster_analysis(algorithm="dynamic")


class TestClustering:
    def test_static_dynamic_single_micelle(self):
        """
        This test checks if the static and dynamic algorithm get the same
        result for this atomistic trajectory (one micelle only). Check if:
        - same amount of clusters
        - same number of surfactant in a cluster
        - same id's of surfactants in cluster.
        Not the best test as it tests a lot of things and also the 
        trajectory is not difficult enough
        """
        for static_clus_list, dynamic_clus_list in zip(
            ClstrEnsStatic.cluster_list, ClstrEnsDynamic.cluster_list
        ):
            diff_clust_count = len(static_clus_list) - len(dynamic_clus_list)

            assert diff_clust_count == 0

            static_molec_count = 0
            for cluster in static_clus_list:
                static_molec_count += len(cluster)
            dynamic_molec_count = 0
            for cluster in dynamic_clus_list:
                dynamic_molec_count += cluster.n_residues

            assert static_molec_count == dynamic_molec_count

            new_s_set = static_clus_list[0]
            for cluster in static_clus_list:
                new_s_set = new_s_set.union(cluster)

            new_d_set = dynamic_clus_list[0]
            for cluster in dynamic_clus_list:
                new_d_set = new_d_set.union(cluster)

            assert static_molec_count == len(new_s_set)
            assert dynamic_molec_count == new_d_set.n_residues

            for idxi, clusteri in enumerate(dynamic_clus_list):
                for idxj, clusterj in enumerate(dynamic_clus_list):
                    print(idxi, idxj)
                    print(clusteri)
                    print(clusteri.n_residues, clusterj.n_residues)
                    assert (clusteri.issuperset(clusterj) and idxi != idxj) == False

