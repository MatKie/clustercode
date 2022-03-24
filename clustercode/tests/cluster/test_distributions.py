import pytest
import numpy as np
from clustercode import ClusterEnsemble

# The trajectory includes a large single micelles split across mutliple PBCs
# It was pretreated with gmx trjconv -pbc atom
traj = "clustercode/tests/cluster/files/traj_pbc_problematic_cluster.xtc"
tpr = "clustercode/tests/cluster/files/topol_no_solv.tpr"

uni = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"])

uni.cluster_analysis()


class TestDistributions:
    def test_distance_distribution(self):
        for clusters in uni.cluster_list:
            for cluster in clusters:
                distances = uni.distance_distribution(cluster, "C1", "C2", unwrap=False)
                assert len(distances) == 382
                assert np.mean(distances) == pytest.approx(4.15, abs=0.025)

    def test_angle_distribution(self):
        for clusters in uni.cluster_list:
            for cluster in clusters:
                angles = uni.angle_distribution(cluster, "C1", "C2", "C3", unwrap=False)
                assert len(angles) == 382
                assert np.mean(angles) == pytest.approx(145, abs=3)
