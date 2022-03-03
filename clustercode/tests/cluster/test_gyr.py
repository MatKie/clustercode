import MDAnalysis as mda
from clustercode import ClusterEnsemble
from pytest import approx
import numpy as np

# The trajectory includes a large single micelles split across mutliple PBCs
# It was pretreated with gmx trjconv -pbc atom
tpr = "clustercode/tests/cluster/files/topol_no_solv.tpr"
traj = "clustercode/tests/cluster/files/traj_pbc_problematic_atom.xtc"
# traj = "clustercode/tests/cluster/files/traj_pbc_problematic_cluster.xtc"
clstr = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"])

clstr.cluster_analysis()

gmx_gyrate = [
    [3.28795, 1.86045, 2.97682, 3.04934],
    [3.22206, 1.86464, 2.9183, 2.96141],
    [3.20438, 1.89937, 2.8877, 2.93081],
    [3.17957, 1.90879, 2.83768, 2.91949],
    [3.17345, 1.86303, 2.87302, 2.90112],
    [3.19677, 1.83934, 2.91293, 2.92753],
]


class TestGyration:
    def test_mass_weighted_rgyr(self):
        """
        This test checks if the result for the radius of gyration calculation
        after a cluster finding and unwrapping of the cluster is the same
        as observed with gmx gyrate -p yes. 
        Accuracy is 0.0005nm
        """
        for clusters, their_gyrate in zip(clstr.cluster_list, gmx_gyrate):
            for cluster in clusters:
                clstr.unwrap_cluster(cluster)
                our_gyrate = clstr.rgyr(cluster, mass=True, components=True, pca=True)
                for item, other_item in zip(their_gyrate, our_gyrate):
                    assert item == approx(other_item, abs=5e-4)

