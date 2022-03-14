import MDAnalysis as mda
from clustercode import ClusterEnsemble
from pytest import approx
import pytest
import numpy as np

# The trajectory includes a large single micelles split across mutliple PBCs
# It was pretreated with gmx trjconv -pbc atom
tpr = "clustercode/tests/cluster/files/topol_no_solv.tpr"
traj = "clustercode/tests/cluster/files/traj_pbc_problematic_atom.xtc"
# traj = "clustercode/tests/cluster/files/traj_pbc_problematic_cluster.xtc"
clstr = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"])
AtomClstr = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"])
clstr.cluster_analysis()
AtomClstr.cluster_analysis(work_in="Atom")

# gmx gyrate -f traj_pbc_problematic_cluster.xtc -s topol_no_solv.tpr -p yes
gmx_gyrate = [
    [3.28795, 1.86045, 2.97682, 3.04934],
    [3.22206, 1.86464, 2.9183, 2.96141],
    [3.20438, 1.89937, 2.8877, 2.93081],
    [3.17957, 1.90879, 2.83768, 2.91949],
    [3.17345, 1.86303, 2.87302, 2.90112],
    [3.19677, 1.83934, 2.91293, 2.92753],
]
gmx_principal = [
    [3.50891250, 8.983379375, 9.426435000],
    [3.52470625, 8.633655000, 8.890634375],
    [3.65722625, 8.453534375, 8.707842500],
    [3.69359125, 8.163197500, 8.640699375],
    [3.51862312, 8.367801875, 8.532266875],
    [3.42970875, 8.601878750, 8.688352500],
]
gmx_gyrate_cmct = [
    [3.11579, 1.66664, 2.85055, 2.91769],
    [3.03193, 1.66153, 2.77456, 2.81538],
    [3.00879, 1.70063, 2.73858, 2.77734],
    [2.98037, 1.70727, 2.68214, 2.76705],
    [2.97118, 1.66493, 2.71059, 2.74528],
    [3.00965, 1.62916, 2.77346, 2.78743],
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

    def test_moment_of_inertia(self):
        """
        This test checks if the result for the inertia tensor is the same
        than in gromacs. There is a relatively high deviation of up to
        0.13 units.
        % WHAT ARE THE UNITS? WHY DO WE DIVIDE BY THE MASS IN .inertia_tensor()?
        """
        for clusters, their_inertia in zip(clstr.cluster_list, gmx_principal):
            for cluster in clusters:
                clstr.unwrap_cluster(cluster)
                our_inertia = clstr.inertia_tensor(cluster)
                for item, other_item in zip(our_inertia, their_inertia):
                    assert item == approx(other_item, abs=0.13)

    @pytest.mark.xfail(
        reason="rgyr was only coded for Residuegroups",
    )
    def test_calc_rgyr_work_in_atomistic_quali(self):
        for clusters in AtomClstr.cluster_list:
            for cluster in clusters:
                clstr.unwrap_cluster(cluster)
                _ = clstr.rgyr(cluster, mass=True, components=True, pca=True)

    @pytest.mark.xfail(
        reason="rgyr was only coded for Residuegroups",
    )
    def test_calc_rgyr_work_in_atomistic_quanti(self):
        """
        This test checks if the result for the radius of gyration calculation
        after a cluster finding and unwrapping of the cluster is the same
        as observed with gmx gyrate -p yes. Only looking at carbon atoms.
        Accuracy is 0.0005nm
        """
        for clusters, their_gyrate in zip(AtomClstr.cluster_list, gmx_gyrate_cmct):
            for cluster in clusters:
                clstr.unwrap_cluster(cluster)
                our_gyrate = clstr.rgyr(cluster, mass=True, components=True, pca=True)
                for item, other_item in zip(their_gyrate, our_gyrate):
                    assert item == approx(other_item, abs=5e-4)


# TestGyration().test_moment_of_inertia()
