"""
Here I test if the COMs of trajectories, pre-processed by different
gmx trjconv -pbc options, result in the same value if processed the same
way. This includes cluster finding and unwrapping and then finding the 
COM with centre_of_mass(pbc=True/False). Both is tested and results
in different results.
The trajectories yield the same COMs to an accuracy of 0.15%.
"""
import MDAnalysis as mda
from clustercode.ClusterEnsemble import ClusterEnsemble
from pytest import approx

# These files comprise the same trajectory but processed with different
# gmx trjconv -pbc options (cluster, atom and mol). The trajectory
# includes a large single micelles split across mutliple PBCs.
tcluster = "clustercode/tests/cluster/files/traj_pbc_problematic_cluster.xtc"
tatom = "clustercode/tests/cluster/files/traj_pbc_problematic_atom.xtc"
tmol = "clustercode/tests/cluster/files/traj_pbc_problematic_mol.xtc"
tpr = "clustercode/tests/cluster/files/topol_no_solv.tpr"

# Find all clusters, should only be one really, and then unwrap it and
# unwrap the cluster and calculate the com.
clstr_ensembles = []
for traj in [tcluster, tatom, tmol]:
    clstr_ensembles.append(ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"]))

for clstr_ensemble in clstr_ensembles:
    clstr_ensemble.cluster_analysis(work_in="Residue")

pbc_COMs = []
nopbc_COMs = []
verbosity = 0
for j, clstr_ensemble in enumerate(clstr_ensembles):
    this_pbc_COMs = []
    this_nopbc_COMs = []
    for i, clusterlist in enumerate(clstr_ensemble.cluster_list):
        for cluster in clusterlist:
            clstr_ensemble.unwrap_cluster(cluster, verbosity=verbosity)
            nopbc = cluster.center_of_mass(pbc=False)
            pbc = cluster.center_of_mass(pbc=True)
            this_pbc_COMs.append(pbc)
            this_nopbc_COMs.append(nopbc)
    pbc_COMs.append(this_pbc_COMs)
    nopbc_COMs.append(this_nopbc_COMs)


def assert_vector(v1, v2):
    for c1, c2 in zip(v1, v2):
        assert c1 == approx(c2, 1.5e-3)


def dont_assert_vector(v1, v2):
    truth_value = []
    for c1, c2 in zip(v1, v2):
        if c1 != approx(c2, abs=1e-3):
            truth_value.append(True)
        else:
            truth_value.append(False)
    if not True in truth_value:
        print(v1, v2)
        raise AssertionError


class TestPBC:
    def test_pbc(self):
        """
        Test, when applying pbc all COMs are the same. 
        The accuracy is 0.15%.
        """
        for v1, v2, v3 in zip(*pbc_COMs):
            assert_vector(v1, v2)
            assert_vector(v3, v2)
            assert_vector(v2, v3)

    def test_nopbc(self):
        """
        Test, when applying pbc all COMs are the same. 
        The accuracy is 0.15%.
        This tests if our unwrap_cluster method produces the same COM 
        for all trajectories
        """
        for v1, v2, v3 in zip(*nopbc_COMs):
            assert_vector(v1, v2)
            assert_vector(v3, v2)
            assert_vector(v2, v3)

    def test_atom(self):
        """
        Check that the atom trajectory is different for pbc and not pbc.
        A better test would check by which boxlength they differ.
        Accuracy is 0.001 nm.
        """
        for v1, v2 in zip(pbc_COMs[1], nopbc_COMs[1]):
            dont_assert_vector(v1, v2)

