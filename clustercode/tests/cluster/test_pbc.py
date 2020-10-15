import MDAnalysis as mda
from clustercode.ClusterEnsemble import ClusterEnsemble
from pytest import approx

tcluster = "clustercode/tests/cluster/files/traj_pbc_problematic_cluster.xtc"
tatom = "clustercode/tests/cluster/files/traj_pbc_problematic_atom.xtc"
tmol = "clustercode/tests/cluster/files/traj_pbc_problematic_mol.xtc"
tpr = "clustercode/tests/cluster/files/topol_no_solv.tpr"

clstr_ensembles = []
for traj in [tcluster, tatom, tmol]:
    clstr_ensembles.append(ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"]))

for clstr_ensemble in clstr_ensembles:
    clstr_ensemble.cluster_analysis(work_in="Residue")

pbc_COMs = []
nopbc_COMs = []
for j, clstr_ensemble in enumerate(clstr_ensembles):
    for i, clusterlist in enumerate(clstr_ensemble.cluster_list):
        if i > 0.5:
            break
        for cluster in clusterlist:
            if j > 0.5:
                clstr_ensemble.unwrap_cluster(cluster)
            nopbc = cluster.center_of_mass(pbc=False)
            pbc = cluster.center_of_mass(pbc=True)
            pbc_COMs.append(pbc)
            nopbc_COMs.append(nopbc)
            break


def assert_vector(v1, v2):
    for c1, c2 in zip(v1, v2):
        assert c1 == approx(c2, 1e-3)


def dont_assert_vector(v1, v2):
    truth_value = []
    for c1, c2 in zip(v1, v2):
        if c1 != approx(c2, abs=1e-3):
            truth_value.append(True)
        else:
            truth_value.append(False)
    if True in truth_value:
        raise AssertionError


def test_pbc():
    assert_vector(pbc_COMs[0], pbc_COMs[1])
    assert_vector(pbc_COMs[2], pbc_COMs[1])


def test_nopbc():
    assert_vector(nopbc_COMs[0], nopbc_COMs[1])
    assert_vector(nopbc_COMs[2], nopbc_COMs[1])


def test_atom():
    dont_assert_vector(pbc_COMs[1], nopbc_COMs[1])


test_pbc()

test_atom()

test_nopbc()
