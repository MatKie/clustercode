import MDAnalysis as mda
from clustercode import ClusterEnsemble
from clustercode.Gyration import Gyration
from pytest import approx
import pytest
import numpy as np

gro = traj = "clustercode/tests/cluster/files/cylinder.gro"
gro_perfect = traj = "clustercode/tests/cluster/files/cylinder_perfect.gro"
universe = mda.Universe(gro)

cluster = universe.residues
perfect_cluster = mda.Universe(gro_perfect).residues


class TestGyrationTensor:
    def test_cog(self):
        r = Gyration._get_reduced_r(cluster, None)
        weights = Gyration._get_weights(cluster, None)
        assert np.abs(np.sum(r * weights)) < 1e-7

    def test_com(self):
        weights = cluster.atoms.masses
        r = Gyration._get_reduced_r(cluster, weights)
        weights = Gyration._get_weights(cluster, weights)
        assert np.abs(np.sum(r * weights)) < 1e-7

    def test_sort_eig_transformation(self):
        gyration_tensor = Gyration._gyration_tensor(cluster, None)
        eig_val, eig_vec = np.linalg.eig(gyration_tensor)
        eig_val, eig_vec = Gyration._sort_eig(eig_val, eig_vec, reverse=False)
        for i in range(3):
            t1 = np.matmul(gyration_tensor, eig_vec[:, i])
            t2 = eig_val[i] * eig_vec[:, i]
            assert t1 == approx(t2)

    def test_sort_eig_order(self):
        gyration_tensor = Gyration._gyration_tensor(cluster, None)
        eig_val, eig_vec = np.linalg.eig(gyration_tensor)
        eig_val, eig_vec = Gyration._sort_eig(eig_val, eig_vec, reverse=False)
        assert eig_val[2] >= eig_val[1]
        assert eig_val[1] >= eig_val[0]

    def test_sort_eig_reverse(self):
        gyration_tensor = Gyration._gyration_tensor(cluster, None)
        eig_val, eig_vec = np.linalg.eig(gyration_tensor)
        eig_val, eig_vec = Gyration._sort_eig(eig_val, eig_vec, reverse=True)
        assert eig_val[2] <= eig_val[1]
        assert eig_val[1] <= eig_val[0]

    def test_gyration(self):
        gyration_values = Gyration().gyration(perfect_cluster)
        assert gyration_values[0] > gyration_values[1]
        assert gyration_values[1] == approx(gyration_values[2])

    def test_rg(self):
        gyration_values = Gyration().rgyr(perfect_cluster)
        assert gyration_values[1] == approx(np.sqrt(0.8))
        assert gyration_values[2] == approx(gyration_values[3])
        assert gyration_values[2] == approx(np.sqrt(5.65))
