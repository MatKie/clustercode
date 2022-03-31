import MDAnalysis as mda
from clustercode import ClusterEnsemble
from clustercode.Gyration import Gyration
from pytest import approx
import pytest
import numpy as np

gro = traj = "clustercode/tests/cluster/files/cylinder.gro"
universe = mda.Universe(gro)

cluster = universe.residues


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
