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
        """
        Test if reduced r & weights calculates centre of geometry accurately.
        """
        r = Gyration._get_reduced_r(cluster, None)
        weights = Gyration._get_weights(cluster, None)
        assert np.abs(np.sum(r * weights)) < 1e-7

    def test_com(self):
        """
        Test if reduced r & weights calculates centre of mass accurately.
        """
        weights = cluster.atoms.masses
        r = Gyration._get_reduced_r(cluster, weights)
        weights = Gyration._get_weights(cluster, weights)
        assert np.abs(np.sum(r * weights)) < 1e-7

    def test_sort_eig_transformation(self):
        """
        Test if eigenvalue sorting results in the correct sorting via
        A*evec_i = eval_i*evec_i
        """
        gyration_tensor = Gyration._gyration_tensor(cluster, None)
        eig_val, eig_vec = np.linalg.eig(gyration_tensor)
        eig_val, eig_vec = Gyration._sort_eig(eig_val, eig_vec, reverse=False)
        for i in range(3):
            t1 = np.matmul(gyration_tensor, eig_vec[:, i])
            t2 = eig_val[i] * eig_vec[:, i]
            assert t1 == approx(t2)

    def test_sort_eig_order(self):
        """
        Test if eigenvalue sorting has the correct sorting
        """
        gyration_tensor = Gyration._gyration_tensor(cluster, None)
        eig_val, eig_vec = np.linalg.eig(gyration_tensor)
        eig_val, eig_vec = Gyration._sort_eig(eig_val, eig_vec, reverse=False)
        assert eig_val[2] >= eig_val[1]
        assert eig_val[1] >= eig_val[0]

    def test_sort_eig_reverse(self):
        """
        Test if eigenvalue sorting has the correct sorting in reverse
        """
        gyration_tensor = Gyration._gyration_tensor(cluster, None)
        eig_val, eig_vec = np.linalg.eig(gyration_tensor)
        eig_val, eig_vec = Gyration._sort_eig(eig_val, eig_vec, reverse=True)
        assert eig_val[2] <= eig_val[1]
        assert eig_val[1] <= eig_val[0]

    def test_gyration(self):
        """
        The gyration value is the root-mean-square distance of
        the radii components along the respecitve axis. The mrs radii
        of xyz are 5.25, 0.4, 0.4.
        """
        gyration_values = Gyration().gyration(perfect_cluster)
        assert gyration_values[0] > gyration_values[1]
        assert gyration_values[1] == approx(gyration_values[2])
        assert gyration_values[0] == approx(5.25)
        assert gyration_values[1] == approx(0.4)

    def test_rg(self):
        """
        The radius of gyration values x,y,z are the root-mean-square of
        the radii components orthogonal to each axis. THe rms radii of
        xyz are 5.25, 0.4, 0.4. Therefore the rg_i^2 must be 0.8, 5.65, 5.65.
        """
        gyration_values = Gyration().rgyr(perfect_cluster)
        assert gyration_values[1] == approx(np.sqrt(0.4 + 0.4))
        assert gyration_values[2] == approx(gyration_values[3])
        assert gyration_values[2] == approx(np.sqrt(5.25 + 0.4))

    def test_rg_mass(self):
        """
        Check if, when all masses in the system are equal, mass and
        number weighted are being the same.
        """
        gyration_values = Gyration().rgyr(perfect_cluster)
        gyration_mass_values = Gyration().rgyr(perfect_cluster, mass=True)
        for non_mass, mass in zip(gyration_values, gyration_mass_values):
            assert non_mass == approx(mass)

    def test_intertia_w_mda(self):
        """
        Check if my inertia_tensor gives same results than MDA
        """
        inertia_tensor = Gyration._inertia_tensor(cluster)
        assert inertia_tensor == approx(cluster.moment_of_inertia())