import numpy as np
from clustercode.UnwrapCluster import UnwrapCluster

# implement this without the unwrap. Then make UnwrapCluster work based
# on the unvierse from cluster.universe


class Gyration(object):
    def __init__(self):
        pass

    def calc_f_factors(self, cluster, unwrap=False, test=False):
        """
        Calculate eigenvalues of gryation tensor (see self.gyration())
        and calculate f_32 and f_21 from their square roots:

        f_32 = (Rg_33 - Rg_22) / Rg_33
        f_21 = (Rg_22 - Rg_11) / Rg_33

        Rg_33 is the eigenvalue belonging to the principal axis -- largest
        value.

        J. Phys. Chem. B 2014, 118, 3864−3880, and:
        MOLECULAR SIMULATION 2020, VOL. 46, NO. 4, 308–322.

        Parameters:
        -----------
        cluster: MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        unwrap: bool, optional
            Wether or not to unwrap cluster around pbc. Default False.

        Returns:
        --------
        f-factors : tuple of float
            f_32 and f_21, as defined above.
        """

        rg_33, rg_22, rg_11 = np.sqrt(self.gyration(cluster, unwrap, test))

        f_32 = (rg_33 - rg_22) / rg_33
        f_21 = (rg_22 - rg_11) / rg_33

        return (f_32, f_21)

    def gyration(self, cluster, unwrap=False, test=False):
        """
        Calculte the gyration tensor defined as:

        Rg_ab = 1/N sum_i a_i*b_i ; a,b = {x,y,z}

        The eigenvalues of these vector are helpful
        to determine the shape of clusters. See:

        J. Phys. Chem. B 2014, 118, 3864−3880, and:
        MOLECULAR SIMULATION 2020, VOL. 46, NO. 4, 308–322.

        Parameters:
        -----------
        cluster: MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        unwrap: bool, optional
            Wether or not to unwrap cluster around pbc. Default False.

        Returns:
        --------
        eigenvalues : tuple of float
            eigenvalues (Rg_11^2, Rg_22^2, Rg_33^2) of the gyration
            tensor in nm, starting with the largest one corresponding to the
            major axis (different than for inertia per gyration definiton).
        """
        gyration_tensor = self._gyration_tensor(cluster, None)

        gyration_tensor /= cluster.n_residues

        eig_val, eig_vec = np.linalg.eig(gyration_tensor)

        # Sort eig_vals and vector
        eig_val, eig_vec = self._sort_eig(eig_val, eig_vec, reverse=True)

        # Return in nm^2
        return eig_val / 100.0

    def inertia_tensor(self, cluster, unwrap=False, test=True):
        """
        Calculte the inertia tensor defined as:

        Ig_ab = 1/M sum_i m_i*(r^2 d_ab - r_a*r_b)
        with a,b = {x,y,z} and r = (x,y,z) a d_ab is the
        kronecker delta. Basically mass weightes distance of a particle
        from an axis.
        The matrix is diagonalised and the eigenvalues are the
        moment of inertia along the principal axis, where the smallest
        value accompanies the major axis (the most mass is
        close to this axis). The largest value accompanies the minor
        axis.

        Parameters:
        -----------
        cluster: MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        unwrap: bool, optional
            Wether or not to unwrap cluster around pbc. Default False.
        test: bool, optional
            Useful to compare some raw data with mdanalysis functions
            on the fly for when you're not sure if you fucke something
            up.
        Returns:
        --------
        eigenvalue : tuple of float
        Starting with the lowest value corresponding to the major axis.
        """
        if unwrap:
            UnwrapCluster().unwrap_cluster(cluster)

        masses = cluster.atoms.masses
        inertia_tensor = self._gyration_tensor(cluster, masses)
        trace = np.trace(inertia_tensor)
        trace_array = trace * np.eye(3)
        inertia_tensor = trace_array - inertia_tensor
        if test:
            assert np.sum(inertia_tensor - cluster.moment_of_inertia() < 1e-6)

        inertia_tensor /= np.sum(cluster.masses)

        eig_val, eig_vec = np.linalg.eig(inertia_tensor)

        # Sort eig_vals and vector
        eig_val, eig_vec = self._sort_eig(eig_val, eig_vec)

        return eig_val / 100.0

    def rgyr(
        self, cluster, mass=False, components=True, pca=True, unwrap=False, test=False
    ):
        """
        Calculate the radius of gyration with mass weightes or non
        mass weighted units (along prinicipal components)

        Rg = sqrt(sum_i mi(xi^2+yi^2+zi^2)/sum_i mi) if mass weighted
        Rg = sqrt(sum_i (xi^2+yi^2+zi^2)/sum_i i) if not mass weighted

        component rg defined like this:

        rg_x = sqrt(sum_i mi(yi^2+zi^2)/sum_i mi),

        Parameters
        ----------
        cluster: MDAnalysis.ResidueGroup
            cluster on which to perform analysis on.
        mass : boolean, optional
            wether or not to mass weight radii, by default False
        components : boolean, optional
            wether or not to calculate rgyr components, by default False
        pca : booelan, optional
            wether or not to calculate rgyr components w.r.t. principal
            component vectors or not, by default True
        unwrap : boolean, optional
            wether or not to unwrap cluster around pbc, by default False
        test : boolean, optional
            wether or not to perform some sanity checks, by default False

        Returns:
        rg : float
           radius of gyration
        rg_i : floats, optional
            If components is True, the components along x, y, z direction
            and if pca is also true along the threeprincipal axis, starting
            with the principal axis with the largest eigenvalue.
        """
        if unwrap:
            UnwrapCluster().unwrap_cluster(cluster)

        if mass:
            weights = cluster.atoms.masses
        else:
            weights = np.ones(cluster.atoms.masses.shape)

        gyration_tensor = self._gyration_tensor(cluster, weights)

        # transform to nm
        factor = 100.0 * sum(weights)
        rg2 = np.trace(gyration_tensor)
        rg2 /= factor

        if components:
            r = np.subtract(cluster.atoms.positions, cluster.atoms.center(weights))

            if pca:
                # Calculate eigenvectors for Karhunen-Loeve Transformation
                eig_val, eig_vec = np.linalg.eig(gyration_tensor)
                eig_val, eig_vec = self._sort_eig(eig_val, eig_vec, reverse=True)
                r = np.matmul(r, eig_vec)  # y = A_t * r w/ A = eig_vec

            weights = np.broadcast_to(weights, (3, weights.shape[0])).transpose()

            if test:
                assert np.abs(np.sum(r * weights)) < 1e-8

            # Although just trace needed, probably fastest
            principal_gyration_tensor = np.matmul(r.transpose(), r * weights)
            principal_rg2 = np.trace(principal_gyration_tensor)
            principal_rg2 /= factor

            if test:
                assert np.abs(rg2 - principal_rg2) < 1e-8

            rg2_1 = principal_rg2 - principal_gyration_tensor[0, 0] / factor
            rg2_2 = principal_rg2 - principal_gyration_tensor[1, 1] / factor
            rg2_3 = principal_rg2 - principal_gyration_tensor[2, 2] / factor

            if test:
                assert (
                    np.abs(principal_rg2 - 0.5 * np.sum([rg2_1, rg2_2, rg2_3])) < 1e-8
                )
            ret = map(np.sqrt, (rg2, rg2_1, rg2_2, rg2_3))
            return tuple(ret)
        return np.sqrt(rg2)

    @staticmethod
    def _sort_eig(eig_val, eig_vec, reverse=False):
        """
        Sort eig_val and eig_vec so that largest eig_value is last and
        smalles is first. Commute eig_vec accordingly.
        """
        for i in range(2, 0, -1):
            index = np.where(eig_val == np.max(eig_val[: i + 1]))[0][0]
            # Switch columns
            eig_vec[:, [i, index]] = eig_vec[:, [index, i]]
            eig_val[i], eig_val[index] = eig_val[index], eig_val[i]

        if reverse:
            eig_vec[:, [0, 2]] = eig_vec[:, [2, 0]]
            eig_val[0], eig_val[2] = eig_val[2], eig_val[0]

        return eig_val, eig_vec

    @staticmethod
    def _gyration_tensor(cluster, weights):
        """
        Calculate gyration tensor either unweighted or mass weighted
        (pass vector of masses for that purpose).
        gyration tensor:

        G_ab = 1/\sum_i wi \sum_i w_i r_a r_b  for a = {x, y, z}
        """
        r = Gyration._get_reduced_r(cluster, weights)
        position_weights = Gyration._get_weights(cluster, weights)
        gyration_tensor = np.matmul(r.transpose(), r * position_weights)

        return gyration_tensor

    @staticmethod
    def _get_reduced_r(cluster, weights):
        r = np.subtract(cluster.atoms.positions, cluster.atoms.center(weights))
        return r

    @staticmethod
    def _get_weights(cluster, weights):
        if weights is None:
            weights = np.ones_like(cluster.atoms.positions)
        else:
            weights = np.broadcast_to(weights, (3, weights.shape[0])).transpose()

        return weights
