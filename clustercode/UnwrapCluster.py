import numpy as np
from MDAnalysis.lib.distances import capped_distance


class UnwrapCluster(object):
    def __init__(self):
        self.universe = None

    def unwrap_cluster(self, resgroup, box=None, unwrap=True, verbosity=0):
        """
        Make cluster which crosses pbc not cross pbc. Algorithm inspired
        by GROMACS but optimised so that it runs somewhat fast in
        python.

        Parameters
        ----------
        resgroup : MDAnalysis.ResidueGroup
            Cluster residues
        box : boxvector, optional
            boxvector. If None is given taken from trajectory,
            by default None
        unwrap : bool, optional
            Wether or not to make molecules whole before treatment
            (only necessary if pbc = atom in trjconv) but doesn't hurt.
            By default True
        verbosity : int, optional
            Chattiness, by default 0
        """
        # cluster is passed as resgroup and boxdimensions as xyz
        # will only support triclinic as of now..
        self.universe = resgroup.universe
        if box is None:
            box = self.universe.dimensions
        # Unwrap position if needed:
        for residue in resgroup:
            residue.atoms.unwrap(reference="cog", inplace=True)
        # Find initial molecule (closest to Centre of box)
        COB = box[:3] / 2
        weights = None  # This selects COG, if we mass weight its COM
        rmin = np.sum(box[:3]) * np.sum(box[:3])
        imin = -1
        for i, residue in enumerate(resgroup):
            tmin = self._pbc(residue.atoms.center(weights=None), COB, box)
            tmin = np.dot(tmin, tmin)
            if tmin < rmin:
                rmin = tmin
                imin = i

        # added and to_be_added store indices of residues
        # in resgroup
        nr_mol = resgroup.n_residues
        to_be_added = [i for i in range(nr_mol)]
        added = [to_be_added.pop(imin)]
        nr_added = 1

        # Setup the COG matrices
        refset_cog = resgroup[added[0]].atoms.center(None)

        configset_cog = np.zeros((nr_mol - 1, 3))
        for i, index in enumerate(to_be_added):
            configset_cog[i, :] = resgroup[index].atoms.center(None)

        while nr_added < nr_mol:
            # Find indices of nearest neighbours of a) res in micelle
            # (imin) and b) res not yet in micelle (jmin). Indices
            # w.r.t. resgroup
            imin, jmin = self._unwrap_ns(
                refset_cog, configset_cog, added, to_be_added, box
            )
            # Translate jmin by an appropriate vector if separated by a
            # pbc from rest of micelle.
            cog_i = resgroup[imin].atoms.center(weights=weights)
            cog_j = resgroup[jmin].atoms.center(weights=weights)

            dx = self._pbc(cog_j, cog_i, box)
            xtest = cog_i + dx
            shift = xtest - cog_j
            if np.dot(shift, shift) > 1e-8:
                if verbosity > 0.5:
                    print(
                        "Shifting molecule {:d} by {:.2f}, {:.2f}, {:.2f}".format(
                            jmin, shift[0], shift[1], shift[2]
                        )
                    )
                for atom in resgroup[jmin].atoms:
                    atom.position += shift
                cog_j += shift

            # Add added res COG to res already in micelle
            refset_cog = np.vstack((refset_cog, cog_j))
            nr_added += 1
            added.append(jmin)

            # Remove added res from res not already in micelle.
            _index = to_be_added.index(jmin)
            configset_cog = np.delete(configset_cog, _index, 0)
            del to_be_added[_index]

    def _unwrap_ns(
        self, refset_cog, configset_cog, added, to_be_added, box, method="pkdtree"
    ):
        """
        Find NN in refset_cog and configset_cog and pass back
        the indices stored in added and to_be added.
        """
        distances = []
        dist = 8.0
        while len(distances) < 1:
            pairs, distances = capped_distance(
                refset_cog,
                configset_cog,
                dist,
                box=box,
                method=method,
                return_distances=True,
            )
            dist += 0.5

        minpair = np.where(distances == np.amin(distances))[0][0]

        imin = added[pairs[minpair][0]]
        jmin = to_be_added[pairs[minpair][1]]
        return imin, jmin

    @staticmethod
    def _pbc(r1, r2, box):
        # Rectangular boxes only
        # Calculate fdiag, hdiag, mhdiag
        fdiag = box[:3]
        hdiag = fdiag / 2

        dx = r1 - r2
        # Loop over dims
        for i in range(3):
            # while loop upper limit: if dx > hdiag shift by - fdiag
            while dx[i] > hdiag[i]:
                dx[i] -= fdiag[i]
            # while loop lower limit: if dx > hdiag shift by + fdiag
            while dx[i] < -hdiag[i]:
                dx[i] += fdiag[i]

        return dx