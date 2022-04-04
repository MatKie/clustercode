from re import L
import MDAnalysis as mda
from clustercode import ClusterEnsemble
from pytest import approx

# These files comprise the same trajectory but processed with different
# gmx trjconv -pbc options (cluster, atom and mol). The trajectory
# includes a large single micelles split across mutliple PBCs.
tcluster = "clustercode/tests/cluster/files/traj_pbc_problematic_cluster.xtc"
tatom = "clustercode/tests/cluster/files/traj_pbc_problematic_atom.xtc"
tpr = "clustercode/tests/cluster/files/topol_no_solv.tpr"

ClusterUniverse = ClusterEnsemble(tpr, tcluster, ["C1", "C2", "C3", "C4"])
AtomUniverse = ClusterEnsemble(tpr, tatom, ["C1", "C2", "C3", "C4"])

ClusterUniverse.cluster_analysis()
AtomUniverse.cluster_analysis()


class TestUnwrap:
    def test_unwrapping(self):
        """
        This test checks if gmx trjcvon -f ... -pbc cluster results in the
        same centre of geometry when applied to the example trajectory.
        Down to 0.005 Angstrom (pretty goood).
        """
        for c_clusters, a_clusters in zip(
            ClusterUniverse.cluster_list, AtomUniverse.cluster_list
        ):
            for c_cluster, a_cluster in zip(c_clusters, a_clusters):
                AtomUniverse.unwrap_cluster(a_cluster)
                c_cog, a_cog = c_cluster.center(None), a_cluster.center(None)
                for c_coord, a_coord in zip(c_cog, a_cog):
                    assert c_coord == approx(a_coord, abs=0.005)
