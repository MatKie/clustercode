from re import L
from click import style
from pytest import approx
from clustercode import ClusterEnsemble
import os

tpr = os.path.join("clustercode", "tests", "cluster", "files", "cg_agg.tpr")
traj = os.path.join("clustercode", "tests", "cluster", "files", "traj_cg_agg.xtc")

MainClstr = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"])
MainClstr.cluster_analysis(
    cut_off=6.5, style="atom", measure="b2b", algorithm="dynamic", work_in="Residue"
)

main_cluster_sizes = MainClstr.cluster_sizes


class TestClustering:
    def test_algorithm_static(self):
        SideClstr = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"])
        SideClstr.cluster_analysis(
            cut_off=6.5,
            style="atom",
            measure="b2b",
            algorithm="static",
            work_in="Residue",
        )
        for main_sizes, side_sizes in zip(main_cluster_sizes, SideClstr.cluster_sizes):
            assert len(main_sizes) == approx(len(side_sizes))
            for main_cluster, side_cluster in zip(main_sizes, side_sizes):
                assert main_cluster == approx(side_cluster)
