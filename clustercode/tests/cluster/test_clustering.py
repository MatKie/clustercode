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

MolClstr = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4", "SU"])
MolClstr.cluster_analysis(
    cut_off=6.5, style="atom", measure="b2b", algorithm="dynamic", work_in="Residue"
)

mol_cluster_sizes = MolClstr.cluster_sizes


class TestClustering:
    def test_generator(self):
        i, j = 0, 0
        for clusters in MainClstr.cluster_list:
            for cluster in clusters:
                i += 1

        for clusters in MainClstr.cluster_list:
            for cluster in clusters:
                j += 1

        assert i == j

    def test_algorithm_static(self):
        """
        Check if static and dynamic keywords get the same cluster
        size distribution
        """
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

    def test_work_in_atom(self):
        """
        Check if work in atom and work in residue keywords get the same cluster
        size distribution. Got to divide by number of atoms for this
        """
        SideClstr = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"])
        SideClstr.cluster_analysis(
            cut_off=6.5,
            style="atom",
            measure="b2b",
            algorithm="static",
            work_in="Atom",
        )
        for main_sizes, side_sizes in zip(main_cluster_sizes, SideClstr.cluster_sizes):
            assert len(main_sizes) == approx(len(side_sizes))
            for main_cluster, side_cluster in zip(main_sizes, side_sizes):
                assert main_cluster == approx(side_cluster / 4)

    def test_measure_COM(self):
        """
        Check if measur distance by COM does result in the same cluster
        size distribution. Not really a good test as it physically
        does not need to result in the same cluster size distribution.
        Apparently it does for this trajectory tho.
        """
        SideClstr = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"])
        SideClstr.cluster_analysis(
            cut_off=6.5,
            style="atom",
            measure="COM",
            algorithm="dynamic",
            work_in="Residue",
        )
        for main_sizes, side_sizes in zip(main_cluster_sizes, SideClstr.cluster_sizes):
            assert len(main_sizes) == approx(len(side_sizes))
            for main_cluster, side_cluster in zip(main_sizes, side_sizes):
                assert main_cluster == approx(side_cluster)

    def test_measure_COG(self):
        """
        Check if measur distance by COG does result in the same cluster
        size distribution. Not really a good test as it physically
        does not need to result in the same cluster size distribution.
        Apparently it does for this trajectory tho.
        """
        SideClstr = ClusterEnsemble(tpr, traj, ["C1", "C2", "C3", "C4"])
        SideClstr.cluster_analysis(
            cut_off=6.5,
            style="atom",
            measure="COG",
            algorithm="dynamic",
            work_in="Residue",
        )
        for main_sizes, side_sizes in zip(main_cluster_sizes, SideClstr.cluster_sizes):
            assert len(main_sizes) == approx(len(side_sizes))
            for main_cluster, side_cluster in zip(main_sizes, side_sizes):
                assert main_cluster == approx(side_cluster)

    def test_style_molecule(self):
        """
        Check if style molecule gets the same result than style atom.
        Need to use different reference, which includes headgroups
        for style atom, as it actually makes a difference for the neighbour
        search.
        """
        SideClstr = ClusterEnsemble(tpr, traj, ["SDS"])
        SideClstr.cluster_analysis(
            cut_off=6.5,
            style="molecule",
            measure="b2b",
            algorithm="dynamic",
            work_in="Residue",
        )
        for main_sizes, side_sizes in zip(mol_cluster_sizes, SideClstr.cluster_sizes):
            assert len(main_sizes) == approx(len(side_sizes))
            for main_cluster, side_cluster in zip(main_sizes, side_sizes):
                assert main_cluster == approx(side_cluster)

    def test_style_molecule_work_in_Atom(self):
        """
        Check if style molecule gets the same result than style atom
        if used with work in atom keyword.
        Need to use different reference, which includes headgroups
        for style atom, as it actually makes a difference for the neighbour
        search.
        """
        SideClstr = ClusterEnsemble(tpr, traj, ["SDS"])
        SideClstr.cluster_analysis(
            cut_off=6.5,
            style="molecule",
            measure="b2b",
            algorithm="dynamic",
            work_in="Atom",
        )
        for main_sizes, side_sizes in zip(mol_cluster_sizes, SideClstr.cluster_sizes):
            assert len(main_sizes) == approx(len(side_sizes))
            for main_cluster, side_cluster in zip(main_sizes, side_sizes):
                assert main_cluster == approx(side_cluster / 5)
