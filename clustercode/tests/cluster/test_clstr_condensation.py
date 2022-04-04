from pytest import approx
import MDAnalysis as mda
from clustercode.ClusterEnsemble import ClusterEnsemble
import os
import copy

# These files comprise the same trajectory but processed with different
# gmx trjconv -pbc options (cluster, atom, whole and mol). The trajectory
# includes a large single micelles, in parts split across multiple pbc.
# It's a superset (of a trajectory) of the pbc_problematic labelled ones.
tpr = os.path.join("clustercode", "tests", "cluster", "files", "topol_no_solv.tpr")
atom = os.path.join("clustercode", "tests", "cluster", "files", "traj_atom.xtc")
mol = os.path.join("clustercode", "tests", "cluster", "files", "traj_mol.xtc")
cluster = os.path.join("clustercode", "tests", "cluster", "files", "traj_cluster.xtc")
whole = os.path.join("clustercode", "tests", "cluster", "files", "traj_whole.xtc")

# The accompanying ClusterEnsemble objects
atom_uni = ClusterEnsemble(tpr, atom, ["C1", "C2", "C3", "C4"])
mol_uni = ClusterEnsemble(tpr, mol, ["C1", "C2", "C3", "C4"])
whole_uni = ClusterEnsemble(tpr, whole, ["C1", "C2", "C3", "C4"])
clstr_uni = ClusterEnsemble(tpr, cluster, ["C1", "C2", "C3", "C4"])

work_in = "Residue"  # atom, Residue
measure = "b2b"  # b2b (bead to bead), COM, COG
pbc = True

mol_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)
whole_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)
clstr_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)
atom_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)

atom_list, whole_list, clstr_list, mol_list = [], [], [], []


def condens(uni, this_list, pbc):
    method = "pkdtree"  # bruteforce, nsgrid, pkdtree
    for clusters in uni.cluster_list:
        for cluster in clusters:
            this_list.append(
                uni.condensed_ions(
                    cluster, "SU", "NA", [4.4, 7.6], method=method, pbc=pbc
                )
            )


condens(atom_uni, atom_list, pbc)
condens(mol_uni, mol_list, pbc)
condens(whole_uni, whole_list, pbc)
condens(clstr_uni, clstr_list, pbc)


class TestCondensation:
    def run_condensed_ions(self, uni):
        """
        Check if pkdtree, nsgrid and bruteforce algorithm in neighbour
        search (in condensed_ions) all get the same result for the mol_iven
        trajectory.
        """
        results = []

        for method in ["pkdtree", "nsgrid", "bruteforce"]:
            ci = []
            for clusters in uni.cluster_list:
                for cluster in clusters:
                    ci.append(
                        uni.condensed_ions(
                            cluster, "SU", "NA", [4.4, 7.6], method=method, pbc=pbc
                        )
                    )
            results.append(copy.deepcopy(ci))

        for ti, mol_i, bi in zip(*results):
            for tii, mol_ii, bii in zip(ti, mol_i, bi):
                assert tii == approx(mol_ii)
                assert tii == approx(bii)
                assert mol_ii == approx(bii)

    def test_atom_uni(self):
        self.run_condensed_ions(atom_uni)

    def test_mol_uni(self):
        self.run_condensed_ions(mol_uni)

    def test_whole_uni(self):
        self.run_condensed_ions(whole_uni)

    def test_cluster_uni(self):
        self.run_condensed_ions(clstr_uni)

    def check_cond1_cond2(self, cond_1, cond_2):
        for cond_1i, cond_2i in zip(cond_1, cond_2):
            for cond_1ii, cond_2ii in zip(cond_1i, cond_2i):
                assert cond_1ii == approx(cond_2ii, abs=1)

    def test_atom_mol_uni(self):
        """
        Test if atom and mol treated trajectories are within 1
        condensed ion
        """
        self.check_cond1_cond2(atom_list, mol_list)

    def test_atom_whole_uni(self):
        """
        Test if atom and whole treated trajectories are within 1
        condensed ion
        """
        self.check_cond1_cond2(atom_list, whole_list)

    def test_atom_clstr_uni(self):
        """
        Test if atom and clstr treated trajectories are within 1
        condensed ion
        """
        self.check_cond1_cond2(atom_list, clstr_list)

    def test_mol_whole_uni(self):
        """
        Test if mol and whole treated trajectories are within 1
        condensed ion
        """
        self.check_cond1_cond2(mol_list, whole_list)

    def test_mol_clstr_uni(self):
        """
        Test if mol and clstr treated trajectories are within 1
        condensed ion
        """
        self.check_cond1_cond2(mol_list, clstr_list)

    def test_whole_clstr_uni(self):
        """
        Test if whole and clstr treated trajectories are within 1
        condensed ion
        """
        self.check_cond1_cond2(whole_list, clstr_list)