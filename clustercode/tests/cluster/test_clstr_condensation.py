from pytest import approx
import MDAnalysis as mda
from clustercode.ClusterEnsemble import ClusterEnsemble
import os

tpr = os.path.join("clustercode", "tests", "cluster", "files", "topol_no_solv.tpr")
atom = os.path.join("clustercode", "tests", "cluster", "files", "traj_atom.xtc")
mol = os.path.join("clustercode", "tests", "cluster", "files", "traj_mol.xtc")
cluster = os.path.join("clustercode", "tests", "cluster", "files", "traj_cluster.xtc")
whole = os.path.join("clustercode", "tests", "cluster", "files", "traj_whole.xtc")

atom_uni = ClusterEnsemble(tpr, atom, ["C1" "C2", "C3", "C4"])
mol_uni = ClusterEnsemble(tpr, mol, ["C1" "C2", "C3", "C4"])
whole_uni = ClusterEnsemble(tpr, whole, ["C1" "C2", "C3", "C4"])
clstr_uni = ClusterEnsemble(tpr, cluster, ["C1" "C2", "C3", "C4"])

work_in = "Residue"  # atom, Residue
measure = "b2b"  # b2b, COM, COG
pbc = True


atom_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)
mol_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)
whole_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)
clstr_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)


def get_lengths(uni):
    uni_list = []
    for item in uni.cluster_list:
        t_list = []
        for it in item:
            t_list.append(len(it))
        uni_list.append(t_list)
    return uni_list


atom_list = get_lengths(atom_uni)
mol_list = get_lengths(mol_uni)
whole_list = get_lengths(whole_uni)
clstr_list = get_lengths(clstr_uni)

for a, m, w, c in zip(atom_list, mol_list, whole_list, clstr_list):
    for ai, mi, wi, ci in zip(a, m, w, c):
        if ai != mi or ai != wi or ai != ci:
            print(ai, mi, wi, ci)
            raise RuntimeError("Different clustering of atom")
        if mi != wi or mi != ci:
            print(ai, mi, wi, ci)
            raise RuntimeError("Different clustering of mol")
        if wi != ci:
            print(ai, mi, wi, ci)
            raise RuntimeError("Different clustering of whole")

pbc = True
method = "nsgrid"  # bruteforce, nsgrid, pkdtree
atom_uni.condensed_ions("SU", "NA", [0.44, 0.76], method=method, pbc=pbc)

