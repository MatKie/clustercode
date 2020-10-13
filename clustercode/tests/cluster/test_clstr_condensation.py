from pytest import approx
import MDAnalysis as mda
from clustercode.ClusterEnsemble import ClusterEnsemble
import os
import copy

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
pbc = True
method = "pkdtree"  # bruteforce, nsgrid, pkdtree
ci = atom_uni.condensed_ions("SU", "NA", [4.4, 7.6], method=method, pbc=pbc)

tree = copy.deepcopy(ci)

method = "nsgrid"  # bruteforce, nsgrid, pkdtree
atom_c = atom_uni.condensed_ions("SU", "NA", [4.4, 7.6], method=method, pbc=pbc)

grid = copy.deepcopy(atom_c)

method = "bruteforce"  # bruteforce, nsgrid, pkdtree
ci = atom_uni.condensed_ions("SU", "NA", [4.4, 7.6], method=method, pbc=pbc)

brute = copy.deepcopy(ci)

for t, g, b in zip(tree, grid, brute):
    for ti, gi, bi in zip(t, g, b):
        for tii, gii, bii in zip(ti, gi, bi):
            assert tii == approx(gii)
            assert tii == approx(bii)
            assert gii == approx(bii)

mol_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)
whole_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)
clstr_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)

method = "pkdtree"  # bruteforce, nsgrid, pkdtree
mol = mol_uni.condensed_ions("SU", "NA", [4.4, 7.6], method=method, pbc=pbc)
whole = whole_uni.condensed_ions("SU", "NA", [4.4, 7.6], method=method, pbc=pbc)
cluster = clstr_uni.condensed_ions("SU", "NA", [4.4, 7.6], method=method, pbc=pbc)

for t, g, b, c in zip(tree, mol, whole, cluster):
    for ti, gi, bi, ci in zip(t, g, b, c):
        for tii, gii, bii, cii in zip(ti, gi, bi, ci):
            assert tii == approx(gii)
            assert tii == approx(bii)
            assert tii == approx(cii)
            assert gii == approx(bii)
            assert gii == approx(cii)
            assert bii == approx(cii)


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

