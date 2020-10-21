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

atom_uni = ClusterEnsemble(tpr, atom, ["C1", "C2", "C3", "C4"])
mol_uni = ClusterEnsemble(tpr, mol, ["C1", "C2", "C3", "C4"])
whole_uni = ClusterEnsemble(tpr, whole, ["C1", "C2", "C3", "C4"])
clstr_uni = ClusterEnsemble(tpr, cluster, ["C1", "C2", "C3", "C4"])

work_in = "Residue"  # atom, Residue
measure = "b2b"  # b2b, COM, COG
pbc = True 


atom_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)

pbc = True
method = "pkdtree"  # bruteforce, nsgrid, pkdtree
ci = []
for clusters in atom_uni.cluster_list:
    for cluster in clusters:
        ci.append(
            atom_uni.condensed_ions(
                cluster, "SU", "NA", [4.4, 7.6], method=method, pbc=pbc
            )
        )

tree = copy.deepcopy(ci)

method = "nsgrid"  # bruteforce, nsgrid, pkdtree
ci = []
for clusters in atom_uni.cluster_list:
    for cluster in clusters:
        ci.append(
            atom_uni.condensed_ions(
                cluster, "SU", "NA", [4.4, 7.6], method=method, pbc=pbc
            )
        )

grid = copy.deepcopy(ci)

ci = []
method = "bruteforce"  # bruteforce, nsgrid, pkdtree
for clusters in atom_uni.cluster_list:
    for cluster in clusters:
        ci.append(
            atom_uni.condensed_ions(
                cluster, "SU", "NA", [4.4, 7.6], method=method, pbc=pbc
            )
        )

brute = copy.deepcopy(ci)

for ti, gi, bi in zip(tree, grid, brute):
    for tii, gii, bii in zip(ti, gi, bi):
        assert tii == approx(gii)
        assert tii == approx(bii)
        assert gii == approx(bii)

mol_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)
whole_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)
clstr_uni.cluster_analysis(work_in=work_in, measure=measure, pbc=pbc)

mol, whole, cluster = [], [], []
def condens(uni, this_list, pbc):
    pbc=pbc
    method = "pkdtree"  # bruteforce, nsgrid, pkdtree
    for clusters in uni.cluster_list:
        for cluster in clusters:
            this_list.append(
                uni.condensed_ions(cluster, 'SU', 'NA', [4.4, 7.6], method=method, pbc=pbc)
                )

condens(mol_uni, mol, pbc)
condens(whole_uni, whole, pbc)
condens(clstr_uni, cluster, pbc)


for i, (ti, gi, bi, ci) in enumerate(zip(tree, mol, whole, cluster)):
    for tii, gii, bii, cii in zip(ti, gi, bi, ci):
        try:
            assert tii == approx(gii)
            assert tii == approx(bii)
            assert tii == approx(cii)
            assert gii == approx(bii)
            assert gii == approx(cii)
            assert bii == approx(cii)
        except:
            print(i)
            print(ti, gi, bi, ci)


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
            print('Different clustering of atom')
            print(ai, mi, wi, ci)
        if mi != wi or mi != ci:
            print("Different clustering of mol")
            print(ai, mi, wi, ci)
        if wi != ci:
            print("Different clustering of clstr")
            print(ai, mi, wi, ci)

