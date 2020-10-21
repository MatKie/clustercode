from pytest import approx
import MDAnalysis as mda
from clustercode.ClusterEnsemble import ClusterEnsemble
import os
import copy


traj = 'clustercode/tests/cluster/files/traj_pbc_problematic_atom.xtc'
tpr  = 'clustercode/tests/cluster/files/topol_no_solv.tpr'

uni = ClusterEnsemble(tpr, traj, ['C1', 'C2', 'C3', 'C4'])

uni.cluster_analysis()
def test_1(uni):
    for clusters in uni.cluster_list:
        for cluster in clusters:
            a,b = uni.condensed_ions(cluster, 'SU', 'NA', [4.4, 7.6])
            ai,bi = uni.condensed_ions(cluster, 'SU', 'NA', [4.4, 7.6], wrap=True) 
            assert a == approx(ai)
            assert b == approx(bi)

test_1(uni)