import MDAnalysis as mda 
from clustercode import ClusterEnsemble
from pytest import approx
import numpy as np
tpr = 'clustercode/tests/cluster/files/topol_no_solv.tpr'
traj = 'clustercode/tests/cluster/files/traj_pbc_problematic_atom.xtc'

clstr = ClusterEnsemble(tpr, traj, ['C1', 'C2', 'C3', 'C4'])

clstr.cluster_analysis()


for clusters in clstr.cluster_list:
    for cluster in clusters:
        #clstr.unwrap_cluster(cluster)
        #print(clstr.gyration(cluster, unwrap=False))
        rg2 = clstr.inertia_tensor(cluster, unwrap=True)
        rg = [np.sqrt(item) for item in rg2]
        #print(rg)
        #print(np.sqrt(sum(rg2)))

