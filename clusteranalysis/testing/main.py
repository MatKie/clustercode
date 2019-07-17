import sys
sys.path.append("../")
from clustering import cluster_analysis  
from ClusterEnsemble import ClusterEnsemble

tpr  = "/home/trl11/Virtual_Share/gromacs_test/npt.tpr"
traj = "/home/trl11/Virtual_Share/gromacs_test/npt.xtc"
#tpr  = "../files/npt.tpr"
#traj = "../files/npt.xtc"
#traj  = "/home/mk8118/OneDrive/2019/simulations/gromacs/SDS/check_ensembles/NVT/PME_revised/nh_10/base/nvt.trr"

#cluster_analysis(tpr, ["CE", "CM"], traj)
ClstrEns = ClusterEnsemble(tpr, traj, ["CE", "CM"])
#ClstrEns.cluster_analysis(algorithm="static") 
#clstr_ens_static = ClstrEns.cluster_list
ClstrEns.cluster_analysis(algorithm="dynamic") 
#clstr_ens_dynamic = ClstrEns.cluster_list

exit()

for idx_time, (static_clus_list, dynamic_clus_list) in enumerate(zip(clstr_ens_static, clstr_ens_dynamic)):
	diff_clust_count = len(static_clus_list) - len(dynamic_clus_list)
	print("_________________________________________________________")
	print("Frame: {:d}".format(idx_time))
	print("Difference in cluster counts (static - dynamic): {:d}".format(
		diff_clust_count))

	static_molec_count = 0
	for cluster in  static_clus_list:
		static_molec_count += len(cluster)

	print("Statis molec count: {:d}".format(static_molec_count))
	
	dynamic_molec_count = 0
	for cluster in  dynamic_clus_list:
		dynamic_molec_count += cluster.n_residues

	print("Dynamic molec count: {:d}".format(dynamic_molec_count))

	print("Difference in molecule counts (static - dynamic): {:d}".format(
		static_molec_count-dynamic_molec_count))

	new_s_set = static_clus_list[0]
	for cluster in  static_clus_list:
		new_s_set = new_s_set.union(cluster)
	
	new_d_set = dynamic_clus_list[0]
	for cluster in  dynamic_clus_list:
		new_d_set = new_d_set.union(cluster)

	print("Static molec double counted: {:d}".format(
		static_molec_count-len(new_d_set)))
	print("Dynamic molec double counted: {:d}".format(
		dynamic_molec_count-new_d_set.n_residues))

	for idxi, clusteri in  enumerate(dynamic_clus_list):
		for idxj, clusterj in  enumerate(dynamic_clus_list):
			if clusteri.issuperset(clusterj) and idxi != idxj:
				print(idxi,idxj)
				print(clusteri)
				print(clusteri.n_residues,clusterj.n_residues)
