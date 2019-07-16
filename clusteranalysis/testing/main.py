import sys
sys.path.append("../")
from clustering import cluster_analysis  
from ClusterEnsemble import ClusterEnsemble

tpr  = "../files/nvt_2018.tpr"
traj = "../files/nvt_short.xtc"
#tpr  = "../files/npt.tpr"
#traj = "../files/npt.xtc"
#traj  = "/home/mk8118/OneDrive/2019/simulations/gromacs/SDS/check_ensembles/NVT/PME_revised/nh_10/base/nvt.trr"

#cluster_analysis(tpr, ["CE", "CM"], traj)
ClstrEns = ClusterEnsemble(tpr, traj, ["CE", "CM"])
ClstrEns.cluster_analysis(algorithm="dynamic") 
