import sys
sys.path.append("../")
from clustering import cluster_analysis  

tpr  = "../files/nvt_2018.tpr"
traj = "../files/nvt_short.xtc"

cluster_analysis(tpr, ["CE", "CM"], traj=traj)

