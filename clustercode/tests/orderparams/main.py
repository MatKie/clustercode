import sys
import numpy as np
sys.path.append("../")
from OrderParameterEnsemble import OrderParameterEnsemble

tpr  = "../files/run.tpr"
traj = "../files/traj.xtc"

times=(1000000,1000000)

#OrderParamEns = OrderParameterEnsemble(tpr, traj, ["ME"])

#OrderParamEns.nematic_op_analysis(times=times, style="molecule", pbc_style="mol")
#OrderParamEns.translational_op_analysis(OrderParamEns.system_director_list,times=times,style="molecule", pbc_style="mol",pos_style="com",plot=False)
#custom_director = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
custom_director = np.asarray([[1,1,0],[0,1,0]])
#custom_director = np.asarray([1,0,0])
custom_director = None
OrderParamEns = OrderParameterEnsemble(tpr, traj, ["CM","CE"])
OrderParamEns.structure_factor_analysis(times=times, style="atom", pos_style="atom", directors=custom_director,chunk_size=10000,q_max=0.5, plot_style="smooth")