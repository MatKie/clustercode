import sys
import numpy as np
sys.path.append("../")
from OrderParameterEnsemble import OrderParameterEnsemble

tpr  = "../files/run.tpr"
traj = "../files/traj.xtc"

times=(0,100000)

OrderParamEns = OrderParameterEnsemble(tpr, traj, ["ME"])
#OrderParamEns.nematic_op_analysis(times=times,style="molecule",pbc_style="mol")
#OrderParamEns.translational_op_analysis(OrderParamEns.system_director_list,times=times,style="molecule",pbc_style="mol",trans_op_style="com",plot=True)
custom_director = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
custom_director = np.asarray([[1,0,0],[0,1,0]])
custom_director = np.asarray([1,0,0])

OrderParamEns.structure_factor_analysis(directors=custom_director)