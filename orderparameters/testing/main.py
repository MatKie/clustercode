import sys
sys.path.append("../")
from OrderParameterEnsemble import OrderParameterEnsemble

tpr  = "../files/run.tpr"
traj = "../files/traj.xtc"

OrderParamEns = OrderParameterEnsemble(tpr, traj, ["ME"])
OrderParamEns.nematic_op_analysis(times=(0,100000),pbc_style="mol")