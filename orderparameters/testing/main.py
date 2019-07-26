import sys
sys.path.append("../")
from OrderParameterEnsemble import OrderParameterEnsemble

tpr  = "../files/run.tpr"
traj = "../files/traj.xtc"

OrderParamEns = OrderParameterEnsemble(tpr, traj, ["ME"])
OrderParamEns.calc_nematic_op()