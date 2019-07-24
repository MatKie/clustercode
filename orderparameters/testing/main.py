import sys
sys.path.append("../")
from OrderParameterEnsemble import OrderParameterEnsemble

tpr  = "../files/run.tpr"
traj = "../files/traj.xtc"

OrderParamEns = OrderParameterEnsemble(tpr, traj, ["CE", "CM"])
print(OrderParamEns.selection)