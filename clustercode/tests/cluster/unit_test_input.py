import sys
sys.path.append("../")
from clustering import get_aggregate_species
import warnings

tpr  = "../files/nvt_2018.tpr"
gro  = "../files/nvt.gro"
traj = "../files/nvt_short.xtc"

try:
    get_aggregate_species(tpr, ["CE"], traj=traj)
except Exception as e:
    print("Could not import tpr with xtc and select CE. Error\
           Message:\n{:s}".format(e)
         )

try:
    get_aggregate_species(tpr, ["CE", "CM"], traj=traj)
except Exception as e:
    print("Could not import tpr with xtc and select CE+CM. Error\
           Message:\n{:s}".format(e)
         )

try:
    get_aggregate_species(tpr, "CE", traj=traj)
except Exception as e:
    print("Could not import tpr with xtc and select CE wo/ list. Error\
           Message:\n{:s}".format(e)
         )

try:
    get_aggregate_species(tpr, ["SDS"], traj=traj, style="molecule")
except Exception as e:
    print("Could not import tpr with xtc and select SDS as molecule. Error\
           Message:\n{:s}".format(e)
         )

#Test Differences


try:
    CE  = get_aggregate_species(tpr, ["CE"], traj=traj)
    CEE = get_aggregate_species(tpr, "CE", traj=traj)
except Exception as e:
    print("Could not import tpr with xtc and select CE. Error\
           Message:\n{:s}".format(e)
         )
