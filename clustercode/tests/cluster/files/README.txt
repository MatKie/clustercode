XTC files:
traj_atom.xtc    : Coarse-grained trajectory of a single micelle 
                   (382 surfactants), split across pbc's in part.
                   Treated with gmx trjconv ... -pbc atom 
                   tpr file: topol_no_solv.tpr
traj_cluster.xtc : As above with pbc cluster
traj_mol.xtc     : As above with pbc atom
traj_whole_xtc   : As above with pbc whole

traj_pbc_problematic_mol.xtc     : As above but v. specifically split at pbc
traj_pbc_problematic_cluster.xtc : As above but v. specifically split at pbc
traj_pbc_problematic_atom.xtc    : As above but v. specifically split at pbc

traj_small.xtc : Atomistic trajectory of a single micelle (100 surfactants) 
                 with extra sodium chloride (104 extra or so).
                 tpr file: topol_small.tpr 
