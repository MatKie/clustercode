import MDAnalysis
import warnings
import os
"""
ToDo:
    Make sure PBC do what we want 
"""

class BaseUniverse():
    """A class used to perform analysis on Clusters of molecules

    Attributes
    ----------
    universe : MDAnalysis universe object
        Universe of the simulated system 
    selection : list of str
        Strings used for the definition of species which form 
        clusters. Can be atom names or molecule names.
    cluster_list : list of list of MDAnalysis ResidueGroups
        a list of ResidueGroups forms one cluster at a given time,
        for multiple times a list of these lists is produced.

    Methods
    -------
    cluster_analysis(cut_off=7.5, style="atom", measure="b2b", 
                     algorithm="static")
        Calculates which molecules are clustering together for all
        timesteps. 
    """
    
    def __init__(self, coord, traj, selection):
        """
        Parameters
        ---------- 
        coord : string 
            Path to a coordinate-like file. E.g. a gromacs tpr or 
            gro file
        traj : string
            Path to a trajectory like file. E.g. a xtc or trr file. 
            Needs to fit the coord file
        selection : list of string
            Strings used for the definition of species to be studied. Can be atom names or molecule names.
        pbc_style : string
            Use gromacs pbc definitions: mol, atom, nojump
        """

        self._coord = coord # Protected Attribute
        self._traj  = traj # Protected Attribute
        self.selection = selection 

    def _get_universe(self, coord, traj=None):
        """Getting the universe when having or not having a trajectory
            
        Parameters
        ----------
        coord : string 
            Path to a coordinate-like file. E.g. a gromacs tpr or 
            gro file
        traj : string
            Path to a trajectory like file. E.g. a xtc or trr file. 
            Needs to fit the coord file

        Returns
        -------
        universe : MDAnalysis universe object
        """
        if traj is not None:
            universe = MDAnalysis.Universe(coord, traj)
        else:
            universe = MDAnalysis.Universe(coord)
        
        return universe
    
    def _select_species(self, atoms, style="atom"):
        """Get an AtomGroup of the selected species 
        
        Parameter
        ---------
        atoms : MDAanalysis Atom(s)Group object
            superset of atoms out of which the aggregating species is
            determined. E.g. atoms could be a whole surfactant and 
            the return just the alkane tail.
        style : string, optional 
            "atom" or "molecule" depending if the aggregating species
            is defined as the whole molecule or just parts of it, 
            e.g. the hydrophobic chain of a surfactant.

        Returns
        -------
        aggregate_species: MDAanalysis Atom(s)Group object
            Just the atoms which define a cluster 
        """
        
        # Cast selection to list if only single string is given
        # this is necessary because of differences in processing strings and 
        # list of strings
        if type(self.selection) is not list: self.selection = [ 
                                                        self.selection 
                                                        ]
        
        # If beads are choosen we look for names instead of resnames 
        if style == "atom":
            aggregate_species  = atoms.select_atoms(
                            "name {:s}".format(" ".join(self.selection))
                            )
        elif style == "molecule":
            aggregate_species  = atoms.select_atoms(
                        "resname {:s}".format(" ".join(self.selection))
                        )
        
        return aggregate_species

    def _set_pbc_style(self, pbc_style):
        """Set the pbc style. If pbc_style is None or if the pbc_style is already set to the desired value the function does nothing, otherwise it converts the trajectory into the desired pbc style 
        
        Parameter
        ---------
        pbc_style : string
            Gromacs pbc definition: mol, atom, nojump
            
        """
        if pbc_style is not None:
            if hasattr(self, "pbc_style"):
                if self.pbc_style != pbc_style:
                    self._change_traj_pbc(pbc_style)
            else:
                self._change_traj_pbc(pbc_style)

    def _change_traj_pbc(self, pbc_style):
        """Changes the pbc style of the trajectory. It uses the gromacs command (gmx trjconv).  
        
        Parameter
        ---------
        pbc_style : string
            Gromacs pbc definition: mol, atom, nojump
            
        """
        new_traj = self._traj[:-4] + "_pbc_{:s}".format(pbc_style) + self._traj[-4:]
        # Gromacs command to convert current trajectory file to a new trajectory file with name new_traj.
        gromacs_command = "gmx trjconv -f {:s} -o {:s} -s {:s} -pbc {:s} <<EOF\n 0\n EOF".format(self._traj, new_traj, self._coord, pbc_style)
        os.system(gromacs_command)
        
        self._traj = new_traj
        self.pbc_style = pbc_style