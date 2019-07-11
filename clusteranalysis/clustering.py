import MDAnalysis

def cluster_analysis(coord, cluster_objects, traj=None, 
                     cut_off=7.5, style="atom", measure="b2b"):

    """High level function clustering molecules together.
        
    Args:
        coords    (str): path to coordinate file. As of now tested: tpr.
        cluster_objects
                  (str): string or list of strings with names of clustered 
                         objects. If style="atom" or "COM", this is a list 
                         of particles, if style="molecule this is a molecule
                         name
                         
        traj      (str): path to trajectory file. Has to fit to coordinate
                         file. As of now tested: xtc. 
        cut_off (float): minimal distance for two particles to be in the 
                         same cluster.
        style     (str): "atom" or "molecule" 
        measure   (str): b2b (bead to bead), COM or COG(center of geometry)
    Returns: 
        Not sure yet

    ToDo:
        Implement List of trajectories, which should facilitate anaylisis
        of replicas.
    """
    
    aggregate_species = get_aggregate_species(coord, cluster_objects, 
                                              traj=traj, style=style)

    print("length of dict cluster_particles is {:d}".format(
                                        len(aggregate_species))
                                        )

    return 0


def get_aggregate_species(coord, cluster_objects, traj=None, style="atom"):
    """Getting a dictionary of the species on which we determine aggregation


    """
    if traj is not None:
        universe = MDAnalysis.Universe(coord, traj)
    else:
        universe = MDAnalysis.Universe(coord)

    # Cast cluster_objects to list if only single string is given
    # this is necessary because of differences in processing strings and list 
    # of strings
    if type(cluster_objects) is not list: cluster_objects = [ cluster_objects ]
    
    # If beads are choosen we look for names instead of resnames 
    if style == "atom":
        aggregate_species  = universe.select_atoms(
                            "name {:s}".format(" ".join(cluster_objects))
                            )
    if style == "molecule":
        aggregate_species  = universe.select_atoms(
                              "resname {:s}".format(" ".join(cluster_objects))
                              )
    
    # Either way, in the end group by resid to get to get a grip on molecules
    # instead of subparts of them 
    return aggregate_species.groupby("resids")

