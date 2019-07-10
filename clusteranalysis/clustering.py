import MDAnalysis

def cluster_analysis(coord, cluster_objects, traj=None, 
                     cut_off=7.5, style="atom", measure="b2b"):

    """High level function clustering molecules together.
        
    Args:
        coords    (str): path to coordinate file. As of now tested: tpr.
        cluster_objects
                  (str): string or list of strings with names of clustered 
                         objects. If style="atom" or "COM", this is a list 
                         of particles, if style="molecule
                         
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
        cluster_particles = universe.select_atoms(
                            "name {:s}".format(" ".join(cluster_objects))
                            )
    if style == "molecule":
        cluster_particles = universe.select_atoms(
                              "resname {:s}".format(" ".join(cluster_objects))
                              )
    
    # Either way, in the end group by resid to get to get a grip on molecules
    # instead of subparts of them 
    cluster_particles = cluster_particles.groupby("resids")
    
    return 0
cluster_analysis("files/nvt_2018.tpr", "CE")


