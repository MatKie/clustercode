import MDAnalysis
def cluster_analysis(coord, cluster_objects, traj, 
                     cut_off=7.5, style="atom", measure="b2b"):
    """High level function clustering molecules together.
        
    Args:
        coord    (str): path to coordinate file. As of now tested: tpr.
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
        Implement List of trajectories, which should facilitate analysis
        of replicas.
    """
    universe          = get_universe(coord, traj=traj)

    aggregate_species = get_aggregate_species(universe, cluster_objects, 
                                              style=style)
    
    cluster_list = []

    for time in universe.trajectory:
        cluster_list.append(get_cluster_list(aggregate_species))

    for time, cluster_time in enumerate(cluster_list):
        print("FRAME: {:d}".format(time))
        for i, cluster in enumerate(cluster_time):
            print(len(cluster))

    return cluster_list
    
def merge_cluster(cluster_list, cluster_temp):
    """Code to merge a cluster into a cluster list

    """
    cluster_list.append(cluster_temp)
    merged_index = []
    for i, cluster in reversed(list(enumerate(cluster_list))):
        if bool(cluster.intersection(cluster_temp)):
            cluster_temp = cluster_temp | cluster
            cluster_list[i] = cluster_temp
            merged_index.append(i)
            if len(merged_index) > 1.5:
                del cluster_list[merged_index[0]]
                del merged_index[0]
            elif len(merged_index) > 1.5:
                print("Somethings wrong with the cluster merging")
    
    return cluster_list
        
def get_cluster_list(aggregate_species, cutoff=7.5):
    """Get Cluster from single frame

    """  
    from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch

    cluster_list = []
    aggregate_species_dict = aggregate_species.groupby("resids")
    
    for atoms in aggregate_species_dict.values():
        cluster_temp = set(AtomNeighborSearch(aggregate_species).search(
                                                atoms=atoms, 
                                                radius=cutoff, 
                                                level="R"
                                                ))
        
        cluster_list = merge_cluster(cluster_list, cluster_temp)   
        
    return cluster_list

def get_universe(coord, traj=None):
    """Getting the universe when having or not having a trajector

    """
    if traj is not None:
        universe = MDAnalysis.Universe(coord, traj)
    else:
        universe = MDAnalysis.Universe(coord)
    
    return universe


def get_aggregate_species(universe, cluster_objects, style="atom"):
    """Getting a dictionary of the species on which we determine aggregation


    """
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
    
    return aggregate_species

