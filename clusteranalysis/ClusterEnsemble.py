import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NeighborSearch
#from MDAnalysis.core.groups import ResidueGroup
"""
ToDo:
    Make sure PBC do what we want 
    Ensure behaviour for gro files
    Add functionality to look only at certain time windows
    Evaluation 
"""

class ClusterEnsemble():
    """Takes a list (of lists) of clusters to perform analysis

    """
    
    def __init__(self, coord, traj, cluster_objects):
        self.coord = coord
        self.traj  = traj
        self.cluster_objects = cluster_objects

    def cluster_analysis(self, cut_off=7.5, style="atom", measure="b2b", algorithm="static"):
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
            algorithm (str): "static" or "dynamic"
        Returns: 
            Not sure yet

        ToDo:
            Implement List of trajectories, which should facilitate analysis
            of replicas.
        """
        self.universe = self._get_universe()

        self.aggregate_species = self._get_aggregate_species(style=style)
        
        self.cluster_list = []

        self.neighbour_search = NeighborSearch.AtomNeighborSearch(
            self.aggregate_species, 
            box=self.universe.dimensions,
            bucket_size=10
            )

        if algorithm == "static":
            cluster_algorithm = self._get_cluster_list_static
        elif algorithm == "dynamic":
            cluster_algorithm = self._get_cluster_list_dynamic
        else:
            print("{:s} is unspecified algorithm".format(algorithm))

        for time in self.universe.trajectory:
            
            self.cluster_list.append(cluster_algorithm())
            print("****TIME: {:8.2f}".format(time.time))
            print("---->Number of clusters {:d}".format(
                    len(self.cluster_list[-1]))
                    )
            if len(self.cluster_list) > 20 :
                break
    def _get_universe(self):
        """Getting the universe when having or not having a trajector

        """
        if self.traj is not None:
            universe = MDAnalysis.Universe(self.coord, self.traj)
        else:
            universe = MDAnalysis.Universe(self.coord)
        
        return universe
    
    def _get_aggregate_species(self, style="atom"):
        """Getting a dictionary of the species on which we determine aggregation


        """
        # Cast cluster_objects to list if only single string is given
        # this is necessary because of differences in processing strings and 
        # list of strings
        if type(self.cluster_objects) is not list: self.cluster_objects = [ 
                                                        self.cluster_objects 
                                                        ]
        
        # If beads are choosen we look for names instead of resnames 
        if style == "atom":
            aggregate_species  = self.universe.select_atoms(
                            "name {:s}".format(" ".join(self.cluster_objects))
                            )
        if style == "molecule":
            aggregate_species  = self.universe.select_atoms(
                        "resname {:s}".format(" ".join(self.cluster_objects))
                        )
        
        return aggregate_species


    def _get_cluster_list_static(self, cut_off=7.5):
        """Get Cluster from single frame

        """   
        cluster_list = []
        aggregate_species_dict = self.aggregate_species.groupby("resids")
        
        for atoms in aggregate_species_dict.values():
            cluster_temp = set(self.neighbour_search.search(
                                                    atoms=atoms, 
                                                    radius=cut_off, 
                                                    level="R"
                                                    ))
            
            cluster_list = self._merge_cluster(cluster_list, cluster_temp)   
            
        return cluster_list

    def _get_cluster_list_dynamic(self, cut_off=7.5):
        """Get Cluster from single frame

        """  
        cluster_list = []
        aggregate_species = self.aggregate_species.residues
        #aggregate_species_set = {atomsi.atoms.residues[0] for atomsi in aggregate_species_dict.values()}
        tempSum = 0
        
        while aggregate_species.n_residues > 0:
            res_temp = aggregate_species[0].atoms.residues
            search_set = res_temp
            cluster_temp = res_temp
            while search_set.n_residues > 0:
                search_set, cluster_temp = self._grow_cluster(cut_off, search_set, cluster_temp)
            cluster_list.append(cluster_temp)
            aggregate_species = aggregate_species.difference(cluster_temp)
            
        return cluster_list

    def _merge_cluster(self, cluster_list, cluster_temp):
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

    def _grow_cluster(self, cut_off, search_set, cluster_temp):
        """Code to grow a cluster (cluster_temp) and obtain new search set
        (search_set)

        Args:
            cut_off    (float):  minimal distance for two particles to be in the 
                                 same cluster.
            search_set   (set):  set of residues in search set
            cluster_temp (set):  set of residues in cluster
        Returns: 
            search_set   (set):  updated set of residues in search set
            cluster_temp (set):  updated set of residues in cluster

        """
        #search_atom_group = MDAnalysis.ResidueGroup(
        #    [residue for residue in search_set]
        #    )
        new_cluster_res = MDAnalysis.core.groups.ResidueGroup(
            self.neighbour_search.search(
                atoms=search_set.atoms.select_atoms('name CM CE'),
                radius=cut_off, 
                level="R"
                )
            )
        search_set = new_cluster_res.difference(cluster_temp)
        cluster_temp = cluster_temp.union(new_cluster_res)

        return search_set, cluster_temp
