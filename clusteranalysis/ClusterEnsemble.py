import MDAnalysis
import MDAnalysis.lib.NeighborSearch as NeighborSearch
import warnings
#from MDAnalysis.core.groups import ResidueGroup
"""
ToDo:
    Make sure PBC do what we want 
    Ensure behaviour for gro files
"""

class ClusterEnsemble():
    """A class used to perform analysis on Clusters of molecules

    Attributes
    ----------
    universe : MDAnalysis universe object
        Universe of the simulated system 
    cluster_objects : list of str
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
    
    def __init__(self, coord, traj, cluster_objects):
        """
        Parameters
        ---------- 
        coord : string 
            Path to a coordinate-like file. E.g. a gromacs tpr or 
            gro file
        traj : string
            Path to a trajectory like file. E.g. a xtc or trr file. 
            Needs to fit the coord file
        cluster_objects : list of string
            Strings used for the definition of species which form 
            clusters. Can be atom names or molecule names.
        """

        self._coord = coord # Protected Attribute
        self._traj  = traj # Protected Attribute
        self.cluster_objects = cluster_objects

    def cluster_analysis(self, cut_off=7.5, style="atom", 
                    measure="b2b", algorithm="dynamics"):
        """High level function clustering molecules together
        
        Example
        -------
        No Example yet

        Note
        ----
        Not all of the functionality is actually coded yet

        Parameters
        ----------
        cut_off : float, optional
            Minimal distance for two particles to be in the
            same cluster, in Angstroem. Results still depend 
            on the measure parameter.
        style : string, optional
            "atom" or "molecule". Dependent on this, the 
            cluster_objects attribute is interpreted as molecule
            or atoms within a molecule. 
        measure : string, optional
            "b2b (bead to bead), COM or COG(center of geometry)
        algorithm : string, optional
            "dynamic" or "static". The static one is slower. It 
            loops over all atoms and then merges cluster, whereas
            the dynamic algorithm grows clusters dynamically.

        Raises
        ------ 
        NotImplementedError
            If an unspecified algorithm is choosen
        
        ToDo
        ----
        -Make static and dynamic algorithms store the same arguments
        -Add plotting capabilities
        -Add capabilities to only look at certain time windows
        -Get rid of traj and coord attributes
        """
        self.universe = self._get_universe(self._coord, traj=self._traj)

        self.aggregate_species = self._get_aggregate_species(self.universe,
                                                            style=style)
        
        self.cluster_list = []

        # Initialise the neighboursearch object
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
            raise NotImplementedError("{:s} is unspecified algorithm".format(
                                                                algorithm))

        # Loop over all trajectory times
        for time in self.universe.trajectory:
            self.cluster_list.append(cluster_algorithm())
            print("****TIME: {:8.2f}".format(time.time))
            print("---->Number of clusters {:d}".format(
                    len(self.cluster_list[-1]))
                    )
    
    def _get_cluster_list_static(self, cut_off=7.5):
        """Get Cluster from single frame with the static method
        
        This code simply loops over all atoms in the aggregate
        species and finds a cluster for each atom (all neighbours).
        This cluster is merged into the already found clusters.

        Parameters
        ----------
        cut_off : float, optional
            Radius around which to search for neighbours 

        Returns
        -------
        cluster_list : list of sets of ClusterIDs
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
    
    def _merge_cluster(self, cluster_list, cluster_temp):
        """Code to merge a cluster into a cluster list
        
        This code merges a cluster into a clusterlist by reverse
        looping over the cluster list. Whenever an overlap is detected
        the new, temporary, cluster is merged into the old one and 
        the new one is deleted. At the same time the temporary cluster
        is updated to be the old one merged with the new one to 
        further look for overlap.

        Parameters
        ----------
        cluster_list : list of sets of clusterids
            All the sets of clusters already present.  
        cluster_temp : set of clusterids
            Last found cluster, to be merged into the cluster_list

        Returns
        -------
        cluster_list : list of sets of clusterids
            Updated cluster_list

        """
        cluster_list.append(cluster_temp)
        merged_index = []
        
        # Loop through cluster_list in reverse with the actual also
        # the revesed indices 
        for i, cluster in reversed(list(enumerate(cluster_list))):
            # If there is an intersection start merging. There is 
            # always an intersection with the first cluster (identical)
            if bool(cluster.intersection(cluster_temp)):
                cluster_temp = cluster_temp | cluster # Updating cluster_temp
                cluster_list[i] = cluster_temp
                merged_index.append(i)
                # If the cluster is not completely new there will be a
                # second intersection and we delete the previous cluster
                # Note that len(merged_index) goes back to 1 afterwards
                # and cluster_temp is the new merged cluster
                if len(merged_index) > 1.5:
                    del cluster_list[merged_index[0]]
                    del merged_index[0]
                elif len(merged_index) > 1.5:
                    warnings.warn("Wrong behaviour in cluster merge",
                                  UserWarning)
        
        return cluster_list

    def _get_cluster_list_dynamic(self, cut_off=7.5):
        """Get Cluster from single frame with dynamic algorithm

        Faster algorithm to find clusters. We single out one 
        molecule and find all its neighbours, then we call the same
        function again and find the neighbours neighbours,
        while excluding already found neighbours. We do this until 
        we cant find any more neighbours. Then we choose another 
        molecule until we do not have any molecules left.

        Parameters
        ----------
        cut_off : float, optional
            Radius around which to search for neighbours 

        Returns
        -------
        cluster_list : list of sets of ClusterIDs

        """  
        cluster_list = []
        aggregate_species = self.aggregate_species.residues
        
        # Loop until all molecules have been in clusters
        while aggregate_species.n_residues > 0:
            # Initialize the search_set and temporary cluster w/
            # one molecule.
            res_temp = aggregate_species[0].atoms.residues
            search_set = res_temp
            cluster_temp = res_temp
            # In this loop search set gets updated to only have new
            # neighours and cluster_temp grows
            while search_set.n_residues > 0:
                search_set, cluster_temp = self._grow_cluster(cut_off, 
                                search_set, cluster_temp)
            # Once no more neighbours are found add the cluster to
            # the list and subtract the cluster from the aggregate
            # species.
            # Possible Improvement: Update NeighbourSearch object
            cluster_list.append(cluster_temp)
            aggregate_species = aggregate_species.difference(cluster_temp)
            
        return cluster_list



    def _grow_cluster(self, cut_off, search_set, cluster_temp):
        """Code to grow a cluster (cluster_temp) and obtain new search set
        (search_set)

        This algorithm looks for neighbours of atoms in search_set 
        and adds them to the temporary cluster. The search_set
        is updated to only include newly found neighbours not 
        already present in search_set. The neighbours searching
        still looks in all the atoms present in the aggregate_species.

        Parameters
        ----------
        cut_off : float, optional
            Radius around which to search for neighbours 
        search_set : MDAnalysis ResidueGroup 
            Atoms for which to look for neighbours
        cluster_temp : MDAanalysis ResidueGroup
            All the atoms/residues currently in the cluster

        Returns
        ------- 
        search_set : MDAnalysis ResidueGroup 
            Atoms for which to look for neighbours updated to be
            the latest found ones
        cluster_temp : MDAanalysis ResidueGroup
            All the atoms/residues currently in the cluster updated 
            to include the latest found ones

        """
        
        # Find neighbours and cast into ResidueGroup
        new_cluster_res = MDAnalysis.core.groups.ResidueGroup(
            self.neighbour_search.search(
                atoms=self._get_aggregate_species(search_set.atoms, style="atom"),
                radius=cut_off, 
                level="R"
                )
            )

        # The new search_set should only have atoms not already in the cluster
        search_set = new_cluster_res.difference(cluster_temp)
        # The new temporary cluster is updated
        cluster_temp = cluster_temp.union(new_cluster_res)

        return search_set, cluster_temp
    
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
    
    def _get_aggregate_species(self, atoms, style="atom"):
        """Getting a dictionary of the species on which we determine aggregation
        
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
        
        # Cast cluster_objects to list if only single string is given
        # this is necessary because of differences in processing strings and 
        # list of strings
        if type(self.cluster_objects) is not list: self.cluster_objects = [ 
                                                        self.cluster_objects 
                                                        ]
        
        # If beads are choosen we look for names instead of resnames 
        if style == "atom":
            aggregate_species  = atoms.select_atoms(
                            "name {:s}".format(" ".join(self.cluster_objects))
                            )
        elif style == "molecule":
            aggregate_species  = atoms.select_atoms(
                        "resname {:s}".format(" ".join(self.cluster_objects))
                        )
        
        return aggregate_species
