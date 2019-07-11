from MDAnalysis.core.groups import ResidueGroup

class ClusterEnsemble(ResidueGroup):
    """Takes a list (of lists) of clusters to perform analysis

    """
    
    def __init__(self, cluster_list)
        cluster_list = []
        if isinstance(cluser_list[0], set):
            for i, cluster in enumerate(cluster_list):
                self.cluster_list.append(ResidueGroup(cluster))
        elif isinstance(cluster_list[0], list):
            for time in len(cluster_list[0]):
                cluster_temp = []
                for i, cluster in enumerate(time):
                    cluster_temp.append(ResidueGroup(cluster))
                self.cluster_list.append(cluster_temp)




