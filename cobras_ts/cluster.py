import itertools


class Cluster:

    def __init__(self, super_instances):
        self.super_instances = super_instances
        self.is_pure = False # in the visual querier, the user can indicate that the entire cluster is pure
        self.is_finished = False

    def distance_to(self, other_cluster):
        super_instance_pairs = itertools.product(self.super_instances, other_cluster.super_instances)
        return min([x[0].distance_to(x[1]) for x in super_instance_pairs])

    def get_comparison_points(self, other_cluster):
        # any super-instance should do, no need to find closest ones!
        return self.super_instances[0], other_cluster.super_instances[0]

    def get_all_points(self):
        all_pts = []
        for super_instance in self.super_instances:
            all_pts.extend(super_instance.indices)
        return all_pts

    def cannot_link_to_other_cluster(self, c, cl):
        medoids_c1 = [si.representative_idx for si in self.super_instances]
        medoids_c2 = [si.representative_idx for si in c.super_instances]

        for x, y in itertools.product(medoids_c1, medoids_c2):
            if (x, y) in cl or (y, x) in cl:
                return True
        return False