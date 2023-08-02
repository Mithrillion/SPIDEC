import numpy as np


def find_exemplar_ids(clusterer):
    if clusterer._prediction_data is None:
        clusterer.generate_prediction_data()

    selected_clusters = clusterer.condensed_tree_._select_clusters()
    raw_condensed_tree = clusterer.condensed_tree_._raw_tree

    exemplars = []
    for cluster in selected_clusters:
        cluster_exemplars = np.array([], dtype=np.int64)
        for leaf in clusterer._prediction_data._recurse_leaf_dfs(cluster):
            leaf_max_lambda = raw_condensed_tree["lambda_val"][
                raw_condensed_tree["parent"] == leaf
            ].max()
            points = raw_condensed_tree["child"][
                (raw_condensed_tree["parent"] == leaf)
                & (raw_condensed_tree["lambda_val"] == leaf_max_lambda)
            ]
            cluster_exemplars = np.hstack([cluster_exemplars, points])
        exemplars.append(cluster_exemplars)
    return exemplars


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def furthest_point_sampling(pts, K):
    farthest_pts = np.zeros((K, 2))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def identify_exemplars(sample, pop):
    return np.where((sample[:, None, :] == pop[None, :, :]).all(-1))[1]


def sample_exemplar_ids(clusterer, emb, K):
    ex_list = []
    for exemplars in clusterer.exemplars_:
        sample = furthest_point_sampling(exemplars, K)
        ids = identify_exemplars(sample, emb.cpu().numpy())
        ex_list += [ids]
    return ex_list
