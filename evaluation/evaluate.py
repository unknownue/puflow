import argparse
import numpy as np
from glob import glob
import re
import csv
from collections import OrderedDict
import logging

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
from tf_ops.nn_distance import tf_nndistance
from tf_ops.approxmatch import tf_approxmatch
from sklearn.neighbors import NearestNeighbors

import math
from time import time
from jsd import jsd_between_point_cloud_sets

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.enable_resource_variables()


def load_xyz(filename, count=None):
    points = np.loadtxt(filename).astype(np.float32)
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = downsample_points(points, count)
    return points

def normalize_point_cloud(pc):
    """
    pc [N, P, 3]
    """
    centroid = tf.reduce_mean(pc, axis=1, keep_dims=True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keep_dims=True)), axis=1, keep_dims=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance

def earth_mover(pcd1, pcd2, radius=1.0):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    cost = cost / radius
    return tf.reduce_mean(cost / num_points)

parser = argparse.ArgumentParser()
parser.add_argument("--pred", type=str, required=True, help=".xyz")
parser.add_argument("--gt", type=str, required=True, help=".xyz")
parser.add_argument("--save_path", type=str, required=True, help="path to save the result")
FLAGS = parser.parse_args()
PRED_DIR = os.path.abspath(FLAGS.pred)
GT_DIR = os.path.abspath(FLAGS.gt)
# print(PRED_DIR)
# NAME = FLAGS.name

# print(GT_DIR)
gt_paths = glob(os.path.join(GT_DIR,'*.xyz'))

gt_names = [os.path.basename(p)[:-4] for p in gt_paths]
# print(len(gt_paths))


def np_normalize(pts):
	centroid = np.mean(pts, axis=1, keepdims=True)
	
	pts = pts - centroid
	furthest = np.amax(np.sqrt(np.sum(pts ** 2, axis=-1)), axis=1, keepdims=True)
	pts = pts / np.expand_dims(furthest, axis=-1)
	# print(np.min(pts), np.max(pts))
	return pts * 0.5


gt = load(gt_paths[0])[:, :3]
pred_placeholder = tf.placeholder(tf.float32, [1, gt.shape[0], 3])
gt_placeholder = tf.placeholder(tf.float32, [1, gt.shape[0], 3])
pred_tensor, centroid, furthest_distance = normalize_point_cloud(pred_placeholder)
gt_tensor, centroid, furthest_distance = normalize_point_cloud(gt_placeholder)

cd_forward, _, cd_backward, _ = tf_nndistance.nn_distance(pred_tensor, gt_tensor)
cd_forward = cd_forward[0, :]
cd_backward = cd_backward[0, :]
emd_forward = earth_mover(pred_tensor, gt_tensor)

precentages = np.array([0.004, 0.006, 0.008, 0.010, 0.012])
# precentages = np.array([0.008, 0.012])

def cal_nearest_distance(queries, pc, k=2):
    """
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    dis,knn_idx = knn_search.kneighbors(queries, return_distance=True)
    return dis[:,1]

def analyze_uniform(idx_file,radius_file,map_points_file):
    start_time = time()
    points = load(map_points_file)[:,4:]
    radius = np.loadtxt(radius_file)
    print('radius:',radius)
    with open(idx_file) as f:
        lines = f.readlines()

    sample_number = 1000
    rad_number = radius.shape[0]

    uniform_measure = np.zeros([rad_number,1])

    densitys = np.zeros([rad_number,sample_number])

	
    expect_number = precentages * points.shape[0]
    expect_number = np.reshape(expect_number, [rad_number, 1])

    for j in range(rad_number):
        uniform_dis = []

        for i in range(sample_number):

            density, idx = lines[i*rad_number+j].split(':')
            densitys[j,i] = int(density)
            coverage = np.square(densitys[j,i] - expect_number[j]) / expect_number[j]

            num_points = re.findall("(\d+)", idx)

            idx = list(map(int, num_points))
            if len(idx) < 5:
                continue

            idx = np.array(idx).astype(np.int32)
            map_point = points[idx]

            shortest_dis = cal_nearest_distance(map_point,map_point,2)
            disk_area = math.pi * (radius[j] ** 2) / map_point.shape[0]
            expect_d = math.sqrt(2 * disk_area / 1.732)##using hexagon

            dis = np.square(shortest_dis - expect_d) / expect_d
            dis_mean = np.mean(dis)
            uniform_dis.append(coverage*dis_mean)

        uniform_dis = np.array(uniform_dis).astype(np.float32)
        uniform_measure[j, 0] = np.mean(uniform_dis)

    print('time cost for uniform :',time()-start_time)
    return uniform_measure

#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
config = tf.ConfigProto()
#config.graph_options.rewrite_options.auto_mixed_precision = 1
#config.gpu_options.allow_growth = False
#config.device_count['GPU'] = 0

with tf.Session() as sess:
    fieldnames = ["name", "CD", "EMD", "hausdorff", "p2f avg", "p2f std", "JSD"]

    fieldnames += ["uniform_%d" % d for d in range(precentages.shape[0])]

    # print("{:60s} ".format("name"), "|".join(["{:>15s}".format(d) for d in fieldnames[1:]]))
    for D in [PRED_DIR]:
        avg_md_forward_value = 0
        avg_md_backward_value = 0
        avg_hd_value = 0
        avg_emd_value = 0
        pred_paths = glob(os.path.join(D, "*.xyz"))

        gt_pred_pairs = []
        for p in pred_paths:
            name, ext = os.path.splitext(os.path.basename(p))
            assert(ext in (".ply", ".xyz"))
            try:
                gt = gt_paths[gt_names.index(name)]
            except ValueError:
                pass
            else:
                gt_pred_pairs.append((gt, p))

        # print("total inputs ", len(gt_pred_pairs))
        tag = re.search("/(\w+)/result", os.path.dirname(gt_pred_pairs[0][1]))
        if tag:
            tag = tag.groups()[0]
        else:
            tag = D

        # print("{:60s}".format(tag), end=' ')
        global_cd  = []
        global_emd = []
        global_jsd = []
        global_hd  = []
        global_p2f = []
        global_density = []
        global_uniform = []
        # print()

        with open(os.path.join(FLAGS.save_path, "evaluation.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
            writer.writeheader()
            for gt_path, pred_path in gt_pred_pairs:
                # print("Evaluating: ", pred_path)
                row = {}
                gt = load(gt_path)[:, :3]
                gt = gt[np.newaxis, ...]
                pred = load_xyz(pred_path)
                pred = pred[:, :3]

                row["name"] = os.path.basename(pred_path)
                pred = pred[np.newaxis, ...]
                
                cd_forward_value, cd_backward_value, emd_forward_value = sess.run([cd_forward, cd_backward, emd_forward], feed_dict={pred_placeholder:pred, gt_placeholder:gt})

                md_value = np.mean(cd_forward_value)+np.mean(cd_backward_value)
                hd_value = np.max(np.amax(cd_forward_value, axis=0)+np.amax(cd_backward_value, axis=0))
                cd_backward_value = np.mean(cd_backward_value)
                cd_forward_value = np.mean(cd_forward_value)
                emd_forward_value = np.mean(emd_forward_value)
                row["CD"] = cd_forward_value+cd_backward_value
                row["EMD"] = emd_forward_value
                row["hausdorff"] = hd_value

                jsd_value = jsd_between_point_cloud_sets(np_normalize(pred), np_normalize(gt))
                
                global_cd.append(cd_forward_value + cd_backward_value)
                global_hd.append(hd_value)
                global_emd.append(emd_forward_value)
                global_jsd.append(jsd_value)

                #print(pred_path[:-4] + "_point2mesh_distance.xyz")
                if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.xyz"):
                    point2mesh_distance = load(pred_path[:-4] + "_point2mesh_distance.xyz")
                    if point2mesh_distance.size > 0:
    	                point2mesh_distance = point2mesh_distance[:, 3]
    	                row["p2f avg"] = np.nanmean(point2mesh_distance)
    	                row["p2f std"] = np.nanstd(point2mesh_distance)
    	                global_p2f.append(point2mesh_distance)

    	                row["JSD"] = jsd_value
    	                if os.path.isfile(pred_path[:-4] + "_disk_idx.txt"):
    	                
    	                    idx_file = pred_path[:-4] + "_disk_idx.txt"
    	                    radius_file = pred_path[:-4] + '_radius.txt'
    	                    map_points_file = pred_path[:-4] + '_point2mesh_distance.txt'
    	                    
    	                    disk_measure = analyze_uniform(idx_file, radius_file, map_points_file)
    	                    global_uniform.append(disk_measure)

    	                    for i in range(len(precentages)):
                                row["uniform_%d" % i] = disk_measure[i, 0]

                writer.writerow(row)

            row = OrderedDict()

            row["CD"] = np.nanmean(global_cd)
            row["EMD"] = np.nanmean(global_emd)
            row["hausdorff"] = np.nanmean(global_hd)
            if global_p2f:
                global_p2f = np.concatenate(global_p2f, axis=0)
                mean_p2f = np.nanmean(global_p2f)
                std_p2f = np.nanstd(global_p2f)
                row["p2f avg"] = mean_p2f
                row["p2f std"] = std_p2f

            row["JSD"] = np.nanmean(global_jsd)
            if global_uniform:
                global_uniform = np.array(global_uniform)
                uniform_mean = np.mean(global_uniform, axis=0)
                for i in range(precentages.shape[0]):
                    row["uniform_%d" % i] = uniform_mean[i, 0]
                
            writer.writerow(row)
            # print(list(row.keys()))
            # print("|".join(["{:>15.8f}".format(row[d]) for d in ['CD', 'EMD', "hausdorff", "p2f avg", 'p2f std', 'JSD']]))

            metrics = []
            print(f'Evaluation: {FLAGS.save_path}')
            for key in ['CD', 'EMD', "hausdorff", "p2f avg", 'p2f std', 'JSD']:
                metrics.append(f'[{key}]{row[key]:>.8f}')
            print('\t' + '  '.join(metrics))
