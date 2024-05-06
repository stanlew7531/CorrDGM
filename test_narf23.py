import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import SurrealTest, SHRECTest, SHRECTest_witout
from models import Net
from tensorboardX import SummaryWriter
from datetime import datetime
from utils.config import cfg
from pathlib import Path
from utils.model_sl import load_model, save_model
import os
import open3d as o3d

def prob_to_corr_test(prob_matrix):
    c = torch.zeros_like(input=prob_matrix)
    idx = torch.argmax(input=prob_matrix, dim=2, keepdim=True)
    for bsize in range(c.shape[0]):
        for each_row in range(c.shape[1]):
            c[bsize][each_row][idx[bsize][each_row]] = 1.0

    return c

def visualize_match(src, tgt, corr):
    tgt_tmp = tgt + 0.5 # shift target over so that the points arent on top of each other

    source_points = src[0]
    target_points = tgt_tmp[0].clone()

    for i in range(corr.shape[1]):
        target_points[i] = tgt_tmp[0, corr[0, i]]
    lines = []
    k = 500
    point = []
    for i in range(k):
        lines.append([2 * i, 2 * i + 1])
        point.append(source_points[i].tolist())
        point.append(target_points[i].tolist())

    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(point),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    point1 = source_points.tolist()
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point1)

    point2 = target_points.tolist()
    print(len(point2))
    pcd2 = o3d.geometry.PointCloud()
    print(tgt_tmp[0].shape)
    pcd2.points = o3d.utility.Vector3dVector(tgt_tmp[0].cpu().numpy())
    o3d.visualization.draw_geometries([line_set, pcd1, pcd2])


def eval_points(net, points_a, points_b, name=None):
    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'resume'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)
    if name is not None:
        print('Loading specific from {}'.format(name))
        load_model(net, name)
    else:
        model_path = str(checkpoint_path / 'params_best.pt')
        print('Loading best model parameters from {}'.format(model_path))
        load_model(net, model_path)

    net.eval()
    total_acc = 0
    num_examples = 0

    src = points_a.cuda()
    tgt = points_b.cuda()
    p = net(src, tgt)
    corr_tensor = prob_to_corr_test(p)
    visualize_match(src, tgt, corr_tensor.to(torch.bool))

if __name__ == '__main__':
    from utils.parse_argspc import parse_args

    TIMESTAMP = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args = parse_args('Non-rigid registration of graph matching training & evaluation code.')

    net = Net.Net()
    net.cuda()

    points_a = np.load("/home/stanlew/src/narf23/point_clouds/points/0_10.npy")#'/home/stanlew/src/narf23/narf/scripts/panda_scene_0_points.npy')
    points_b = np.load("/home/stanlew/src/narf23/point_clouds/points/3_10.npy")#'/home/stanlew/src/narf23/narf/scripts/panda_scene_1_points.npy')
    print(points_a.shape)
    print(points_b.shape)
    points_a = torch.from_numpy(points_a).float().unsqueeze(dim = 0)
    points_b = torch.from_numpy(points_b).float().unsqueeze(dim = 0)

    print(points_a.shape)
    print(points_b.shape)
    num_points = 10000
    rand_indices = torch.randperm(points_a.shape[1])[:num_points]
    print(rand_indices.shape)
    points_a = points_a[:, rand_indices, :]

    rand_indices = torch.randperm(points_b.shape[1])[:num_points]
    print(rand_indices.shape)
    points_b = points_b[:, rand_indices, :]

    print(points_a.shape)
    print(points_b.shape)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_a.cpu().numpy().tolist())
    # o3d.visualization.draw_geometries([pcd])

    pcd = o3d.geometry.PointCloud()
    pb_viz = points_b.cpu().numpy()
    print(pb_viz.shape)
    pcd.points = o3d.utility.Vector3dVector(pb_viz[0])
    o3d.visualization.draw_geometries([pcd])

    eval_points(net, points_a, points_b, name="pretrained.pt")
    