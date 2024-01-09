#
# Created on Wed Oct 18 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: This file supports to generate dataset based on a config file
# Version: 7.1
# Note: You may need to increase/ decrease values in the config to get stable states
# Tip: Start with gen_demand=True, set off other gen_* flags
# Tip: set debug=True for more details
# Tip: don't change static hydraulic values
# ------------------------------
#

import os
import sys

# setting path
abs_file = __file__
prefix = os.path.abspath(os.path.dirname(abs_file))
sys.path.append(prefix)
print("now path: {}".format(sys.path))

# from TokenGeneratorByRange import *
import argparse

import shutil
import datetime
import zarr
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
# from Executorv7 import *
import networkx as nx
import configparser
import pint
import epynet
from epynet import Network, epanet2, ObjectCollection
from enum import Enum
import glob
import wntr
import wntr.epanet.util as wutils
from copy import deepcopy
import ray
import pandas as pd
import numpy as np
from ray.exceptions import RayError
from numcodecs import Blosc
from sklearn.cluster import k_means
import json
from collections import defaultdict


# ------------------------------------
#
# Token Generator Part
#
# ------------------------------------

EPSILON = 1e-12

def update_object_by_json_string(json_string, object_dict, expected_shape):
    is_success = True
    overridden_values = None
    try:
        value_dict = json.loads(json_string)

        overridden_values = np.zeros(shape=expected_shape)

        uids = object_dict.uid

        in_uids_mask = np.isin(uids, value_dict, assume_unique=True, invert=True)

        tmp = np.array(list(value_dict.values())).T
        overridden_values[:, in_uids_mask] = tmp
        if in_uids_mask.shape[0] != overridden_values.shape[1]:
            print(
                f'WARNING! in mask shape is not equal to expected shape! in mask shape = {in_uids_mask.shape}, expected shape = {expected_shape.shape}!')
            print(f'Missing values will be replaced by zeros!')
    except Exception as e:
        print(f'Error in update_demand_json - Error: {e}')
        is_success = False
        overridden_values = None

    return overridden_values, is_success


def compute_contineous_values_by_range(tokens, ratios, ori_vals=None, **kwargs):
    range_lo, range_hi = ratios[0], ratios[1]
    new_values = range_lo + tokens * (range_hi - range_lo)
    return new_values


def compute_boolean_values(tokens, ratios, **kwargs):
    open_prob = ratios[0]
    new_values = np.less(tokens, open_prob).astype(tokens.dtype)
    return new_values


def compute_contineous_values_by_ratio(ori_vals, tokens, ratios, **kwargs):
    param_minmax = [0, np.max(ori_vals)]
    ratio_lo, ratio_hi = ratios[0], ratios[1]
    new_values = ori_vals + np.sign(tokens) * (ratio_lo + (np.abs(tokens) * (ratio_hi - ratio_lo))) * ori_vals
    new_values = np.clip(new_values, param_minmax[0], param_minmax[1])
    return new_values


def compute_contineous_diameter_by_ratio(ori_vals, tokens, ratios, **kwargs):
    param_minmax = [np.min(ori_vals), np.max(ori_vals)]
    ratio_lo, ratio_hi = ratios[0], ratios[1]
    new_values = ori_vals + np.sign(tokens) * (ratio_lo + (np.abs(tokens) * (ratio_hi - ratio_lo))) * ori_vals
    # new_values_mask = (new_values <= 0) #np.clip(new_values, param_minmax[0], param_minmax[1])
    new_values = np.where(new_values <= param_minmax[0], ori_vals, new_values)
    return new_values


def compute_contineous_values_by_ran_cluster(ori_vals, tokens, ratios, **kwargs):
    """Require kwargs :
        use_existing_clusters (bool): flag,\n
        num_clusters_lo (int): lowest n_clusters,\n
        num_clusters_hi (int): highest n_clusters,\n
    Optional kwargs:
        kmean_params (np.ndrray): nodal coordinates or link coordinates or any feature vectors, only used if use_existing_clusters = False\n
        sigma (float): standard deviation from local centroid values. If None, take std from ori_vals\n
        cluster_num_clusters (int): a specific number of cluster, only used if use_existing_clusters = True\n
        cluster_labels (np.ndrray): a specific labels assigned to elements, only used if use_existing_clusters = True\n
    Args:
        ori_vals (list): old/ original values
        tokens (np.ndrray): random tokens
        ratios (list): range or ratio list

    Returns:
        np.ndrray: new updated values
    """

    use_existing_clusters = kwargs['use_existing_clusters']
    num_clusters_lo, num_clusters_hi = kwargs['num_clusters_lo'], kwargs['num_clusters_hi']
    chunk_size = tokens.shape[0]

    range_lo, range_hi = ratios[0], ratios[1]

    num_elements = len(ori_vals)
    if not use_existing_clusters:

        kmean_params = kwargs['kmean_params']
        num_clusters = num_clusters_lo + np.random.random([chunk_size, 1]) * (num_clusters_hi - num_clusters_lo)
        if num_clusters_hi < num_elements:
            chunk_labels = []
            for c in range(chunk_size):
                centroids, labels, _ = k_means(kmean_params, n_clusters=int(num_clusters[c]), n_init='auto')
                chunk_labels.append(labels)
            labels = np.array(chunk_labels).reshape([chunk_size, -1])
        else:
            labels = np.arange(num_elements).reshape([1, -1]).repeat(chunk_size, axis=0)

    else:
        num_clusters = kwargs['cluster_num_clusters']
        labels = kwargs['cluster_labels']

    if num_clusters_hi < num_elements:
        local_tokens = range_lo + np.random.random(size=[chunk_size, num_clusters_hi]) * (range_hi - range_lo)
    else:
        local_tokens = range_lo + np.random.random(size=[chunk_size, num_elements]) * (range_hi - range_lo)

    sign = np.where(np.random.random(size=tokens.shape) >= 0.5, 1.0, -1.0)

    cluster_vals = np.take_along_axis(local_tokens, labels, axis=1)

    if 'sigma' in kwargs:
        sigma = kwargs['sigma']
    else:
        sigma = np.std(ori_vals.flatten())
        # sigma = abs(ori_vals - cluster_vals)

    new_values = cluster_vals + sign * tokens * sigma

    new_values = np.clip(new_values, a_min=range_lo, a_max=range_hi)
    return new_values


def get_node_coordinates(wn_g: nx.Graph, wn: Network, do_normalize: bool = True) -> dict:
    pos_dict = nx.get_node_attributes(wn_g, "pos")

    node_coords = [pos_dict[node.uid] for node in wn.nodes if node.uid in pos_dict]

    assert len(wn.nodes) == len(node_coords)
    node_coords = np.array(node_coords)
    if do_normalize:
        node_coords = node_coords / (np.linalg.norm(node_coords) + EPSILON)

    return dict(zip(wn.nodes.uid, node_coords.tolist()))


def get_link_coordinates(wn_g: nx.Graph, wn: Network, do_normalize: bool = True) -> dict:
    pos_dict = nx.get_node_attributes(wn_g, "pos")

    link_coords = []
    for link in wn.links:
        upstream_node = link.upstream_node.uid
        downstream_node = link.upstream_node.uid
        if upstream_node in pos_dict and downstream_node in pos_dict:
            xy = pos_dict[upstream_node]
            zt = pos_dict[downstream_node]
            link_coords.append([xy[0], xy[1], zt[0], zt[1]])

    assert len(wn.links) == len(link_coords)
    link_coords = np.array(link_coords)
    if do_normalize:
        link_coords = link_coords / (np.linalg.norm(link_coords) + EPSILON)

    return dict(zip(wn.links.uid, link_coords.tolist()))


def generate_params(
        tokens,
        ratios,
        target_object_collection,
        get_original_values_fn,
        update_formula_fn,
        component_key,
        update_json=None,
        **kwargs,
):
    new_values = None
    if update_json is not None:
        new_values, _ = update_object_by_json_string(json_string=update_json,
                                                     object_dict=target_object_collection,
                                                     expected_shape=tokens.shape)

    if new_values is None:
        if kwargs is not None and 'coords' in kwargs:
            coord_dict = kwargs['coords']
            kwargs['kmean_params'] = [coord_dict[obj.uid] for obj in target_object_collection]

        # ori_vals = da.from_array(list(map(get_original_values_fn,target_object_collection)))
        # new_demands = ori_dmds + sign(tokens) *  (dmd_lo + abs(tokens) * (dmd_hi - dmd_lo) )
        # tokens = da.from_array(tokens,chunks=self.num_chunks)
        ori_vals = np.array(list(map(get_original_values_fn, target_object_collection)))
        if sum(ratios) == 0.:
            new_values = ori_vals
        else:
            new_values = update_formula_fn(tokens=tokens, ratios=ratios, ori_vals=ori_vals, **kwargs)

    return new_values


@ray.remote
def ray_batch_update(chunk_size, num_features, featlen_dict, args):
    return batch_update(chunk_size, num_features, featlen_dict, args)


def batch_update(chunk_size, num_features, featlen_dict, args):
    tokens = np.random.uniform(
        low=0.0,  # -1.0,
        high=1.0,
        size=(chunk_size, num_features)
    )

    config = configparser.ConfigParser()
    config.read(args.config)
    config_keys = dict(config.items()).keys()

    wn_inp_path = config.get('general', 'wn_inp_path')
    wn = Network(wn_inp_path)
    wn_g = wntr.network.WaterNetworkModel(wn_inp_path).get_graph()

    ragged_tokens = RaggedArrayDict.from_keylen_and_stackedarray(featlen_dict, tokens)

    node_coord_dict = get_node_coordinates(wn_g, wn, do_normalize=True)
    link_coord_dict = get_link_coordinates(wn_g, wn, do_normalize=True)

    update_formula_args = {
        'use_existing_clusters': False,
        'num_clusters_lo': 4,
        'num_clusters_hi': 50,
        'sigma': 1.0,
    }
    node_update_formula_args = deepcopy(update_formula_args)
    link_update_formula_args = deepcopy(update_formula_args)
    node_update_formula_args['coords'] = node_coord_dict
    link_update_formula_args['coords'] = link_coord_dict

    new_tokens = defaultdict()
    if 'junction' in config_keys:
        if args.gen_demand:
            def get_origin_dmd(junc):
                return junc.basedemand * junc.pattern.values[0] if ENhasppatern(junc) else 0.

            new_tokens[ParamEnum.JUNC_DEMAND] = generate_params(
                tokens=ragged_tokens[ParamEnum.JUNC_DEMAND],
                ratios=[
                    config.getfloat('junction', 'demand_lo'),
                    config.getfloat('junction', 'demand_hi')
                ],
                target_object_collection=wn.junctions,
                get_original_values_fn=get_origin_dmd,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.JUNC_DEMAND,
                update_json=args.update_demand_json,
                **node_update_formula_args
            )

        if args.gen_elevation:
            new_tokens[ParamEnum.JUNC_ELEVATION] = generate_params(
                tokens=ragged_tokens[ParamEnum.JUNC_ELEVATION],
                ratios=[
                    config.getfloat('junction', 'ele_lo'),
                    config.getfloat('junction', 'ele_hi')
                ],
                target_object_collection=wn.junctions,
                get_original_values_fn=lambda junc: junc.elevation,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.JUNC_ELEVATION,
                update_json=args.update_elevation_json,
                **node_update_formula_args
            )

    if 'pump' in config_keys:
        if args.gen_pump_init_status:
            new_tokens[ParamEnum.PUMP_STATUS] = generate_params(
                tokens=ragged_tokens[ParamEnum.PUMP_STATUS],
                ratios=[
                    config.getfloat('pump', 'open_prob'),
                ],
                target_object_collection=wn.pumps,
                get_original_values_fn=lambda pump: pump.initstatus,
                update_formula_fn=compute_boolean_values,
                component_key=ParamEnum.PUMP_STATUS,
                update_json=args.update_pump_init_status_json,
            )

        if args.gen_pump_speed:
            new_tokens[ParamEnum.PUMP_SPEED] = generate_params(
                tokens=ragged_tokens[ParamEnum.PUMP_SPEED],
                ratios=[
                    config.getfloat('pump', 'speed_lo'),
                    config.getfloat('pump', 'speed_hi'),
                ],
                target_object_collection=wn.pumps,
                get_original_values_fn=lambda pump: pump.speed,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.PUMP_SPEED,
                update_json=args.update_pump_speed_json,
                **link_update_formula_args,
            )

        if args.gen_pump_length:
            new_tokens[ParamEnum.PUMP_LENGTH] = generate_params(
                tokens=ragged_tokens[ParamEnum.PUMP_LENGTH],
                ratios=[
                    config.getfloat('pump', 'length_lo'),
                    config.getfloat('pump', 'length_hi'),
                ],
                target_object_collection=wn.pumps,
                get_original_values_fn=lambda pump: pump.length,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.PUMP_LENGTH,
                update_json=args.update_pump_length_json,
                **link_update_formula_args,
            )

    if 'tank' in config_keys:
        if args.gen_tank_level:
            new_tokens[ParamEnum.TANK_LEVEL] = generate_params(
                tokens=ragged_tokens[ParamEnum.TANK_LEVEL],
                ratios=[
                    config.getfloat('tank', 'level_lo'),
                    config.getfloat('tank', 'level_hi'),
                ],
                target_object_collection=wn.tanks,
                get_original_values_fn=lambda tank: tank.tanklevel,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.TANK_LEVEL,
                update_json=args.update_tank_level_json,
                **node_update_formula_args,
            )

        if args.gen_tank_elevation:
            new_tokens[ParamEnum.TANK_ELEVATION] = generate_params(
                tokens=ragged_tokens[ParamEnum.TANK_ELEVATION],
                ratios=[
                    config.getfloat('tank', 'ele_lo'),
                    config.getfloat('tank', 'ele_hi'),
                ],
                target_object_collection=wn.tanks,
                get_original_values_fn=lambda tank: tank.elevation,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.TANK_ELEVATION,
                update_json=args.update_tank_elevation_json,
                **node_update_formula_args,
            )

        if args.gen_tank_diameter:
            new_tokens[ParamEnum.TANK_DIAMETER] = generate_params(
                tokens=ragged_tokens[ParamEnum.TANK_DIAMETER],
                ratios=[
                    config.getfloat('tank', 'dia_lo'),
                    config.getfloat('tank', 'dia_hi'),
                ],
                target_object_collection=wn.tanks,
                get_original_values_fn=lambda tank: tank.diameter,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.TANK_DIAMETER,
                update_json=args.update_tank_diameter_json,
                **node_update_formula_args,
            )

    if 'valve' in config_keys:
        if args.gen_valve_setting:
            '''
            ratio_los = []
            ratio_his = []
            for v in wn.valves:
                key= v.valve_type.lower()
                ratio_lo,ratio_hi = config.getfloat('valve',f'{key}_ratio_lo'),config.getfloat('valve',f'{key}_ratio_hi')
                ratio_los.append(ratio_lo)
                ratio_his.append(ratio_hi)
            ratios = np.stack([ratio_los,ratio_his],axis=0)
            '''
            valve_type_ratio_dict = {}
            valve_type_uid_dict = {}
            for v in wn.valves:
                if v.valve_type not in valve_type_ratio_dict:
                    key = v.valve_type.lower()
                    ratio_lo, ratio_hi = config.getfloat('valve', f'setting_{key}_lo'), config.getfloat('valve',
                                                                                                        f'setting_{key}_hi')
                    valve_type_ratio_dict[v.valve_type] = ratio_lo, ratio_hi
                    valve_type_uid_dict[v.valve_type] = []
                valve_type_uid_dict[v.valve_type].append(v.uid)
            overridden_values = np.zeros(shape=[chunk_size, len(wn.valves)])

            for valve_type in valve_type_ratio_dict:
                ratios = valve_type_ratio_dict[valve_type]
                uids = valve_type_uid_dict[valve_type]
                target_object_collection = ObjectCollection({k: wn.valves[k] for k in uids if k in wn.valves})
                # print(f'len target = {len(target_object_collection)} | len all = {len(wn.valves)}')
                in_uids_mask = np.isin(list(wn.valves.keys()), uids)
                # print(f'min in_uids_mask = {np.min(in_uids_mask)}')
                valve_type_new_tokens = generate_params(
                    tokens=ragged_tokens[ParamEnum.VALVE_SETTING][:, in_uids_mask],
                    ratios=ratios,
                    target_object_collection=target_object_collection,
                    get_original_values_fn=lambda v: v.setting,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key=ParamEnum.VALVE_SETTING,
                    update_json=args.update_valve_setting_json,
                    **link_update_formula_args
                )

                overridden_values[:, in_uids_mask] = valve_type_new_tokens

            new_tokens[ParamEnum.VALVE_SETTING] = overridden_values

        if args.gen_valve_init_status:
            new_tokens[ParamEnum.VALVE_STATUS] = generate_params(
                tokens=ragged_tokens[ParamEnum.VALVE_STATUS],
                ratios=[
                    config.getfloat('valve', 'open_prob'),
                ],
                target_object_collection=wn.valves,
                get_original_values_fn=lambda v: v.initstatus,
                update_formula_fn=compute_boolean_values,
                component_key=ParamEnum.VALVE_STATUS,
                update_json=args.update_valve_init_status_json,
            )

        if args.gen_valve_diameter:
            new_tokens[ParamEnum.VALVE_DIAMETER] = generate_params(
                tokens=ragged_tokens[ParamEnum.VALVE_DIAMETER],
                ratios=[
                    config.getfloat('valve', 'dia_lo'),
                    config.getfloat('valve', 'dia_hi'),
                ],
                target_object_collection=wn.valves,
                get_original_values_fn=lambda v: v.diameter,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.VALVE_DIAMETER,
                update_json=args.update_valve_diameter_json,
                **link_update_formula_args
            )

    if 'pipe' in config_keys:
        if args.gen_roughness:
            new_tokens[ParamEnum.PIPE_ROUGHNESS] = generate_params(
                tokens=ragged_tokens[ParamEnum.PIPE_ROUGHNESS],
                ratios=[
                    config.getfloat('pipe', 'roughness_lo'),
                    config.getfloat('pipe', 'roughness_hi'),
                ],
                target_object_collection=wn.pipes,
                get_original_values_fn=lambda p: p.roughness,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.PIPE_ROUGHNESS,
                update_json=args.update_pipe_roughness_json,
                **link_update_formula_args,
            )

        if args.gen_diameter:
            new_tokens[ParamEnum.PIPE_DIAMETER] = generate_params(
                tokens=ragged_tokens[ParamEnum.PIPE_DIAMETER],
                ratios=[
                    config.getfloat('pipe', 'diameter_lo'),
                    config.getfloat('pipe', 'diameter_hi'),
                ],
                target_object_collection=wn.pipes,
                get_original_values_fn=lambda p: p.diameter,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.PIPE_DIAMETER,
                update_json=args.update_pipe_diameter_json,
                **link_update_formula_args,
            )

        if args.gen_length:
            new_tokens[ParamEnum.PIPE_LENGTH] = generate_params(
                tokens=ragged_tokens[ParamEnum.PIPE_LENGTH],
                ratios=[
                    config.getfloat('pipe', 'length_lo'),
                    config.getfloat('pipe', 'length_hi'),
                ],
                target_object_collection=wn.pipes,
                get_original_values_fn=lambda p: p.length,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.PIPE_LENGTH,
                update_json=args.update_pipe_length_json,
                **link_update_formula_args
            )

        if args.gen_minorloss:
            new_tokens[ParamEnum.PIPE_MINORLOSS] = generate_params(
                tokens=ragged_tokens[ParamEnum.PIPE_MINORLOSS],
                ratios=[
                    config.getfloat('pipe', 'minorloss_lo'),
                    config.getfloat('pipe', 'minorloss_hi'),
                ],
                target_object_collection=wn.pipes,
                get_original_values_fn=lambda p: p.minorloss,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.PIPE_MINORLOSS,
                update_json=args.update_pipe_minorloss_json,
                **link_update_formula_args
            )

    if 'reservoir' in config_keys:
        if args.gen_res_total_head:
            def get_original_res_head(res):
                base_head = res.elevation
                try:
                    p_index = res.get_object_value(epanet2.EN_PATTERN)
                    head = wn.ep.ENgetpatternvalue(int(p_index), 1)  # get first value only
                except epanet2.ENtoolkitError:
                    head = 1.

                return base_head * head

            new_tokens[ParamEnum.RESERVOIR_TOTALHEAD] = generate_params(
                tokens=ragged_tokens[ParamEnum.RESERVOIR_TOTALHEAD],
                ratios=[
                    config.getfloat('reservoir', 'head_lo'),
                    config.getfloat('reservoir', 'head_hi'),
                ],
                target_object_collection=wn.reservoirs,
                get_original_values_fn=get_original_res_head,
                update_formula_fn=compute_contineous_values_by_range,
                component_key=ParamEnum.RESERVOIR_TOTALHEAD,
                update_json=args.update_res_total_head_json,
                **node_update_formula_args
            )

    # ensure the order
    concated_arrays = [new_tokens[k] for k in featlen_dict.keys()]
    return np.concatenate(concated_arrays, axis=-1)


class RayTokenGenerator:
    def __init__(self, store, num_scenes, featlen_dict, num_chunks):
        self.store = store
        self.num_scenes = num_scenes
        self.featlen_dict = featlen_dict
        self.num_chunks = num_chunks
        self.num_features = sum(self.featlen_dict.values())

    def update(self, args):
        chunk_size = args.batch_size
        num_chunks = self.num_scenes // chunk_size
        progressbar = tqdm(total=num_chunks)
        start_index = 0
        worker_ids = [ray_batch_update.remote(chunk_size, self.num_features, self.featlen_dict, args) for _ in
                      range(num_chunks)]
        done_ids, undone_ids = ray.wait(worker_ids)
        num_out_features = sum(self.featlen_dict.values())

        while done_ids:
            result = ray.get(done_ids[0])
            if start_index == 0:
                z_tokens = zarr.empty([self.num_scenes, num_out_features],
                                      chunks=(chunk_size, num_out_features),
                                      dtype='f8',
                                      store=os.path.join(self.store.path, ParamEnum.RANDOM_TOKEN),
                                      overwrite=True,
                                      synchronizer=zarr.ThreadSynchronizer(),
                                      compressor=Blosc(cname='lz4', clevel=5)
                                      )
            z_tokens[start_index: start_index + chunk_size] = result
            start_index += chunk_size
            del result
            done_ids, undone_ids = ray.wait(undone_ids)
            progressbar.update(1)
        progressbar.close()
        ray.shutdown()
        print('OK')

    def sequential_update(self, args):
        chunk_size = args.batch_size
        num_chunks = self.num_scenes // chunk_size
        num_out_features = sum(self.featlen_dict.values())
        start_index = 0
        progressbar = tqdm(total=num_chunks)
        for _ in range(num_chunks):
            result = batch_update(chunk_size, self.num_features, self.featlen_dict, args)
            if start_index == 0:
                z_tokens = zarr.empty([self.num_scenes, num_out_features],
                                      chunks=(chunk_size, num_out_features),
                                      dtype='f8',
                                      store=os.path.join(self.store.path, ParamEnum.RANDOM_TOKEN),
                                      overwrite=True,
                                      synchronizer=zarr.ThreadSynchronizer(),
                                      compressor=Blosc(cname='lz4', clevel=5)
                                      )
            z_tokens[start_index: start_index + chunk_size] = result
            start_index += chunk_size
            del result
            progressbar.update(1)

        progressbar.close()
        print('OK')

    def load_computed_params(self):
        param = zarr.open_array(
            store=os.path.join(self.store.path, ParamEnum.RANDOM_TOKEN),
            mode='r'
        )
        return param


# ------------------------------------
#
# Token Generator Part
#
# ------------------------------------

# ------------------------------------
# 
# epynet utils
# 
# ------------------------------------

def get_networkx_graph(wn, include_reservoir=True, graph_type='multi_directed'):
    """support to form a networkx graph from the Epynet water network
    ref: https://github.com/BME-SmartLab/GraphConvWat/blob/be97b45fbc7dfdba22bb1ee406424a7c568120e5/utils/graph_utils.py
    :param Epynet.Network wn: water network object
    :param bool include_reservoir: Flag indicates involve links from reservoirs, defaults to True
    """
    if graph_type == 'undirected':
        G = nx.Graph()
    elif graph_type == 'directed':
        G = nx.DiGraph()
    elif graph_type == 'multi_undirected':
        G = nx.MultiGraph()
    elif graph_type == 'multi_directed':
        G = nx.MultiDiGraph()
    else:
        raise NotImplementedError()

    node_list = []
    collection = wn.junctions if not include_reservoir else wn.nodes
    for node in collection:
        node_list.append(node.uid)

    for pipe in wn.pipes:
        if (pipe.from_node.uid in node_list) and (pipe.to_node.uid in node_list):
            G.add_edge(pipe.from_node.uid, pipe.to_node.uid, weight=1., length=pipe.length)
        else:
            print(f'WARNING! pipe {pipe.uid} is not connect to any node in node list')
    for pump in wn.pumps:
        if (pump.from_node.uid in node_list) and (pump.to_node.uid in node_list):
            G.add_edge(pump.from_node.uid, pump.to_node.uid, weight=1., length=0.)
        else:
            print(f'WARNING! pump {pump.uid} is not connect to any node in node list')
    for valve in wn.valves:
        if (valve.from_node.uid in node_list) and (valve.to_node.uid in node_list):
            G.add_edge(valve.from_node.uid, valve.to_node.uid, weight=1., length=0.)
        else:
            print(f'WARNING! valve {valve.uid} is not connect to any node in node list')

    return G


def args2config(args, config_name, config_path=None):
    """convert argparse args to config for saving

    :param dict args: arguments 
    :param str config_name: name of the new config
    :param str config_path: optional storage path 
    """
    config = configparser.ConfigParser()
    if config_path is not None:
        os.makedirs(config_path, exist_ok=True)

    config.defaults().update(vars(args))

    full_path = os.path.join(config_path, config_name)
    with open(full_path, 'w') as configfile:
        config.write(configfile)

    return full_path


def config2args(config_path):
    """convert config to args

    :param str config_name: name of the new config
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    defaults = config.defaults()
    parser = argparse.ArgumentParser()
    parser.set_defaults(**defaults)
    return parser.parse_args()


def set_object_value_wo_ierror(obj, code, value):
    """this function is similar to setattrr(obj,code,value) but omit the ierror flag causes crash when running RAY
    It is a sequence of [setattr -> set_static_property -> ENsetnodevalue] but we alternate to:
    [setattr -> set_static_property -> ENsetnodevalue2] to skip ierror
    :param Object obj: object from EPYNET network
    :param int code: EPANET Param Code
    :param Any value: value
    """
    # tank._values[epanet2.EN_TANKLEVEL] = tank_level
    # self.wn.ep.ENsetnodevalue2(tank.index, epanet2.EN_TANKLEVEL, tank_level)

    assert hasattr(obj, '_values') and hasattr(obj, 'index') and obj.network() is not None
    # ref https://github.com/Vitens/epynet/blob/992ce792c6b6427ee0d35325645c8185bc888928/epynet/baseobject.py#L42
    obj.network().solved = False
    obj._values[code] = value
    try:
        if isinstance(obj, epynet.Node):
            ENsetnodevalue2(obj.network().ep, obj.index, code, value)
        else:
            ENsetlinkvalue2(obj.network().ep, obj.index, code, value)
    except Exception as e:
        print(f'ERROR AT OBJ = {obj.uid},code = {code} , value = {value}')
        raise Exception(e)


def ENhasppatern(obj):
    if not isinstance(obj, epynet.Junction):
        return False

    try:
        p = obj.pattern
        return p is not None
    except Exception:
        return False


def ENsetnodevalue2(ep, index, paramcode, value):
    """Sets the value of a parameter for a specific node.
    Arguments:
    index:  node index
    paramcode: Node parameter codes consist of the following constants:
                  EN_ELEVATION  Elevation
                  EN_BASEDEMAND ** Base demand
                  EN_PATTERN    ** Demand pattern index
                  EN_EMITTER    Emitter coeff.
                  EN_INITQUAL   Initial quality
                  EN_SOURCEQUAL Source quality
                  EN_SOURCEPAT  Source pattern index
                  EN_SOURCETYPE Source type (See note below)
                  EN_TANKLEVEL  Initial water level in tank
                       ** primary demand category is last on demand list
               The following parameter codes apply only to storage tank nodes
                  EN_TANKDIAM      Tank diameter
                  EN_MINVOLUME     Minimum water volume
                  EN_MINLEVEL      Minimum water level
                  EN_MAXLEVEL      Maximum water level
                  EN_MIXMODEL      Mixing model code
                  EN_MIXFRACTION   Fraction of total volume occupied by the inlet/outlet
                  EN_TANK_KBULK    Bulk reaction rate coefficient
    value:parameter value"""
    ierr = ep._lib.EN_setnodevalue(ep.ph, epanet2.ctypes.c_int(index), epanet2.ctypes.c_int(paramcode),
                                   epanet2.ctypes.c_float(value))
    if ierr != 0: raise Exception(ierr)
    del ierr
    # raise ENtoolkitError(self,100) if self._lib.EN_setnodevalue(self.ph, ctypes.c_int(index), ctypes.c_int(paramcode), ctypes.c_float(value)) != 0 else None
    # if ierr!=0: raise ENtoolkitError(self, ierr)


def ENsetlinkvalue2(ep, index, paramcode, value):
    ierr = ep._lib.EN_setlinkvalue(ep.ph, epanet2.ctypes.c_int(index), epanet2.ctypes.c_int(paramcode),
                                   epanet2.ctypes.c_float(value))
    if ierr != 0: raise Exception(ierr)
    del ierr


def ENdeletepattern(wn, pattern_uid, delete_pattern_in_rules=True):
    patten_index = epanet2.ctypes.c_int()

    wn.ep._lib.EN_getpatternindex(wn.ep.ph, epanet2.ctypes.c_char_p(pattern_uid.encode(wn.ep.charset)),
                                  epanet2.ctypes.byref(patten_index))
    wn.ep._lib.EN_deletepattern(wn.ep.ph, patten_index)
    if delete_pattern_in_rules:
        wn.ep._lib.EN_deleterule(wn.ep.ph, patten_index)


def ENdeletepatternbyindex(wn, pattern_index, delete_pattern_in_rules=True):
    wn.ep._lib.EN_deletepattern(wn.ep.ph, epanet2.ctypes.c_int(pattern_index))
    if delete_pattern_in_rules:
        wn.ep._lib.EN_deleterule(wn.ep.ph, epanet2.ctypes.c_int(pattern_index))


def ENsetdemandpattern(wn, node_index, demand_category, pattern_index):
    wn.ep._lib.EN_setdemandpattern(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category),
                                   epanet2.ctypes.c_int(pattern_index))


def ENsetdemandpatterntoallcategories(wn, node_index, base_demand, pattern_index):
    demand_category = 1
    ierr = wn.ep._lib.EN_setdemandpattern(wn.ep.ph, epanet2.ctypes.c_int(node_index),
                                          epanet2.ctypes.c_int(demand_category), epanet2.ctypes.c_int(pattern_index))
    ierr = wn.ep._lib.EN_setbasedemand(wn.ep.ph, epanet2.ctypes.c_int(node_index),
                                       epanet2.ctypes.c_int(demand_category), epanet2.ctypes.c_double(base_demand))
    while ierr == 0:
        demand_category += 1
        ierr = wn.ep._lib.EN_setdemandpattern(wn.ep.ph, epanet2.ctypes.c_int(node_index),
                                              epanet2.ctypes.c_int(demand_category),
                                              epanet2.ctypes.c_int(pattern_index))
        if ierr == 0:
            ierr = wn.ep._lib.EN_setbasedemand(wn.ep.ph, epanet2.ctypes.c_int(node_index),
                                               epanet2.ctypes.c_int(demand_category),
                                               epanet2.ctypes.c_double(base_demand))


def ENsetheadcurveindex(wn, pump_index, curve_index):
    ierr = wn.ep._lib.EN_setheadcurveindex(wn.ep.ph, epanet2.ctypes.c_int(pump_index),
                                           epanet2.ctypes.c_int(curve_index))
    if ierr != 0: raise Exception(ierr)


def ENdeletecontrol(ep, control_index):
    ierr = ep._lib.EN_deletecontrol(ep.ph, epanet2.ctypes.c_int(control_index))
    if ierr != 0: raise Exception(ierr)  # epanet2.ENtoolkitError(ep, ierr)
    del ierr


def ENdeleterule(ep, rule_index):
    ierr = ep._lib.EN_deleterule(ep.ph, epanet2.ctypes.c_int(rule_index))
    if ierr != 0: raise Exception(ierr)
    del ierr


def ENdeleteallcontrols(wn):
    current_control_index = 1  # start index
    ierr = wn.ep._lib.EN_deletecontrol(wn.ep.ph, epanet2.ctypes.c_int(current_control_index))
    # print(f'pre-ENdeleteallcontrols i = {current_control_index}, ierr = {ierr}')
    while (ierr == 0):  # no error
        current_control_index = 1
        ierr = wn.ep._lib.EN_deletecontrol(wn.ep.ph, epanet2.ctypes.c_int(current_control_index))
        # print(f'pre-ENdeleteallcontrols i = {current_control_index}, ierr = {ierr}')
    # [wn.ep._lib.EN_deletecontrol( wn.ep.ph,  epanet2.ctypes.c_int(trial_index)) for trial_index in range(1,20)]

    del ierr

    # ctype = epanet2.ctypes.c_int()
    # lindex = epanet2.ctypes.c_int()
    # setting= epanet2.ctypes.c_int()
    # nindex = epanet2.ctypes.c_int()
    # level= epanet2.ctypes.c_int()
    # for i in range(1,10):
    #    err=  wn.ep._lib.EN_getcontrol( wn.ep.ph, epanet2.ctypes.c_int(i), epanet2.ctypes.byref(ctype), 
    #                    epanet2.ctypes.byref(lindex), epanet2.ctypes.byref(setting), 
    #                    epanet2.ctypes.byref(nindex), epanet2.ctypes.byref(level) )
    #    print(f'After-ENdeleteallcontrols i = {i}, ierr = {err}')


def ENdeleteallpatterns(wn):
    current_pattern_index = 1  # start index
    ierr = wn.ep._lib.EN_deletepattern(wn.ep.ph, epanet2.ctypes.c_int(current_pattern_index))

    print(f'1-ENdeleteallpatterns id {current_pattern_index}-ierr = {ierr}')
    while (ierr == 0):  # no error
        current_pattern_index = 1
        ierr = wn.ep._lib.EN_deleterule(wn.ep.ph, epanet2.ctypes.c_int(current_pattern_index))
    print(f'1-ENdeleteallpatterns id {current_pattern_index}-ierr = {ierr}')
    del ierr


def ENdeleteallrules(wn):
    current_rule_index = 1  # start index
    ierr = wn.ep._lib.EN_deleterule(wn.ep.ph, epanet2.ctypes.c_int(current_rule_index))

    # print(f'1-ENdeleteallrules id {current_rule_index}-ierr = {ierr}')
    while (ierr == 0):  # no error
        current_rule_index = 1
        ierr = wn.ep._lib.EN_deleterule(wn.ep.ph, epanet2.ctypes.c_int(current_rule_index))
        # print(f'ENdeleteallrules id {current_rule_index}-ierr = {ierr}')

    del ierr


def ENconvert(from_unit, to_unit, hydraulic_param, values):
    """ENconvert helps convert value 

    :param str from_unit: original flow unit should be in []
    :param str to_unit: target flow unit should be in []
    :param str hydraulic_param: currently support ['pressure', 'demand', 'head',' velocity', 'flow']
    :param np.array values: _description_
    """

    us_flow_units = ['CFS', 'GPM', 'MGD', 'IMGD', 'AFD']
    si_flow_units = ['LPS', 'LPM', 'MLD', 'CMH', 'CMD']
    supported_flow_units = list(set(us_flow_units).union(si_flow_units))
    assert from_unit in supported_flow_units
    assert to_unit in supported_flow_units
    assert hydraulic_param in ['pressure', 'demand', 'head', ' velocity', 'flow']
    assert isinstance(values, np.ndarray)

    ureg = pint.UnitRegistry()

    # define quantity
    # basic units: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
    # flow and demand
    ureg.define('GPM = gallon / minute')
    ureg.define('cubic_meter = meter**3')
    ureg.define('CMH = cubic_meter / hour')
    ureg.define('meter_H2O = 100 * centimeter_H2O')
    ureg.define('CFS = cubic_feet / second')
    ureg.define('MGD = 1000000 * gallon / day')
    ureg.define('IMGD = 1000000 * imperial_gallon / day')
    ureg.define('AFD =  acre_feet / day')
    ureg.define('LPS = liter / second = lps')
    ureg.define('LPM =  liter / minute')
    ureg.define('MLD =  1000000 * liter / day')
    ureg.define('CMD =  cubic_meter / day')

    if hydraulic_param in ['demand', 'flow']:
        leg1 = ureg.Quantity(values, from_unit)
    else:
        if (from_unit in us_flow_units and to_unit in us_flow_units) \
                or (from_unit in si_flow_units and to_unit in si_flow_units):
            return values

        if hydraulic_param == 'pressure':
            leg1_punit = 'psi' if from_unit in us_flow_units else 'meter_H2O'
            leg1 = ureg.Quantity(values, leg1_punit)
        elif hydraulic_param == 'head':
            leg1_punit = 'feet_H2O' if from_unit in us_flow_units else 'meter_H2O'
            leg1 = ureg.Quantity(values, leg1_punit)
        elif hydraulic_param == 'velocity':
            leg1_punit = 'fps' if from_unit in us_flow_units else 'mps'
            leg1 = ureg.Quantity(values, leg1_punit)

    if hydraulic_param in ['demand', 'flow']:
        leg2 = leg1.to(to_unit)
    elif hydraulic_param == 'pressure':
        leg2_punit = 'psi' if to_unit in us_flow_units else 'meter_H2O'
        leg2 = leg1.to(leg2_punit)
    elif hydraulic_param == 'head':
        leg2_punit = 'feet_H2O' if to_unit in us_flow_units else 'meter_H2O'
        leg2 = leg1.to(leg2_punit)
    elif hydraulic_param == 'velocity':
        leg2_punit = 'fps' if to_unit in us_flow_units else 'mps'
        leg2 = leg1.to(leg2_punit)

    return leg2.magnitude
    # return leg2


class RaggedArrayList(object):
    def stack_ragged(self, array_list, axis=1):
        lengths = [arr.shape[axis] for arr in array_list]
        idx = np.cumsum(lengths[:-1])
        stacked = np.concatenate(array_list, axis)
        return stacked, idx, lengths

    def __init__(self, array_list, axis=1) -> None:
        self.axis = axis
        if array_list:
            self._stacked_array, self._indices, self._lengths = self.stack_ragged(array_list, axis=self.axis)
        else:
            self._stacked_array = None
            self._indices = None
            self._lengths = []

    def split(self):
        return np.split(self._stacked_array, self._indices, axis=1) if len(self._indices) > 0 else [self._stacked_array]

    def __len__(self):
        if self._lengths:
            return self._indices[-1] + self._lengths[-1]
        else:
            return 0

    def __getitem__(self, index):
        assert index < len(self._lengths)
        cur_length = self._lengths[index]

        if index < 0:
            index = len(self._lengths) + index

        if index < len(self._lengths) - 1:
            next_idx = self._indices[index]
            if self.axis == 0:
                return self._stacked_array[next_idx - cur_length: next_idx]
            else:
                return self._stacked_array[:, next_idx - cur_length: next_idx]
        else:
            if self.axis == 0:
                return self._stacked_array[-cur_length:]
            else:
                return self._stacked_array[:, -cur_length:]

    def __setitem__(self, index, value):
        assert index < len(self._lengths)
        cur_length = self._lengths[index]

        if index < 0:
            index = len(self._lengths) + index

        if index < len(self._lengths) - 1:
            next_idx = self._indices[index]

            if self.axis == 0:
                post_segment = self._stacked_array[next_idx:]
                prev_segment = self._stacked_array[:next_idx - cur_length]
            else:
                post_segment = self._stacked_array[:, next_idx:]
                prev_segment = self._stacked_array[:, :next_idx - cur_length]
            self._stacked_array = np.concatenate([prev_segment, value, post_segment], axis=self.axis)

            self._lengths[index] = value.shape[self.axis]
            self._indices = np.cumsum(self._lengths[:-1])
        else:
            self.pop()
            self.append(value)

    def append(self, new_array):
        if self._lengths:
            assert self._stacked_array.shape[self.axis - 1] == new_array.shape[self.axis - 1]
            new_length = new_array.shape[self.axis]
            self._indices = np.append(self._indices, self._indices[-1] + self._lengths[-1])
            self._lengths.append(new_length)
            self._stacked_array = np.concatenate([self._stacked_array, new_array],
                                                 axis=self.axis)  # np.concatenate(, new_array)
        else:
            self._stacked_array, self._indices, self._lengths = self.stack_ragged([new_array], axis=self.axis)

    def pop(self):
        out_array = None
        if self._lengths:
            print(f'self._lengths = {self._lengths}')
            if len(self._lengths) == 1:
                out_array = self._stacked_array
                self._stacked_array = None
            else:
                if self.axis == 0:
                    out_array = self._stacked_array[-self._lengths[-1]:]
                    self._stacked_array = self._stacked_array[:self._indices[-1]]
                else:
                    out_array = self._stacked_array[:, -self._lengths[-1]:]
                    self._stacked_array = self._stacked_array[:, :self._indices[-1]]
            self._indices = self._indices[:-1]
            self._lengths.pop()

        return out_array


class RaggedArrayDict(RaggedArrayList):
    def __init__(self, array_dict, axis=1) -> None:
        if array_dict:
            self._keys = list(array_dict.keys())
            array_list = list(array_dict.values())
            super().__init__(array_list, axis)
        else:
            self._keys = []
            super().__init__(None, axis)

    @staticmethod
    def from_keylen_and_stackedarray(keylen_dict, stacked_array, axis=1):
        lengths = list(keylen_dict.values())
        indices = np.cumsum(lengths[:-1])
        ragged_tokens = np.split(stacked_array, indices, axis=axis) if len(indices) > 0 else [stacked_array]
        feed_dict = {k: ragged_tokens[i] for i, k in enumerate(keylen_dict)}
        return RaggedArrayDict(feed_dict, axis=axis)

    @staticmethod
    def from_keylen_and_daskstackedarray(keylen_dict, stacked_array, axis=1):
        def da_split(stacked_array, indices, axis):
            splitted_arrays = []
            start_index = 0
            for ind in indices:
                a = stacked_array[start_index:ind]
                splitted_arrays.append(a)
                start_index = ind
            a = stacked_array[start_index:]
            splitted_arrays.append(a)
            return splitted_arrays

        lengths = list(keylen_dict.values())
        indices = np.cumsum(lengths[:-1])
        ragged_tokens = da_split(stacked_array, indices, axis=axis) if len(indices) > 0 else [stacked_array]
        feed_dict = {k: ragged_tokens[i] for i, k in enumerate(keylen_dict)}
        return RaggedArrayDict(feed_dict, axis=axis)

    @staticmethod
    def from_RaggedArrayDict(keys, stacked_array, indices, lengths, axis=1):
        x = RaggedArrayDict(None, axis)
        x._keys = list(keys)
        x._stacked_array = np.array(stacked_array, dtype=stacked_array.dtype)
        x._indices = np.array(indices, dtype=indices.dtype)
        x._lengths = list(lengths)
        return x

    def split(self):
        tokens = super().split()
        return tokens, self._keys

    def get_chunk(self, from_id, to_id):
        return self.from_RaggedArrayDict(self._keys, self._stacked_array[from_id:to_id], self._indices, self._lengths,
                                         axis=self.axis)

    def __getitem__(self, key):
        assert isinstance(key, str)
        if key in self._keys:
            return super().__getitem__(self._keys.index(key))
        else:
            return None

    def pop(self):
        out_array = super().pop()
        self._keys = self._keys[:-1]
        return out_array

    def __setitem__(self, key, value):
        if key not in self._keys:
            self._keys.append(key)
            super().append(value)
        else:
            index = self._keys.index(key)
            super().__setitem__(index, value)


FlowUnits = {0: "CFS",  # cubic feet / sec
             1: "GPM",  # gallons / min
             2: "AFD",  # acre-feet / day
             3: "MGD",  # million gallon / day
             4: "IMGD",  # Imperial MGD
             5: "LPS",  # liters / sec
             6: "LPM",  # liters / min
             7: "MLD",  # megaliters / day
             8: "CMH",  # cubic meters / hr
             9: "CMD"}  # cubic meters / day)


# -------------------------------------
#
# Executor part
#
# -------------------------------------

class ParamEnum(str, Enum):
    RANDOM_TOKEN = 'token'
    JUNC_DEMAND = 'junc_demand'
    JUNC_ELEVATION = 'junc_elevation'
    PUMP_STATUS = 'pump_status'
    PUMP_SPEED = 'pump_speed'
    PUMP_LENGTH = 'pump_speed'
    TANK_LEVEL = 'tank_level'
    TANK_ELEVATION = 'tank_elevation'
    TANK_DIAMETER = 'tank_diameter'
    VALVE_SETTING = 'valve_setting'
    VALVE_STATUS = 'valve_status'
    VALVE_DIAMETER = 'valve_diameter'
    PIPE_ROUGHNESS = 'pipe_roughness'
    PIPE_DIAMETER = 'pipe_diameter'
    PIPE_LENGTH = 'pipe_length'
    PIPE_MINORLOSS = 'pipe_minor_loss'
    RESERVOIR_TOTALHEAD = 'reservoir_totalhead'


class WDNExecutor(object):

    def __init__(self, featlen_dict, config, valve_type_dict, args, wn=None):
        self.sort_node_name = False
        self.min_valve_setting = 1e-4
        self.ordered = False
        wn_to_copy = wn
        self.custom_base_index = 100

        self.featlen_dict = deepcopy(featlen_dict)
        self.config = config
        # self.valve_type_dict        = valve_type_dict
        self.ele_std = args.ele_std
        self.ele_kmean_init = args.ele_kmean_init
        self.update_elevation_method = args.update_elevation_method

        self.expected_attr = args.att.strip().split(',')
        self.pressure_upperbound = args.pressure_upperbound
        self.pressure_lowerbound = args.pressure_lowerbound
        self.flowrate_threshold = args.flowrate_threshold
        self.init_valve_state = args.init_valve_state
        self.init_pipe_state = args.init_pipe_state
        self.accept_warning_code = args.accept_warning_code

        self.min_diameter = self.config.getfloat('pipe', 'diameter_lo')
        wn_inp_path = self.config.get('general', 'wn_inp_path')

        self.skip_nodes = self.config.get('general', 'skip_nodes').strip().split(',') if self.config.has_option(
            'general', 'skip_nodes') else []
        self.skip_links = self.config.get('general', 'skip_links').strip().split(',') if self.config.has_option(
            'general', 'skip_links') else []

        ###################################
        self.gen_demand = args.gen_demand
        self.gen_elevation = args.gen_elevation
        self.gen_diameter = args.gen_diameter
        self.gen_pipe_roughness = args.gen_roughness  # False#True
        self.gen_valve_init_status = args.gen_valve_init_status  # False#True
        self.gen_valve_setting = args.gen_valve_setting

        self.gen_pump_init_status = args.gen_pump_init_status
        self.gen_pump_speed = args.gen_pump_speed
        self.gen_tank_level = args.gen_tank_level
        self.gen_res_total_head = args.gen_res_total_head

        self.gen_tank_elevation = args.gen_tank_elevation
        self.gen_tank_diameter = args.gen_tank_diameter
        self.gen_pipe_length = args.gen_length
        self.gen_pipe_minorloss = args.gen_minorloss
        self.gen_pump_length = args.gen_pump_length
        self.gen_valve_diameter = args.gen_valve_diameter

        self.replace_nonzero_basedmd = args.replace_nonzero_basedmd
        self.update_totalhead_method = args.update_totalhead_method
        self.mean_cv_threshold = args.mean_cv_threshold  # 5. #10. #args.mean_group_var_threshold #10.0
        self.neighbor_std_threshold = args.neighbor_std_threshold
        self.allow_error = args.allow_error
        self.debug = args.debug

        self.remove_control = args.remove_control
        self.remove_rule = args.remove_rule
        self.remove_pattern = args.remove_pattern
        self.convert_results_by_flow_unit = args.convert_results_by_flow_unit
        ##################################
        if wn_to_copy:
            self.wn = wn_to_copy
        else:
            self.wn = Network(wn_inp_path)

        self.flow_unit = wutils.FlowUnits(self.wn.ep.ENgetflowunits())

        if args.skip_resevoir_result:
            self.skip_nodes.extend(self.wn.reservoirs.uid.to_list())

        if self.remove_pattern:
            patterns = self.wn.patterns
            if len(patterns) > 0:
                for p in patterns:
                    uid = p.uid
                    ENdeletepattern(self.wn, uid)

        if self.remove_rule:
            ENdeleteallrules(self.wn)

        if self.remove_control:
            ENdeleteallcontrols(self.wn)

        wntr_wn = wntr.network.WaterNetworkModel(wn_inp_path)
        self.wn_g = wntr_wn.get_graph().to_undirected()
        self.ori_diameters = [p.diameter for p in self.wn.pipes]
        self.ori_elevations = [p.elevation for p in self.wn.junctions]

        # Headloss formula enum: 0-HW, 1-DW, 2-CM
        self.head_loss_type = wntr_wn.options.hydraulic.headloss  # EN_HEADLOSSFORM = 7

        patterns = self.wn.patterns
        for i, _ in enumerate(self.wn.junctions):
            if not str(self.custom_base_index + i) in patterns:
                self.wn.add_pattern(str(self.custom_base_index + i), values=[0])

        self.custom_res_pattern_base_index = self.custom_base_index + len(self.wn.junctions)
        for i, _ in enumerate(self.wn.reservoirs):
            if not str(self.custom_res_pattern_base_index + i) in patterns:
                self.wn.add_pattern(str(self.custom_res_pattern_base_index + i), values=[0])

        # curves = self.wn.curves
        # self.custom_pump_curve_base_index = self.custom_res_pattern_base_index + len(self.wn.reservoirs)
        # for i,pump  in enumerate(self.wn.pumps):

        #     if not str(self.custom_pump_curve_base_index + i) in curves:
        #         self.wn.add_curve(str(self.custom_pump_curve_base_index + i) ,values=[[0,0]])
        #         c_index= self.wn.ep.ENgetcurveindex(str( self.custom_res_pattern_base_index + i))
        #         self.wn.ep.ENsetheadcurveindex(pump.index,c_index )

    def filter_skip_elements(self, df, skip_list):
        mask = df.index.isin(skip_list)
        ret = df.loc[np.invert(mask)]
        return ret

    def split_token_to_features(self, t, featlen_dict, axis=1):
        features = []
        start = 0
        for length in featlen_dict.values():
            end = start + length
            if axis == 0:
                features.append(t[start:end] if length > 0 else None)
            else:
                features.append(t[:, start:end] if length > 0 else None)
            start += length
        return features

    def epynet_simulate2(self,
                         tokens,
                         scene_id):
        """EN_DURATION 0 Simulation duration
        EN_HYDSTEP 1 Hydraulic time step
        EN_QUALSTEP 2 Water quality time step
        EN_PATTERNSTEP 3 Time pattern time step
        EN_PATTERNSTART 4 Time pattern start time
        EN_REPORTSTEP 5 Reporting time step
        EN_REPORTSTART 6 Report starting time
        EN_RULESTEP 7 Time step for evaluating rule-based controls
        EN_STATISTIC 8 Type of time series post-processing to use:
                                    EN_NONE (0) = none
                                    EN_AVERAGE (1) = averaged
                                    EN_MINIMUM (2) = minimums
                                    EN_MAXIMUM (3) = maximums
                                    EN_RANGE (4) = ranges

        :param _type_ tokens: _description_
        :param _type_ scene_id: _description_
        """
        self.wn.reset()

        ragged_tokens = RaggedArrayDict.from_keylen_and_stackedarray(self.featlen_dict, tokens, axis=0)

        junc_demands = ragged_tokens[ParamEnum.JUNC_DEMAND]
        junc_elevations = ragged_tokens[ParamEnum.JUNC_ELEVATION]
        pump_statuses = ragged_tokens[ParamEnum.PUMP_STATUS]
        pump_speed = ragged_tokens[ParamEnum.PUMP_SPEED]
        pump_lengths = ragged_tokens[ParamEnum.PUMP_LENGTH]
        pipe_roughness = ragged_tokens[ParamEnum.PIPE_ROUGHNESS]
        pipe_lengths = ragged_tokens[ParamEnum.PIPE_LENGTH]
        pipe_minorlosses = ragged_tokens[ParamEnum.PIPE_MINORLOSS]
        pipe_diameters = ragged_tokens[ParamEnum.PIPE_DIAMETER]
        tank_elevations = ragged_tokens[ParamEnum.TANK_ELEVATION]
        tank_diameters = ragged_tokens[ParamEnum.TANK_DIAMETER]
        tank_levels = ragged_tokens[ParamEnum.TANK_LEVEL]
        valve_statuses = ragged_tokens[ParamEnum.VALVE_STATUS]
        valve_settings = ragged_tokens[ParamEnum.VALVE_SETTING]
        valve_diameters = ragged_tokens[ParamEnum.VALVE_DIAMETER]
        res_heads = ragged_tokens[ParamEnum.RESERVOIR_TOTALHEAD]

        self.wn.ep.ENsettimeparam(epanet2.EN_DURATION, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_QUALSTEP, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_PATTERNSTEP, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_PATTERNSTART, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_REPORTSTEP, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_REPORTSTART, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_RULESTEP, 1)

        support_node_attr_keys = ['demand', 'head', 'pressure']
        support_link_attr_keys = ['velocity', 'flow']  # 'flowrate','status',

        for i, junc in enumerate(self.wn.junctions):
            if self.gen_demand:
                # this only affects if we have zero/one demand category
                if not self.replace_nonzero_basedmd or (junc.basedemand != 0 and self.replace_nonzero_basedmd):
                    junc.basedemand = 1.0

                junc.pattern = str(self.custom_base_index + i)
                junc.pattern.values = [junc_demands[i]]
                # In EPANET <=2.2, we have no way to delete the demand category if there are more than 1 exists...
                # Thus, we copy the base_demand and pattern into each demand category
                ENsetdemandpatterntoallcategories(self.wn, junc.index, junc.basedemand, junc.pattern.index)

            # static features
            if self.gen_elevation:
                junc.elevation = junc_elevations[i]

        for i, pump in enumerate(self.wn.pumps):
            # if self.remove_pattern:
            #     pump.curve.values = np.zeros_like(pump.curve.values).tolist()

            if self.gen_pump_init_status:
                pump.initstatus = int(pump_statuses[i])
                # print(f'pump {pump.uid} expected status = {bool(pump_statuses[i])} | actual status = {pump.initstatus}')
                # set_object_value_wo_ierror(pump,epanet2.EN_INITSTATUS,  bool(pump_statuses[i]) )

            if self.gen_pump_speed:
                pump.speed = pump_speed[i]

            if self.gen_pump_length:
                set_object_value_wo_ierror(pump, epanet2.EN_LENGTH, pump_lengths[i])

        for i, tank in enumerate(self.wn.tanks):
            if self.gen_tank_level:
                tank_level = tank_levels[i]  # tank.minlevel + tank_levels[i] * (tank.maxlevel - tank.minlevel)
                set_object_value_wo_ierror(obj=tank, code=epanet2.EN_TANKLEVEL, value=tank_level)
            if self.gen_tank_elevation:
                set_object_value_wo_ierror(obj=tank, code=epanet2.EN_ELEVATION, value=tank_elevations[i])
            if self.gen_tank_diameter:
                set_object_value_wo_ierror(obj=tank, code=epanet2.EN_TANKDIAM, value=tank_diameters[i])

        debug_flag = False

        tmp_graph = get_networkx_graph(wn=self.wn, include_reservoir=True, graph_type='undirected')
        for i, valve in enumerate(self.wn.valves):
            if self.init_valve_state is not None:
                valve.initstatus = int(self.init_valve_state)

            if self.gen_valve_init_status:
                if self.init_valve_state is not None and self.debug and not debug_flag:
                    print(f'WARN! init value state is overrided')
                    debug_flag = True
                if not bool(valve_statuses[i]):
                    tmp_graph.remove_edge(valve.from_node.uid, valve.to_node.uid)
                    if nx.is_connected(tmp_graph):
                        valve.initstatus = int(valve_statuses[i])
                    else:
                        if self.debug:
                            print(f'WARN! Unable to off valve {valve.uid} due to the graph disconnection')
                        tmp_graph.add_edge(valve.from_node.uid, valve.to_node.uid)
                        valve.initstatus = True
                else:
                    valve.initstatus = int(valve_statuses[i])

            if self.gen_valve_setting:  # open
                # scaled_valve_setting  =  self.valve_type_dict[valve.valve_type][0] +  valve_settings[i] * (self.valve_type_dict[valve.valve_type][1] - self.valve_type_dict[valve.valve_type][0])
                # valve.setting = max(self.min_valve_setting, scaled_valve_setting)
                if valve_settings[i] > 0:  # 0 mean unused
                    # valve.setting = valve_settings[i]
                    set_object_value_wo_ierror(obj=valve, code=epanet2.EN_INITSETTING, value=valve_settings[i])

            if self.gen_valve_diameter:
                set_object_value_wo_ierror(obj=valve, code=epanet2.EN_DIAMETER,
                                           value=np.maximum(valve_diameters[i], 1e-12))

        pipe_uids = []
        for i, pipe in enumerate(self.wn.pipes):
            # if  self.init_pipe_state is not None and not pipe.check_valve:
            if self.init_pipe_state is not None and not pipe.check_valve:
                # pipe.initstatus = self.init_pipe_state
                set_object_value_wo_ierror(obj=pipe, code=epanet2.EN_INITSTATUS, value=int(self.init_pipe_state))
                pipe_uids.append(pipe.uid)
            if self.gen_pipe_roughness:
                pipe.roughness = pipe_roughness[i]

            if self.gen_pipe_length:
                set_object_value_wo_ierror(obj=pipe, code=epanet2.EN_LENGTH,
                                           value=np.maximum(pipe_lengths[i], 1e-12))
            if self.gen_pipe_minorloss:
                set_object_value_wo_ierror(obj=pipe, code=epanet2.EN_MINORLOSS,
                                           value=np.maximum(pipe_minorlosses[i], 1e-12))

            if self.gen_diameter:
                # pipe.diameter =  pipe_diameters[i]

                set_object_value_wo_ierror(obj=pipe, code=epanet2.EN_DIAMETER,
                                           value=np.maximum(pipe_diameters[i], 1e-12))

                # new_diameter = self.ori_diameters[i] +  pipe_diameters[i]
                # if new_diameter > 10:
                #    pipe.diameter = new_diameter

        # total head should be the last element
        if self.gen_res_total_head:
            for i, res in enumerate(self.wn.reservoirs):

                res.set_object_value(epanet2.EN_ELEVATION, 1.0)
                if self.update_totalhead_method is None:
                    tmp = res_heads[i]
                elif self.update_totalhead_method == 'add_max_elevation':
                    elevations = [n.elevation for n in
                                  self.wn.junctions]  # [n.elevation for n in self.wn.nodes]# [self.wn.nodes[n].elevation for n in neighbors]
                    tmp = max(elevations) + res_heads[i]

                p_index = self.wn.ep.ENgetpatternindex(str(self.custom_res_pattern_base_index + i))
                self.wn.ep.ENsetpattern(p_index, [tmp])
                res.set_object_value(epanet2.EN_PATTERN, p_index)

        sim_results = {}

        prefix_name = 'tmp_' + str(scene_id)

        for file in glob.glob(f"{prefix_name}.*"):
            os.remove(file)

        # RUN SIM
        def ENrunH(ep):
            """Runs a single period hydraulic analysis,
            retrieving the current simulation clock time t"""
            ierr = ep._lib.EN_runH(ep.ph, epanet2.ctypes.byref(ep._current_simulation_time))
            # if ierr>=100:
            #    raise epanet2.ENtoolkitError(ep, ierr)
            return ierr

        def solve_return_error(wn, simtime=0):
            if wn.solved and wn.solved_for_simtime == simtime:
                return
            # wn.reset()
            wn.ep.ENsettimeparam(4, simtime)
            wn.ep.ENopenH()
            wn.ep.ENinitH(0)
            code = ENrunH(wn.ep)
            assert code is not None
            wn.ep.ENcloseH()
            wn.solved = True
            wn.solved_for_simtime = simtime
            return code

        code = solve_return_error(self.wn)

        # out_feature_lens
        # skipped_uids =  self.filter_skip_elements(self.wn.nodes.uid,self.skip_nodes)
        # print(f'before_skip-pressure_results shape  = {self.wn.nodes.pressure.shape}')

        if self.skip_nodes is not None:
            pressure_df = self.wn.nodes.pressure
            pressure_results = self.filter_skip_elements(pressure_df, self.skip_nodes).values
            pressure_results = np.reshape(pressure_results, [1, -1])
        else:
            pressure_results = self.wn.nodes.pressure.values
            pressure_results = np.reshape(pressure_results, [1, -1])

        if self.convert_results_by_flow_unit is not None:
            from_unit = FlowUnits[self.wn.ep.ENgetflowunits()]
            to_unit = self.convert_results_by_flow_unit

            if from_unit != to_unit:
                pressure_results = ENconvert(from_unit=from_unit,
                                             to_unit=to_unit,
                                             hydraulic_param='pressure',
                                             values=pressure_results)

        is_nan = np.isnan(pressure_results).any()
        if is_nan and self.debug:
            print('is nan')
        error = is_nan

        if code > 0:
            if self.accept_warning_code:
                error = error or code > 6
            else:
                error = error or code > 0
            if self.debug:
                print(f'Detected abnormal code- code {code}')

        if self.pressure_lowerbound is not None:
            negative_error = any(pressure_results.min(axis=1) < self.pressure_lowerbound)
            if negative_error and self.debug:
                print(f'negative_error')  # at { skipped_uids[np.argmin(pressure_results,axis=1)].to_list()  }
            error = error or negative_error

        if self.pressure_upperbound is not None:
            extreme_error = any(pressure_results.max(axis=1) > self.pressure_upperbound)
            if extreme_error and self.debug:
                print(f'extreme_error ')  # { skipped_uids[np.argmax(pressure_results,axis=1)].to_list()  }
            error = error or extreme_error

        if self.neighbor_std_threshold is not None:

            hop = 2
            uids = np.array(list(self.wn.nodes.uid))
            tmp = self.wn.nodes.pressure.values

            def get_neighbor_std(node):
                test_neighbor_uids = list(
                    set(nx.single_source_shortest_path_length(self.wn_g, node.uid, cutoff=hop).keys()).difference(
                        [node.uid]))
                neighbor_ids = np.where(np.isin(uids, test_neighbor_uids))[0]
                neighbor_values = np.take(tmp, neighbor_ids)
                neighbor_variance = np.std(neighbor_values)
                return neighbor_variance

            # neighbor_variances = []
            # for _,node in enumerate(self.wn.nodes) :
            #    neighbor_variance = get_neighbor_variance(node)
            #    neighbor_variances.append(neighbor_variance)
            neighbor_stds = list(map(get_neighbor_std, self.wn.nodes))

            neighbor_stds = np.array(neighbor_stds)
            mean_neighbor_stds = np.mean(neighbor_stds)
            error = error or mean_neighbor_stds > self.neighbor_std_threshold

            print(f' neighbor std = {mean_neighbor_stds}')
            if mean_neighbor_stds > self.neighbor_std_threshold and self.debug:
                print(f'high neighbor std = {mean_neighbor_stds}')

        if self.mean_cv_threshold is not None:
            tmp = pressure_results  # self.wn.nodes.pressure.values
            cv = float(np.var(tmp, axis=1) / np.mean(tmp, axis=1))
            error = error or cv > self.mean_cv_threshold
            if cv > self.mean_cv_threshold and self.debug:
                print(f'too high cv = {cv}')

        # flowrate_results = results.link['flowrate']
        # error = error | any(flowrate_results.min(axis=1) < self.flowrate_threshold)
        sim_result_indices = None
        for attr in self.expected_attr:
            if attr in support_node_attr_keys:
                sim_result = getattr(self.wn.nodes, attr) if hasattr(self.wn.nodes, attr) else getattr(
                    self.wn.junctions, attr)
                if self.skip_nodes is not None:
                    sim_result = self.filter_skip_elements(sim_result, self.skip_nodes)

            elif attr in support_link_attr_keys:
                sim_result = getattr(self.wn.links, attr)
                if self.skip_links is not None:
                    sim_result = self.filter_skip_elements(sim_result, self.skip_links)

            if self.sort_node_name:
                sim_result = sim_result.sort_index(axis=1)

            sim_result_indices = sim_result.index.tolist()
            sim_result = np.reshape(sim_result.to_numpy(), [1, -1])
            if self.convert_results_by_flow_unit is not None:
                from_unit = FlowUnits[self.wn.ep.ENgetflowunits()]
                to_unit = self.convert_results_by_flow_unit

                if from_unit != to_unit:
                    if self.debug:
                        print(f'detected unit convension from {from_unit} to {to_unit}...')
                    sim_result = ENconvert(from_unit=from_unit,
                                           to_unit=to_unit,
                                           hydraulic_param=attr,
                                           values=sim_result)

            sim_results[attr] = sim_result

        # debug
        if sim_results['pressure'] is None or len(sim_results['pressure']) <= 0:
            print('weriddd')
        return sim_results, error, sim_result_indices

    def update_batch_dict(self, batch_dict, single_dict):
        for key, value in single_dict.items():
            if key not in batch_dict:
                batch_dict[key] = value
            else:
                batch_dict[key] = np.concatenate([batch_dict[key], value], axis=0)
        return batch_dict

    def check_order(self, l1, l2):
        if len(l1) != len(l2):
            return False
        else:
            for i in range(len(l1)):
                if l1[i] != l2[i]:
                    return False
            return True

    def simulate(self, batch_tokens, scence_ids):
        batch_results = {}
        batch_size = batch_tokens.shape[0]
        do_saved = False
        stored_ordered_name_list = None
        for id in range(batch_size):
            tokens = batch_tokens[id]
            single_result, error, ordered_name_list = self.epynet_simulate2(tokens, scence_ids[
                id])  # self.epynet_simulate2(tokens,scence_ids[id])

            if stored_ordered_name_list is not None:
                assert self.check_order(ordered_name_list, stored_ordered_name_list)
            stored_ordered_name_list = ordered_name_list

            if not error or self.allow_error:
                batch_results = self.update_batch_dict(batch_results, single_result)

                # if not do_saved:
                #    do_saved=True
                #    self.wn.ep.ENsaveinpfile(f'{scence_ids[id]}.inp')
        return batch_results, stored_ordered_name_list


@ray.remote(num_cpus=0)
class WDNRayExecutor(WDNExecutor):
    pass


# -------------------------------------
#
# Executor part
#
# -------------------------------------

program_start = time()
parser = argparse.ArgumentParser()
# main config
parser.add_argument('--config',
                    default=r"D:\GithubRepository\gnn-pressure-estimation\configs\v7.1\ctown_7v1__EPYNET_config.ini",
                    type=str, help='configuration path')

# initial valve/pipe states
parser.add_argument('--init_valve_state', default=1, type=int,
                    help='init status = CheckedValve(3) Active(2) Open(1) Closed(0) KeepInitStatus(None)')
parser.add_argument('--init_pipe_state', default=None, type=int,
                    help='init status = CheckedValve(3) Active(2) Open(1) Closed(0) KeepInitStatus(None)')

# removal flags
parser.add_argument('--remove_pattern', default=True, type=bool,
                    help='flag indicates to remove any pattern in input file')
parser.add_argument('--remove_control', default=False, type=bool,
                    help='flag indicates to remove any control in input file')
parser.add_argument('--remove_rule', default=False, type=bool,
                    help='flag indicates to remove any rule in input file! Note EPANET authors confuse control and rule')

# demands settings
parser.add_argument('--gen_demand', default=True, type=bool,
                    help='If true, replacing nonzero base demand to 1.0 | ELSE, replacing ALL base demands to 1.0. Default is False')
parser.add_argument('--replace_nonzero_basedmd', default=False, type=bool,
                    help='If true, replacing nonzero base demand to 1.0 | ELSE, replacing ALL base demands to 1.0. Default is False')
parser.add_argument('--update_demand_json', default=None, type=str,
                    help='JSON string. Overriding demand values (Note: demand = base_dmd * multipliers) according to the JSON file. Set None if unsed. Default is None')

# elevation settings
parser.add_argument('--gen_elevation', default=False, type=bool, help='flag indicates to change the nodal elevation')
parser.add_argument('--ele_kmean_init', default='k-means++', type=str,
                    help='Initialization of K-mean for elevation cluster = k-means++ | random')
parser.add_argument('--update_elevation_method', default='ran_cluster', type=str,
                    help='update elevation if gen_elevation is True, options: ran_cluster | ran_local | ran | idw_dist | idw_ran')
parser.add_argument('--ele_std', default=1., type=float, help='the std apart from the elevation of local region')
parser.add_argument('--update_elevation_json', default=None, type=str,
                    help='JSON string. Overriding elevation values according to the JSON file. Set None if unsed. Default is None')

# pipe settings
parser.add_argument('--gen_roughness', default=True, type=bool, help='flag indicates to change the pipe roughness')
parser.add_argument('--gen_diameter', default=False, type=bool, help='flag indicates to change the pipe diameter')
parser.add_argument('--dia_kmean_init', default='k-means++', type=str,
                    help='(UNSED)Initialization of K-mean for diameter cluster = k-means++ | random')
parser.add_argument('--gen_length', default=False, type=bool, help='flag indicates to change the pipe roughness')
parser.add_argument('--gen_minorloss', default=False, type=bool, help='flag indicates to change the pipe diameter')
parser.add_argument('--update_pipe_roughness_json', default=None, type=str,
                    help='JSON string. Overriding pipe roughness values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_pipe_diameter_json', default=None, type=str,
                    help='JSON string. Overriding pipe_diameter values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_pipe_length_json', default=None, type=str,
                    help='JSON string. Overriding pipe length values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_pipe_minorloss_json', default=None, type=str,
                    help='JSON string. Overriding pipe minorloss values according to the JSON file. Set None if unsed. Default is None')

# valve settings
parser.add_argument('--gen_valve_init_status', default=True, type=bool,
                    help='flag indicates to change the valve init status')
parser.add_argument('--gen_valve_setting', default=True, type=bool, help='flag indicates to change the valve settings')
parser.add_argument('--gen_valve_diameter', default=False, type=bool,
                    help='flag indicates to change the valve diameter')
parser.add_argument('--update_valve_init_status_json', default=None, type=str,
                    help='JSON string. Overriding valve status values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_valve_setting_json', default=None, type=str,
                    help='JSON string. Overriding valve setting values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_valve_diameter_json', default=None, type=str,
                    help='JSON string. Overriding valve diameter values according to the JSON file. Set None if unsed. Default is None')

# pump settings
parser.add_argument('--gen_pump_init_status', default=False, type=bool,
                    help='flag indicates to change the pump init status')
parser.add_argument('--gen_pump_speed', default=True, type=bool, help='flag indicates to change the pump speed')
parser.add_argument('--gen_pump_length', default=False, type=bool, help='flag indicates to change the pump length')
parser.add_argument('--update_pump_init_status_json', default=None, type=str,
                    help='JSON string. Overriding pump init status values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_pump_speed_json', default=None, type=str,
                    help='JSON string. Overriding pump speed values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_pump_length_json', default=None, type=str,
                    help='JSON string. Overriding pump length values according to the JSON file. Set None if unsed. Default is None')

# tank settings
parser.add_argument('--gen_tank_level', default=True, type=bool, help='flag indicates to change the tank level')
parser.add_argument('--gen_tank_elevation', default=False, type=bool,
                    help='flag indicates to change the tank elevation')
parser.add_argument('--gen_tank_diameter', default=False, type=bool, help='flag indicates to change the tank diameter')
parser.add_argument('--update_tank_level_json', default=None, type=str,
                    help='JSON string. Overriding tank level values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_tank_elevation_json', default=None, type=str,
                    help='JSON string. Overriding tank elevation values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_tank_diameter_json', default=None, type=str,
                    help='JSON string. Overriding tank diameter values according to the JSON file. Set None if unsed. Default is None')

# reservoir settings
parser.add_argument('--gen_res_total_head', default=True, type=bool,
                    help='flag indicates to change the total head of reservoir')
parser.add_argument('--skip_resevoir_result', default=True, type=bool,
                    help='flag indicates to skip the resevoirs result after simulation')
parser.add_argument('--update_totalhead_method', default=None, type=str,
                    help='update total head of reservor only if gen_res_total_head is True, options: add_max_elevation | None')
parser.add_argument('--update_res_total_head_json', default=None, type=str,
                    help='JSON string. Overriding reservoir totalHead values according to the JSON file. Set None if unsed. Default is None')

# settings
parser.add_argument('--debug', default=True, type=bool, help='flag allows to print some useful measurements')
parser.add_argument('--allow_error', default=False, type=bool,
                    help='flag allows to bypass error scenarios (useful for debug ), defaults to False')
parser.add_argument('--convert_results_by_flow_unit', default='CMH', type=str,
                    help='CMH Convert all results according to the SI flow units that includes LPS, LPM, MLD, CMH, CMD. Set None to keep original unit')
parser.add_argument('--change_dmd_by_junc_indices_path', default=None, type=str,
                    help='selected_sensitivity_by_cv_2023-02-13.pkl|Path to the indices of junctions used to change demand only. The one which is not in this list has the minimum value. Setting None if not used')  #

# conditions
parser.add_argument('--accept_warning_code', default=False, type=bool,
                    help='flag allows to accept warning codes (0 < code < 6)')
parser.add_argument('--pressure_lowerbound', default=None, type=float,
                    help='threshold value to filter small pressure values - 5mH2O - 7.1 psi. Set None if unused')
parser.add_argument('--pressure_upperbound', default=None, type=float,
                    help='threshold value to filter high pressure values - 100mH2O - 142.23 psi. Set None if unused')
parser.add_argument('--flowrate_threshold', default=None, type=float,
                    help='threshold value to filter valid flowarate values')
parser.add_argument('--mean_cv_threshold', default=None, type=float, help='5.0 threshold value to filter high cv')
parser.add_argument('--neighbor_std_threshold', default=None, type=float,
                    help='threshold value to filter high neighbor std')

# general info
parser.add_argument('--batch_size', default=5, type=int, help='batch size')
parser.add_argument('--executors', default=2, type=int, help='number of executors')
parser.add_argument('--att', default='pressure,head', type=str,
                    help='list of simulation attributes you want to extract. Supported attributes: demand, pressure, head, flow, velocity')
parser.add_argument('--train_ratio', default=0.6, type=float, help='the ratio of training scenarios and total')
parser.add_argument('--valid_ratio', default=0.2, type=float, help='the ratio of validation scenarios and total')
parser.add_argument('--is_single_thread', default=False, type=bool,
                    help='run the generation with only a single thread for debugging only. Defaults is False')

args = parser.parse_args([])

config = configparser.ConfigParser()
config.read(args.config)
config_keys = dict(config.items()).keys()

wn_inp_path = config.get('general', 'wn_inp_path')
storage_dir = config.get('general', 'storage_dir')

zarr_storage_dir = os.path.join(storage_dir, 'zarrays')
random_array_dir = os.path.basename(wn_inp_path)[:-4] + '_random_array_' + datetime.datetime.now().strftime(
    '%m_%d_%Y_%H_%M')  # get input name
random_array_dir = os.path.join(storage_dir, random_array_dir)
os.makedirs(storage_dir, exist_ok=True)
os.chdir(storage_dir)
shutil.rmtree(path=storage_dir, ignore_errors=True)
os.makedirs(zarr_storage_dir, exist_ok=False)

saved_path = storage_dir
num_scenarios = config.getint('general', 'num_scenarios')  # 1000
backup_num_scenarios = num_scenarios * 10
batch_size = args.batch_size
num_executors = args.executors
expected_attributes = args.att.strip().split(',')  # ['pressure','head','flowrate','velocity']
train_ratio = args.train_ratio
valid_ratio = args.valid_ratio
num_batches = backup_num_scenarios // batch_size
num_chunks = backup_num_scenarios // batch_size
support_node_attr_keys = ['head', 'pressure', 'demand']
support_link_attr_keys = ['flow', 'velocity']  # 'flowrate',
support_keys = list(set(support_node_attr_keys).union(support_link_attr_keys))
for a in expected_attributes:
    if a not in support_keys:
        raise AttributeError(f'{a} is not found or not supported!')
###################################################

print(wn_inp_path)
wn = Network(wn_inp_path)
# wn_g = wntr.network.WaterNetworkModel(wn_inp_path).get_graph()
skip_nodes = config.get('general', 'skip_nodes').strip().split(',') if config.has_option('general',
                                                                                         'skip_nodes') else None

valve_type_dict = {}
# for valve in wn.valves:
#     valve_type = str(valve.valve_type) 
#     if valve_type not in valve_type_dict:
#         valve_type_dict[valve_type] = np.array(config.get('valve',f'setting_{valve_type}').strip().split(','),dtype=float)


featlen_dict = dict()

if len(wn.junctions) > 0:
    if args.gen_demand:
        featlen_dict[ParamEnum.JUNC_DEMAND] = len(wn.junctions)

    if args.gen_elevation:
        featlen_dict[ParamEnum.JUNC_ELEVATION] = len(wn.junctions)

if len(wn.pipes) > 0:
    num_pipes = len(wn.pipes)
    if args.gen_roughness:
        featlen_dict[ParamEnum.PIPE_ROUGHNESS] = num_pipes
    if args.gen_diameter:
        featlen_dict[ParamEnum.PIPE_DIAMETER] = num_pipes
    if args.gen_length:
        featlen_dict[ParamEnum.PIPE_LENGTH] = num_pipes
    if args.gen_minorloss:
        featlen_dict[ParamEnum.PIPE_MINORLOSS] = num_pipes

if len(wn.pumps) > 0:
    num_pumps = len(wn.pumps)
    if args.gen_pump_init_status:
        featlen_dict[ParamEnum.PUMP_STATUS] = num_pumps
    if args.gen_pump_speed:
        featlen_dict[ParamEnum.PUMP_SPEED] = num_pumps
    if args.gen_pump_length:
        featlen_dict[ParamEnum.PUMP_LENGTH] = num_pumps

if len(wn.tanks) > 0:
    num_tanks = len(wn.tanks)
    if args.gen_tank_level:
        featlen_dict[ParamEnum.TANK_LEVEL] = num_tanks
    if args.gen_tank_elevation:
        featlen_dict[ParamEnum.TANK_ELEVATION] = num_tanks
    if args.gen_tank_diameter:
        featlen_dict[ParamEnum.TANK_DIAMETER] = num_tanks

if len(wn.valves) > 0:
    num_valves = len(wn.valves)
    if args.gen_valve_init_status:
        featlen_dict[ParamEnum.VALVE_STATUS] = num_valves
    if args.gen_valve_setting:
        featlen_dict[ParamEnum.VALVE_SETTING] = num_valves
    if args.gen_valve_diameter:
        featlen_dict[ParamEnum.VALVE_DIAMETER] = num_valves

if args.gen_res_total_head and len(wn.reservoirs) > 0:
    featlen_dict[ParamEnum.RESERVOIR_TOTALHEAD] = len(wn.reservoirs)

########################################################
last_results = []

print('Start simulation...')
print('saved_path = ', saved_path)
skip_nodes = skip_links = []
num_skip_nodes = num_skip_links = 0
if config.has_option('general', 'skip_nodes'):
    skip_nodes = config.get('general', 'skip_nodes').strip().split(',')

if args.skip_resevoir_result:
    skip_nodes.extend(wn.reservoirs.uid.to_list())

num_skip_nodes = len(skip_nodes)

print(f'skip nodes = {skip_nodes}')
print(f'#skip_nodes = {num_skip_nodes}')

if config.has_option('general', 'skip_links'):
    skip_links = config.get('general', 'skip_links').strip().split(',')
    num_skip_links = len(skip_links)
print(f'#skip_links = {num_skip_links}')

node_uids = wn.nodes.uid
num_result_nodes = len(node_uids.loc[~node_uids.isin(skip_nodes)]) if skip_nodes else len(node_uids)
print(
    f'exepected #result_nodes = {num_result_nodes} | Note that if attribute is \'demand\', #results_nodes should be #junctions')

link_uids = wn.links.uid
num_result_links = len(link_uids.loc[~link_uids.isin(skip_links)]) if skip_links else len(link_uids)
print(f'exepected #result_links = {num_result_links}')
###########################################################

# store = zarr.DirectoryStore(zarr_storage_dir)
store = zarr.DirectoryStore('test')
tg = RayTokenGenerator(store=store,
                       num_scenes=backup_num_scenarios,
                       featlen_dict=featlen_dict,
                       num_chunks=num_chunks)

# tg.init()
tg.sequential_update(args=args)
ragged_tokens = tg.load_computed_params()
root_group = zarr.open_group(store,
                             synchronizer=zarr.ThreadSynchronizer())

tmp_group = root_group.create_group('tmp', overwrite=True)
for att in expected_attributes:
    if att in support_node_attr_keys:
        if att == 'demand':
            uids = wn.junctions.uid
            num_junctions = len(uids.loc[~uids.isin(skip_nodes)]) if skip_nodes else len(uids)
            tmp_group.create(att, shape=[num_scenarios, num_junctions],
                             chunks=[batch_size, num_result_nodes],
                             overwrite=True)
        else:
            tmp_group.create(att, shape=[num_scenarios, num_result_nodes],
                             chunks=[batch_size, num_result_nodes],
                             overwrite=True)

    elif att in support_link_attr_keys:
        tmp_group.create(att, shape=[num_scenarios, num_result_links],
                         chunks=[batch_size, num_result_links],
                         overwrite=True)


def single_thread_executor(batch_size,
                           ragged_tokens,
                           new_featlen_dict,
                           config,
                           valve_type_dict,
                           args,
                           tmp_group,
                           num_batches,
                           ):
    token_ids = []
    scene_ids = []

    for batch_id in range(num_batches):
        start_id = batch_id * batch_size
        end_id = start_id + batch_size
        batch_ragged_tokens = ragged_tokens[start_id:end_id]
        token_ids.append(batch_ragged_tokens)
        scene_ids.append([start_id + x for x in range(batch_size)])

    sim_start = time()
    executor = WDNExecutor(
        featlen_dict=new_featlen_dict,
        config=config,
        valve_type_dict=valve_type_dict,
        args=args,
    )

    start_index = 0
    progressbar = tqdm(total=num_batches)

    ordered_names_dict = {}
    success_scenarios = 0
    while len(token_ids) > 0:
        catch_error = False
        try:
            result, ordered_name_list = executor.simulate(token_ids.pop(), scene_ids.pop())
        except Exception as e:
            print(e)
            catch_error = True

        if not catch_error:
            success_size = 0
            for key, value in result.items():
                if key not in ordered_names_dict:
                    ordered_names_dict[key] = ordered_name_list
                if start_index + value.shape[0] < tmp_group[key].shape[0]:
                    success_size = value.shape[0]
                    tmp_group[key][start_index:start_index + success_size] = value
                else:
                    success_size = tmp_group[key].shape[0] - start_index
                    tmp_group[key][start_index:start_index + success_size] = value[:success_size]

            del result
            start_index += success_size
            success_scenarios += success_size
        progressbar.update(1)

    progressbar.close()
    ray.shutdown()
    elapsed_time = time() - sim_start
    print(f'\nSimulation time: {elapsed_time} seconds')

    print(f'Success/Total: {success_scenarios}/{num_scenarios} scenes')
    return success_scenarios, ordered_names_dict


try:
    sim_start = time()

    '''
        #use it for debug
        success_scenarios, ordered_names_dict= single_thread_executor(
                                                    batch_size,
                                                    ragged_tokens,
                                                    featlen_dict,
                                                    config,
                                                    valve_type_dict,
                                                    args,
                                                    tmp_group,
                                                    num_batches)
    '''

    token_ids = []
    scene_ids = []

    for batch_id in range(num_batches):
        start_id = batch_id * batch_size
        end_id = start_id + batch_size
        batch_ragged_tokens = ragged_tokens[start_id:end_id]
        token_ids.append(ray.put(batch_ragged_tokens))
        scene_ids.append(ray.put([start_id + x for x in range(batch_size)]))
    new_featlen_dict = featlen_dict  # we don't create new features

    executors = [WDNRayExecutor.remote(
        featlen_dict=new_featlen_dict,
        config=config,
        valve_type_dict=valve_type_dict,
        args=args,
    ) for _ in range(num_executors)]

    start_index = 0
    progressbar = tqdm(total=num_batches)  # tqdm(total=num_batches,desc="batch" ,leave=False, colour='red')
    # successbar = tqdm(total= num_scenarios,desc="scase",leave=False, colour='green')
    result_worker_dict = {e.simulate.remote(token_ids.pop(), scene_ids.pop()): e for e in executors if scene_ids}
    done_ids, _ = ray.wait(list(result_worker_dict), num_returns=1)

    ordered_names_dict = {}
    success_scenarios = 0
    while done_ids and success_scenarios < num_scenarios:
        done_worker_id = done_ids[0]
        catch_error = False
        try:
            result, ordered_name_list = ray.get(done_worker_id)
        except RayError as e:
            print(f'WARNING! Ray error {e}')
            catch_error = True
        worker = result_worker_dict.pop(done_worker_id)
        if scene_ids:
            result_worker_dict[worker.simulate.remote(token_ids.pop(), scene_ids.pop())] = worker

        if not catch_error:
            success_size = 0
            # write_id = dataset[start_index:start_index + batch_size].write(result) #.result()
            for key, value in result.items():
                if key not in ordered_names_dict:
                    ordered_names_dict[key] = ordered_name_list

                # print(f'key = {key}, value shape = {value.shape}')
                if start_index + value.shape[0] < tmp_group[key].shape[0]:
                    success_size = value.shape[0]
                    tmp_group[key][start_index:start_index + success_size] = value
                else:
                    success_size = tmp_group[key].shape[0] - start_index
                    tmp_group[key][start_index:start_index + success_size] = value[:success_size]

            del result
            start_index += success_size
            success_scenarios += success_size
            # successbar.update(success_size)
        progressbar.update(1)
        done_ids, _ = ray.wait(list(result_worker_dict), num_returns=1)

    # successbar.close()
    progressbar.close()
    ray.shutdown()

    elapsed_time = time() - sim_start
    print(f'\nSimulation time: {elapsed_time} seconds')
    print(f'Process run on {num_batches} batches, total scenes: {backup_num_scenarios}')
    print(f'Success/Expected: {success_scenarios}/{num_scenarios} scenes')

    del root_group[ParamEnum.RANDOM_TOKEN]
    if success_scenarios > 0:
        for name in list(tmp_group.keys()):
            name_group = root_group.create_group(name, overwrite=True)
            if success_scenarios != num_scenarios:
                # reshape
                tmp_group[name].resize(success_scenarios, tmp_group[name].shape[-1])
                if success_scenarios < batch_size:
                    # TODO: rechunk to maximize the r/w speed
                    pass

        train_index = int(success_scenarios * train_ratio)
        valid_index = train_index + int(success_scenarios * valid_ratio)

        key_list = list(tmp_group.keys())

        config_dict = {sect: dict(config.items(sect)) for sect in config.sections()}
        if skip_nodes:
            config_dict['skip_nodes'] = skip_nodes
        if skip_links:
            config_dict['skip_links'] = skip_links

        root_group.attrs['config'] = config_dict
        root_group.attrs['args'] = vars(args)
        root_group.attrs['ordered_names_by_attr'] = ordered_names_dict

        for key in key_list:
            a = tmp_group[key]
            # print(f'\n{key}.info: {a.info}')
            train_a, valid_a, test_a = a[:train_index], a[train_index:valid_index], a[valid_index:]

            train_a_df = pd.DataFrame(train_a).astype(float)
            train_min = train_a.min()
            train_max = train_a.max()
            train_mean = train_a.mean()
            train_std = train_a.std()

            train_mean_feat_coef = train_a_df.corr().mean().mean()  # np.corrcoef(train_a.T).mean()
            train_mean_batch_coef = train_a_df.T.corr().mean().mean()  # np.corrcoef(train_a).mean()
            train_cv = (train_a.var(axis=-1) / train_a.mean(axis=-1)).mean()

            root_group[key].attrs['min'] = train_min
            root_group[key].attrs['max'] = train_max
            root_group[key].attrs['mean'] = train_mean
            root_group[key].attrs['std'] = train_std
            root_group[key].attrs['mcoef'] = train_mean_feat_coef
            root_group[key].attrs['bcoef'] = train_mean_batch_coef
            root_group[key].attrs['cv'] = train_cv

            print(f'##############################{key}###############################################')
            print(f'min         : {train_min}')
            print(f'max         : {train_max}')
            print(f'mean        : {train_mean}')
            print(f'std         : {train_std}')
            print(f'mean fcoef  : {train_mean_feat_coef}')
            print(f'mean bcoef  : {train_mean_batch_coef}')
            print(f'cv          : {train_cv}')

            key_train = os.path.join(key, 'train')
            root_group.empty_like(key_train, train_a, chunks=(batch_size, a.chunks[-1]))
            root_group[key_train][:] = train_a
            print(f'\n{key_train}.info: {root_group[key_train].info}')

            key_valid = os.path.join(key, 'valid')
            root_group.empty_like(key_valid, valid_a, chunks=(batch_size, a.chunks[-1]))
            root_group[key_valid][:] = valid_a
            print(f'\n{key_valid}.info: {root_group[key_valid].info}')

            key_test = os.path.join(key, 'test')
            root_group.empty_like(key_test, test_a, chunks=(batch_size, a.chunks[-1]))
            root_group[key_test][:] = test_a
            print(f'\n{key_test}.info: {root_group[key_test].info}')

            # del tmp_group[key]

        del root_group['tmp']

        elapsed_time = time() - program_start
        print(f'\nExecution time: {elapsed_time} seconds')

        store2 = zarr.ZipStore(saved_path + '.zip', mode='w')
        zarr.copy_store(store, store2, if_exists='replace')
        store2.close()
        print(root_group.tree())

        if args.debug:
            f, axs = plt.subplots(2, 1)
            axs[0].hist(np.mean(train_a, axis=0), bins=100, alpha=1)
            axs[0].set_title('Histogram-  mean axis = 0')
            axs[1].hist(np.mean(train_a, axis=1), bins=100, alpha=1)
            axs[1].set_title('Histogram-  mean axis = 1')
            plt.show()
except Exception as e:
    print(e)
# finally:
#    shutil.rmtree(random_array_dir)
