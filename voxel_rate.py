import sys
# add unimapper to the path
# unimapper is a private repo: https://github.com/castacks/unimapper/tree/dev/interfaces
# use the interfaces branch 
sys.path.append('/home/wenshan/workspace/pytorch/unimapper') # this is hard coded, we can also change it to a submodule in the future

from unimapper.map_world_converter import VoxelMapWorldConverter
from unimapper.feature_meta_data import FeatureMetadata
from unimapper.feature_mapping import VoxelMapper
from unimapper.feature_fusion import FeatureFusionAverage

import torch


def unimapper_count_voxels(points, voxel_resolution = 0.1):
    '''
    Count the number of voxels in a point cloud using unimapper.
    points: (N, 3) tensor of points
    '''

    # some initial parameters for the voxel map, this does not matter too much, as the map will be updated to fit the points
    origin = [0.,0.,0.]
    maprange = [[-10.,10],
            [-10.,10],
            [-2,2.]]
    
    resolution = [voxel_resolution, voxel_resolution, voxel_resolution]

    metadata = VoxelMapWorldConverter(origin = origin,
                                    maprange = maprange,
                                    resolution = resolution,
                                    device="cuda")

    feature_metadata = FeatureMetadata(field_names=[], 
                                        field_idxs=[],
                                        fusion_method=['Average'])

    fusion = FeatureFusionAverage()

    localmapper = VoxelMapper(
                metadata,
                feature_metadata=feature_metadata,
                fusions=fusion,
                device='cuda',
            )
    dummy_features = torch.zeros((points.shape[0], 0)).float().cuda()
    dummy_confidence = torch.ones((points.shape[0],)).float().cuda()
    points = torch.tensor(points).float().cuda()

    localmapper.insert_points_features(points, dummy_features, confidence=dummy_confidence, update_range=True)

    voxel_num = len(localmapper.voxel_grid.indices)

    return voxel_num, localmapper.voxel_grid


if __name__ == "__main__":
    points = torch.rand((10000, 3)).float().cuda() * 2 - 1  # random points in [-1, 1]
    print(points)
    voxel_num, voxel_grid = unimapper_count_voxels(points, voxel_resolution=0.1)
    print(f"Number of voxels: {voxel_num}") 
    voxel_rate = voxel_num / points.shape[0]
    print(f"Voxel rate: {voxel_rate}")