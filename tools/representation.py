import math

import torch as th

from . import event




def to_trilinear_voxel_grid(event_sequence, number_of_time_bins=5, dtype=th.float32):
    """Returns voxel grid representation of event sequence.
    
    In voxel grid representation, temporal dimension is discretized 
    into "number_of_time_bins" bins. The events polarities are
    interpolated between eight near-by spatio-temporal bins using
    trilinear interpolation and summed up. If event sequence is 
    empty, the voxel grid will be empty.
    
    Args:
        events: object of EventSequence class.
        number_of_time_bins: number of time bins in voxel grid.
    """
    voxel_grid = th.zeros(
        number_of_time_bins,
        event_sequence._image_height,
        event_sequence._image_width,
        dtype=dtype,
        device="cpu",
    )
    duration = event_sequence.duration()
    if len(event_sequence) == 0 or duration == 0:
        # Duration of voxel grid might be zero even
        # if there are several events in voxel grid, e.g. in
        # case of events with same timestamp.
        return voxel_grid
    voxel_grid_flat = voxel_grid.flatten()
    # Convert timestamps to [0, nb_of_time_bins] range.
    start_timestamp = event_sequence.start_time()
    features = th.from_numpy(event_sequence._features)
    x = features[:, event.X_COLUMN]
    y = features[:, event.Y_COLUMN]
    polarity = features[:, event.POLARITY_COLUMN].float()
    t = (
        (features[:, event.TIMESTAMP_COLUMN] - start_timestamp)
        * (number_of_time_bins - 1)
        / duration
    )
    t = t.float()
    left_t, right_t = t.floor(), t.floor() + 1
    left_x, right_x = x.floor(), x.floor() + 1
    left_y, right_y = y.floor(), y.floor() + 1
    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                mask = (
                    (0 <= lim_x)
                    & (0 <= lim_y)
                    & (0 <= lim_t)
                    & (lim_x <= event_sequence._image_width - 1)
                    & (lim_y <= event_sequence._image_height - 1)
                    & (lim_t <= number_of_time_bins - 1)
                )
                # we cast to long here otherwise the mask is not computed correctly
                lin_idx = (
                    lim_x.long()
                    + lim_y.long() * event_sequence._image_width
                    + lim_t.long() * event_sequence._image_width * event_sequence._image_height
                )
                weight = (
                    polarity
                    * (1 - (lim_x - x).abs())
                    * (1 - (lim_y - y).abs())
                    * (1 - (lim_t - t).abs())
                )
                voxel_grid_flat.index_add_(
                    dim=0, index=lin_idx[mask], source=weight[mask]
                )
    return voxel_grid
