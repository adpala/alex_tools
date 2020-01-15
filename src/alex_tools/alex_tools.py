import numpy as np
import matplotlib.pyplot as plt
import xarray_behave as xb
import itertools
from matplotlib import cm
from leap_utils.preprocessing import export_boxes, crop_frame

def frame_number2index(ffs,dataset):
    """ takes (a list of, or any iterable object of) frame numbers (as per video logic) and gives corresponding index (as per dataset and metrics_dataset logic)."""
    return [(np.abs(dataset.nearest_frame.values - ff)).argmin() for ff in ffs]

def index2frame_number(idxs,dataset):
    """ takes (a list of, or any iterable object of) indexes (as per dataset and metrics_dataset logic) and gives corresponding frame numbers (as per video logic)."""
    return dataset.nearest_frame[idxs].values.tolist()

def plot_boxes(vr, dataset, frame_numbers, nflies):
    """
    time in seconds
    """
    frames = list(vr[frame_numbers.tolist()])
    times = [dataset.time[dataset.nearest_frame == ff][0].values for ff in frame_numbers]

    for ff, frame in enumerate(frames):
        plt.figure(figsize=[10, 10])
        plt.imshow(frame, cmap='Greys')
        plt.plot(dataset.pose_positions_allo.loc[times[ff], :, :, 'x'], dataset.pose_positions_allo.loc[times[ff], :, :, 'y'], '.')
        for ifly in range(nflies):
            plt.text(dataset.pose_positions_allo.loc[times[ff], ifly, 'thorax', 'x']+30, dataset.pose_positions_allo.loc[times[ff], ifly, 'thorax', 'y']-30, f"{ifly}", color='red', weight='bold')
        plt.xlim(0, frame.shape[1])
        plt.title(f"time {times[ff]}")
        plt.axis('off')
        plt.show()

def assemble_datasets(datename, root="Z:/#Common", expsetup='backlight', target_sampling_rate=1000,include_dataset:bool=False):
    try:
        dataset = xb.assemble(datename, root=root+f'/{expsetup}', target_sampling_rate=target_sampling_rate)
    except:
        dataset = xb.assemble(datename, dat_path='dat.processed', root=root+f'/{expsetup}', target_sampling_rate=target_sampling_rate)
    metrics_dataset = xb.assemble_metrics(dataset)
    if include_dataset:
        return dataset, metrics_dataset
    else:
        return metrics_dataset

def remove_all_nans(x):
    """Interpolates nans from a matrix of traces which have time as first dimension."""
    
    xx = np.swapaxes(x,0,-1)
    for qq in itertools.product(*[range(kk) for kk in xx.shape[:-1]]):
        mask = np.isnan(xx[qq])
        xx[qq][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), xx[qq][~mask])
    return np.swapaxes(xx,0,-1)


def smarter_plot_boxes(dataset, vr, frame_numbers = None, indexes = None, ifly: int = 0, box_size: int = 100, thorax_idx: int = 8, reduced_parts: bool = True, plot_lines: bool = True, extra_text = None, verbose:bool=False):

    if (frame_numbers == None) and (indexes == None): 
        print('you forgot to give frame_numbers or indexes')
        return None
    elif frame_numbers == None:
        frame_numbers = index2frame_number(indexes, dataset)
    if indexes == None: 
        indexes = frame_number2index(frame_numbers, dataset)

    if verbose:   
        print(f"frame_numbers: {frame_numbers}")
        print(f"indexes: {indexes}")
    
    # video frames
    frames = list(vr[frame_numbers])
    center = remove_all_nans(dataset.pose_positions_allo[:, ifly, thorax_idx,:].astype(np.uintp).values)

    # collecting data and frames, and plot together
    for jj, ff in enumerate(frame_numbers):
        # collecting
        myposes = dataset.pose_positions_allo[indexes[jj], ifly, :,:].values
        myrelposes = myposes - myposes[thorax_idx,:] + box_size/2
        if reduced_parts:
            myrelposes = myrelposes[[0,8,9,10],:]
            head_idx = 0
            thorax_idx = 1
            wL_idx = 2
            wR_idx = 3
        else:
            head_idx = 0
            thorax_idx = 8
            wL_idx = 9
            wR_idx = 10            
        croppedframe = crop_frame(frame=frames[jj], center=center[indexes[jj],:], box_size=np.array([box_size,box_size]), mode = 'clip')

        # plot
        bpcmp = cm.get_cmap('gist_rainbow', myrelposes.shape[0])
        plt.imshow(croppedframe, cmap='Greys')
        plt.scatter(myrelposes[:,1],myrelposes[:,0], c=bpcmp(range(myrelposes.shape[0])))
        if plot_lines:
            for part_idx in [wL_idx, wR_idx,head_idx]:
                main_axis_x, main_axis_y = line_2pointform(myrelposes[thorax_idx,::-1],myrelposes[part_idx,::-1],box_size)
                plt.plot(main_axis_x, main_axis_y)
        plt.title(f"frame: {ff}, index: {indexes[jj]}")
        plt.legend(extra_text)
        plt.axis('off')
        plt.show()

def line_2pointform(p1,p2,xmax):
    """line that passes through points p1 and p2 (which are arrays with 2 elements, [x,y]), extending to cover the x range from 0 to xmax."""

    x = np.arange(xmax)
    y = p2[1] + (x-p2[0])*(p2[1]-p1[1])/(p2[0]-p1[0])
    inside_image = np.where((y < xmax)*(y > 0))[0]
    x= x[inside_image]
    y = y[inside_image]
    return x, y


def internal_angle(A: np.array, B: np.array, C: np.array, deg:bool=True, array_logic: str='tfc'):
    """Calculates internal angle (∠ABC) between three points. If A,B,C are lists or arrays, calculation happens element-wise.
    
    Args:
        A, B, C ([type]): position of points between which the angle is calculated.
        deg ([type]): Return angle in degrees if True, radians if False (default).

    Returns:
        angles ([type]): internal angle between lines AB and BC.
    """

    v1s = B-A
    v2s = C-A

    # reshape vector arrays to be [time, coordinates, ...]
    if A.ndim == 3:
        if array_logic == 'tfc':
            v1s = np.swapaxes(v1s,1,2)
            v2s = np.swapaxes(v2s,1,2)
        elif array_logic == 'ftc':
            v1s = np.swapaxes(v1s,1,2)
            v1s = np.swapaxes(v1s,0,2)
            v2s = np.swapaxes(v2s,1,2)
            v2s = np.swapaxes(v2s,0,2)
        elif array_logic == 'fct':
            v1s = np.swapaxes(v1s,0,2)
            v2s = np.swapaxes(v2s,0,2)
    elif A.ndim > 3:
        print('Result might not be correct, only tested for arrays with 2 or 3 dimensions. Contact Adrian for help, if required.')

    dot_v1_v2 = np.einsum('ij...,ij...->i...', v1s, v2s)
    dot_v1_v1 = np.einsum('ij...,ij...->i...', v1s, v1s)
    dot_v2_v2 = np.einsum('ij...,ij...->i...', v2s, v2s)

    angles = np.arccos(dot_v1_v2/np.sqrt(dot_v1_v1*dot_v2_v2))

    if deg:
        angles *= 180/np.pi

    return angles

def correct_wing_angles(dataset):
    """ Recalculates wing angles to be correct using pose information from dataset. Angle range between 0-360 degrees.
    
    Warning: Small angles to the opposite side will also be positive, but the are never big enough to be confused as wing-extension events,
    therefore this won't be a problem, but for other applications this might have to be taken into account.
    """
    heads = dataset.pose_positions_allo[:, :, 0,:]
    thorax = dataset.pose_positions_allo[:, :, 8,:]
    wingL = dataset.pose_positions_allo[:, :, 9,:]
    wingR = dataset.pose_positions_allo[:, :, 10,:]
    left_angles = internal_angle(wingL,thorax,heads)
    right_angles = internal_angle(wingR,thorax,heads)
    sum_angles = left_angles + right_angles
    return left_angles, right_angles, sum_angles

def alex_rotate_points(x, y, degrees, origin=(0, 0)):
    """Rotate (x,y) a point around a given point given by origin."""
    radians = degrees / 180 * np.pi
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy