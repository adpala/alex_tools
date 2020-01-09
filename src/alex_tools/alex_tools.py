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


def smarter_plot_boxes(dataset, vr, frame_numbers = None, indexes = None, ifly: int = 0, box_size: int = 100, thorax_idx: int = 8, reduced_parts: bool = True):

    if (frame_numbers == None) and (indexes == None): 
        print('you forgot to give frame_numbers or indexes')
        return None
    elif frame_numbers == None:
        frame_numbers = index2frame_number(indexes, dataset)
    if indexes == None: 
        indexes = frame_number2index(frame_numbers, dataset)
        
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
        croppedframe = crop_frame(frame=frames[jj], center=center[indexes[jj],:], box_size=np.array([box_size,box_size]), mode = 'clip')

        # plot
        bpcmp = cm.get_cmap('gist_rainbow', myrelposes.shape[0])
        plt.imshow(croppedframe, cmap='Greys')
        plt.scatter(myrelposes[:,1],myrelposes[:,0], c=bpcmp(range(myrelposes.shape[0])))
        plt.title(ff)
        plt.axis('off')
        plt.show()