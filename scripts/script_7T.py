"""
This script aims at getting quality measure 
on a 7T data acquired at Neurospin.

It implements a stablity metric and considers 
spatial and temporal autocorrelation in the data

Author: Bertrand Thirion, 2023

Todo: make it more modular, extract functions to be reused in further situations
"""
import os
from glob import glob
import numpy as np
import pingouin as pg
from joblib import Memory
from sklearn.covariance import GraphicalLassoCV
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.connectome import GroupSparseCovarianceCV
from nilearn.image import high_variance_confounds
from nilearn.plotting import find_parcellation_cut_coords
from nilearn.connectome import sym_matrix_to_vec
from nilearn.plotting import plot_connectome, plot_matrix

from ibc_public.utils_data import get_subject_session, DERIVATIVES
import itertools

from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from scipy.interpolate import interp1d
import statsmodels.api as sm

# Data directory
data_dir = '/neurospin/optimed/BlancheBapst/derivatives'

# cache and output directory
cache = OUT_ROOT = "/neurospin/tmp/resting_state"

# overwrite existing files
OVERWRITE = True

# number of parallel workers
N_JOBS = 4

# we will use the resting state and all the movie-watching sessions
task = "task-RestingState"

# get atlas
atlas = datasets.fetch_atlas_schaefer_2018(
    data_dir=cache, resolution_mm=2, n_rois=400
)
# give atlas a custom name
atlas["name"] = "schaefer400"
correlation_measure = ConnectivityMeasure(kind="correlation")
repetition_time = 1.0 # CHECK
sub_ses = {
    'sub-01': ['r000014', 'r000017', 'r000019', 'r000020']
    }

# define regions using the atlas
masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize=True,
    low_pass=0.2,
    high_pass=0.01,
    t_r=repetition_time,
    verbose=1,
    memory=cache,
).fit()


def load_clean_data(sub, run, masker):
    run_num = os.path.basename(run)[2:8]
    confounds = glob(
        os.path.join(data_dir, sub, "func", f"rp*{run_num}*.txt")
    )[0]
    # calculate high-variance confounds
    compcor = high_variance_confounds(run)
    # load confounds and append high-variance confounds
    confounds = np.hstack((np.loadtxt(confounds), compcor))
    # extract time series for atlas defined regions
    time_series = masker.transform(run, confounds=confounds)
    return time_series

mem = Memory(location=cache, verbose=0)
load_clean_data_cached = mem.cache(load_clean_data)

# iterate over subjects
all_time_series = []
all_subjects = []
all_runs = []
all_params = []

for sub, runs_ in sub_ses.items():
    # iterate over sessions for each subject
    for run in runs_:
        # setup tmp dir for saving figures and calculated matrices
        tmp_dir = os.path.join(OUT_ROOT, sub, "func")
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        runs = sorted(glob(
            os.path.join(data_dir, sub, "func", f"w{run}*.nii.gz")
        ))
        param = [x for x in runs[0].split('-') if 'FA' in x and len(x) < 5]
        time_series = load_clean_data_cached(sub, runs[0], masker)
        all_time_series.append(time_series)
        all_subjects.append(sub)
        all_runs.append(run)
        all_params.append(param)

runs =  np.array(all_runs)
all_subjects = np.array(all_subjects)
all_params = np.array(all_params)
df = pd.DataFrame({"subject": all_subjects,
                   "runs": runs})


############################################################################
# Compute correlation matrices

n_samples = 1000

all_csvs = []
correlation_matrices = []
for ts, sub, sess, run in zip(
        all_time_series, all_subjects, all_runs, all_params
):
    # name of csv file to save matrix as
    correlation_csv = os.path.join(
        tmp_dir,
        (
            f"{sub}_{sess}_{task}_{run}_{atlas.name}"
            "_pearsons_corr.csv"from nilearn.plotting import plot_connectome, plot_matrix
    # skip calculating correlation if already done
    if os.path.isfile(correlation_csv) and not OVERWRITE:
        pass
    else:
        # calculate pearson correlation matrix
        correlation_matrix = correlation_measure.fit_transform(
            [ts[:n_samples]]
        )[0]
        correlation_matrices.append(correlation_matrix)
        # save pearson correlation matrix as csv
        np.savetxt(correlation_csv, correlation_matrix, delimiter=",")
        all_csvs.append(correlation_csv)


#############################################################################
# Run the Poldrack measure
# need a cross validation
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

correlation_measure = ConnectivityMeasure(from nilearn.plotting import plot_connectome, plot_matrix
    discard_diagonal=True,
    # standardize='zscore_sample'
)

lengths = np.linspace(100, 1000, 10).astype(int)

similarity = {}
for subject in np.unique(all_subjects):
    tseries = [x for x, y in zip(all_time_series, all_subjects) if y == subject]
    kf = KFold(n_splits = len(tseries))
    corrs = []
    for i, (train_index, test_index) in enumerate(kf.split(tseries)):
        # compute a template correlation from the others
        ref_conn = correlation_measure.fit_transform(
            [np.vstack([tseries[i] for i in train_index])])[0]
        corr_length = []
        for length in lengths:
            test_conn = correlation_measure.fit_transform(
                [tseries[test_index[0]][:length]])[0]
            corr_length.append(np.corrcoef(ref_conn, test_conn)[0, 1])
        corrs.append(corr_length)
    similarity[subject] = np.array(corrs)

n_params = len(all_params)
facecolor = [.8, .8, .8]
colors = plt.cm.gist_ncar(np.linspace(1./ n_params, 1, n_params))
plt.figure(facecolor=facecolor)
for i, color in enumerate(colors):
    plt.plot(similarity[subject][i].T, color=color, label=all_params[i])
plt.xticks(range(len(lengths)), lengths)
plt.legend(loc='center')
ax = plt.subplot(111)
ax.set_facecolor(facecolor)
plt.xlabel('time series length')
plt.ylabel('correlation of connectivity matrices')


#############################################################################
# Compute the autocorrelation of the time series


def compute_ar_coefs(x, order):
    """ """
    denom = x.shape[-1] - np.arange(order + 1)
    n = np.prod(np.array(x.shape[:-1], int))
    r = np.zeros((n, order + 1), np.float64)
    y = x - x.mean()
    y.shape = (n, x.shape[-1])  # inplace
    r[:, 0] += (y[:, np.newaxis, :] @ y[:, :, np.newaxis])[:, 0, 0]
    for k in range(1, order + 1):
        r[:, k] += (y[:, np.newaxis, 0:-k] @ y[:, k:, np.newaxis])[:, 0, 0]
    return r[:, 1:] / r[:, :1]

all_ar_coefs = []
order = 3
for x in all_time_series:from nilearn.plotting import plot_connectome, plot_matrix

bins = 20
n = len(all_ar_coefs)
n_subjects = n // 4

plt.figure(figsize=(8, 3 * n_subjects))
for  k in range(order):
    for i, ar_coefs in enumerate(all_ar_coefs):
        hist, bin_edges = np.histogram(
            ar_coefs[:, k],
            bins,
            range = [0, 1],
            density=True)
        plt.subplot(n_subjects, 4, i + 1)
        plt.bar(bin_edges[: -1], hist / bins, .05)


##########################################################################
# Compute ICC of average coefficients
"""
all_ar_coefs = np.array(all_ar_coefs)
mean_ar = all_ar_coefs.mean(1)

for k in range(order):
    df[f"mean_ar{k}"] = mean_ar[:, k]
    icc = pg.intraclass_corr(
        data=df,
        targets="subject",
        raters="runs",
        ratings=f"mean_ar{k}")
    print(k)
    print(icc)
"""
##########################################################################
# Compute ICC of per-region coefficients

"""
all_iccs = [[], [], []]
for r in range(all_ar_coefs.shape[1]):
    for k in range(order):
        df[f"ar{k}"] = all_ar_coefs[:, r, k]
        icc = pg.intraclass_corr(
            data=df,
        targets="subject",
        raters="runs",
        ratings=f"ar{k}")
        all_iccs[k].append(icc.ICC[0])

all_iccs = np.array(all_iccs)
print(np.mean(all_iccs, 1))
"""
#############################################################################
# plot correlation vs distance"Yeo Atlas 17 thick (func)"

# compute distance between regions

labels_img = masker.labels_img_
cut_coords = find_parcellation_cut_coords(labels_img, background_from nilearn.plotting import plot_connectome, plot_matrix
plt.figure(figsize=(8, 3 * n_subjects))
SALambda = []
SAInf = []
lambd = 15
for i, csv in enumerate(all_csvs):
    correlation = pd.read_csv(csv, index_col=False, header=None).values
    triu_connectivity = sym_matrix_to_vec(correlation, discard_diagonal=False)
    plt.subplot(n_subjects, 4, i + 1)
    plt.scatter(triu_distances, triu_connectivity, marker='.')
    # lowess will return our "smoothed" data with a y value for at every x-value
    lowess = sm.nonparametric.lowess(triu_connectivity, triu_distances, frac=.1)

    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]

    # run scipy's interpolation. There is also extrapolation I believe
    f = interp1d(lowess_x, lowess_y, bounds_error=False)
    
    # this this generate y values for our xvalues by our interpolator
    # it will MISS values outsite of the x window (less than 3, greater than 33)
    # There might be a better approach, but you can run a for loop
    #and if the value is out of the range, use f(min(lowess_x)) or f(max(lowess_x))
    ynew = f(xnew)
    plt.plot(lowess_x, lowess_y, 'r*')
    plt.plot(xnew, ynew, 'r-')
    SALambda.append(f(15))
    SAInf.append(f(100))

# compute the ICC
"""
df[f"SALambda"] = SALambda
df[f"SAInf"] =  SAInf
icc = pg.intraclass_corr(
        data=df,
        targets="subject",
        raters="runs",
        ratings="SALambda")
print("SALambda")
print(icc)
icc = pg.intraclass_corr(
        data=df,
        targets="subject",
        raters="runs",
        ratings="SAInf")
print("SAInf")
print(icc)

# plt.savefig(os.path.join(tmp_dir, 'correlation_distance.png')
"""

##############################################################################
#  Plot connectomes


# plot connectome with 80% edge strength in the connectivity
for param, correlation_matrix in zip(all_params, correlation_matrices):
    plot_connectome(
        correlation_matrix,
        cut_coords,
        edge_threshold="99.5%",
        title=param[0],
    )

    plot_matrix(
        correlation_matrix,
        figure=(9, 7),
        vmax=1,
        vmin=-1,
        title=param[0], 
)
plt.show()

    
