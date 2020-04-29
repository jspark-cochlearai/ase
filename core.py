#############################################
# core.py                                   #
# - This file contains functions that are   #
#   frequently used in a.s.e. series.       #
#              Jeongsoo Park, Cochlear.ai   #
#############################################
import csv
import numpy as np
import scipy
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy import interpolate
from scipy.optimize import least_squares


def speed_parsing(fileName):
  speed = 1
  if '1um' in fileName:
    speed = 1
  elif '5um' in fileName:
    speed = 5
  elif '10um' in fileName:
    speed = 10
  elif '20um' in fileName:
    speed = 20
  elif '100nm' in fileName:
    speed = 0.1
  elif '200nm' in fileName:
    speed = 0.2
  else:
    print('Speed not indicated in the filename!!!')
    raise ValueError
  return speed


def time_axis_parsing(y_length, speed):
  if speed == 1:
    t_rt = np.arange((y_length))*0.9461548828125*512/y_length
  elif speed == 5:
    t_rt = np.arange((y_length))*0.1908802734375*512/y_length
  elif speed == 10:
    t_rt = np.arange((y_length))*0.0949671875000001*512/y_length
  elif speed == 20:
    t_rt = np.arange((y_length))*0.047019140625*512/y_length
  elif speed == 0.1:
    t_rt = np.arange((y_length))*9.721661328125*512/y_length
  elif speed == 0.2:
    t_rt = np.arange((y_length))*4.878356640625*512/y_length
  else:
    print('Not supported speed!!!')
    raise ValueError
  return t_rt


def baseline_corr1(x_rt, y_rt):
  analysis_window = int(50/512*len(y_rt))
  sd = np.zeros((10))
  for r in range(10):
    sd[r] = np.std(y_rt[r*analysis_window:(r+1)*analysis_window])

  sd_flat = np.min(sd)
  r_flat = np.argmin(sd)
  y_flat = y_rt[r_flat*analysis_window:(r_flat+1)*analysis_window]
  m_flat = np.mean(y_flat)

  # 0 mean
  y_rt = y_rt-m_flat

  # flat region
  x_flat = x_rt[r_flat*analysis_window:(r_flat+1)*analysis_window]

  # minimum value
  min_val = np.min(y_rt)
  min_idx = np.argmin(y_rt)
  return (y_rt, x_flat, sd_flat, min_val, min_idx)


def baseline_corr2(y_rt, baseline_order, allowed_deviation):
  t_train_tmp = np.where(  np.logical_or(y_rt>0, np.logical_and(y_rt<=0, np.abs(y_rt)<allowed_deviation ))  )[0]
  t_train = np.array([t for t in t_train_tmp if t>1])
  y_train = y_rt[t_train]

  x0 = np.ones(baseline_order+1)
  t_test = np.linspace(0,len(y_rt)-1,len(y_rt))
  if baseline_order == 1:
    res_robust = least_squares(fun1, x0, loss='soft_l1', f_scale=allowed_deviation, args=(t_train,y_train))
    y_robust = generate_data1(t_test, *res_robust.x)
  elif baseline_order == 2:
    res_robust = least_squares(fun2, x0, loss='soft_l1', f_scale=allowed_deviation, args=(t_train,y_train))
    y_robust = generate_data2(t_test, *res_robust.x)
  return y_robust


def is_local_minima(y_rt, y_idx, idx_range):
  is_lm = 1
  for current_idx in range(idx_range):
    if y_rt[y_idx+current_idx] > y_rt[y_idx+current_idx+1]:
      is_lm = 0
  for current_idx in range(idx_range):
    if y_rt[y_idx-current_idx] > y_rt[y_idx-current_idx-1]:
      is_lm = 0
  return is_lm


def is_minima(y_rt, y_idx):
  if y_idx == 0 or y_idx == len(y_rt)-1:
    return 0
  if y_rt[y_idx] < y_rt[y_idx-1] and y_rt[y_idx] < y_rt[y_idx+1]:
    return 1
  else:
    return 0


def fun1(x, t, y):
  return x[0]+x[1]*t-y


def fun2(x, t, y):
  return x[0]+x[1]*t+x[2]*t**2-y


def generate_data1(t, x_0, x_1):
  return x_0+x_1*t


def generate_data2(t, x_0, x_1, x_2):
  return x_0+x_1*t+x_2*t**2


# slope_ESTI, min_value_ESTI = ase(txt_name=txt_file_name)
def ase(fileName=None):
  ############# Parameters ###############
  # Baseline correction
  baseline_order=1
  allowed_deviation=5 # 5 times of sd_flat

  # Variables related to meaningful data filtering
  allowed_min_val = -10 # Criterion for the global minimum value

  # Variables related to multiple binding selection
  # -> "or" condition: don't need to satify all the conditions below.
  min_val_ratio = 0.5 # if min_val = -90, peaks under -45 are selected
  binding_threshold = -50 # peaks under this value are selected
  sd_flat_ratio = -3  # if sd_flat=8, peaks under -24 are selected

  # Multiple binding validity check
  neighbor_boundary = [-8,4]

  # Peak inverval (from the global min val)
  peak_interval = 20

  # 1/3 or 1/4 or 1/6 ...
  n_divide = 6.0
  #########################################

  # Speed parsing from filename
  speed = speed_parsing(fileName)

  # Data parsing
  y_rt = np.array([])
  x_rt = np.array([])
  with open(fileName,'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    first_column = 1
    for row in reader:
      if first_column != 0:
        first_column -= 1
      else:
        y_rt = np.concatenate((y_rt,[float(row[5])] ))
        x_rt = np.concatenate((x_rt,[float(row[3])] ))
  y_rt = y_rt[:-1]

  # Since the 'time' data in the txt files is messed up, it is manually
  # generated based on the time interval between two consecutive samples.
  t_rt = time_axis_parsing(len(y_rt) ,speed)

  # x-axis = time
  x_rt = t_rt

  ##### Baseline correction 1 
  # Flat region -> normalize to have 0-mean
  y_rt, x_flat, sd_flat, min_val, min_idx = baseline_corr1(x_rt,y_rt)
  
  
  ##### Baseline correction 2: close value enrichment
  # Robust nonlinear regression
  y_robust = baseline_corr2(y_rt, baseline_order=baseline_order, allowed_deviation=allowed_deviation*sd_flat)
  y_rt_original = y_rt
  y_rt = y_rt-y_robust
  
  
   # # Baseline correction 3
  y_rt, x_flat, sd_flat, min_val, min_idx = baseline_corr1(x_rt,y_rt)
  
  
  
  # Data validity check
  ##### Multiple binding handling
  # Check if there are other bindings or not.
  local_minima_idx = list(np.where( np.min((min_val_ratio*min_val,binding_threshold,sd_flat_ratio*sd_flat)) > y_rt)[0])  # many bindings
  lmi_tobe_del = list()
  for lmi in local_minima_idx:
    if is_minima(y_rt, lmi):
      pass
    else:
      lmi_tobe_del.append(lmi)

  for ltd in lmi_tobe_del:
    local_minima_idx.remove(ltd)


  # Check if those are the min values
  # by comparing them with the neighboring values.
  local_minima_idx2 = list()
  for lmi in local_minima_idx:
    if y_rt[lmi] == np.min(y_rt[lmi+neighbor_boundary[0]:lmi+neighbor_boundary[1]]):
      local_minima_idx2.append(lmi)
  local_minima_idx = np.array(local_minima_idx2)


  # when the interval is larger than peak_interval, the indices are considered as different ones.
  if np.sum(np.abs(local_minima_idx-min_idx) > peak_interval): # at least one of the lmi
    min_idx = np.max(local_minima_idx)
    min_val = y_rt[min_idx]



  ##### ROI (region of interest) selection
  # picking rollback index
  y_tmp = y_rt[0:np.minimum(min_idx+1,len(y_rt))]

  # get rollback index
  rollback_idx = min_idx
  while y_tmp[rollback_idx] <= 0:
    rollback_idx -= 1

  total_dist = min_idx-rollback_idx
  
  # Real ROI
  one_third = rollback_idx + int(total_dist*(n_divide-1)/n_divide)
  x_ROI = x_rt[one_third:min_idx+1]
  y_ROI = y_rt[one_third:min_idx+1]


  ##### Get the trendline & slope of it
  p = np.polyfit(x_ROI,y_ROI,1)
  trendline = p[1]+p[0]*x_ROI

  slope = p[0]
  slope = round(1000*slope,4)
  min_val = np.abs(round(min_val,4))

  return (slope, min_val)






