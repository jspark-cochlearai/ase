#############################################
# ase.py series                             #
# - The code below can be used to estimate  #
#   the binding slopes.                     #
#              Jeongsoo Park, Cochlear.ai   #
#############################################



import csv
import numpy as np
import scipy
import os
import time
import matplotlib.pyplot as plt
from core import *

# # # # # # # # User-defined variables # # # # # # # # 
experiment_name = 'exp1'
data_folder = './txt'
data_type = 'txt' # either txt or npy

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


plt.figure(figsize=(8,6)) # Figure size


# Folders
success_folder = 'png_success'
fail_folder = 'png_fail'

# # # # # # # # # # # # # # # # # # # # # # # # # # # 


try:
  os.mkdir(os.path.join(data_folder,fail_folder))
except:
  pass

try:
  os.mkdir(os.path.join(data_folder,success_folder))
except:
  pass





# csv file ingredients
my_foldernames = list()
my_filenames = list()
my_slopes = list()
my_min_vals = list()
total_cnt = 0

for dirpath, dirnames, fileNames in os.walk(data_folder):
  fileNames.sort()
  for fileName in [f for f in fileNames if f.endswith(data_type)]:
    print(str(dirpath+' / '+fileName))
    

    if data_type == 'txt' or data_type == '.txt':
      # Speed parsing from filename
      speed = speed_parsing(fileName)


      # Data parsing
      y_rt = np.array([])
      x_rt = np.array([])
      with open(dirpath+'/'+fileName,'r') as csvfile:
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
    
    elif data_type == 'npy' or data_type == '.npy':
      X = np.load(dirpath+'/'+fileName)
      x_rt = X[0:511]
      y_rt = X[511:1022]

    
    
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
    error_flag = 0
    if min_val > allowed_min_val:
      error_flag = 1
      print('Error! Minimum value is : '+str(min_val))

    else:
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



    ##### plot
    if error_flag == 1:
      plt.ion()
      plt.plot(x_rt,y_rt)
      plt.plot(x_rt,y_rt_original)
      plt.title(fileName[:30]+'\n'+fileName[30:])
      plt.savefig(os.path.join(data_folder,fail_folder,fileName[:-4]+'.png'))
      plt.clf()
      

      my_foldernames.append(dirpath)
      my_filenames.append(fileName)
      my_slopes.append(1)
      my_min_vals.append(1)

    
    else:
      # plot
      plt.ion()
      plt.plot(x_rt[rollback_idx],y_rt[rollback_idx],'o')
      plt.plot(x_flat,np.ones((len(x_flat)))*50)
      plt.plot(x_rt,y_rt_original)
      plt.plot(x_rt[one_third],trendline[0])
      plt.plot(x_rt,y_rt, linewidth=1)
      plt.plot(x_ROI,trendline, 'r')
      plt.legend(['Rollback point', 'Flat region', 'Original', '1/3 point', 'Baseline_corrected', 'Trendline in ROI'])
      plt.plot(x_rt,np.zeros((len(x_rt))))
      plt.title(fileName[:30]+'\n'+fileName[30:])
      plt.text(np.min(x_rt), (np.max(y_rt)+np.min(y_rt))/2, 'slope : '+str(slope)+', '+'min value : '+str(min_val) )
      #plt.show()
      plt.savefig( os.path.join(data_folder,success_folder,fileName[:-4]+'.png') )
      plt.clf()

      # csv file ingredients
      my_foldernames.append(dirpath)
      my_filenames.append(fileName)
      my_slopes.append(slope)
      my_min_vals.append(min_val)
    total_cnt += 1
    input('.........')

plt.close()

import csv
with open('result_'+experiment_name+'.csv','w') as csvfile:
  mywriter = csv.writer(csvfile, delimiter=',')

  for ii in range(len(my_filenames)):
    mywriter.writerow([my_foldernames[ii],my_filenames[ii],str(-my_slopes[ii]),str(my_min_vals[ii])])



