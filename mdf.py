import csv
import numpy as np
import scipy
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy import interpolate
from scipy.optimize import least_squares



# 2nd order polynomial
def fun(x, t, y):
  return x[0]+x[1]*t+x[2]*t**2-y
def generate_data(t, x_0, x_1, x_2):
    return x_0+x_1*t+x_2*t**2
baseline_order=2






txt_folder_name = 'txt'
spm_folder_name = 'spm'

(loc, _, fileNames) = os.walk(txt_folder_name).next()
fileNames.sort()
fileNames = [f for f in fileNames if f.endswith(".txt")]

try:
  os.mkdir(txt_folder_name+'/useful')
  os.mkdir(txt_folder_name+'/useless')
except:
  pass

try:
  os.mkdir(spm_folder_name+'/useful')
  os.mkdir(spm_folder_name+'/useless')
except:
  pass


for fileName in fileNames:
  print(fileName)

  # Data parsing
  y_rt = np.array([])

  with open(txt_folder_name+'/'+fileName,'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    first_column = 1
    for row in reader:
      if first_column != 0:
        first_column -= 1
      else:
        y_rt = np.concatenate((y_rt,[float(row[5])] ))

  y_rt = y_rt[:-1]




  # # # Baseline correction 1
  # Flat region -> normalize to have 0-mean
  sd = np.zeros((10))
  for r in range(10):
    sd[r] = np.std(y_rt[r*50:(r+1)*50])

  sd_flat = np.min(sd)
  r_flat = np.argmin(sd)
  y_flat = y_rt[r_flat*50:(r_flat+1)*50]
  m_flat = np.mean(y_flat)

  # 0 mean
  y_rt = y_rt-m_flat


  # minimum value
  min_val = np.min(y_rt)
  min_idx = np.argmin(y_rt)



  # # # Baseline correction 2
  # Robust nonlinear regression
  x0 = np.ones(baseline_order+1)
  t_train = np.linspace(0, 510, 510-0+1)
  y_train = y_rt[0:510+1]
  t_test = np.linspace(0,len(y_rt)-1,len(y_rt))
  res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.05*sd_flat, args=(t_train,y_train))
  y_robust = generate_data(t_test, *res_robust.x)
    
  y_rt_original = y_rt
  y_rt = y_rt-y_robust


  # # # Baseline correction 3
  # Flat region once more
  sd = np.zeros((10))
  for r in range(10):
    sd[r] = np.std(y_rt[r*50:(r+1)*50])

  sd_flat = np.min(sd)
  r_flat = np.argmin(sd)
  y_flat = y_rt[r_flat*50:(r_flat+1)*50]
  m_flat = np.mean(y_flat)

    # # # Baseline correction 2
  # Robust nonlinear regression
  x0 = np.ones(baseline_order+1)
  t_train = np.linspace(0, 510, 510-0+1)
  y_train = y_rt[0:510+1]
  t_test = np.linspace(0,len(y_rt)-1,len(y_rt))
  res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.05*sd_flat, args=(t_train,y_train))
  y_robust = generate_data(t_test, *res_robust.x)
    
  y_rt_original = y_rt
  y_rt = y_rt-y_robust


  # # # Baseline correction 3
  # Flat region once more
  sd = np.zeros((10))
  for r in range(10):
    sd[r] = np.std(y_rt[r*50:(r+1)*50])

  sd_flat = np.min(sd)
  r_flat = np.argmin(sd)
  y_flat = y_rt[r_flat*50:(r_flat+1)*50]
  m_flat = np.mean(y_flat)


  # 0 mean
  y_rt = y_rt-m_flat
  y_rt[0:35]= 0
  m_flat = 0


  # minimum value
  min_val = np.min(y_rt[6:])
  min_idx = np.argmin(y_rt[6:])+6


  

  # # # head
  max_head = 0
  head = 0
  buff = 0
  for idx in range(6, len(y_rt)-2):
    if y_rt[idx] < y_rt[idx+1]:
      head += y_rt[idx+1]-y_rt[idx]
    else:
      if head != 0 and buff == 0:
        buff = 1
        head += y_rt[idx+1]-y_rt[idx]
      elif head != 0 and buff == 1:
        if y_rt[idx-1] > y_rt[idx]:
          head -= y_rt[idx]-y_rt[idx-1]
        if head > max_head:
          max_head = head
        head = 0
        buff = 0
      else:
        head = 0
        buff = 0






  error_flag = 0
  if min_val > -20:
    error_flag = 1
  # elif max_head < 2*(np.max(y_flat-m_flat)-np.min(y_flat-m_flat)):
  #   error_flag = 1
  elif min_val > -2.75*np.max(np.abs(y_flat-m_flat)):
    error_flag = 1







  # # # plot
  if error_flag == 1:
    # move the txt file
    os.rename(txt_folder_name+'/'+fileName, txt_folder_name+'/useless/'+fileName)
    os.rename(spm_folder_name+'/'+fileName[:-4], spm_folder_name+'/useless/'+fileName[:-4])



    # plt.ion()
    # plt.plot(x_rt,y_rt)

    # try:
    #   os.mkdir('png_fail')
    # except:
    #   pass

    # plt.savefig('png_fail/'+fileName[:-4]+'.png')
    # plt.clf()

  else:
    os.rename(txt_folder_name+'/'+fileName, txt_folder_name+'/useful/'+fileName)
    os.rename(spm_folder_name+'/'+fileName[:-4], spm_folder_name+'/useful/'+fileName[:-4])


  # # # plot
  # plt.ion()
  # plt.plot(y_rt)
  # plt.show()
  # print('min_val: '+str(min_val)+' , '+'max_head: '+str(max_head))
  # raw_input('Enter...')
  # plt.clf()
# plt.close()
