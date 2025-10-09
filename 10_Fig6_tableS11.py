# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %%
# Fig6
fig, ax = plt.subplots(1, 2, figsize=(9, ((8.27 / 2))), sharey=True)

times = {
    'naive':[7.7, 25.5, 1*60+36, 7*60+21, 61*60+41, 285*60+26],
    'naive_gpu':[3.9, 5.6, 14.6, 47.3, 2*60+21, 10*60+8],
    'rle':[2*60+9, 8*60+33, 34*60+9, 135*60+37, 539*60+23, 2500*60],
    'cwa':[6*60+27, 25*60+58, 107*60+41, 420*60],
    'asa':[75*60+36, 252*60+16, 981*60+41],
    'cotr':[8, 15, 1*60+7, 4*60+1, 15*60+50, 60*60],
    'cotr_gpu':[6, 6, 14, 37, 1*60+56, 7*60+30],
    'sev':[10, 46, 2*60+25, 8*60+33, 31*60+52, 127*60+25],
    'sev_gpu':[9, 9, 40, 1*60+21, 6*60+18, 19*60+4]
}
genes = [2000, 4000, 8000, 16000, 32000, 64000]


colors = ['black', 'black', '#F1C40F', '#03AF7A', '#005AFF', 
          '#4DC4FF', '#4DC4FF', '#FF4B00', '#FF4B00']


counter = 0
for meth, time in times.items(): 
    log_x = np.log10(genes[:len(time)])
    log_y = np.log10(time)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

    ax[0].scatter(log_x, log_y, color=colors[counter])
    if meth.endswith('gpu'):
        ax[0].plot(log_x, slope * log_x + intercept, c=colors[counter], linestyle="--")
    else:
        ax[0].plot(log_x, slope * log_x + intercept, c=colors[counter], label=meth)
        
    counter += 1
ax[0].set_xticks(log_x) 
ax[0].set_xticklabels(["2K", "4K", "8K", "16K", "32K", "64K"])
ax[0].set_xlabel('Number of genes')
ax[0].set_yticks([1, 2, 3, 4, 5])
ax[0].set_yticklabels(['10s', '1.7m', '17m', '2.8h', '1.1d'])
ax[0].set_ylabel('Runtime')
ax[0].grid()

num_species =[0.5, 1, 2, 4, 8, 16]
times = {
    'naive':[13.3, 25.5, 1*60+15, 5*60+33, 13*60+4, 28*60+12],
    'naive_gpu':[5.8, 5.8, 6.9, 8.1, 9.7, 13.7],
    'RLE':[8*60+10, 8*60+33, 9*60+4, 10*60+17, 12*60+29, 17*60+5],
    'CWA':[16*60+55, 25*60+58, 48*60+22, 91*60+4, 178*60+44, 354*60+19],
    'ASA':[120*60+39, 252*60+16, 532*60+38, 1185*60+23],
    'cotransitions':[10, 32, 1*60+3, 1*60+7, 1*60+56, 3*60+49],
    'cotr_gpu':[8, 9, 12, 13, 17, 24],
    'SEV':[26, 46 ,1*60+15, 2*60+48, 4*60+13, 8*60+23],
    'SEV_gpu':[6, 9, 11, 20, 29, 58]
}


counter = 0
for meth, time in times.items(): 
    log_x = np.log10(num_species[:len(time)])
    log_y = np.log10(time)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

    ax[1].scatter(log_x, log_y, color=colors[counter])
    if meth.endswith('gpu'):
        ax[1].plot(log_x, slope * log_x + intercept, c=colors[counter], linestyle="--")
    else:
        ax[1].plot(log_x, slope * log_x + intercept, c=colors[counter], label=meth)
    counter += 1

ax[1].set_xticks(log_x) 
ax[1].set_xticklabels(["0.5K", "1K", "2K", "4K", "8K", "16K"])
ax[1].set_yticks([1, 2, 3, 4, 5])
ax[1].set_yticklabels(['10s', '1.7m', '17m', '2.8h', '1.1d'])
ax[1].grid()
ax[1].set_xlabel('Number of species')
ax[1].legend(loc='center left', bbox_to_anchor=(1, .5))
plt.tight_layout()
plt.savefig('Fig6.png', dpi=300)

# %%
methods = {
    'naive':[7.7, 25.5, 1*60+36, 7*60+21, 61*60+41, 285*60+26, 13.3, 25.5, 1*60+15, 5*60+33, 13*60+4, 28*60+12],
    'naive_gpu':[3.9, 5.6, 14.6, 47.3, 2*60+21, 10*60+8, 5.8, 5.8, 6.9, 8.1, 9.7, 13.7],
    'rle':[2*60+9, 8*60+33, 34*60+9, 135*60+37, 539*60+23, 2500*60, 8*60+10, 8*60+33, 9*60+4, 10*60+17, 12*60+29, 17*60+5],
    'cotr':[8, 15, 1*60+7, 4*60+1, 15*60+50, 60*60, 10, 32, 1*60+3, 1*60+7, 1*60+56, 3*60+49],
    'cotr_gpu':[6, 6, 14, 37, 1*60+56, 7*60+30, 8, 9, 12, 13, 17, 24],
    'sev':[10, 46, 2*60+25, 8*60+33, 31*60+52, 127*60+25, 44.7, 1*60+38, 3*60+18, 6*60+16, 12*60+15, 24*60+38],
    'sev_gpu':[14, 23, 47, 1*60+21, 6*60+18, 19*60+4, 33, 55, 1*60+55, 3*60+10, 6*60+4, 11*60+59],
}

times = {
    'naive':[7.7, 25.5, 1*60+36, 7*60+21, 61*60+41, 285*60+26,
             13.3, 25.5, 1*60+15, 5*60+33, 13*60+4, 28*60+12],
    'naive_gpu':[3.9, 5.6, 14.6, 47.3, 2*60+21, 10*60+8, 
                 5.8, 5.8, 6.9, 8.1, 9.7, 13.7],
    'rle':[2*60+9, 8*60+33, 34*60+9, 135*60+37, 539*60+23, 2500*60,
           8*60+10, 8*60+33, 9*60+4, 10*60+17, 12*60+29, 17*60+5],
    'cotr':[8, 15, 1*60+7, 4*60+1, 15*60+50, 62*60+41,
            10, 32, 1*60+3, 1*60+7, 1*60+56, 3*60+49],
    'cotr_gpu':[6, 6, 14, 37, 1*60+56, 7*60+30, 8, 9, 12, 13, 17, 24],
    'sev':[10, 46, 2*60+25, 8*60+33, 31*60+52, 127*60+25,
           26, 46 ,1*60+15, 2*60+48, 4*60+13, 8*60+23],
    'sev_gpu':[9, 9, 40, 1*60+21, 6*60+18, 19*60+4,
               6, 9, 11, 20, 29, 58]
}

x = [1000, 1000, 1000, 1000, 1000, 1000, 
     500, 1000, 2000, 4000, 8000, 16000]
y = [2000, 4000, 8000, 16000, 32000, 64000,
     4000, 4000, 4000, 4000, 4000, 4000]


# %%
def cal_times(i):
    if i >= 31536000:
        i = i/31536000
        unit ='years'
    elif i >= 86400*3:
        i = i/86400
        unit = 'days'
    elif i >= 3600:
        i = i/3600
        unit = 'hours'
    elif i >= 60:
        i = i/60
        unit = 'mins'
    else:
        unit = 'secs'
    return f'{i:.0f}' + unit


# %%
log_x = np.log(x)
log_y = np.log(y)
X = np.column_stack((log_x, log_y))
for meth, time in times.items():
    z = np.log(time)
    model = LinearRegression()
    model.fit(X, z)
    z_pred = model.predict(X)
    r2 = r2_score(z, z_pred)
    new_x = np.log([107000, 6000])
    new_y = np.log([28000, 61000])
    new_X = np.column_stack((new_x, new_y))
    z_pred = model.predict(new_X)
    print(meth, r2, np.exp(z_pred)/60, model.coef_, model.intercept_)

# %%
log_x = np.log(x)
log_y = np.log(y)
X = np.column_stack((log_x, log_y))

print('method R_square OrthoDB GTDB ')
for meth, time in times.items():
    z = np.log(time)
    model = LinearRegression()
    model.fit(X, z)
    z_pred = model.predict(X)
    r2 = r2_score(z, z_pred)
    new_x = np.log([107000, 6000])
    new_y = np.log([28000, 61000])
    new_X = np.column_stack((new_x, new_y))
    z_pred = model.predict(new_X)
    t1 = cal_times(np.exp(z_pred[0]))
    t2 = cal_times(np.exp(z_pred[1]))
    print(meth, f'{r2:.3f}', t1, t2, model.coef_, model.intercept_)

# %%
times = [6*60+27, 25*60+58, 107*60+41, 420*60, 16*60+55, 25*60+58, 48*60+22, 91*60+4, 178*60+44, 354*60+19]
x = [1000, 1000, 1000, 1000,
     500, 1000, 2000, 4000, 8000, 16000]
y = [2000, 4000, 8000, 16000,
     4000, 4000, 4000, 4000, 4000, 4000]

log_x = np.log(x)
log_y = np.log(y)
X = np.column_stack((log_x, log_y))

print('method R_square OrthoDB GTDB ')
z = np.log(times)
model = LinearRegression()
model.fit(X, z.transpose())
z_pred = model.predict(X)
r2 = r2_score(z, z_pred)
new_x = np.log([107000, 6000])
new_y = np.log([28000, 61000])
new_X = np.column_stack((new_x, new_y))
z_pred = model.predict(new_X)
t1 = cal_times(np.exp(z_pred[0]))
t2 = cal_times(np.exp(z_pred[1]))
print('cwa', f'{r2:.3f}', t1, t2, model.coef_, model.intercept_)

# %%
times = [75*60+36, 252*60+16, 981*60+41, 120*60+39, 252*60+16, 532*60+38, 1185*60+23]
x = [1000, 1000, 1000, 
     500, 1000, 2000, 4000]
y = [2000, 4000, 8000,
     4000, 4000, 4000, 4000]

log_x = np.log(x)
log_y = np.log(y)
X = np.column_stack((log_x, log_y))

print('method R_square OrthoDB GTDB ')
z = np.log(times)
model = LinearRegression()
model.fit(X, z)
z_pred = model.predict(X)
r2 = r2_score(z, z_pred)
new_x = np.log([107000, 6000])
new_y = np.log([28000, 61000])
new_X = np.column_stack((new_x, new_y))
z_pred = model.predict(new_X)
t1 = cal_times(np.exp(z_pred[0]))
t2 = cal_times(np.exp(z_pred[1]))
print('asa', f'{r2:.3f}', t1, t2, model.coef_, model.intercept_)

# %%
times = [11*60+23, 16*60+30, 31*60+30, 80*60+4, 171*60+35, 432*60+3,
         4*60+23, 16*60+30, 73*60+8, 289*60+28]

x = [500, 1000, 2000, 4000, 8000, 16000,
     1000, 1000, 1000, 1000]

y = [4000, 4000, 4000, 4000, 4000, 4000,
     2000, 4000, 8000, 16000]

# %%
log_x = np.log(x)
log_y = np.log(y)
X = np.column_stack((log_x, log_y))

z = np.log(times)
model = LinearRegression()
model.fit(X, z)
z_pred = model.predict(X)
r2 = r2_score(z, z_pred)
new_x = np.log([107000, 6000])
new_y = np.log([28000, 61000])
new_X = np.column_stack((new_x, new_y))
z_pred = model.predict(new_X)
t1 = cal_times(np.exp(z_pred[0])/2)
t2 = cal_times(np.exp(z_pred[1])/2)
print(f'G/L Distance {r2:.3f}', t1, t2, model.coef_, model.intercept_)

# %%
