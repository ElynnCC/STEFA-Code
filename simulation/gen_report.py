import numpy as np
import pandas as pd
import os
import re

df = pd.DataFrame(columns=['name', 'type',
                           'A_error_tucker 1', 'A_error_tucker 2', 'A_error_tucker 3',
                           # 'F_error_tucker',
                           'Y_error_tucker',
                           'A_error_ptucker 1', 'A_error_ptucker 2', 'A_error_ptucker 3',
                           # 'F_error_ptucker',
                           'Y_error_ptucker',
                           'G_1_error 1', 'G_1_error 2', 'G_1_error 3'])

index = 0
for filename in np.sort(os.listdir('output')):
    result = re.findall('(.+).pkl', filename)
    if len(result) > 0:
        name = result[0]
        data = pd.read_pickle('output/' + name + '.pkl')
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        df.loc[index, 2:] = mean
        df.loc[index, 'name'] = name
        df.loc[index, 'type'] = 'mean'
        index += 1
        df.loc[index, 2:] = std
        df.loc[index, 'name'] = name
        df.loc[index, 'type'] = 'std'
        index += 1

df.to_csv('report/summary.csv', index=None)

#### Separate reports

methods = ['Projected Tucker', 'Vanilla Tucker']
evalations = ['A1', 'A2', 'A3', 'Y']

correspondence = [(('Projected Tucker', 'A1'), 'A_error_ptucker 1'),
                  (('Projected Tucker', 'A2'), 'A_error_ptucker 2'),
                  (('Projected Tucker', 'A3'), 'A_error_ptucker 3'),
                  # (('Projected Tucker', 'F'), 'F_error_ptucker'),
                  (('Projected Tucker', 'Y'), 'Y_error_ptucker'),
                  (('Vanilla Tucker', 'A1'), 'A_error_tucker 1'),
                  (('Vanilla Tucker', 'A2'), 'A_error_tucker 2'),
                  (('Vanilla Tucker', 'A3'), 'A_error_tucker 3'),
                  # (('Vanilla Tucker', 'F'), 'F_error_tucker'),
                  (('Vanilla Tucker', 'Y'), 'Y_error_tucker'), ]

correspondence_g = [('g_1,1', 'G_1_error 1'),
                    ('g_1,2', 'G_1_error 2'),
                    ('g_1,3', 'G_1_error 3'), ]

### Growing Dimension


df_growing_dimension = pd.DataFrame(columns=pd.MultiIndex.from_product([methods, evalations],
                                                                       names=['methods', 'errors']),
                                    index=pd.MultiIndex.from_product([[0.1, 0.3, 0.5], [3], [100, 200, 300]],
                                                                     names=['alpha', 'R', 'I']),
                                    dtype=np.str)

for index, row in df_growing_dimension.iterrows():
    name = f"growing_dimension_I_{index[2]}_R_{index[1]}_SNR_{index[0]}"
    mean = df.loc[(df.name == name) & (df.type == 'mean')]
    std = df.loc[(df.name == name) & (df.type == 'std')]
    for ind, colname in correspondence:
        row[ind] = f"{mean[colname].iloc[0]:.3f} ({std[colname].iloc[0]:.3f})"

df_growing_dimension.to_csv('report/Growing_Dimension.csv')

### Growing Dimension - G

df_growing_dimension_g = pd.DataFrame(columns=['g_1,1', 'g_1,2', 'g_1,3'],
                                      index=pd.MultiIndex.from_product([[0.1, 0.3, 0.5],
                                                                        [3],
                                                                        [100, 200, 300]],
                                                                       names=['alpha', 'R', 'I']),
                                      dtype=np.str)

for index, row in df_growing_dimension_g.iterrows():
    name = f"growing_dimension_I_{index[2]}_R_{index[1]}_SNR_{index[0]}"
    mean = df.loc[(df.name == name) & (df.type == 'mean')]
    std = df.loc[(df.name == name) & (df.type == 'std')]
    for ind, colname in correspondence_g:
        row[ind] = f"{mean[colname].iloc[0]:.3f} ({std[colname].iloc[0]:.3f})"

df_growing_dimension_g.to_csv('report/Growing_Dimension_G.csv')

### Unbalanced Tensor

df_unbalanced = pd.DataFrame(columns=pd.MultiIndex.from_product([methods, evalations],
                                                                names=['methods', 'errors']),
                             index=pd.MultiIndex.from_product([[0.3, 0.5],
                                                               [3],
                                                               [(100, 100, 200),
                                                                (100, 100, 400),
                                                                (100, 200, 200),
                                                                (100, 200, 400)
                                                                ]],
                                                              names=['alpha', 'R', 'I']),
                             dtype=np.str)

for index, row in df_unbalanced.iterrows():
    name = f"unbalanced_dimension_I_{index[2][0]}_{index[2][1]}" \
           f"_{index[2][2]}_R_{index[1]}_SNR_{index[0]}"
    mean = df.loc[(df.name == name) & (df.type == 'mean')]
    std = df.loc[(df.name == name) & (df.type == 'std')]
    for ind, colname in correspondence:
        row[ind] = f"{mean[colname].iloc[0]:.3f} ({std[colname].iloc[0]:.3f})"

df_unbalanced.to_csv('report/Unbalanced_Dimension.csv')

### J Effect

df_j_effect = pd.DataFrame(columns=pd.MultiIndex.from_product([methods, evalations],
                                                              names=['methods', 'errors']),
                           index=pd.Index([2, 4, 8, 16], name='J'),
                           dtype=np.str)

for index, row in df_j_effect.iterrows():
    name = f"J_effect_I_200_R_3_SNR_0.3_J_{index}_16"
    mean = df.loc[(df.name == name) & (df.type == 'mean')]
    std = df.loc[(df.name == name) & (df.type == 'std')]
    for ind, colname in correspondence:
        row[ind] = f"{mean[colname].iloc[0]:.3f} ({std[colname].iloc[0]:.3f})"

df_j_effect.to_csv('report/J_Effect.csv')


df_j_effect_new = pd.DataFrame(columns=pd.MultiIndex.from_product([methods, evalations],
                                                              names=['methods', 'errors']),
                           index=pd.Index([2, 4, 8, 16], name='J'),
                           dtype=np.str)

for index, row in df_j_effect_new.iterrows():
    name = f"J_effect_I_200_R_3_SNR_0.5_J_{index}_16"
    mean = df.loc[(df.name == name) & (df.type == 'mean')]
    std = df.loc[(df.name == name) & (df.type == 'std')]
    for ind, colname in correspondence:
        row[ind] = f"{mean[colname].iloc[0]:.3f} ({std[colname].iloc[0]:.3f})"

df_j_effect_new.to_csv('report/J_Effect_new.csv')

### Gamma Effect


df_gamma_effect = pd.DataFrame(columns=pd.MultiIndex.from_product([methods, evalations],
                                                                  names=['methods', 'errors']),
                               index=pd.Index(['0.00', '0.01', '0.1', '1'], name='eta'),
                               dtype=np.str)

for index, row in df_gamma_effect.iterrows():
    name = f"Gamma_effect_I_200_R_3_SNR_0.3_ETA_{index}"
    mean = df.loc[(df.name == name) & (df.type == 'mean')]
    std = df.loc[(df.name == name) & (df.type == 'std')]
    for ind, colname in correspondence:
        row[ind] = f"{mean[colname].iloc[0]:.3f} ({std[colname].iloc[0]:.3f})"

df_gamma_effect.to_csv('report/Gamma_Effect.csv')
