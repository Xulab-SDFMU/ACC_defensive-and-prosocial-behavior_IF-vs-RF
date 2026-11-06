# Functions.py
#
# Author: Xiaoqian Sun, Apr 2023
#
# Functions for projection project

from IPython.core.debugger import set_trace


# Import Packages
#========================================================================================
import os 
import cv2
import math
import itertools
import scipy as sp
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import tifffile as tiff
import matplotlib as mpl
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
# from oasis.functions import deconvolve
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage.filters import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.signal import chirp, find_peaks, peak_widths


import warnings
warnings.filterwarnings('ignore')





# Functions
#========================================================================================
def process_frame(df):
    
    '''
    Aim:
        frame 0 at trace file may not sart from time 0
        called in function `add_event_label()`
    '''
    
    # get frame list
    frame_list = df.index.values.tolist()
    
    # get starting value
    a = frame_list[0]
    new_frame_list = [(i-a) for i in frame_list]
    
    return(new_frame_list)

def process_event_interval(df):
    
    '''
    Aim:
        time in event file doesn't always start from 0s, fix that
        called in function `add_event_label()`
    '''
    
    new_col = []
    for col in df.columns.tolist():
        new_col.append(col.strip())
    df.columns = new_col
    
    event_list = df[['From Second','To Second']].values.tolist()
    startTime = df['From Second'].values.tolist()[0]
    new_event_list = []
    for event in event_list:
        new_event = [x -startTime for x in event]
        new_event_list.append(new_event)
    return(new_event_list)

# +
def get_event_label(event_df):
    
    '''
    Aim:
        for OFT
        translate 'event' in eventFile to integer label
        each row (corresponding to each time interval) of event is assigned a label
        called in function `add_event_label()`
    '''
    
    # get event status list
    Event_status = event_df['Event'].values.tolist()
    # compact event label list for three sessions
    event_label = []
    
#     ['In_all',
#  'In_right',
#  'Investigating_RIGHT_SNIFF',
#  'In_left',
#  'Investigating_LEFT_SNIFF']

#     if which_status_to_check_frame == 'Investigating_RIGHT_SNIFF':
#         which_status_to_check_frame = str(11)
#     elif which_status_to_check_frame == 'Investigating_LEFT_SNIFF':
#         which_status_to_check_frame = str(12)
#     elif which_status_to_check_frame == 'In_right':
#         which_status_to_check_frame = str(21)
#     elif which_status_to_check_frame == 'In_all':
#         which_status_to_check_frame = str(22)
#     elif which_status_to_check_frame == 'In_left':
#         which_status_to_check_frame = str(23)
#     else:

    for i in Event_status:   
        if i.strip() == 'free_allogroom':
            event_label.append(11)
        elif i.strip() == 'sniff' :
            event_label.append(12)
        elif i.strip() == 'rear_selfgroom':
            event_label.append(21)
        elif i.strip() == 'all':
            event_label.append(22)
        else:
            event_label.append(23)
                
#     for i in Event_status:   
#         if i.strip() == 'Area:Mouse 1 Center In floor':
#             event_label.append(10)
#         elif i.strip() == 'Area:Mouse 1 Center In All ' :
#             event_label.append(10)
#         elif i.strip() == 'Area:Mouse 1 Center In central area':
#             event_label.append(11)
#         elif i.strip() == 'Area:Mouse 1 Center In corner':
#             event_label.append(12)
            

#         elif i.strip() == 'Mouse 1 sniffing On wall':
#             event_label.append(30)
#         elif i.strip() == 'Mouse 1 Grooming in Area All':
#             event_label.append(31)
#         elif i.strip() == 'Mouse 1 Grooming in Area floor':
#             event_label.append(31)
#         elif i.strip() ==  'Mouse 1 Locomotion in Area All':
#             event_label.append(32)

#         else:
#             event_label.append(90)
                
    return(event_label)


# -

def add_event_label(trace_df, event_path):

    '''
    Aim:
        add event_label to each frame
        one frame can have several label (since mouse can in a location and do something)
    
    Retrun:
        trace dataframe with one label column
    '''
    
    frame_list = process_frame(trace_df)
    #get event dataframe/eventIntervals/eventLabel
#     event_df = pd.read_csv(event_path)
    event_df = pd.read_excel(event_path)
    interval_list = process_event_interval(event_df)
    event_label = get_event_label(event_df)

    
    # add label to each frame
    frame_label_list = []
    for frame in frame_list:
        frame_label = str()
        
        for i in range(len(interval_list)): # one frame can be in several interval
            if min(interval_list[i])  <= frame <= max(interval_list[i]) :
                frame_label = frame_label+ str(event_label[i])

        # convert str to int
        if len(frame_label) == 0:
            frame_label = np.nan
        else:
            frame_label = int(frame_label)

        # append frame label to list
        frame_label_list.append(frame_label)
    
    
    # add label list to trace df 
    new_df = trace_df.copy()
    new_df['Frame_Label']  =  frame_label_list
    new_df = new_df.dropna()
    

    return(new_df)


def get_minMaxForPlot_One(WT_summary):
    
    
    '''
    get necessary info for plot
    used in averageTrace plot
    '''
    
    # wt
    WT = WT_summary['ALL_MICE_MEAN'].values.tolist()
    WT_SEM = WT_summary['MEAD_SEM'].values.tolist()
    WTSEM = WT_summary['MEADSEM'].values.tolist()
    
    vmin = min(WT_SEM)-0.2
    vmin = math.floor(vmin*100)/100
    vmax = max(WTSEM)*1.2
    vmax = math.ceil(vmax*100)/100
    
    
    x = WT_summary.index.values.tolist()
    
    return(x, WT, WT_SEM, WTSEM, vmin, vmax)

def find_compositeIntervals(df,
                            which_status_to_check,
                            intervalLength=20,
                            if_CombineAdjacentChunks=True,
                            gapMax=10):
    
    '''
    Aim:
        a mouse might get into a location/behavior several time during a whole session. this function is to 
        find all intervals that the mouse gets into certain state.
    
    Arguments:
        - df: neuron activity df with status label for each frame (1-in state)
        - which_status_to_check: behavior/location (choose from center/corner/sniffing/grooming)
        - intervalLength: minimum size of state-interval, that is the interval must longer than #_frames, otherwise, this interval will be dumped
        - if_CombineAdjacentChunks: if merge 2 neighbours
        - gapMax: if gap between two chunks less than gapMax, merge the two
    
    Return:
        - kept chunks, sublists in list
    '''
    
    
    df_status = check_frame_interval(df, which_status_to_check, ifKeep=True)
    # get status list
    status_list = df_status['status'].values.tolist()
    # get 1 index
    index_1_list = [i for i, x in enumerate(status_list) if x == 1]
    
    # pick all consecutive chunks
    chunks = []
    try:
        for k, g in groupby(enumerate(index_1_list), lambda ix:ix[0]-ix[1]):
            chunk = list((map(itemgetter(1), g)))
            chunks.append([chunk[0], chunk[-1]])

        # combine small chunks
        if if_CombineAdjacentChunks:
            combinedchunks = []
            combinedchunks.append(chunks[0]) # initialize with first chunk
            #print('starting point', combinedchunks)
            combinedIndex = 0
            for chunkIndex in range(1, len(chunks)):
                chunkPre = combinedchunks[combinedIndex]
                chunkLat = chunks[chunkIndex]
                #print('1st round, pre:',chunkPre, ' lat:',chunkLat, ', diff=', chunkLat[0]-chunkPre[1] )

                if chunkLat[0]-chunkPre[1] <= gapMax:
                    del combinedchunks[combinedIndex]
                    #print('after del 0,', combinedchunks)
                    combinedchunk = chunkPre+chunkLat
                    #print('combined chunk is', combinedchunk)
                    combinedchunks.append([combinedchunk[0], combinedchunk[-1]])
                else:
                    combinedchunks.append(chunkLat)
                    combinedIndex = combinedIndex + 1 
        else:
            combinedchunks = chunks

        # remove small chunks
        keep_combinedchunks = []
        for chunk in combinedchunks:
            if chunk[1]-chunk[0] >= intervalLength:
                keep_combinedchunks.append(chunk)
    
    except Exception as e:
        keep_combinedchunks = chunks
        print('  ', e)

    return (keep_combinedchunks)


def normBA(df,df_B_list,df_A_list,beforeB=False,baseSecond=2):
    
    '''
    Aim:
        normalize trace based on some frames before/start of the trace
        
    Arguments:
        - df: trace df
        - df_B_list: dataframe list BEFORE behavior/location onset
        - df_A_list: dataframe list AFTER behavior/location onset
        - beforeB: if use a period before df_B to calculate baseline. 
                   If True, than use baseSecond before de_B to calculate baselinse, otherwise use the entire df_B as baseline
        - baseSecond: how long before df_B are used to calculate baseline
        
    Return:
        - normalized df_B_list and df_A_list
        
    '''
    
    
    df_BN_list = []; df_AN_list = []
    if len(df_B_list)!=0 and len(df_A_list) !=0:
        
        for dfIndex in range(len(df_B_list)):
            dfB = df_B_list[dfIndex]
            dfA = df_A_list[dfIndex]
            
            # find baseline
            if beforeB:
                dfBM = df.loc[(dfB.index[0]-baseSecond):dfB.index[0]]
            else:
                dfBM = dfB.copy()
            dfBM.loc['mean'] = dfBM.mean(axis=0)
            
            # norm first/second half based on baseline
            dfBN = dfB-dfBM.loc['mean']
            dfAN = dfA-dfBM.loc['mean']

            df_BN_list.append(dfBN)
            df_AN_list.append(dfAN)
        
    
    return (df_BN_list, df_AN_list)


def get_BonsetA(df,
                which_status_to_check,
                fps,
                secForward=10,
                secBackward=10,
                beforeB=False,baseSecond=2,intervalLength=20,if_CombineAdjacentChunks=True,gapMax=10,ifNorm=True):
    
    '''
    Aim:
        Through function find_compositeIntervals() we locate interval that mouse was in certain state.
        By calling this function, we enlarge each interval forward #secForward seconds and make it last #secBackward seconds
        For example (1 interval C),  C=[second_10, second_25] or C=[second_10, second_15].
        By calling this function, we make C=[second_8, second_20]
        
    Arguments:
        df: dff
        which_status_to_check: the behavior of onset
        secForward: seconds to enlarge forward from onset
        secBackward: seconds to include backward from onset
        other arguments are either from function find_compositeIntervals() or normBA(), check there for more details
        
    '''
    
    
    df_status = check_frame_interval(df, which_status_to_check, ifKeep=True)
    combinedchunks = find_compositeIntervals(df,which_status_to_check,intervalLength=intervalLength,
                                             if_CombineAdjacentChunks=if_CombineAdjacentChunks,gapMax=gapMax)
    
    
    df_Bonset_list = []
    df_onsetA_list = []
    if len(combinedchunks) > 0:
    
        # enlargy each interval 100 frames (50 forward, 50 afterward)
        for interval in combinedchunks:
            if interval[0] >= secForward*fps:
                Bonset_start = interval[0] - secForward*fps
                Bonset_end = interval[0]
                onsetA_start = interval[0]
                if interval[0] + secBackward*fps < len(df):
                    onsetA_end = interval[0] + secBackward*fps 
                
                
                    #----------------locate----------------#
                    Bonset_df = df_status.iloc[Bonset_start:Bonset_end]
                    Bonset_df = Bonset_df.drop('status', axis = 1)
                    df_Bonset_list.append(Bonset_df)

                    onsetA_df = df_status.iloc[onsetA_start:onsetA_end]
                    onsetA_df = onsetA_df.drop('status', axis = 1)
                    df_onsetA_list.append(onsetA_df)
        
    # norm
    if ifNorm:
        df_Bonset_listN, df_onsetA_listN = normBA(df_status,df_Bonset_list,df_onsetA_list,beforeB=beforeB,baseSecond=baseSecond)
    else:
        df_Bonset_listN, df_onsetA_listN = df_Bonset_list,df_onsetA_list
        
    
    return (df_Bonset_listN, df_onsetA_listN)


def check_frame_interval(df, which_status_to_check_frame, ifKeep=True):
#     set_trace()
    # set up which_status_to_check_frame
    
    if which_status_to_check_frame == 'free_allogroom':
        which_status_to_check_frame = str(11)
    elif which_status_to_check_frame == 'sniff':
        which_status_to_check_frame = str(12)
    elif which_status_to_check_frame == 'rear_selfgroom':
        which_status_to_check_frame = str(21)
    elif which_status_to_check_frame == 'all':
        which_status_to_check_frame = str(22)
    else:
        raise ValueError('Not approved status. choose from: Investigating_RIGHT_SNIFF, Investigating_LEFT_SNIFF, In_right, In_all, In_left')

    #df = add_event_label(df, event_1_path, event_2_path, event_3_path, mice_path)[which_session]
    frame_label_list = df['Frame_Label'].values.tolist()
    
    # to separate status label, eg:20214111
    n = 2
    
    check_frame_label_list = []
    
    for line in frame_label_list:
        line = str(line)
        line_list= [line[i:i+n] for i in range(0, len(line), n)]
        if which_status_to_check_frame in line_list:
            check_frame_label_list.append(1)
        else:
            check_frame_label_list.append(0)       
            
            
    df_status = df.copy()
    df_status['status'] = check_frame_label_list

    if ifKeep:
        df_status = df_status.drop('Frame_Label', axis = 1)    # check frame interval before and after the event onset!
        print('the shape of the check frame interval before and after onset is:', df_status.shape)
        return(df_status)
    else:
        df_keep = df_status[(df_status['status'] == 1)]  # check frame interval within event period only!
        df_keep = df_keep.drop('status', axis = 1)
        df_keep = df_keep.drop('Frame_Label', axis = 1)

        print('the shape of the check frame interval only is:', df_keep.shape)
        return(df_keep)
            
    #df['status'] = check_frame_label_list
    #df_keep = df[ (df['status'] == 1)]
#     df_keep = df_keep.drop('status', axis = 1)
#     df_keep = df_keep.drop('Frame_Label', axis = 1)
    
    #print('the shape of the dataframe in the status is:', df_keep.shape)
    #return(df_keep)

       
# -

def add_event_label(trace_df, event_path):

    '''
    Aim:
        add event_label to each frame
        one frame can have several label (since mouse can in a location and do something)
    
    Retrun:
        trace dataframe with one label column
    '''
    
    frame_list = process_frame(trace_df)
    #get event dataframe/eventIntervals/eventLabel
#     event_df = pd.read_csv(event_path)
    event_df = pd.read_excel(event_path)
    interval_list = process_event_interval(event_df)
    event_label = get_event_label(event_df)

    
    # add label to each frame
    frame_label_list = []
    for frame in frame_list:
        frame_label = str()
        
        for i in range(len(interval_list)): # one frame can be in several interval
            if min(interval_list[i])  <= frame <= max(interval_list[i]) :
                frame_label = frame_label+ str(event_label[i])

        # convert str to int
        if len(frame_label) == 0:
            frame_label = np.nan
        else:
            frame_label = int(frame_label)

        # append frame label to list
        frame_label_list.append(frame_label)
    
    
    # add label list to trace df 
    new_df = trace_df.copy()
    new_df['Frame_Label']  =  frame_label_list
    new_df = new_df.dropna()
    

    return(new_df)




def find_compositeIntervals(df,
                            which_status_to_check,
                            intervalLength=20,
                            if_CombineAdjacentChunks=True,
                            gapMax=10):
    
    '''
    Aim:
        a mouse might get into a location/behavior several time during a whole session. this function is to 
        find all intervals that the mouse gets into certain state.
    
    Arguments:
        - df: neuron activity df with status label for each frame (1-in state)
        - which_status_to_check: behavior/location (choose from center/corner/sniffing/grooming)
        - intervalLength: minimum size of state-interval, that is the interval must longer than #_frames, otherwise, this interval will be dumped
        - if_CombineAdjacentChunks: if merge 2 neighbours
        - gapMax: if gap between two chunks less than gapMax, merge the two
    
    Return:
        - kept chunks, sublists in list
    '''
    
    
    df_status = check_frame_interval(df, which_status_to_check, ifKeep=True)
    # get status list
    status_list = df_status['status'].values.tolist()
    # get 1 index
    index_1_list = [i for i, x in enumerate(status_list) if x == 1]
    
    # pick all consecutive chunks
    chunks = []
    try:
        for k, g in groupby(enumerate(index_1_list), lambda ix:ix[0]-ix[1]):
            chunk = list((map(itemgetter(1), g)))
            chunks.append([chunk[0], chunk[-1]])

        # combine small chunks
        if if_CombineAdjacentChunks:
            combinedchunks = []
            combinedchunks.append(chunks[0]) # initialize with first chunk
            #print('starting point', combinedchunks)
            combinedIndex = 0
            for chunkIndex in range(1, len(chunks)):
                chunkPre = combinedchunks[combinedIndex]
                chunkLat = chunks[chunkIndex]
                #print('1st round, pre:',chunkPre, ' lat:',chunkLat, ', diff=', chunkLat[0]-chunkPre[1] )

                if chunkLat[0]-chunkPre[1] <= gapMax:
                    del combinedchunks[combinedIndex]
                    #print('after del 0,', combinedchunks)
                    combinedchunk = chunkPre+chunkLat
                    #print('combined chunk is', combinedchunk)
                    combinedchunks.append([combinedchunk[0], combinedchunk[-1]])
                else:
                    combinedchunks.append(chunkLat)
                    combinedIndex = combinedIndex + 1 
        else:
            combinedchunks = chunks

        # remove small chunks
        keep_combinedchunks = []
        for chunk in combinedchunks:
            if chunk[1]-chunk[0] >= intervalLength:
                keep_combinedchunks.append(chunk)
        
        print(keep_combinedchunks)
    
    except Exception as e:
        keep_combinedchunks = chunks
        print('  ', e)
        print("no results")

    return (keep_combinedchunks)
    


def normBA(df,df_B_list,df_A_list,beforeB=False,baseSecond=2):
    
    '''
    Aim:
        normalize trace based on some frames before/start of the trace
        
    Arguments:
        - df: trace df
        - df_B_list: dataframe list BEFORE behavior/location onset
        - df_A_list: dataframe list AFTER behavior/location onset
        - beforeB: if use a period before df_B to calculate baseline. 
                   If True, than use baseSecond before de_B to calculate baselinse, otherwise use the entire df_B as baseline
        - baseSecond: how long before df_B are used to calculate baseline
        
    Return:
        - normalized df_B_list and df_A_list
        
    '''
    
    
    df_BN_list = []; df_AN_list = []
    if len(df_B_list)!=0 and len(df_A_list) !=0:
        
        for dfIndex in range(len(df_B_list)):
            dfB = df_B_list[dfIndex]
            dfA = df_A_list[dfIndex]
            
            # find baseline
            if beforeB:
                dfBM = df.loc[(dfB.index[0]-baseSecond):dfB.index[0]]
            else:
                dfBM = dfB.copy()
            dfBM.loc['mean'] = dfBM.mean(axis=0)
            
            # norm first/second half based on baseline
            dfBN = dfB-dfBM.loc['mean']
            dfAN = dfA-dfBM.loc['mean']

            df_BN_list.append(dfBN)
            df_AN_list.append(dfAN)
        
    
    return (df_BN_list, df_AN_list)


def get_BonsetA(df,
                which_status_to_check,
                fps,
                secForward=10,
                secBackward=10,
                beforeB=False,baseSecond=2,intervalLength=20,if_CombineAdjacentChunks=True,gapMax=10,ifNorm=True):
    
    '''
    Aim:
        Through function find_compositeIntervals() we locate interval that mouse was in certain state.
        By calling this function, we enlarge each interval forward #secForward seconds and make it last #secBackward seconds
        For example (1 interval C),  C=[second_10, second_25] or C=[second_10, second_15].
        By calling this function, we make C=[second_8, second_20]
        
    Arguments:
        df: dff
        which_status_to_check: the behavior of onset
        secForward: seconds to enlarge forward from onset
        secBackward: seconds to include backward from onset
        other arguments are either from function find_compositeIntervals() or normBA(), check there for more details
        
    '''
    
    
    df_status = check_frame_interval(df, which_status_to_check, ifKeep=True)
    combinedchunks = find_compositeIntervals(df,which_status_to_check,intervalLength=intervalLength,
                                             if_CombineAdjacentChunks=if_CombineAdjacentChunks,gapMax=gapMax)
    
    
    df_Bonset_list = []
    df_onsetA_list = []
    if len(combinedchunks) > 0:
    
        # enlargy each interval 100 frames (50 forward, 50 afterward)
        for interval in combinedchunks:
            if interval[0] >= secForward*fps:
                Bonset_start = interval[0] - secForward*fps
                Bonset_end = interval[0]
                onsetA_start = interval[0]
                if interval[0] + secBackward*fps < len(df):
                    onsetA_end = interval[0] + secBackward*fps 
                
                
                    #----------------locate----------------#
                    Bonset_df = df_status.iloc[Bonset_start:Bonset_end]
                    Bonset_df = Bonset_df.drop('status', axis = 1)
                    df_Bonset_list.append(Bonset_df)

                    onsetA_df = df_status.iloc[onsetA_start:onsetA_end]
                    onsetA_df = onsetA_df.drop('status', axis = 1)
                    df_onsetA_list.append(onsetA_df)
        
    # norm
    if ifNorm:
        df_Bonset_listN, df_onsetA_listN = normBA(df_status,df_Bonset_list,df_onsetA_list,beforeB=beforeB,baseSecond=baseSecond)
    else:
        df_Bonset_listN, df_onsetA_listN = df_Bonset_list,df_onsetA_list
        
    
    return (df_Bonset_listN, df_onsetA_listN)





def get_chunksInDF(keep_combinedChunks, df_frame_label):

    chunk_list = []
    for chunk in keep_combinedChunks:
        chunk_list.append(df_frame_label.iloc[chunk[0]:chunk[1]])


    focus_df = pd.concat(chunk_list)
    return(focus_df)

def calculate_avg_FF(df_list,fps,seconds):
    
    '''
    Aim:
        get average of all entries from all neurons around onset
    
    Steps:
        - one df in df_list with shape [num_frames, num_neurons]
        - get mean of all neurons with shape [num_frames,] for one df, append to Mean_DF
        - get mean of [df1'mean, df2'mean, ...], append to Mean_DF

    Return:
        a dataframe with all df's mean and MEAN of all df's-means
    '''
    
    frames = seconds * fps
    
    
    Mean_DF = pd.DataFrame()
    if len(df_list) > 0:
        for i in range(len(df_list)):
            if df_list[i].shape[0] == frames:
                Mean_DF['mean'+str(i)] = df_list[i].mean(axis=1).values.tolist()
        Mean_DF['MEAN'] = Mean_DF.mean(axis=1)
    
    return(Mean_DF)





def deconvolve_df(df, ifSave=True, savePath=None, baseName=None):
    
    '''
    Aim:
        deconvolute trace for each neuron & save
    
    Note:
        used function: https://github.com/j-friedrich/OASIS/blob/master/oasis/functions.py#L110
        c: nferred denoised fluorescence signal at each timebin
        b: fluorescence baseline
        s: spikes

    Return:
        df_denoised(b+c), df_spikes(s)

    '''
    
    
    df_spikes = df.copy(); df_denoised = df.copy()
    columns = df.columns
    Frames, n_neuron = df.shape

    # start deconvolution
    if Frames != 0 and n_neuron != 0:

        # get dfarray
        df_array = df.to_numpy()

        # get decovoluted trace for each neuron
        for i in range(n_neuron):
            y = df_array[:,i][~np.isnan(df_array[:,i])]
            c, s, b, g, lam =deconvolve(y, penalty=1)
            df_denoised[columns[i]] = b+c
            df_spikes[columns[i]] = s
        
        # save to path
        if ifSave:
            df_denoised.to_csv(os.path.join(savePath, baseName+'-D.csv'))
            df_spikes.to_csv(os.path.join(savePath, baseName+'-S.csv'))
        
        return (df_denoised, df_spikes)
            
    else:
        raise Exception("Empty DataFrame!")
        
        
               

def mixMaxScale_dff(df, feature_range=(0, 1)):
    
    '''
    Aim:
        Transform features by scaling each feature to a given range using MinMaxScaler()
        note that the input dataframe should be in shape [num_frames, num_neurons]
    '''
    
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(df.copy())
    
    dfN = pd.DataFrame(scaler.transform(df))
    dfN.columns = df.columns
    dfN.index = df.index
    
    return(dfN)    



# +
def average_nActEntries(df_list,neuron_list,fps,seconds):
    
    '''
    Aim:
        get average activity around onset for each neuron across entries
    
    Return:
        a dataframe with all mean activity of each neuron [num_frames, num_neurons]
    '''
    
    frames = seconds*fps
    avgNeuron_actEntries = pd.DataFrame()
    if len(df_list) > 0:
        for nCol in neuron_list:

            # get mean activity of this neuron across entries]
            nAct_entries = []
            for i in range(len(df_list)):
                if df_list[i].shape[0] == frames:
                    # nAct_entries.append(df_list[i][nCol])
                    nAct_entries.append(df_list[i][nCol].astype(float))
                    
#             set_trace()
            nAct_mean = np.mean(np.array(nAct_entries), axis=0)
            avgNeuron_actEntries[nCol] =  nAct_mean
    
    
    return(avgNeuron_actEntries)
# -






def filterOut_nonceONce(df, fps, secF):
    
    '''
    remove neurons that not showing ceON neurons around center
    specificly designed for 3-AverageTrace_4ONs_heatmap
    
    '''
    
    df_keep = df.copy()
    f, n = df.shape

    for i in range(n):
        act = df.iloc[:, i]
        Bavg = np.mean(act[:fps*secF])
        avgA = np.mean(act[fps*secF:fps*secF*3])

        if avgA<Bavg:
            df_keep = df_keep.drop(df.columns[i], axis=1)
            
    return(df_keep)



def filterOut_nonceONco(df, fps, secF):
    
    '''
    remove neurons that not showing ceON neurons around corner (drop onset)
    specificly designed for 3-AverageTrace_4ONs_heatmap
    
    '''
    
    df_keep = df.copy()
    f, n = df.shape

    for i in range(n):
        act = df.iloc[:, i]
        Bavg = np.mean(act[:fps*secF])
        avgA = np.mean(act[fps*secF:fps*secF*3])

        if avgA>Bavg:
            df_keep = df_keep.drop(df.columns[i], axis=1)
            
    return(df_keep)



def get_overlap(lists):
    ls=[set(l) for l in lists]
    fst=ls[0]
    for l in ls[1:]:
        fst.intersection_update(l)
    return (len(fst), fst)



def fit_apply_PCA(input_df_Std, ifPrint=True):
    '''
    Aim:
        apply PCA
    
    '''
    pcaModel = PCA(n_components=4)
    input_Reduced_Std = pcaModel.fit_transform(input_df_Std)

    if ifPrint:
        print('first 3PCs explain', round(sum((pcaModel.explained_variance_ratio_[0:3])),3), \
                'PC1='+str(round(pcaModel.explained_variance_ratio_[0],3))+\
                ' PC2='+str(round(pcaModel.explained_variance_ratio_[1],3))+\
                ' PC3='+str(round(pcaModel.explained_variance_ratio_[2],3)))
        #print('explained_variance_ratio_', pcaModel.explained_variance_ratio_)
    
    # reduced df
    input_Reduced_df = pd.DataFrame(input_Reduced_Std)
    input_Reduced_df.columns = ['PC1', 'PC2', 'PC3', 'PC4']

    return(input_Reduced_df, pcaModel)

def generate_PCA_summaryDF(reduced_df, pcaModel, behaviorList, mouseName):
    '''
    Aim:
        generate PCA summary, 4 PCs
        eigenvector=features, eigenvalues=coefficient, mean=X.mean
        also generate df for running log, just explained variance ratio
    
    '''
    # PC summary
    #----------------------------------------------------
    dic_eigen = {'Eigenvector 1':pcaModel.components_[0], 
                 'Eigenvector 2':pcaModel.components_[1],
                 'Eigenvector 3':pcaModel.components_[2],
                 'Eigenvector 4':pcaModel.components_[3],
                 'Sample Mean':pcaModel.mean_,
                 'Eigenvalues':pcaModel.explained_variance_, 
                 'Explained Ratio':pcaModel.explained_variance_ratio_,
                 'singular Values':pcaModel.singular_values_,
                 'behavior':behaviorList}
    dic_eigen = dict([(k,pd.Series(v)) for k,v in dic_eigen.items() ])
    df_eigen = pd.DataFrame(dic_eigen)
    # concat PCs
    PCA_summary = pd.concat([df_eigen, reduced_df], axis=1)
    
    # running Log
    #----------------------------------------------------
    runningLog = pcaModel.explained_variance_ratio_
    runningLog_dic = {'mouseName':mouseName,
                      'first_3PCs': sum(runningLog),
                      'PC1':runningLog[0],
                      'PC2': runningLog[1],
                      'PC3':runningLog[2]}
    runningLog_dic = dict([(k,pd.Series(v)) for k,v in runningLog_dic.items() ])
    runningLog_df = pd.DataFrame(runningLog_dic)

    
    return(PCA_summary, runningLog_df)





# +
def average_nActEntries(df_list,neuron_list,fps,seconds):
    
    '''
    Aim:
        get average activity around onset for each neuron across entries
    
    Return:
        a dataframe with all mean activity of each neuron [num_frames, num_neurons]
    '''
    
    frames = seconds*fps
    avgNeuron_actEntries = pd.DataFrame()
    if len(df_list) > 0:
        for nCol in neuron_list:

            # get mean activity of this neuron across entries]
            nAct_entries = []
            for i in range(len(df_list)):
                if df_list[i].shape[0] == frames:
                    # nAct_entries.append(df_list[i][nCol])
                    nAct_entries.append(df_list[i][nCol].astype(float))
                    
#             set_trace()
            nAct_mean = np.mean(np.array(nAct_entries), axis=0)
            avgNeuron_actEntries[nCol] =  nAct_mean
    
    
    return(avgNeuron_actEntries)
# -



def randomSelect_avg(df_toSelect, numFrames, repeatNum=50):
    
    '''
    get averaged df of repeated-randomly-selected frames
    
    '''
    randomMean_DF = pd.DataFrame()
    for i in range(repeatNum):
        a_sample = df_toSelect.sample(n=numFrames).reset_index(drop=True)
        randomMean_DF = randomMean_DF.add(a_sample, fill_value=0)
    
    return(randomMean_DF/repeatNum)

def getSummary(df_toGenerateSummary):
    
    '''
    Aim:
        generate summary for each behavior, including, mean, mean-sem, mean+sem, sem, max, min
    
    '''
    
    
    Summary = pd.DataFrame()
    Summary['ALL_MICE_MEAN'] = df_toGenerateSummary.mean(axis = 1)
    Summary['MEAD_SEM'] = df_toGenerateSummary.mean(axis = 1) - df_toGenerateSummary.sem(axis = 1) 
    Summary['MEADSEM'] = df_toGenerateSummary.mean(axis = 1) + df_toGenerateSummary.sem(axis = 1) 
    Summary['SEM'] = df_toGenerateSummary.sem(axis = 1)
    Summary['MAX'] = df_toGenerateSummary.max(axis = 1)
    Summary['MIN'] = df_toGenerateSummary.min(axis = 1)
    
    
    return(Summary)


def getList(dict): 
      return [*dict]
    
def reorder_df(df,ifReverse=False):
    '''
    Aim:
        Reorder dataframe by neruon firing happending frame
        The resulting heatmap will show a diagonal-like firing trend 
    
    Note:
        the input dataframe should be in shape (#_frames, #_neurons)
    '''

    frame_list = df.T.columns.tolist()
    neuron_columns = df.columns.tolist()

    i = 0
    firing_neuorn_index  = {}
    for row in df.T.values.tolist():
        # get firing point of the cell trace
        firing = max(row)

        # get the index of the cell trace
        index = row.index(firing)
        firing_neuorn_index[neuron_columns[i]] = index

        i = i+1

    #reorder neuorn by firing point index
    firing_neuorn_index_sorted = {k: v for k, v in sorted(firing_neuorn_index.items(), key=lambda item: item[1],reverse=ifReverse)}

    
    # keys of the dic
    new_neuorn_columns = getList(firing_neuorn_index_sorted)
    #reorder the dataframe
    df_reorder = df[new_neuorn_columns]

    return(df_reorder)




def filterOut_nonceONce(df, fps, secF):
    
    '''
    remove neurons that not showing ceON neurons around center
    specificly designed for 3-AverageTrace_4ONs_heatmap
    
    '''
    
    df_keep = df.copy()
    f, n = df.shape

    for i in range(n):
        act = df.iloc[:, i]
        Bavg = np.mean(act[:fps*secF])
        avgA = np.mean(act[fps*secF:fps*secF*3])

        if avgA<Bavg:
            df_keep = df_keep.drop(df.columns[i], axis=1)
            
    return(df_keep)



def filterOut_nonceONco(df, fps, secF):
    
    '''
    remove neurons that not showing ceON neurons around corner (drop onset)
    specificly designed for 3-AverageTrace_4ONs_heatmap
    
    '''
    
    df_keep = df.copy()
    f, n = df.shape

    for i in range(n):
        act = df.iloc[:, i]
        Bavg = np.mean(act[:fps*secF])
        avgA = np.mean(act[fps*secF:fps*secF*3])

        if avgA>Bavg:
            df_keep = df_keep.drop(df.columns[i], axis=1)
            
    return(df_keep)



def get_overlap(lists):
    ls=[set(l) for l in lists]
    fst=ls[0]
    for l in ls[1:]:
        fst.intersection_update(l)
    return (len(fst), fst)


def get_contour(one_neuron_path, thres=10):
    
    '''
    get contour for each neuron through 'C*.tiff'
    '''

    # read an image in the currect TIFF directory as a numpy array
    img = tiff.imread(one_neuron_path)

    # set up new file path
    new_file=one_neuron_path[:-5]+'.jpg'

    # save image to a specified file, (filename, img to ve saved)
    cv2.imwrite(new_file, img)

    # re-read the image using cv2
    img = cv2.imread(new_file)

    # convert BGR to Gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold explain
    # find the most shining neuron
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    ret , binary = cv2.threshold (gray , thres , 255 , cv2.THRESH_BINARY)

    # detect contour
    # for better accuracy, use binary image
    # finding contours is like finding white object from black background - gray scale
    # RETR_EXTERNAL - returns only extreme outer flags
    # CHAIN_APPROX_SIMPLE - only save necessary points to represent the line
    # https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html
    contours , hierarchy = cv2.findContours(binary , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # delete new_file in the folder
    os.remove(new_file)
    
    return(contours)

def get_index(ONOFF, trace_df, neuronType='Investigating_RIGHT_SNIFF_ON'): 
     
        ####################################### Which neuron type?
    '''
    get neuron index in all neuron list, which is used locate corresponding contour in contour_list
    
    '''

    # get all neuron list 
    all_neurons = trace_df.columns.tolist()
    all_neurons = [neuron.strip() for neuron in all_neurons]
    ONNeurons = ONOFF[ONOFF[neuronType] == 1].index.tolist()
    
    # get neuronType neuron index list
    nIndex_list = []
    for n in ONNeurons:
        index = all_neurons.index(n)
        nIndex_list.append(index)
            
        
    return(nIndex_list)



def Std(dataframe, numEachFeature=771):
    '''
    The input shape is #_frames * #_neurons
    The standardization should be for each neuron
    
    '''
    values = np.array(dataframe.T.values)

    if int(len(values[0])) !=numEachFeature:
        raise ValueError('wrong input dataframe structure, rows=frames, columns=features')
        dataframe_std=None
    else:
        values_norm = [(x-np.mean(x))/np.std(x) for x in values] #len(x)=#frames
        dataframe_std = pd.DataFrame(values_norm).T
        dataframe_std.columns = dataframe.columns
        dataframe_std.index = dataframe.index
    
    
    return(dataframe_std)



def fit_apply_PCA(input_df_Std, ifPrint=True):
    '''
    Aim:
        apply PCA
    
    '''
    pcaModel = PCA(n_components=4)
    input_Reduced_Std = pcaModel.fit_transform(input_df_Std)

    if ifPrint:
        print('first 3PCs explain', round(sum((pcaModel.explained_variance_ratio_[0:3])),3), \
                'PC1='+str(round(pcaModel.explained_variance_ratio_[0],3))+\
                ' PC2='+str(round(pcaModel.explained_variance_ratio_[1],3))+\
                ' PC3='+str(round(pcaModel.explained_variance_ratio_[2],3)))
        #print('explained_variance_ratio_', pcaModel.explained_variance_ratio_)
    
    # reduced df
    input_Reduced_df = pd.DataFrame(input_Reduced_Std)
    input_Reduced_df.columns = ['PC1', 'PC2', 'PC3', 'PC4']

    return(input_Reduced_df, pcaModel)

def generate_PCA_summaryDF(reduced_df, pcaModel, behaviorList, mouseName):
    '''
    Aim:
        generate PCA summary, 4 PCs
        eigenvector=features, eigenvalues=coefficient, mean=X.mean
        also generate df for running log, just explained variance ratio
    
    '''
    # PC summary
    #----------------------------------------------------
    dic_eigen = {'Eigenvector 1':pcaModel.components_[0], 
                 'Eigenvector 2':pcaModel.components_[1],
                 'Eigenvector 3':pcaModel.components_[2],
                 'Eigenvector 4':pcaModel.components_[3],
                 'Sample Mean':pcaModel.mean_,
                 'Eigenvalues':pcaModel.explained_variance_, 
                 'Explained Ratio':pcaModel.explained_variance_ratio_,
                 'singular Values':pcaModel.singular_values_,
                 'behavior':behaviorList}
    dic_eigen = dict([(k,pd.Series(v)) for k,v in dic_eigen.items() ])
    df_eigen = pd.DataFrame(dic_eigen)
    # concat PCs
    PCA_summary = pd.concat([df_eigen, reduced_df], axis=1)
    
    # running Log
    #----------------------------------------------------
    runningLog = pcaModel.explained_variance_ratio_
    runningLog_dic = {'mouseName':mouseName,
                      'first_3PCs': sum(runningLog),
                      'PC1':runningLog[0],
                      'PC2': runningLog[1],
                      'PC3':runningLog[2]}
    runningLog_dic = dict([(k,pd.Series(v)) for k,v in runningLog_dic.items() ])
    runningLog_df = pd.DataFrame(runningLog_dic)

    
    return(PCA_summary, runningLog_df)



def get_ONNeurons(root_path, ROI, project, mouse_summary, statusToCheck, otherStatus, fileName):

    '''
    read in OFT ON/OFF summary dataframe
    '''

    file_name = project+fileName
    
    on_off_df = pd.read_csv( os.path.join(root_path, ROI, project, mouse_summary, file_name)) ### input pathway!!!
    columns = on_off_df.columns.to_list()
    columns[0] = 'Neurons'
    on_off_df.columns = columns
    on_off_df = on_off_df.set_index('Neurons')
    
    # get center/corner on neurons
    statusON_list = on_off_df[(on_off_df[statusToCheck] == 1)&(on_off_df[otherStatus] != 1)].index.to_list()
    
    return(statusON_list)

def read_sameNeuron(fileName, root_path, ROI, mouse_name):
    '''
    read in same neuron summary
    '''
    sameNeuron_summary = pd.read_excel(os.path.join(root_path, ROI, fileName)).iloc[:, 1:]   ### input pathway!!!
    this_mouse = sameNeuron_summary[sameNeuron_summary['mouse_name']==mouse_name]
    
    return(this_mouse)



def ASAM(df_frame_label, coordinate, savePath, fps):
    
    '''
    Aim:
        Get neuron averaged neuron activity on each spot, since the mouse might visit the spot several times
        Also record the frequency of visiting that spot
    
    Note 
        - one might encounter xy like (-1, -1) which is also included in the output
        - at this step, nothing got deleted
        - there is no frame-time-line here, since time visiting one spot got averaged

    '''
    
    coor, coor_x, coor_y, center = get_coordinate(coordinate, df_frame_label, fps)
    
    # add x and y to dataframe
    df_Coor = df_frame_label.head(len(coor))
    df_Coor['X'] = coor_x
    df_Coor['Y'] = coor_y
    
    # get coor set
    coor_uni = np.unique(coor, axis=0) # the mouse might visit same spot several times
    
    # calculate norm activity for each neuron in each location
    df_xy_neurons_norm_activity_list = []
    for x_y in coor_uni:
        df_xy = df_Coor[(df_Coor['X'] == x_y[0]) & (df_Coor['Y'] == x_y[1])]

        # get df only with neurons, and calculate norm neuron activity
        df_xy_neurons = df_xy.iloc[:, :-3]
        df_xy_neurons.loc['norm_act'] = df_xy_neurons.sum(axis = 0)/(df_xy_neurons.shape[0])
        df_xy_neurons['X'] = x_y[0]
        df_xy_neurons['Y'] = x_y[1]
        df_xy_neurons['Frequency'] = df_xy_neurons.shape[0]

        # only keep norm activity for each neuron in that location
        df_xy_neurons_norm_activity = df_xy_neurons.iloc[[-1]]
        df_xy_neurons_norm_activity_list.append(df_xy_neurons_norm_activity)

    
    df_norm_act_all_location = pd.concat(df_xy_neurons_norm_activity_list)
    # remove any negative coordinates
    df_norm_act_all_location = df_norm_act_all_location[(df_norm_act_all_location['X']>0) & (df_norm_act_all_location['Y']>0)]
    df_norm_act_all_location = df_norm_act_all_location.reset_index(drop=True)
    df_norm_act_all_location.to_csv(savePath)
    
    
    return(df_norm_act_all_location)

def get_coordinate(corrdinate_file_path, df, fps):
    
    '''
    Aim:
        get coordinates from mouse location file
        Note that the behavior recording fps ~ 30, need to translate between behavior recording and neuron recording
        That is to downsampling behavior
    
    Return:
        coor, x, y, center
    '''
    
    rawdata = []
    with open(corrdinate_file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 7:
                videoFPS = float(line.split(':')[1])
            if i == 9:
                start_frame = int(line.split(':')[1])
            if i == 20:
                temp = line.split('\t')
                left, right, top, bottom = float(temp[0].split(':')[2]), float(temp[1].split(':')[1]), \
                                           float(temp[2].split(':')[1]), float(temp[3].split(':')[1].split('\n')[0])
            lines = line.replace('"', '').replace("'", "").split('   ')
            try:
                rawdata.append([int(lines[0]), int(lines[1]), int(lines[2]),
                                float(lines[5]), float(lines[6]), float(lines[9])])
                # 0-frame number; 1-center X; 2-center Y; 5-motion; 6-Orientation; 9-TurningAngleSum
            except:
                pass
    rawdata = np.array(rawdata)


    center = [(left + right) / 2, (top + bottom) / 2]
    a = np.asarray(np.arange(df.shape[0])/fps * videoFPS + (rawdata[0, 0]-start_frame))
    # just in case the recording is not to 600s
    new_a = a[(a <= len(rawdata))]
    coordinates = rawdata[np.floor(new_a).astype('int'), 1:3]
    coordinates_x = rawdata[np.floor(new_a).astype('int'), 1]
    coordinates_y = rawdata[np.floor(new_a).astype('int'), 2]
    
    
    return (coordinates, coordinates_x, coordinates_y, center)



def img_array(Norm_Act_df, neuron_index, ifSmooth=False, sigma_x=2, sigma_y=2, 
              PAD=False, padder=None, binned=False, binned_size=None):

    '''
    Aim:
        translate ASAM to image array for later plot
    
    '''

    # remove any negative coordinates
    Norm_Act_df = Norm_Act_df[Norm_Act_df['X']>0]
    Norm_Act_df = Norm_Act_df[Norm_Act_df['Y']>0]
    # get this neuron norm array
    A_Neuron = Norm_Act_df.iloc[:, [-2, -1, neuron_index]]
    A_Neuron_array = A_Neuron.to_numpy()
    
    # get x,y,z ranges
    xmin = A_Neuron_array[:,0].min()
    xmax = A_Neuron_array[:,0].max()
    ymin = A_Neuron_array[:,1].min()
    ymax = A_Neuron_array[:,1].max()
    zmin = A_Neuron_array[:,2].min()
    zmax = A_Neuron_array[:,2].max()

    a_number_to_fill_blank = zmin-2
    
    print('neuron_index: ', neuron_index)
    print('xmin:', xmin,' xmax:', xmax,' ymin:', ymin, ' ymax:', ymax,' neuronActivityMin:', zmin,' neuronActivityMax:', zmax)
    
    # create array for image : zmax+1 is the default value
    shape = (int(xmax-xmin+2),int(ymax-ymin+3))
    # build an image array fill with default value
    img_array = np.array(np.ones(shape)*(a_number_to_fill_blank))  # rows-x, col-y, .T when return
    # fit neuron firing rate
    for inp in A_Neuron_array:
        img_array[int(inp[0]-xmin),int(inp[1]-ymin)]=inp[2]
    print('img_array shape: ', img_array.shape)
        
    # pad
    if PAD:
        # pad the original image with 0 to make it 512
        img_array = np.pad(img_array, 19, pad_with, padder = padder)
        print('img_array_pad shape', img_array.shape)        
        
    # smooth
    if ifSmooth:
        sigma = [sigma_y, sigma_x]
        img_array_sm = sp.ndimage.filters.gaussian_filter(img_array, sigma, mode='constant')
    else:
        img_array_sm = img_array
        
    # bin image
    if binned:
        img_array_sm = bin_ndarray(img_array_sm, binned_size)
    print('smoothed z min', img_array_sm.min(), 'smoothed z max', img_array_sm.max())
    print('--------------------------------------------')
    
    return(img_array.T, img_array_sm.T)

def bin_ndarray(a, shape):
    '''
    bin 2D array, half
    '''

    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def pad_with(vector, pad_width, iaxis, kwargs):
    '''
    pad array around with a number
    '''
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def get_xy_limit(df_list, colX, colY):
    
    '''
    Aim:
        get grid limit
    '''

    DF = pd.concat(df_list)
    return( round(DF[colX].max()+5), round(DF[colX].min()-5), round(DF[colY].max()+5), round(DF[colY].min()-5) )
        
def generate_imgGrid(df_ASAM,neuronIndex,nGrid,xMax,xMin,yMax,yMin):
    
    '''
    Aim:
        generate the normalized grid (n*n) for neuron_paw_spatial_mapping

    Arguments:

        df_ASAM: neuron activity + [x, y] location info
        neuronIndex : which neuron to plot
        nGrid : should be larger than 100

    Explanation:
        plot a grid heatmap of how one single neuron fired at different paw locations
        the numbers along x and y are coordinates of one paw location ('Lx'/'Ly', or, 'Rx'/'Ry')
        Basically:
        - divide the original paw movement panel into 100*100 
        - build an empty square matrix (100*100)
        - average the neuron firing rate of each grid and update the value to matrix
        - update the column and index to coordinates 
        (e.g., before columns=[0, 1, ..., 99] to [22, 25...] which represnet the x direction location)
        - plot heatmap of the matrix (100*100), if smooth, use gaussian smooth

    Note:
        xMax/xMin/yMax/yMin are the limits of paw movement, it can either of a whole day, or one speed, or 2-same-speed
        not sure which makes more sense
    '''


    # prepare
    #------------------------------------------------------------------
    xRange = xMax-xMin
    yRange = yMax-yMin
    xGridStep = xRange/nGrid
    yGridStep = yRange/nGrid
    
    
    # create an empty matrix
    #------------------------------------------------------------------
    gridMatrix = np.zeros((nGrid, nGrid))
    #gridMatrix = np.full((nGrid,nGrid), fill_value=-1)
    # update matrix
    for i in range(nGrid):
        y1=yMin + yGridStep*i
        y2=yMin + yGridStep*(i+1)

        for j in range(nGrid):
            x1=xMin + xGridStep*j
            x2=xMin + xGridStep*(j+1)

            # calculate average firing rate for this grid
            gridIndex = df_ASAM[(df_ASAM['X']<=x2) & (df_ASAM['X']>x1) & (df_ASAM['Y']<=y2) & (df_ASAM['Y']>y1)].index.tolist()
            
            # if the mouse has been there
            if len(gridIndex) > 0:
                gridActivity = df_ASAM.iloc[gridIndex].iloc[:, neuronIndex]
                avgActivity = gridActivity.sum()/len(gridActivity)
                # update value
                gridMatrix[i, j]=avgActivity    
                
                
                
    # reindex/recolumn grid dataframe
    #------------------------------------------------------------------
    pawLocX = [round(xMin + xGridStep*(j+1)) for j in range(nGrid)]
    pawLocY = [round(yMin + yGridStep*(i+1)) for i in range(nGrid)]

    gridDF = pd.DataFrame(gridMatrix)
    gridDF.columns=pawLocX
    gridDF[''] = pawLocY
    gridDF = gridDF.set_index('')
    
    
    return(gridDF)



def get_BonsetAwithLabel(df,which_status_to_check,fps,secForward=5,secBackward=5,
                             intervalLength=20,if_CombineAdjacentChunks=True,gapMax=10):
    
    df_status = check_frame_interval(df, which_status_to_check, ifKeep=True)
    combinedchunks = find_compositeIntervals(df,which_status_to_check,intervalLength=intervalLength,
                                             if_CombineAdjacentChunks=if_CombineAdjacentChunks,gapMax=gapMax)
    
    
    df_Bonset_list = []
    df_onsetA_list = []
    if len(combinedchunks) > 0:
        # enlargy each interval 100 frames (50 forward, 50 afterward)
        for interval in combinedchunks:
            if interval[0] >= secForward*fps:
                Bonset_start = interval[0] - secForward*fps
                Bonset_end = interval[0]
                onsetA_start = interval[0]
                if interval[0] + secBackward*fps < len(df):
                    onsetA_end = interval[0] + secBackward*fps 
                
                
                    #----------------locate----------------#
                    Bonset_df = df_status.iloc[Bonset_start:Bonset_end]
                    Bonset_df = Bonset_df.drop('status', axis = 1)
                    Bonset_df['label'] = 0
                    df_Bonset_list.append(Bonset_df)

                    onsetA_df = df_status.iloc[onsetA_start:onsetA_end]
                    onsetA_df = onsetA_df.drop('status', axis = 1)
                    onsetA_df['label'] = 1
                    df_onsetA_list.append(onsetA_df)
        
        
    if len(df_Bonset_list) > 0:
        df_Bstatus = pd.concat(df_Bonset_list)
        df_statusA = pd.concat(df_onsetA_list)
        df_status = pd.concat([df_Bstatus, df_statusA])
        labels = df_status['label'].values.tolist()
        df_status = df_status.drop('label', axis = 1)
    else:
        df_status = pd.DataFrame(columns = df_status.columns )
        labels = []
        
    return(df_status, labels)

def RF_trainPredictSummary(neuronList, mouseName, trainType, df_train, train_Labels, dfs_toPredict, labels_toVarify):
    
    
    '''
    Aim:
        use 1 df to train RF model, then predict on a list of dfs
        E.g., 
            - use OFT ceON neurons in center to train RF, labels are 0/1s, 0-not in center 1-in center
            - use that RF to predict mouse behavior/location in corner/sniffing/grooming/openP/openNP/closeP/closeNP
            - note that the neurons in training set must be the same with that in dataset to be predicted
    
    Arguments:
        - neuronList: neurons in the training df
        - mouseName: same mouse in OFT/EPM
        - trainType: either 'allNeurons' or 'ceONNeurons' in our case
        - dfs_toPredict: a list of dfs to be predicted on
    
    '''
    
    
    if len(neuronList) > 3: # if any neuron in commen between OFT/EPM
        
        if len(df_train) > 0: # if the mouse spent frames large enough  
            
            # train_test_split
            X_train, X_test, y_train, y_test = train_test_split(df_train.values.tolist(),
                                                                train_Labels, test_size=0.33, random_state=42)
            # RF, fit, test
            clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, criterion = 'entropy')
            clf.fit(X_train, y_train)
            test_accu = clf.score(X_test, y_test)
            
            # feature importance 
            neuron_im_dic = {}
            neuron_im_dic['mouseName'] = [mouseName] * len(neuronList)
            neuron_im_dic[trainType] = neuronList
            neuron_im_dic['neuronImportance'] = clf.feature_importances_
            neuron_im_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in neuron_im_dic.items() ]))

            
            # predict
            accus = []
            for dfIndex in range(len(dfs_toPredict)):
                df = dfs_toPredict[dfIndex]
                label = labels_toVarify[dfIndex]
                
                if len(df) > 0:
                    accu = clf.score(df.values.tolist(), label)
                else:
                    accu = np.nan
                    
                accus.append(accu)

        else:
            test_accu = np.nan
            accus = np.nan
            neuron_im_df = pd.DataFrame()
            
    else:
        test_accu = np.nan
        accus = np.nan
        neuron_im_df = np.nan
                
            
            
    return(test_accu, accus, neuron_im_df)
            






# Plot
# ========================================================================================


def plot_3D_4Behaviors(PCA_df,behaviors=['center', 'corner', 'sniffing', 'grooming'],cs=['r','g','b','black'],
                       alphas=[0.6, 0.1, 0.8, 1], ifSave=True,savePath=None,filename=None):
    '''
    Aim:
        plot single 3D line plot and save
        different behavior, differetn scatter plot
    '''
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    for bIndex in range(len(behaviors)):
        b = behaviors[bIndex]
        x = PCA_df[PCA_df['behavior']==b]['PC1']
        y = PCA_df[PCA_df['behavior']==b]['PC2']
        z = PCA_df[PCA_df['behavior']==b]['PC3']
        ax.scatter(x, y, z, c=cs[bIndex], marker='o', alpha=alphas[bIndex], label=b)
        plt.legend()
    

    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename), dpi=300)
        plt.close()
    else:
        plt.show()

def plot_2D_4Behaviors(PCA_df,
                       behaviors=['center', 'corner', 'sniffing', 'grooming'],
                       cs=['r','g','b','black'],
                       alphas=[0.6, 0.1, 0.8, 1], 
                       ifSave=True,savePath=None,filename=None):
    '''
    Aim:
        plot single 3D line plot and save
        different behavior, differetn scatter plot
    '''
    
    fig = plt.figure(figsize=(10, 10))
    
    for bIndex in range(len(behaviors)):
        b = behaviors[bIndex]
        x = PCA_df[PCA_df['behavior']==b]['PC1']
        y = PCA_df[PCA_df['behavior']==b]['PC2']
        plt.scatter(x, y, marker='o', color=cs[bIndex], alpha=alphas[bIndex], label=b)
        plt.legend()
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename), dpi=300)
        plt.close()
    else:
        plt.show()



# Plot
# ========================================================================================
def overlaid_histogram(figsize, data1, data2, n_bins=200, data1_name=None, data1_color="#539caf", 
                       data2_name=None, data2_color="#7663b0", x_label=None, y_label=None,
                       y_limit=None,x_limit = None,ifSave=False,savePath=None,filename=None):
    
    
    '''
    Aim:
        plot 2 histograms on one plot
    
    Refs:
        - https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
        - https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/fonts_demo.html
        - https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib

    '''
    
    # Set the bounds for the bins so that the two distributions are fairly compared
    max_nbins = 10
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    binwidth = (data_range[1] - data_range[0]) / max_nbins

    weights1 = np.ones_like(data1)/float(len(data1))
    weights2 = np.ones_like(data2)/float(len(data2))

    if n_bins == 0:
        bins = np.arange(data_range[0], data_range[1]+binwidth, binwidth)
    else: 
        bins = n_bins

    
    # plot
    _, ax = plt.subplots(figsize = figsize)
    ax.hist(data1, bins = bins, color = data1_color, alpha = 1, 
            label = data1_name, weights=weights1)
    ax.hist(data2, bins = bins, color = data2_color, alpha = 0.75, 
            label = data2_name ,weights=weights2)
    
    if y_limit:
        ax.set_ylim(0, y_limit)
    if x_limit:
        ax.set_xlim(-x_limit, x_limit)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    
    ax.tick_params(axis='both', which='major', labelsize=20)
    labels = ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    labels = ax.get_xticklabels()
    [label.set_fontname('Arial') for label in labels]
    #ax.set_ylabel(y_label,fontname="Arial",fontsize = 20)
    #ax.set_xlabel(x_label,fontname="Arial",fontsize = 20)
    #ax.set_title(title, fontsize = 16, fontweight = 'bold')
    #ax.legend(loc = 'best')
    
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plotPeaks(df, prominence=5, width=2, threshold=None, height=None, ifSave=False,savePath=None,filename=None):
    
    # get array
    Frames, n_neuron = df.shape
    df_array = df.to_numpy()
    
    # plot
    fig, ax = plt.subplots(n_neuron,1, figsize=(20,4*n_neuron),facecolor='w', edgecolor='k')
    for i in range(n_neuron):

        Array = df_array[:,i]
        peaks, _ = find_peaks(Array, height=height, threshold=threshold, prominence=prominence, width=width)
        
        # plot
        ax[i].plot(Array, color='black')
        ax[i].plot(peaks, np.array(Array)[peaks], "X", markersize=10, label = 'peak')
        ax[i].set_title('neuron '+str(i), fontsize=20)
       
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()
        
        
        
        
def get_minMaxForPlot(WT_summary, NULL_summary):
    
    
    '''
    get necessary info for plot
    used in averageTrace plot
    '''
    
    # wt
    WT = WT_summary['ALL_MICE_MEAN'].values.tolist()
    WT_SEM = WT_summary['MEAD_SEM'].values.tolist()
    WTSEM = WT_summary['MEADSEM'].values.tolist()
    
    # null
    NULL = NULL_summary['ALL_MICE_MEAN'].values.tolist()
    NULL_SEM = NULL_summary['MEAD_SEM'].values.tolist()
    NULLSEM = NULL_summary['MEADSEM'].values.tolist()
    
    vmin = min(min(WT_SEM), min(NULL_SEM))*1.2
    vmin = math.floor(vmin*100)/100
    vmax = max(max(WTSEM), max(NULLSEM))*1.2
    vmax = math.ceil(vmax*100)/100
    
    
    x = WT_summary.index.values.tolist()
    
    return(x, WT, WT_SEM, WTSEM, NULL, NULL_SEM, NULLSEM, vmin,vmax)

def get_minMaxForPlot_One(WT_summary):
    
    
    '''
    get necessary info for plot
    used in averageTrace plot
    '''
    
    # wt
    WT = WT_summary['ALL_MICE_MEAN'].values.tolist()
    WT_SEM = WT_summary['MEAD_SEM'].values.tolist()
    WTSEM = WT_summary['MEADSEM'].values.tolist()
    
    vmin = min(WT_SEM)-0.2
    vmin = math.floor(vmin*100)/100
    vmax = max(WTSEM)*1.2
    vmax = math.ceil(vmax*100)/100
    
    
    x = WT_summary.index.values.tolist()
    
    return(x, WT, WT_SEM, WTSEM, vmin, vmax)

def plotAgeTrace(x,yWT,yNULL,yWT_SEM,yWTSEM,yNULL_SEM,yNULLSEM,ymin,ymax,seconds,project,behavior,ifSave=False,savePath=None,filename=None):
    
    '''
    plot averaged dff trace with SEM filled
    '''
    
    plt.figure(figsize = (8,4))

    plt.plot(x, yWT, color = 'black', lw=2)
    plt.plot(x, yNULL, color = '#9933ff', lw=2)
    plt.plot([x[int(len(x)/2)],x[int(len(x)/2)]],[ymin*0.8,ymax*0.8] , '--r', lw = 2)

    # fill
    plt.fill_between(x, yWT_SEM, yWTSEM, color = 'darkgrey', alpha = 0.5)
    plt.fill_between(x, yNULL_SEM, yNULLSEM, color = 'violet', alpha = 0.5)

    plt.ylim(ymin, ymax)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, project+"_"+behavior+"_"+str(seconds)+'s_AVG_FF.png'))
        plt.close()
    else:
        plt.show()

def plotAgeTrace_One(x,yWT,yWT_SEM,yWTSEM,ymin,ymax,cWT,cFWT,xVline=None,ifSave=False,savePath=None,filename=None):
    
    '''
    plot averaged dff trace with SEM filled
    '''
    
    plt.figure(figsize = (8,4))

    plt.plot(x, yWT, color=cWT, lw=2)
    if xVline != None:
        plt.plot([xVline,xVline],[ymin*0.8,ymax*0.8] , '--r', lw = 2)
    else:
        plt.plot([x[int(len(x)/2)],x[int(len(x)/2)]],[ymin*0.8,ymax*0.8] , '--r', lw = 2)

    # fill
    plt.fill_between(x, yWT_SEM, yWTSEM, color=cFWT, alpha = 0.5)

    plt.ylim(ymin, ymax)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_neuronTrace(DFF,ifSave=False,savePath=None,filename=None):
    
    '''
    plot dff trace for all neurons

    '''
    f, n = DFF.shape
    fig, ax = plt.subplots(n,1, figsize=(20,4*n),facecolor='w', edgecolor='k')

    for i in range(n):
    
        
        label='neuron'+str(i)
        ax[i].plot(DFF.iloc[:, i], c='b', lw=1, label=label)
        ax[i].axhline(y=0, ls='--',c='gray')
        ax[i].legend(loc=1)   
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_colomap(df,ifCenter='True',cmap='seismic',figsize=(10,8),vmin=-0.5,vmax=4,ifSave=False,savePath=None,filename=None):
    
    
    '''
    Aim:
        plot re-ordered dataframe as a heatmap
    
    '''
    
    plt.figure(figsize = figsize)
    if ifCenter:
        divnorm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        plt.pcolormesh(df.T, cmap=cmap, norm=divnorm) 
    else:
        plt.pcolormesh(df.T, cmap=cmap) 
    plt.colorbar() 
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()

def plot_line(lineValues,vLineX,lColor='black',lw=2,vLcolor='grey',vLymin=0,vLymax=1,ifSave=False,savePath=None,filename=None):
    
    '''
    plot a line with vertical dashed line
    '''
    
    plt.figure(figsize=(20, 4))
    plt.plot(lineValues, lw=lw, c=lColor)
    plt.vlines(x = vLineX, ymin=vLymin, ymax=vLymax, colors=vLcolor, ls='--')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()
        
        
        
        




def plot_3D_4Behaviors(PCA_df,behaviors=['center', 'corner', 'sniffing', 'grooming'],cs=['r','g','b','black'],
                       alphas=[0.6, 0.1, 0.8, 1], ifSave=True,savePath=None,filename=None):
    '''
    Aim:
        plot single 3D line plot and save
        different behavior, differetn scatter plot
    '''
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    for bIndex in range(len(behaviors)):
        b = behaviors[bIndex]
        x = PCA_df[PCA_df['behavior']==b]['PC1']
        y = PCA_df[PCA_df['behavior']==b]['PC2']
        z = PCA_df[PCA_df['behavior']==b]['PC3']
        ax.scatter(x, y, z, c=cs[bIndex], marker='o', alpha=alphas[bIndex], label=b)
        plt.legend()
    

    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename), dpi=300)
        plt.close()
    else:
        plt.show()

def plot_2D_4Behaviors(PCA_df,
                       behaviors=['center', 'corner', 'sniffing', 'grooming'],
                       cs=['r','g','b','black'],
                       alphas=[0.6, 0.1, 0.8, 1], 
                       ifSave=True,savePath=None,filename=None):
    '''
    Aim:
        plot single 3D line plot and save
        different behavior, differetn scatter plot
    '''
    
    fig = plt.figure(figsize=(10, 10))
    
    for bIndex in range(len(behaviors)):
        b = behaviors[bIndex]
        x = PCA_df[PCA_df['behavior']==b]['PC1']
        y = PCA_df[PCA_df['behavior']==b]['PC2']
        plt.scatter(x, y, marker='o', color=cs[bIndex], alpha=alphas[bIndex], label=b)
        plt.legend()
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename), dpi=300)
        plt.close()
    else:
        plt.show()


def plot_ASAM(img_array, img_array_sm, mouseName, mouseType, cmap='jet', ifSave=True, savePath=None, filename=None):
    
    '''
    Aim:
        plot img_array and smoothed img_array side by side
    
    '''
    
    
    fig, ax = plt.subplots(1,2, figsize=(20,6),facecolor='w', edgecolor='k')

    img = ax[0].contourf(img_array,  cmap = cmap)
    fig.colorbar(img)

    # smoothed
    imgSM = ax[1].imshow(img_array_sm, cmap = cmap,
               interpolation='nearest', aspect='auto',origin='lower', )
    fig.colorbar(imgSM)

    fig.suptitle(mouseName+'-'+mouseType+' ceON in OFT', fontsize=20)
    
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()
    
def plot_gridASAM(img_array, sigma=1.5, cmap='jet', ifSave=True, savePath=None, filename=None):
    
    '''
    Aim:
        plot img_array and smoothed img_array side by side
    
    '''
    
    
    fig, ax = plt.subplots(1,2, figsize=(20,6),facecolor='w', edgecolor='k')

    img = ax[0].contourf(img_array,  cmap = cmap, vmin=0)
    fig.colorbar(img)

    # smoothed
    img_array_sm = gaussian_filter(img_array, sigma=sigma)
    imgSM = ax[1].imshow(img_array_sm, cmap = cmap,
               interpolation='nearest', aspect='auto',origin='lower', vmin=0)
    fig.colorbar(imgSM)

    fig.suptitle(filename, fontsize=20)
    
    
    if ifSave:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        plt.savefig(os.path.join(savePath, filename))
        plt.close()
    else:
        plt.show()
    
def plot_allNeuronTrace_withColoredInterval(x, xCe, xCo, xCe_keep, dfO, dfD, on_off, mouse_name, fig_path):

    all_neurons = on_off.index.tolist()
    
    
    fig, ax = plt.subplots(len(all_neurons),1, figsize=(40,3*len(all_neurons)),facecolor='w', edgecolor='k')
    for i in range(len(all_neurons)):
        neuronName = all_neurons[i]
        neurontitle = on_off[(on_off.loc[neuronName]==1).index[on_off.loc[neuronName]==1]].columns.tolist()
        y = dfO.iloc[:,i].values.tolist()
        yD = dfD.iloc[:,i].values.tolist()


        for xco in xCo:
            ax[i].axvline(x=xco, color='#7fc792', alpha=0.2)
        for xce in xCe:
            ax[i].axvline(x=xce, color='#fac5a0', alpha=0.2)
        if i==0 or i== math.ceil(len(all_neurons)/2) or i==(len(all_neurons)-1):
            for xce in xCe_keep:
                ax[i].axvline(x=xce, color='#f27c7c', alpha=0.2)
            
            
        ax[i].plot(x, y, lw=6, color = 'grey', alpha=0.5)
        ax[i].plot(x, yD, lw=4, color = 'black')
        ax[i].set_title(neuronName+" ONOFF:"+str(neurontitle), fontsize=28, fontweight='bold')


        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].axes.get_xaxis().set_visible(False)
        ax[i].axes.get_yaxis().set_visible(False)

    
    fig.savefig(os.path.join(fig_path, mouse_name+'_allNeuronsTrace.png'))
    
    plt.close()









