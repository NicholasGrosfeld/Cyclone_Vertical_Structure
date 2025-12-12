# This code takes a .txt file containing individual cyclone centers identified using the code in script 
# 1_detect_closed_lows.py, and joins individual low centers into tracks through time by assigning a common
# track id number when consecutive lows are within a distance threshold of each other. 
#
# It outputs the located cyclone centers in a .txt file containing the following columns:
# -track_id number;
# -cyclone_id number; 
# -year;
# -month; 
# -day; 
# -hour; 
# -hours since 1900 (the time convention of some reanalyses);
# -lat; 
# -lon; 
# -central gph; 
# -inner geopotential height gradient; 
# -outer geopotential height gradient.

#============================== Import modules ===============================
import numpy as np
import argparse

#================== Define Key Functions ========================================
def join_low_tracks(low_array, tstep_col, lat_col, lon_col, center_id_col, spac_threshold):
    
    '''This function joins low centers across multiple timesteps into a track
    of a single system. It does this by looping through each identified low
    center, and looking forward through the next low centers to find the closest
    low center within a certain time and space threshold.'''

    # get the number of rows in the array to loop through
    l = np.shape(low_array)[0]
    
    # to set up a loop through each low entry.
    entry_range = np.arange(0,l)

    # a variable to keep track of the latest event number that has been assigned, will be updated
    # as new tracks are assigned. 
    latest_event = 0

    for r in entry_range:

        current_low_entry = low_array[r,:] #gets the current row being tested for 
        #any previous events that it belongs to. NOTE: THIS IS AN EXACT (NON-CONDITIONAL)
        #REFERENCE. IT SIMPLY CREATES A LINK TO THAT ROW IN 'low_array'. 
        
        e = current_low_entry[0]
        t = current_low_entry[tstep_col]
        lat = current_low_entry[lat_col]
        lon = current_low_entry[lon_col]

        # this part initiates a new track id number if the current low entry has not been assigned
        # a track number yet
        if np.isnan(e): 

            latest_event = latest_event + 1
            e = latest_event
            current_low_entry[0] = e

        # now search forward in time to find low centers at the next 6 hours, within the distance threshold. We only test the latitude
        # distance here to avoid the problem of crossing the limits of longitude, but we handle those cases later. 
        ii = (low_array[:,tstep_col] ==  t + 6) & (abs(lat - low_array[:,lat_col]) <= spac_threshold)
        next_event_entries = low_array[ii,:] #Note: This IS a copy (indexed with a Boolean)!
        
        #This section calculates the distance between the center of the current low and the
        #center of the low reccorded in each row of next_event_entries by pythagoras's theorem. 
        #Note we already have the condition that each of the lows in prev_event_entries
        #are within the distance box_size2 of the current low center. 
        # These distances are stored in the last 3 columns of the array. 
        next_event_entries[:, -3] = abs(next_event_entries[:,lat_col] - lat)

        # These three lines handle the longitude differences that cross 0 degrees longitude
        lon_diff1 = abs(next_event_entries[:,lon_col] - lon)
        lon_diff2 = 360 - abs(next_event_entries[:,lon_col] - lon)

        next_event_entries[:, -2] = np.min((lon_diff1, lon_diff2), axis = 0)

        # compute the diagonal distance by pythagoras, and store it in the last column of the array
        next_event_entries[:, -1] = ((next_event_entries[:, -3]) ** 2 + (next_event_entries[:, -2]) ** 2) ** 0.5
        
        if np.shape(next_event_entries)[0] > 0:

            # find the distance of the closest low center
            min_distance = np.min(next_event_entries[:, -1])

            # test if the minimum distance is below the spatial threshold to join
            if min_distance <= spac_threshold:

                # locate the row of the closest cyclone in the next timestep
                i_min = next_event_entries[:, -1] == min_distance
                next_track_location = next_event_entries[i_min,:]
    
                # INCLUDE A FEATURE THAT TESTS IF THERE ARE MORE THAN ONE FUTURE LOW CENTERS IDENTIFIED
                # AS THE CLOSEST ONE, AND CHOOSE ONE
                if np.shape(next_track_location)[0] > 1:
    
                    # arbitrarily choose the first one in the array. This part might change if I find
                    # a better way to do it. the index is written as [:1,:] to trick next_track_location 
                    # into having 2 dimensions. 
                    next_track_location = next_track_location[:1,:]
    
                # Now we find the row in the low_array that matches the next track location
                # (since that row is a copy) and write the new event number to it. 
                next_id = next_track_location[0,center_id_col]
                i_next = (low_array[:,center_id_col] ==  next_id)
                
                # write the new event number e
                low_array[i_next, 0] = e
            
    return low_array

def reorder_lows(array, tstep_col):

    '''This last section re-orders the low entries for each event in time-order,
    just in case anything got put out of order.'''

    # get the cyclone track numbers to loop through
    event_nums = np.unique(array[:,0])

    # set up a variable to indicate which row of the array to start writing the reorganised track data
    # to. This variable will get updated with each itteration of the loop.
    row = 0

    # create the new array to store the reorganised data.
    new_array = np.full(np.shape(array), np.nan)
    
    for e in event_nums:

        # pull out the event number e
        event_rows = array[(array[:,0] == e), :]

        # le will give us the number of rows to write into the new array
        le = np.shape(event_rows)[0]

        # re-order the rows of the cyclone by the timestep
        ie = event_rows[:,tstep_col].argsort()
        event_rows = event_rows[ie,:]
        
        # this is the part where we put the event rows in the new array2
        new_array[row:row + le, :] = event_rows

        # update row for the next cyclone event
        row = row + le

    # trim off the extra 3 columns for distance computations before returning the 
    # new_array2 out of the function
    new_array = new_array[:,:-3]

    return new_array

# =========================== Take Input Parameters from Shell Script =====================

# generate the object that receives the input variables from the linux script
parser = argparse.ArgumentParser(description = "Pressure level of input dataset")
parser.add_argument("--p_level", required = True, type = int)
    
args = parser.parse_args()
    
p_level = args.p_level

#============================= Parameters set by User ===================================
data_path = '/g/data/m35/nxg561/02_Cyclone_Vertical_Structure/Cyclone_Identification/Nov2025/'
save_path = data_path

dataset = 'era5'

TRACK_JOINING_SPATIAL_THRESHOLD = 7.5 #For joining CONSECUTIVE low centers into tracks. The maximum 
            #distance (in degees lat/lon) between one low center in the 
            #'current' pressure field, and another low center in the next
            #pressure fields to join them to the same track. 

# ================================== Hard-coded and computed Parameters ========================
P_LEVEL = p_level

# set the column numbers of the cyclone arrays as they will be handled by the track joining function. The input arrays will have an
# empty column appended at position 0 of the cyclone array, hence the column numbers are advanced by 1 after the previous script. 
# These will also become the column numbers for the output array. 
CENTER_ID_COL = 1
YEAR_COL = 2
MONTH_COL = 3
DAY_COL = 4
HOUR_COL = 5
TSTEP_COL = 6
LAT_COL = 7
LON_COL = 8
CELL_COL = 9
TEST1_COL = 10
TEST2_COL = 11
# ================== Load and Wrangle Cyclone Data ================================

refined_lows = np.loadtxt(data_path + 'rawlows_' + str(P_LEVEL) + '_sh.txt', delimiter = ',')

# Append an empty column to the start of the cyclone array to store the track id numbers, and three empty columns at the end
# to store the distances between pairs of low centers (used in the tracking process). 
l = np.shape(refined_lows)[0]
track_col = np.full((l,1), np.nan)
dist_cols = np.full((l,3), np.nan)
refined_lows = np.concatenate((track_col, refined_lows, dist_cols), axis = 1)

#========== Join Low Entries into Tracks ==============================

# This code combines all the low centers from the same system, spread over multiple
# time steps, into a single track for the duration of the system. 

closed_low_tracks = join_low_tracks(refined_lows, TSTEP_COL, LAT_COL, LON_COL, CENTER_ID_COL, spac_threshold = TRACK_JOINING_SPATIAL_THRESHOLD) 

# re-order output
low_tracks_out = reorder_lows(closed_low_tracks, TSTEP_COL)

#====================== Save Output ===================================    
                                       
output_filename = str(save_path) + 'low_tracks_' + str(P_LEVEL) + '_sh.txt'
np.savetxt(output_filename, low_tracks_out, delimiter = ',') 