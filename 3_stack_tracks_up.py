# This code takes the .txt files containing single level cyclone tracks from the script 2_join_low_tracks.py for several different 
# levels, and joins them together into vertical stacks where they are determined to form different levels of the same vertical system.
#
# It outputs the located cyclone centers in a .txt file containing the following columns:
# -stack_id_number;
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


# Note: This code has been adapted to run in blocks of 10 years, as running the whole 44 year dataset in one go takes longer than
# the time allowance on our compute nodes. Future work to improve the runtime of this code may remove the need to run it in blocks.

#============================== Import modules ===============================
import numpy as np
from datetime import datetime
import matplotlib.dates as dt
import argparse

#================== Define Key Functions ========================================
def stack_summary_data(current_stack_num, track_data):

    ''' This function assembles an array of data to summarise the first
    time step of a cyclone stack. '''

    # Pull out the year, month, day, hour, timestep, level, lat and lon components from the cyclone track data
    y = track_data[0,3]
    m = track_data[0,4]
    d = track_data[0,5]
    h = track_data[0,6]
    t = track_data[0,7]
    level = int(str(track_data[0,2])[:3]) # extract the level data from the cyclone id string
    lat = track_data[0,8]
    lon = track_data[0,9]

    out_data = np.array([current_stack_num, y, m, d, h, t, level, lat, lon])

    return out_data

def update_stack_number(old_stack_num, new_stack_num, track_arrays_all, stack_summary_array):

    '''This function locates all the instances of a cyclone stack across all previously processed levels and changes the
    stack number. track_arrays_all is a list containing the cyclone arrays that have previously been processed and need to be
    searched through and updated, and stack_summary_array is a separate array containing summary data of the first timestep of each 
    identified cyclone stack, which will also need to be updated.'''

    # find the instances of the old stack number in each of the cyclone arrays at a single level and
    # change them to the new number.
    for array in track_arrays_all:

        old_stack_ii = array[:,0] == old_stack_num
        array[old_stack_ii, 0] = new_stack_num

    # remove the old stack number from the stack summary array
    old_stack_ii = stack_summary_array[:,0] == old_stack_num
    stack_summary_array[old_stack_ii,:] = np.nan
    
    return track_arrays_all, stack_summary_array

def upper_level_track_search(track_arrays_all, lower_level, upper_level, lower_track_num, current_stack_num, t, distance_threshold, found, stack_origin_array):
    
    ''' This is the function that takes a single point in time of a cyclone track in a lower level, and searches an array
    of cyclone centers at a higher level for the closest cyclone point located within a specified distance threshold, 
    and marks the upper cyclone point as being part of the same vertical system. track_arrays_all is a list containing 
    all the arrays of low tracks from each individual level, and lower_level and upper_level are integers to index that list.'''

    lower_tracks_array = track_arrays_all[lower_level]
    
    # current_lower_track_array will also be a copy, it is indexed with a boolean. Make sure 
    # current_lower_track_array is written back into lower_tracks_array at the end.
    current_lower_track_array = lower_tracks_array[lower_tracks_array[:,1] == lower_track_num, :]
    
    # This will be a row of current_lower_track_array containing a single point
    lower_cyclone_point = current_lower_track_array[current_lower_track_array[:,7] == t, :]

    # Now get the array of cyclone tracks at the upper level
    upper_array = track_arrays_all[upper_level]

    # This will be every cyclone center in the upper array at that timestep
    upper_cyclone_points = upper_array[upper_array[:,7] == t, :]

    # reference to the lat/lon of the current cyclone in the lower level track at this time
    lat_lower = lower_cyclone_point[0,8]
    lon_lower = lower_cyclone_point[0,9]

    # Compute the lat/lon distance between every cyclone center in the upper array
    # and the current cyclone center in the lower array
    upper_cyclone_points[:,13] = abs(upper_cyclone_points[:,8] - lat_lower)

    # These three lines handle the longitude differences that cross 0 degrees longitude
    lon_diff1 = abs(upper_cyclone_points[:,9] - lon_lower)
    lon_diff2 = 360 - abs(upper_cyclone_points[:,9] - lon_lower)
    upper_cyclone_points[:,14] = np.min((lon_diff1, lon_diff2), axis = 0)
    
    upper_cyclone_points[:,15] = ((upper_cyclone_points[:,13]) ** 2 + (upper_cyclone_points[:,14]) ** 2) ** 0.5

    # Assign the upper track with the shortest separation distance by a minimum threshold 
    # to the same stack. If no upper cyclone points are found, nothing else happens after this. 
    if np.shape(upper_cyclone_points)[0] > 0:

        # Locate the cyclone in the upper level with the closest distance at this point in time
        min_distance = np.min(upper_cyclone_points[:,15])
        i_min = upper_cyclone_points[:,15] == min_distance
        upper_cyclone_match_point = upper_cyclone_points[i_min,:] # A copy due to being indexed with a boolean

        # Test if the closest upper level cyclone point is within the specified distance threshold. If it is not, 
        # nothing else happens after this. 
        if min_distance <= distance_threshold:
                      
            found = True

            # get the whole track data of the matching upper level cyclone track
            upper_track_num = upper_cyclone_match_point[0,1]
            iupper = upper_array[:,1] == upper_track_num
            whole_upper_track_array = upper_array[iupper, :]

            # THIS PART IS WHERE THE FIRST TIME STEP OF THE UPPER LEVEL TRACK SHOULD BE CHECKED IF IT BEGAN
            # BEFORE THE LOWER LEVEL TRACK. IT DOES NOT MATTER IF THE UPPER LEVEL TRACK HAS A STACK NUMBER OR NOT.
            first_t_stack = stack_origin_array[stack_origin_array[:,0] == current_stack_num, 5]
            first_t_upper = whole_upper_track_array[0,7]

            # If the matching upper level track began at an earlier time than the current lower track, 
            # update the summary data for this stack id.
            if first_t_upper <= first_t_stack:

                #print('updating stack_origin_array')
                stack_origin_array[stack_origin_array[:,0] == current_stack_num, :] = stack_summary_data(current_stack_num, whole_upper_track_array)
                
            # One final branch to handle cases where the upper level track already has a stack number assigned.
            upper_cyclone_stack_num = upper_cyclone_match_point[0,0]
            
            if (upper_cyclone_stack_num > 0) and (upper_cyclone_stack_num != current_stack_num):

                # If the lower stack came earlier or at the same time, change the upper level stack number
                if first_t_stack <= first_t_upper:
                    
                    old_stack_num = upper_cyclone_stack_num 
                    new_stack_num = current_stack_num

                    # locate the records of the old cyclone stack, and change the stack number to the new number. 
                    track_arrays_all, stack_origin_array = update_stack_number(old_stack_num, new_stack_num, track_arrays_all, stack_origin_array)

                # If the upper track came earlier, change all lower level stack numbers
                else:

                    old_stack_num = current_stack_num 
                    new_stack_num = upper_cyclone_stack_num

                    # locate the records of the old cyclone stack, and change the stack number to the new number. 
                    track_arrays_all, stack_origin_array = update_stack_number(old_stack_num, new_stack_num, track_arrays_all, stack_origin_array)
                    
                    # update current_stack_num so we are now working on a new stack id number
                    current_stack_num = upper_cyclone_stack_num
                    
            else:
                
                # Just write the new stack number into the upper cyclone track
                upper_array[iupper,0] = current_stack_num
            
    return found, track_arrays_all, stack_origin_array, current_stack_num

def stack_tracks_up3(track_arrays_all, level, highest_stack_num, distance_threshold, stack_origin_array):

    ''' This function works through the array of cyclone tracks at the base level, and organises the search
    in the next 3 levels up for matching cyclone tracks. It loops through each track at the base level, and for each track:
    - it assigns a new stack id number if it does not have one already;
    - for each timestep of each track, it calls the function upper_level_track_search that searches in the 
    array of tracks at higher levels for a matching track to join to. '''

    # Get the array of tracks for the lower level
    lower_array = track_arrays_all[level]
    
    # get the unique track numbers of the lower_array to loop through
    track_nums_lower = np.unique(lower_array[:,1])

    # for each track in lower_array, generate a new stack number if required, and then step through
    # each timestep and call the function that searches the upper levels for a matching track.
    for lower_track_num in track_nums_lower:

        # Get the track at the base level to process
        
        # current_lower_track_array will be a copy, because lower_array is indexed with a boolean. This pulls out the
        # data of the whole track from the lower_array. 
        current_lower_track_array = lower_array[lower_array[:,1] == lower_track_num, :]
        current_lower_track_tsteps = current_lower_track_array[:,7]
        
        current_stack_num = current_lower_track_array[0,0] 
        
        # some tracks may already have a stack number assigned from
        # being matched with a lower level track. If so, the program will move to the function that begins the search
        # for corresponding tracks in the upper levels. If not, the next section will assign a new stack number. 

        if current_stack_num == 0:

            highest_stack_num = highest_stack_num + 1
            current_stack_num = highest_stack_num
            current_lower_track_array[:,0] = current_stack_num

            # Write the new stack number into the summary array, along with the other data of the initial
            # cyclone instance.
            stack_origin_array[int(current_stack_num) - 1, :] = stack_summary_data(current_stack_num, current_lower_track_array)

            # re-write current_lower_track_array back into the lower_array (it now has a stack number)
            lower_array[lower_array[:,1] == lower_track_num, :] = current_lower_track_array

            # Make sure the list of all levels is updated before it is passed into the next function
            track_arrays_all[level] = lower_array

        ######### LOOP THROUGH EACH TIMESTEP OF THE LOWER TRACK AND SEARCH FOR THE UPPER LEVEL MATCH #########
        
        # loop through each timestep of current_tsteps, and call the function that searches the upper level for a
        # track to match with.
        for t in current_lower_track_tsteps:

            # This boolean variable changes to True when an upper level track has been found to join to the lower 
            # level track at each time step.
            found = False

            # integers to refer to the track array in track_arrays_all
            lower_level = level
            upper_level = level + 1
            
            found, track_arrays_all, stack_origin_array, current_stack_num = upper_level_track_search(track_arrays_all, lower_level, upper_level, lower_track_num, current_stack_num, t, distance_threshold, found, stack_origin_array)

            if found:
                continue

            upper_level = level + 2
            
            found, track_arrays_all, stack_origin_array, current_stack_num = upper_level_track_search(track_arrays_all, lower_level, upper_level, lower_track_num, current_stack_num, t, distance_threshold, found, stack_origin_array)

            if found:
                continue

            upper_level = level + 3
            
            found, track_arrays_all, stack_origin_array, current_stack_num = upper_level_track_search(track_arrays_all, lower_level, upper_level, lower_track_num, current_stack_num, t, distance_threshold, found, stack_origin_array)

            if found:
                continue

    # return the lower_array, upper_array and latest_track_num out of the function
    return track_arrays_all, highest_stack_num, stack_origin_array

def stack_tracks_up2(track_arrays_all, level, highest_stack_num, distance_threshold, stack_origin_array):

    ''' This function works through the array of cyclone tracks at the base level, and organises the search
    in the next 2 levels up for matching cyclone tracks. It loops through each track at the base level, and for each track:
    - it assigns a new stack id number if it does not have one already;
    - for each timestep of each track, it calls the function upper_level_track_search that searches in the 
    array of tracks at higher levels for a matching track to join to. '''

    # Get the array of tracks for the lower level
    lower_array = track_arrays_all[level]
    
    # get the unique track numbers of the lower_array to loop through
    track_nums_lower = np.unique(lower_array[:,1])

    # for each track in lower_array, generate a new stack number if required, and then step through
    # each timestep and call the function that searches the upper levels for a matching track.
    for lower_track_num in track_nums_lower:

        # Get the track at the base level to process
        
        # current_lower_track_array will be a copy, because lower_array is indexed with a boolean. This pulls out the
        # data of the whole track from the lower_array. 
        current_lower_track_array = lower_array[lower_array[:,1] == lower_track_num, :]
        current_lower_track_tsteps = current_lower_track_array[:,7]
        
        current_stack_num = current_lower_track_array[0,0] 
        
        # some tracks may already have a stack number assigned from
        # being matched with a lower level track. If so, the program will move to the function that begins the search
        # for corresponding tracks in the upper levels. If not, the next section will assign a new stack number. 

        if current_stack_num == 0:

            highest_stack_num = highest_stack_num + 1
            current_stack_num = highest_stack_num
            current_lower_track_array[:,0] = current_stack_num

            # Write the new stack number into the summary array, along with the other data of the initial
            # cyclone instance.
            stack_origin_array[int(current_stack_num) - 1, :] = stack_summary_data(current_stack_num, current_lower_track_array)

            # re-write current_lower_track_array back into the lower_array (it now has a stack number)
            lower_array[lower_array[:,1] == lower_track_num, :] = current_lower_track_array

            # Make sure the list of all levels is updated before it is passed into the next function
            track_arrays_all[level] = lower_array

        ######### LOOP THROUGH EACH TIMESTEP OF THE LOWER TRACK AND SEARCH FOR THE UPPER LEVEL MATCH #########
        
        # loop through each timestep of current_tsteps, and call the function that searches the upper level for a
        # track to match with.
        for t in current_lower_track_tsteps:

            # This boolean variable changes to True when an upper level track has been found to join to the lower 
            # level track at each time step.
            found = False

            # integers to refer to the track array in track_arrays_all
            lower_level = level
            upper_level = level + 1
            
            found, track_arrays_all, stack_origin_array, current_stack_num = upper_level_track_search(track_arrays_all, lower_level, upper_level, lower_track_num, current_stack_num, t, distance_threshold, found, stack_origin_array)

            if found:
                continue

            upper_level = level + 2
            
            found, track_arrays_all, stack_origin_array, current_stack_num = upper_level_track_search(track_arrays_all, lower_level, upper_level, lower_track_num, current_stack_num, t, distance_threshold, found, stack_origin_array)

            if found:
                continue

    # return the lower_array, upper_array and latest_track_num out of the function
    return track_arrays_all, highest_stack_num, stack_origin_array

def stack_tracks_up1(track_arrays_all, level, highest_stack_num, distance_threshold, stack_origin_array):

    ''' This function works through the array of cyclone tracks at the base level, and organises the search
    in the next 2 levels up for matching cyclone tracks. It loops through each track at the base level, and for each track:
    - it assigns a new stack id number if it does not have one already;
    - for each timestep of each track, it calls the function upper_level_track_search that searches in the 
    array of tracks at higher levels for a matching track to join to. '''

    # Get the array of tracks for the lower level
    lower_array = track_arrays_all[level]
    
    # get the unique track numbers of the lower_array to loop through
    track_nums_lower = np.unique(lower_array[:,1])

    # for each track in lower_array, generate a new stack number if required, and then step through
    # each timestep and call the function that searches the upper levels for a matching track.
    for lower_track_num in track_nums_lower:

        # Get the track at the base level to process
        
        # current_lower_track_array will be a copy, because lower_array is indexed with a boolean. This pulls out the
        # data of the whole track from the lower_array. 
        current_lower_track_array = lower_array[lower_array[:,1] == lower_track_num, :]
        current_lower_track_tsteps = current_lower_track_array[:,7]
        
        current_stack_num = current_lower_track_array[0,0] 
        
        # some tracks may already have a stack number assigned from
        # being matched with a lower level track. If so, the program will move to the function that begins the search
        # for corresponding tracks in the upper levels. If not, the next section will assign a new stack number. 

        if current_stack_num == 0:

            highest_stack_num = highest_stack_num + 1
            current_stack_num = highest_stack_num
            current_lower_track_array[:,0] = current_stack_num

            # Write the new stack number into the summary array, along with the other data of the initial
            # cyclone instance.
            stack_origin_array[int(current_stack_num) - 1, :] = stack_summary_data(current_stack_num, current_lower_track_array)

            # re-write current_lower_track_array back into the lower_array (it now has a stack number)
            lower_array[lower_array[:,1] == lower_track_num, :] = current_lower_track_array

            # Make sure the list of all levels is updated before it is passed into the next function
            track_arrays_all[level] = lower_array

        ######### LOOP THROUGH EACH TIMESTEP OF THE LOWER TRACK AND SEARCH FOR THE UPPER LEVEL MATCH #########
        
        # loop through each timestep of current_tsteps, and call the function that searches the upper level for a
        # track to match with.
        for t in current_lower_track_tsteps:

            # This boolean variable changes to True when an upper level track has been found to join to the lower 
            # level track at each time step.
            found = False

            # integers to refer to the track array in track_arrays_all
            lower_level = level
            upper_level = level + 1
            
            found, track_arrays_all, stack_origin_array, current_stack_num = upper_level_track_search(track_arrays_all, lower_level, upper_level, lower_track_num, current_stack_num, t, distance_threshold, found, stack_origin_array)

            if found:
                continue

    # return the lower_array, upper_array and latest_track_num out of the function
    return track_arrays_all, highest_stack_num, stack_origin_array

# ======================= Take Input Parameters ================================

# generate the object that receives the input variables from the linux script
parser = argparse.ArgumentParser(description = "Start and End Years")
parser.add_argument("--year_start", required = True, type = int)
parser.add_argument("--year_end", required = True, type = int)
parser.add_argument("--year_previous", required = False, type = int)
    
args = parser.parse_args()
    
y_start = args.year_start
y_end = args.year_end
y_previous = args.year_previous

#============================= Parameters set by User ===================================

TIME_Y0 = 1900 # This is required to re-compute the tstep values to loop over

DISTANCE_THRESHOLD = 5 # degrees lat/lon

# Data I/O parameters
data_path = '/g/data/m35/nxg561/02_Cyclone_Vertical_Structure/Cyclone_Identification/Nov2025/'

# =============================== Load Cyclone Data ===========================================

# input all the files
input_filename_900 = 'low_tracks_900_sh.txt'
input_filename_800 = 'low_tracks_800_sh.txt'
input_filename_700 = 'low_tracks_700_sh.txt'
input_filename_600 = 'low_tracks_600_sh.txt'
input_filename_500 = 'low_tracks_500_sh.txt'
input_filename_400 = 'low_tracks_400_sh.txt'
input_filename_300 = 'low_tracks_300_sh.txt'
input_filename_200 = 'low_tracks_200_sh.txt'

z900_tracks = np.loadtxt(data_path + input_filename_900, delimiter = ',')
z800_tracks = np.loadtxt(data_path + input_filename_800, delimiter = ',')
z700_tracks = np.loadtxt(data_path + input_filename_700, delimiter = ',')
z600_tracks = np.loadtxt(data_path + input_filename_600, delimiter = ',')
z500_tracks = np.loadtxt(data_path + input_filename_500, delimiter = ',')
z400_tracks = np.loadtxt(data_path + input_filename_400, delimiter = ',')
z300_tracks = np.loadtxt(data_path + input_filename_300, delimiter = ',')
z200_tracks = np.loadtxt(data_path + input_filename_200, delimiter = ',')

# This part makes sure the stack numbers build after the last stack number from the previously run files. A y_previous
# value of '0' will evaluate as 'false', any other year will evaluate as 'true'. 
if y_previous:

    # get the last stack number to be assigned in the previous block of years
    z200_tracks_old = np.loadtxt(data_path + 'low_stacks_200_sh_' + str(y_previous) + '.txt', delimiter = ',')
    z300_tracks_old = np.loadtxt(data_path + 'low_stacks_300_sh_' + str(y_previous) + '.txt', delimiter = ',')
    z400_tracks_old = np.loadtxt(data_path + 'low_stacks_400_sh_' + str(y_previous) + '.txt', delimiter = ',')
    z500_tracks_old = np.loadtxt(data_path + 'low_stacks_500_sh_' + str(y_previous) + '.txt', delimiter = ',')
    z600_tracks_old = np.loadtxt(data_path + 'low_stacks_600_sh_' + str(y_previous) + '.txt', delimiter = ',')
    z700_tracks_old = np.loadtxt(data_path + 'low_stacks_700_sh_' + str(y_previous) + '.txt', delimiter = ',')
    z800_tracks_old = np.loadtxt(data_path + 'low_stacks_800_sh_' + str(y_previous) + '.txt', delimiter = ',')
    z900_tracks_old = np.loadtxt(data_path + 'low_stacks_900_sh_' + str(y_previous) + '.txt', delimiter = ',')
    
    previous_stack_nums = np.concatenate((z200_tracks_old[:,0], z300_tracks_old[:,0], z400_tracks_old[:,0], 
                                         z500_tracks_old[:,0], z600_tracks_old[:,0], z700_tracks_old[:,0], 
                                         z800_tracks_old[:,0], z900_tracks_old[:,0]), axis = 0)
    highest_stack_num = int(np.max(previous_stack_nums))

else: 
    
    highest_stack_num = 0

# Extract only the tracks from the block of years
z900_tracks = z900_tracks[(z900_tracks[:,2] >= y_start) & (z900_tracks[:,2] < y_end), :]
z800_tracks = z800_tracks[(z800_tracks[:,2] >= y_start) & (z800_tracks[:,2] < y_end), :]
z700_tracks = z700_tracks[(z700_tracks[:,2] >= y_start) & (z700_tracks[:,2] < y_end), :]
z600_tracks = z600_tracks[(z600_tracks[:,2] >= y_start) & (z600_tracks[:,2] < y_end), :]
z500_tracks = z500_tracks[(z500_tracks[:,2] >= y_start) & (z500_tracks[:,2] < y_end), :]
z400_tracks = z400_tracks[(z400_tracks[:,2] >= y_start) & (z400_tracks[:,2] < y_end), :]
z300_tracks = z300_tracks[(z300_tracks[:,2] >= y_start) & (z300_tracks[:,2] < y_end), :]
z200_tracks = z200_tracks[(z200_tracks[:,2] >= y_start) & (z200_tracks[:,2] < y_end), :]

# append a new column of zeros to the start of each array, and 3 extra columns at the end 
# before they go into the stack_tracks function
z900_tracks = np.concatenate((np.zeros((len(z900_tracks), 1)), z900_tracks, np.zeros((len(z900_tracks), 3))), axis = 1)
z800_tracks = np.concatenate((np.zeros((len(z800_tracks), 1)), z800_tracks, np.zeros((len(z800_tracks), 3))), axis = 1)
z700_tracks = np.concatenate((np.zeros((len(z700_tracks), 1)), z700_tracks, np.zeros((len(z700_tracks), 3))), axis = 1)
z600_tracks = np.concatenate((np.zeros((len(z600_tracks), 1)), z600_tracks, np.zeros((len(z600_tracks), 3))), axis = 1)
z500_tracks = np.concatenate((np.zeros((len(z500_tracks), 1)), z500_tracks, np.zeros((len(z500_tracks), 3))), axis = 1)
z400_tracks = np.concatenate((np.zeros((len(z400_tracks), 1)), z400_tracks, np.zeros((len(z400_tracks), 3))), axis = 1)
z300_tracks = np.concatenate((np.zeros((len(z300_tracks), 1)), z300_tracks, np.zeros((len(z300_tracks), 3))), axis = 1)
z200_tracks = np.concatenate((np.zeros((len(z200_tracks), 1)), z200_tracks, np.zeros((len(z200_tracks), 3))), axis = 1)

# Set up an array to store the summary data of each stack in
total_tracks = int(np.max(z900_tracks[:,1]) + np.max(z800_tracks[:,1]) + np.max(z700_tracks[:,1]) + np.max(z600_tracks[:,1]) + 
        np.max(z500_tracks[:,1]) + np.max(z400_tracks[:,1]) + np.max(z300_tracks[:,1]) + np.max(z200_tracks[:,1]))

# Create an empty array of zeros to store the data of the first instance of each stack. 
# it will contain stack_num, Y, M, D, H, t, level, lat, lon. 
stack_origin_array = np.zeros((total_tracks, 9))

# Call the function to search 3 levels up for each layer
print('processing 900 hPa tracks')
# Gather all the arrays of tracks at a single level into a list
track_arrays_all = [z900_tracks, z800_tracks, z700_tracks, z600_tracks,z500_tracks, z400_tracks, z300_tracks, z200_tracks]
# Call the track stacking function to search 3 levels up based at 900 hPa
track_arrays_all, highest_stack_num, stack_origin_array = stack_tracks_up3(track_arrays_all, level = 0, highest_stack_num = highest_stack_num, distance_threshold = DISTANCE_THRESHOLD, stack_origin_array = stack_origin_array)

print('processing 800 hPa tracks')
# Gather all the arrays of tracks at a single level into a list
track_arrays_all = [z900_tracks, z800_tracks, z700_tracks, z600_tracks,z500_tracks, z400_tracks, z300_tracks, z200_tracks]
# Call the track stacking function to search 3 levels up based at 800 hPa
track_arrays_all, highest_stack_num, stack_origin_array = stack_tracks_up3(track_arrays_all, level = 1, highest_stack_num = highest_stack_num, distance_threshold = DISTANCE_THRESHOLD, stack_origin_array = stack_origin_array)

print('processing 700 hPa tracks')
# Gather all the arrays of tracks at a single level into a list
track_arrays_all = [z900_tracks, z800_tracks, z700_tracks, z600_tracks,z500_tracks, z400_tracks, z300_tracks, z200_tracks]
# Call the track stacking function to search 3 levels up based at 700 hPa
track_arrays_all, highest_stack_num, stack_origin_array = stack_tracks_up3(track_arrays_all, level = 2, highest_stack_num = highest_stack_num, distance_threshold = DISTANCE_THRESHOLD, stack_origin_array = stack_origin_array)

print('processing 600 hPa tracks')
# Gather all the arrays of tracks at a single level into a list
track_arrays_all = [z900_tracks, z800_tracks, z700_tracks, z600_tracks,z500_tracks, z400_tracks, z300_tracks, z200_tracks]
# Call the track stacking function to search 3 levels up based at 600 hPa
track_arrays_all, highest_stack_num, stack_origin_array = stack_tracks_up3(track_arrays_all, level = 3, highest_stack_num = highest_stack_num, distance_threshold = DISTANCE_THRESHOLD, stack_origin_array = stack_origin_array)

print('processing 500 hPa tracks')
# Gather all the arrays of tracks at a single level into a list
track_arrays_all = [z900_tracks, z800_tracks, z700_tracks, z600_tracks,z500_tracks, z400_tracks, z300_tracks, z200_tracks]
# Call the track stacking function to search 3 levels up based at 500 hPa
track_arrays_all, highest_stack_num, stack_origin_array = stack_tracks_up3(track_arrays_all, level = 4, highest_stack_num = highest_stack_num, distance_threshold = DISTANCE_THRESHOLD, stack_origin_array = stack_origin_array)

# Call the stacking function for the remaining levels
print('processing 400 hPa tracks')
# Gather all the arrays of tracks at a single level into a list
track_arrays_all = [z900_tracks, z800_tracks, z700_tracks, z600_tracks,z500_tracks, z400_tracks, z300_tracks, z200_tracks]
# Call the track stacking function to search 2 levels up based at 400 hPa
track_arrays_all, highest_stack_num, stack_origin_array = stack_tracks_up2(track_arrays_all, level = 5, highest_stack_num = highest_stack_num, distance_threshold = DISTANCE_THRESHOLD, stack_origin_array = stack_origin_array)

print('processing 300 hPa tracks')
# Gather all the arrays of tracks at a single level into a list
track_arrays_all = [z900_tracks, z800_tracks, z700_tracks, z600_tracks,z500_tracks, z400_tracks, z300_tracks, z200_tracks]
# Call the track stacking function to search 1 level up based at 300 hPa
track_arrays_all, highest_stack_num, stack_origin_array = stack_tracks_up1(track_arrays_all, level = 6, highest_stack_num = highest_stack_num, distance_threshold = DISTANCE_THRESHOLD, stack_origin_array = stack_origin_array)

# Go through the remaining 200 hpa tracks and assign new stack numbers. 
print('processing 200 hPa tracks')
z200_cyclones_unassigned = z200_tracks[z200_tracks[:,0] == 0, :]
z200_unassigned_track_nums = np.unique(z200_cyclones_unassigned[:,1])

for track_num in z200_unassigned_track_nums:

    highest_stack_num = highest_stack_num + 1
    z200_tracks[z200_tracks[:,1] == track_num, 0] = highest_stack_num

# save output
np.savetxt(data_path + 'low_stacks_900_sh_' + str(y_start) + '.txt', z900_tracks, delimiter = ',')
np.savetxt(data_path + 'low_stacks_800_sh_' + str(y_start) + '.txt', z800_tracks, delimiter = ',') 
np.savetxt(data_path + 'low_stacks_700_sh_' + str(y_start) + '.txt', z700_tracks, delimiter = ',') 
np.savetxt(data_path + 'low_stacks_600_sh_' + str(y_start) + '.txt', z600_tracks, delimiter = ',') 
np.savetxt(data_path + 'low_stacks_500_sh_' + str(y_start) + '.txt', z500_tracks, delimiter = ',') 
np.savetxt(data_path + 'low_stacks_400_sh_' + str(y_start) + '.txt', z400_tracks, delimiter = ',') 
np.savetxt(data_path + 'low_stacks_300_sh_' + str(y_start) + '.txt', z300_tracks, delimiter = ',') 
np.savetxt(data_path + 'low_stacks_200_sh_' + str(y_start) + '.txt', z200_tracks, delimiter = ',') 