# This code locates cyclone centers as local minima in geopotential height in reanalysis data. 
#
# It outputs the located cyclone centers in a .txt file containing the following columns:
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
import xarray as xr
from datetime import datetime
import matplotlib.dates as dt
import pandas as pd
import math
from geopy.distance import geodesic as gd
import argparse

#================== Define Key Functions ========================================
def preprocessing(ds):
    
    '''This function is applied in the call of xarray open_mfdataset, so it reduces the size of the 
    output xarray dataset by only selecting the data at the required pressure level straight away.'''
    
    # Pre-select the dimensions
    ds = ds.sel(level=p_level)
    
    return ds
    
def netcdf_time_scale(y,m,d,h,y0):
    
    '''This function takes a date in y,m,d,h format and returns the value of the
    hours since jan 1, y0 (the time convention of reanalysis). Inputs are integers. 
    Output is a float.''' 
    
    d0 = datetime(y0,1,1,0) # the datetime object representing 12am, jan 1 of the baseline year of the reanalysis
    d = datetime(y,m,d,h) # the datetime object for the day and time being processed
    
    t = 24 * (dt.date2num(d) - dt.date2num(d0)) # number of hours since 12am, jan 1 of the baseline year
    # dt.date2num gives the number of days since the python datetime time origin. 
    
    return t

def date_from_dt64(dt64):
    
    '''This function takes a datetime64 object and converts it to a date in y,m,d,h format, for when you want to
    have the year, month, day, and hour values as separate variables. Output is a tuple. '''

    # the pandas function Timestamp creates an object with year, month, day, and hour from a datetime64 object.
    date = pd.Timestamp(dt64)
    
    y = date.year
    m = date.month
    d = date.day
    h = date.hour
    
    return y,m,d,h

def horizontal_distance(low1, low2):
    
    '''This function takes a pair of (lat,lon) coordinates (i.e. low1 and low2), and computes the linear distance between 
    them using Pythagoras's theorem. Output is a float. '''

    # latitude components of the two points
    lat1 = low1[0]
    lat2 = low2[0]

    # longitude components of the two points
    lon1 = low1[1]
    lon2 = low2[1]

    # compute the difference in latitude between the two points.
    lat_distance = abs(lat2 - lat1)

    # compute the difference in longitude between the two points. This part handles the situation where the two lows are 
    # close on either side of the meridian of 0 degrees longitude. 
    lon_dist1 = abs(lon2 - lon1)
    lon_dist2 = 360 - abs(lon2 - lon1)
    lon_distance = np.minimum(lon_dist1, lon_dist2) # np.minimum is important
    # here so this function can be vectorised across a whole array of low centers.

    # apply Pythagoras's theorem to compute the distance between the two points
    distance = (lat_distance ** 2 + lon_distance ** 2) ** 0.5
    
    return distance

def eliminate_secondary_lows(low_array, tstep_col, lat_col, lon_col, cell_col, test2_col, distance_threshold):
    
    '''This function removes shallow low centers that are close to a deeper parent
    low center. It does this by looping through each low center identified in a field, 
    and comparing it to each other low center in the same field. If the two lows are 
    within a given distance threshold, the shallower low is overwritten with np.nan. Only 
    the non-nan lows are then returned out of the function. Output is a numpy array.'''
    
    # get each individual timestep with at least 1 cyclone center to loop through
    timerange = np.unique(low_array[:,tstep_col]).astype(int)
    
    for tstep in timerange: 
        
        # extracts all of the rows in the low_array for the timestep being tested. fieldlows is a copy because 
        # it is indexed with a boolean.
        ifield = (low_array[:,tstep_col] == tstep)
        fieldlows = low_array[ifield, :]

        # set up a loop through each of the low centers in the array fieldlows
        arraylen = np.shape(fieldlows)[0]
        arrayloop = np.arange(0,arraylen,1)

        # loop through each low in fieldlows, pull out each other low, and compare their depths and the distances between them.
        for a in arrayloop:

            # get the low at row a, and loop through the other lows in the remaining rows (i.e. row b). 
            # Low1 and low2 will be references to the data in-place, because they are indexed with integers.
            low1 = fieldlows[a,:]
            array_subloop = np.arange(a + 1, arraylen, 1)
                
            for b in array_subloop:

                # get the other low to compare the depth with
                low2 = fieldlows[b,:]
                    
                # These conditional tests indicate whether the two closed low centers are within the spatial 
                # proximity of each other, and which low center is the deepest. 

                # compute the distance between the two low centers using the function we defined earlier
                lin_dist = horizontal_distance((low1[lat_col], low1[lon_col]), (low2[lat_col], low2[lon_col]))

                # boolean test for whether low 1 and low2 are within the spatial proximity
                distance_test = lin_dist <= distance_threshold 

                # booleans for whether low1 or low2 are deeper, or if they have the same depth
                low1_shallower = (low1[cell_col] > low2[cell_col])
                low2_shallower = (low2[cell_col] > low1[cell_col])
                equal_lowtest = (low2[cell_col] == low1[cell_col])
                    
                if distance_test and low1_shallower:
                        
                    low1[:] = np.nan
                        
                elif distance_test and low2_shallower:
                        
                    low2[:] = np.nan

                # this part handles cases where two close low centers actually have the same central depth
                elif distance_test and equal_lowtest:
                        
                    # retain the cyclone center with the strongest test2 pressure gradient  
                    if low1[test2_col] < low2[test2_col]:
                            
                        low1[:] = np.nan
                            
                    elif low2[test2_col] < low1[test2_col]:
                            
                        low2[:] = np.nan
                                         
                    else:
                        
                        # It may be helpful to receive an allert when two low centers of equal depth are 
                        # located in the array (which does occasionally happen). In this case we just choose
                        # to keep the low1
                        print('FOUND TWO LOW CENTERS OF EQUAL DEPTH IN FIELD.')
                        print(low1)
                        print(low2)

                        # because we just have to choose one
                        low2[:] = np.nan
            
        # The filtered low data (including np.nan rows) are written back into 
        # the base low array. Fieldlows will be modified, because its rows were indexed with integers. 
        low_array[ifield,:] = fieldlows
    
    # This stage removes all the rows from the array that were blanked out with
    # np.nan. 
    ir = np.isnan(low_array[:,1])
    refined_lows = low_array[(ir == False),:]
    
    return refined_lows
    
def assign_cyclone_id(array, tstep_col, level):

    ''' This function assigns a unique number to each cyclone center consisting of the following information:
     - 4 digits for the pressure level;
     - date in the format of YYYYMMDDHH;
     - 2 digits for the cyclone center number at that pressure level at that timestep (hopefully there are never more than 99
     cyclone centers found somewhere in the world at any one time).
     The function outputs the array with the id numbers added at the end. This is helpful for when you identify and track cyclones
     different levels, and then combine them in the same program later for further analysis. '''
    
    max_field_cyclones = 0 # This variable will keep track of the greatest number of cyclone centers that are ever reccorded
    # at one timestep. If there are ever more than 99 cyclone centers in a single timestep, the centre_id function will need
    # to be expanded to add a third digit for the cyclone centre number. 

    # get each individual timestep to loop through
    for t in np.unique(array[:,tstep_col]):

        # extracts all of the rows in the low_array for the timestep being tested
        ifield = (array[:,tstep_col] == t)
        fieldlows = array[ifield, :]

        # Set up a loop through each of the lows in fieldlows
        fieldlen = np.shape(fieldlows)[0]
        fieldloop = np.arange(0,fieldlen,1)

        # loop through each low in fieldlows, and assign levlevlevlevyyyymmddhh##.    
        for row in fieldloop:

            # get the integer components of the strings
            t = int(fieldlows[row, tstep_col])
            n = row + 1 # this gets the count number for the cyclone center in that field. + 1 is added so the first
            # row (i.e. 0th row) is written as the first cyclone in the field.

            # convert them to strings
            llll = str(level) # this will lose any leading zeros when it is written back into the array as a number :(
            ttttttt = str(t).zfill(7) 
            nn = str(n).zfill(2) 

            # assemble the cyclone id here
            cyc_id = llll + ttttttt + nn

            # convert the cyc_id back to an integer and write it into fieldlows
            fieldlows[row, 0] = int(cyc_id)

        # write the fieldlows back into the orriginal array
        array[ifield, :] = fieldlows

        # check if the number of lows in that field is greater than the previous maximum
        if fieldlen > max_field_cyclones:

            max_field_cyclones = fieldlen

    print('Max cyclones in a field: ', max_field_cyclones)

    return array

# =========================== Take Input Parameters from Shell Script =====================

# generate the object that receives the input variables from the linux script
parser = argparse.ArgumentParser(description = "Pressure level to identify Cyclones")
parser.add_argument("--p_level", required=True, type=int)
    
args = parser.parse_args()
    
p_level = args.p_level

#============================= Parameters set by User ===================================

dataset = 'era5' 

data_path = '/g/data/w40/nxg561/ERA5/' # location of reanalysis files
save_path = '/g/data/m35/nxg561/02_Cyclone_Vertical_Structure/Cyclone_Identification/Nov2025/' # location to save output text file

#Time period setup
year_range = (1979, 2022) #(start year,end year)
startdate = (1,1) #(month,day)
enddate = (12,31) #(month,day)

#Spatial domain setup
lat_range = (-4.5,-78) #Boundary of region in latitude
lon_range = (-180, 178.5) #Boundary of region in longitude 

# Reanalysis data properties
t_res = 6 #hrs
spacial_res = 1.5 #degrees lat/lon

#p_level = 900 #comment this out if taking input parameters from a shell script  

time_y0 = 1900
    
lat_variable_name = 'lat'
lon_variable_name = 'lon'
vertical_coordinate_name = 'level' # 'level' for ERA5
time_variable_name = 'time'
gph_variable_name = 'z' # 'z' for ERA5

#cyclone diagnostic properties setup
hgt_threshold = 0 #gpm. The minimum difference in pressure between the center
            #square and each of the surrounding 8 squares to identify a closed 
            #low.
            
pg_threshold = 0  #hPa. Similar as above, in case the code is run on mslp data.

#Filtering parameters
filter_minor_lows = True

low_filtering_spatial_threshold = 10 #The maximum distance (degrees lat/lon) between two low centers 
            #in the same pressure field to consider them the same system.

# file save parameters
concatenate_years = True # set this to false if you are happy to have the cyclones recorded in a file for each year separately.

# ================================== Hard-coded and computed Parameters ========================

g = 9.8065 # geopotential data needs to be divided by gravity to get height

# column names and numbers for the output array
CENTER_ID_COL = 0
YEAR_COL = 1
MONTH_COL = 2
DAY_COL = 3
HOUR_COL = 4
TSTEP_COL = 5
LAT_COL = 6
LON_COL = 7
CELL_COL = 8
TEST1_COL = 9
TEST2_COL = 10

# This part computes the conversion factor between the resolution of ncep1 (the 
# original dataset that this code was developed for) and the resolution of the 
# dataset being used here. 
scale_factor = 2.5 / spacial_res
    
r1 = math.floor(1 * scale_factor) # number of grid spaces for 1 NCEP1 grid space equivalent
r2 = math.floor(2 * scale_factor) # number of grid spaces for 2 NCEP1 grid spaces equivalent

#====================== Data Input and Cyclone Detection =======================================

# loop through each year separately, identify the local minima, and output the array as a .txt file for that year. 
for year in range(year_range[0], year_range[1] + 1):

    # create the raw_lows_z array here for each new year. I just guessed 500 x the total number of 
    # timesteps in one year would be enough rows.
    raw_lows_z = np.full((500 * 365 * 4, 11), np.nan) 
    
    # Load all the reanalysis files in a multi-file dataset, and find all the locations of local minima:
    xr_obj = xr.open_mfdataset(data_path + 'z_era5_oper_pl_' + str(year) + '*.nc', combine='nested', parallel=True, preprocess=preprocessing, concat_dim='time', engine='netcdf4')
    
    # Extract the data for the search domain, as well as the domain shifted one and
    # two grid spaces to the north and south. The east and west shifted data will
    # be obtained with a np.roll() function later. These variables are an xarray data array. 
    hgt_data = xr_obj[gph_variable_name].sel(lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1]))
    
    hgt_data_north = xr_obj[gph_variable_name].sel(lat=slice(lat_range[0] + r1 * spacial_res, lat_range[1] + r1 * spacial_res), lon=slice(lon_range[0], lon_range[1]))
    hgt_data_north2 = xr_obj[gph_variable_name].sel(lat=slice(lat_range[0] + r2 * spacial_res, lat_range[1] + r2 * spacial_res), lon=slice(lon_range[0], lon_range[1]))
    
    hgt_data_south = xr_obj[gph_variable_name].sel(lat=slice(lat_range[0] - r1 * spacial_res, lat_range[1] - r1 * spacial_res), lon=slice(lon_range[0], lon_range[1]))
    hgt_data_south2 = xr_obj[gph_variable_name].sel(lat=slice(lat_range[0] - r2 * spacial_res, lat_range[1] - r2 * spacial_res), lon=slice(lon_range[0], lon_range[1]))

    # Extract the numpy arrays to work with the actual data
    hgt_field = hgt_data.values
    hgt_field_north = hgt_data_north.values
    hgt_field_north2 = hgt_data_north2.values
    hgt_field_south = hgt_data_south.values
    hgt_field_south2 = hgt_data_south2.values
    
    # Some reanalysis datasets need to have geopotential divided by g = 9.8 m/s^2 to get geopotential height
    if dataset == 'era5' or dataset == 'eraint':
            
            hgt_field = hgt_field / 9.8
            hgt_field_north = hgt_field_north / 9.8
            hgt_field_north2 = hgt_field_north2 / 9.8
            hgt_field_south = hgt_field_south / 9.8
            hgt_field_south2 = hgt_field_south2 / 9.8
            
    # Do the rolls to set up the east, west, north-east, north-west, south-east and
    # south-west shifted data here        
    hgt_field_west = np.roll(hgt_field, r1, axis=2)
    hgt_field_east = np.roll(hgt_field, -r1, axis=2)
    
    hgt_field_northeast = np.roll(hgt_field_north, -r1, axis=2)
    hgt_field_northwest = np.roll(hgt_field_north, r1, axis=2)
    hgt_field_southeast = np.roll(hgt_field_south, -r1, axis=2)
    hgt_field_southwest = np.roll(hgt_field_south, r1, axis=2)
    
    hgt_field_west2 = np.roll(hgt_field, r2, axis=2)
    hgt_field_east2 = np.roll(hgt_field, -r2, axis=2)
    
    hgt_field_northeast2 = np.roll(hgt_field_north2, -r2, axis=2)
    hgt_field_northwest2 = np.roll(hgt_field_north2, r2, axis=2)
    hgt_field_southeast2 = np.roll(hgt_field_south2, -r2, axis=2)
    hgt_field_southwest2 = np.roll(hgt_field_south2, r2, axis=2)
    
    # compute the differences between the points in the central field and the points in the shifted fields:      
    t1n = hgt_field_north - hgt_field
    t1ne = (hgt_field_northeast - hgt_field) / (2 ** 0.5)
    t1e = hgt_field_east - hgt_field
    t1se = (hgt_field_southeast - hgt_field) / (2 ** 0.5)
    t1s = hgt_field_south - hgt_field
    t1sw = (hgt_field_southwest - hgt_field) / (2 ** 0.5)
    t1w = hgt_field_west - hgt_field
    t1nw = (hgt_field_northwest - hgt_field) / (2 ** 0.5)
    
    t2n = hgt_field_north2 - hgt_field
    t2ne = (hgt_field_northeast2 - hgt_field) / (2 ** 0.5)
    t2e = hgt_field_east2 - hgt_field
    t2se = (hgt_field_southeast2 - hgt_field) / (2 ** 0.5)
    t2s = hgt_field_south2 - hgt_field
    t2sw = (hgt_field_southwest2 - hgt_field) / (2 ** 0.5)
    t2w = hgt_field_west2 - hgt_field
    t2nw = (hgt_field_northwest2 - hgt_field) / (2 ** 0.5)
    
    # low_test is a numpy array of True/False values, where a True value corresponds to a point in the 
    # geopotential height field that is lower than each of the 16 surrounding points. The shape of the 
    # low_test array corresponds to the shape of the hgt_field (i.e. time, lat, lon.)
    low_test = (t1n >= 0) & (t1ne >= 0) & (t1e >= 0) & (t1se >= 0) & (t1s >= 0) & (t1sw >= 0) & (t1w >= 0) & (t1nw >= 0) & (t2n >= hgt_threshold) & (t2ne >= hgt_threshold) & (t2e >= hgt_threshold) & (t2se >= hgt_threshold) & (t2s >= hgt_threshold) & (t2sw >= hgt_threshold) & (t2w >= hgt_threshold) & (t2nw >= hgt_threshold)

    # Get the values of cell, test1 and test2 values for the cyclone centers here. cell refers to the central geopotential height
    # of the cyclone, test1 is the mean difference in geopotential height between the center and the points r1 grid spaces away, and 
    # test2 is the mean difference in geopotential height between the center and the points r2 grid spaces away.
    cell_values = hgt_field[low_test]
    
    test1_array = np.mean((t1n, t1ne, t1e, t1se, t1s, t1sw, t1w, t1nw), axis=0)
    test1_values = test1_array[low_test]

    test2_array = np.mean((t2n, t2ne, t2e, t2se, t2s, t2sw, t2w, t2nw), axis=0)
    test2_values = test2_array[low_test]

    # the time, lat and lon scales of the xarray object to index
    domain_lats = hgt_data['lat'].values
    domain_lons = hgt_data['lon'].values
    domain_time = hgt_data['time'].values

    # loop through the identified local minima, and compute the other entries for the output array here (i.e. year, month
    # day, hour etc. unfortunately the time stamp computation cannot be vectorised :(

    # cyclone_inds is a tuple of three arrays consisting of the time, lat and lon
    # indices of each of the 'True' values in low_test.
    cyclone_inds = np.where(low_test)

    # set up the loop of cyclones 
    local_minima_loop = range(0,len(cyclone_inds[0]), 1)

    low_center_count = 0 # will get advanced by 1 as each cyclone center is processed

    for ic in local_minima_loop:

        # get the time, lat and lon indices for the cyclone number ic
        t_ind = cyclone_inds[0][ic]
        lat_ind = cyclone_inds[1][ic]
        lon_ind = cyclone_inds[2][ic]

        # get the year, month, day, hour, and hours since y0 from the xarray timestamp
        y,m,d,h = date_from_dt64(domain_time[t_ind])

        # write the cyclone data into the array raw_lows_z
        raw_lows_z[low_center_count,YEAR_COL] = y  
        raw_lows_z[low_center_count,MONTH_COL] = m
        raw_lows_z[low_center_count,DAY_COL] = d
        raw_lows_z[low_center_count,HOUR_COL] = h
        raw_lows_z[low_center_count,TSTEP_COL] = netcdf_time_scale(y,m,d,h,time_y0)

        # get the lat and lon values for the cyclone ic and write into the array raw_lows_z
        raw_lows_z[low_center_count,LAT_COL] = domain_lats[lat_ind]
        raw_lows_z[low_center_count,LON_COL] = domain_lons[lon_ind]

        # get the central cell value, test1 and test2 values for the cyclone ic and write into the array raw_lows_z
        raw_lows_z[low_center_count,CELL_COL] = cell_values[ic]
        raw_lows_z[low_center_count,TEST1_COL] = test1_values[ic]
        raw_lows_z[low_center_count,TEST2_COL] = test2_values[ic]

        # advance the cyclone number count by 1
        low_center_count = low_center_count + 1

    # Remove any blank rows of raw_lows_z
    ir = np.isnan(raw_lows_z[:,1]) == False
    raw_lows_z = raw_lows_z[ir, :]

    # Remove weaker lows in close proximity to a deeper parent low
    if filter_minor_lows:
    
        raw_lows_z = eliminate_secondary_lows(raw_lows_z, TSTEP_COL, LAT_COL, LON_COL, CELL_COL, TEST2_COL, low_filtering_spatial_threshold)

    # assign the cyclone id here
    raw_lows_id = assign_cyclone_id(raw_lows_z, TSTEP_COL, p_level)

    # save output
    output_filename = str(save_path) + 'rawlows_' + str(p_level) + '_' + str(year) + '_sh.txt'
    np.savetxt(output_filename, raw_lows_id, delimiter=',') 

# ======================== Concatenate year files into one file ==================================
if concatenate_years: 
    
    # create an empty list to store the cyclone arrays for each year
    array_list = []

    # loop through each year in the range, open the file of cyclone centers, and append the cyclones to the list
    for year in range(year_range[0],year_range[1] + 1):

        # open the data
        year_array = np.loadtxt(data_path + 'rawlows_' + str(p_level) + '_' + str(year) + '_sh.txt', delimiter=',')

        # append the data to the list
        array_list.append(year_array)

    # combine the list of arrays into one array
    full_array = np.concatenate(array_list, axis=0)

    # save final output file of full_array
    output_filename = str(save_path) + 'rawlows_' + str(p_level) + '_sh.txt'
    np.savetxt(output_filename, full_array, delimiter=',') 