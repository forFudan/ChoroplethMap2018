#%%
# Packages
from lxml import etree
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from pysal.esda.mapclassify import Natural_Breaks as nb
from descartes import PolygonPatch
import fiona
import geopandas
#from itertools import chain
from copy import deepcopy
#%matplotlib inline

# Import functions
from include.map import cmap_discretize
from include.map import colorbar_index
from include.map import self_categorize

"""
Start of the main program
"""
# paths

path_china = 'source/shape/CHN_adm_shp/CHN_adm3'
path_taiwan = 'source/shape/TWN_adm_shp/TWN_adm2'
path_china_shape = 'source/shape/CHN_adm_shp/CHN_adm3.shp'
path_taiwan_shape = 'source/shape/TWN_adm_shp/TWN_adm2.shp'

path_data_file = 'source/data/price.csv'
path_output = 'output/landchina_2013_2016.png'
# Border adjustment
extra = 0.01


#%%
"""
Fiona first read the map shape file downloaded from https://gadm.org/.
When download the shapefile, be careful in which level of admin region you are downloading.
Then read the bounds of the shape file to calculate the height and the width.
"""

shape = fiona.open(path_china_shape)    # Read in shape
bounds = shape.bounds    # Read in bounds
shape.close()    # Close shapes

# South 0, West 1.
south, west, north, east = bounds
(width, height) = (north - south, east - west)

"""
lower_left = (bounds[0], bounds[1]) # lower-left
upper_right = (bounds[2], bounds[3]) # upper-right

coordinates = list(chain(lower_left, upper_right))
(width, height) = (coordinates[2] - coordinates[0], coordinates[3] - coordinates[1])
print(coordinates)
"""

print('The map has width = {}, height = {}'.format(width, height))

"""
Use Basemap to create the frame of the map (empty).
Then use this instance to read in the shape files.
"""

map_china = Basemap(
    projection = 'tmerc',
    lon_0 = (south + north) / 2, # Longitude center
    lat_0 = (west + east) / 2, # Latitude center
    ellps = 'WGS84',
    # Adjustment to the border of the map
    llcrnrlon = south - extra + 0.05 * width,
    llcrnrlat = west - extra - 0.1 * height,
    urcrnrlon = north + extra + 0.05 * width,
    urcrnrlat = east + extra - 0.1 * height,
    lat_ts = 0,
    resolution = 'i',
    suppress_ticks = True
    )

map_taiwan = deepcopy(map_china)

print('The frame of the map frame created!')

map_china.readshapefile(
    path_china,
    name = 'china',
    color = 'none',
    zorder = 2 )

map_taiwan.readshapefile(
    path_taiwan,
    name= 'taiwan' ,
    color= 'none' ,
    zorder = 2)

print('Basemap has read the shape files.')

#%%
"""
Prepare for the data in each region.
The data is prepared using DataFrame.
Calculate the region of polies and regions.
    "shape" in the map_china.china is a list of x y coordinates.
    It can be visualized by Polygon.
"""
"""
This test polygon is strange but needed.
If you directly use Polygon on the data, error raises:
Shell is not a LinearRing
So you first run the test polygon. Then you use it on the data.
"""
test_polygon = Polygon([(0,0),(1,1),(1,0)])

poly_china = [Polygon(shape) for shape in map_china.china] 
poly_taiwan = [Polygon(shape) for shape in map_taiwan.taiwan]
region_china = [region['NAME_3'] for region in map_china.china_info]
region_taiwan = [region['NAME_2'] for region in map_taiwan.taiwan_info]

# set up a map dataframe for China
df_map_china = pd.DataFrame({
    'poly': poly_china,
    'region': region_china})

df_map_china['poly_area_m2'] = df_map_china['poly'].map(lambda x: x.area)
df_map_china['poly_area_km2'] = df_map_china['poly_area_m2'] / 1000000
df_map_china['poly_area_100km2'] = df_map_china['poly_area_km2'] / 100
df_map_china['region_area_m2'] = df_map_china.region.apply(df_map_china.groupby('region').sum().poly_area_m2.get_value)
df_map_china['region_area_km2'] = df_map_china['region_area_m2'] / 1000000
df_map_china['region_area_100km2'] = df_map_china['region_area_km2'] / 100

# set up a map dataframe for Taiwan
df_map_taiwan = pd.DataFrame({
    'poly': poly_taiwan,
    'region': region_taiwan})
df_map_taiwan['poly_area_m2'] = df_map_taiwan['poly'].map(lambda x: x.area)
df_map_taiwan['poly_area_km2'] = df_map_taiwan['poly_area_m2'] / 1000000
df_map_taiwan['poly_area_100km2'] = df_map_taiwan['poly_area_km2'] / 100
df_map_taiwan['region_area_m2'] = df_map_taiwan.region.apply(df_map_taiwan.groupby('region').sum().poly_area_m2.get_value)
df_map_taiwan['region_area_km2'] = df_map_taiwan['region_area_m2'] / 1000000
df_map_taiwan['region_area_100km2'] = df_map_taiwan['region_area_km2'] / 100

print('Area calculated!')

rows_china = df_map_china.poly.count()
rows_taiwan = df_map_taiwan.poly.count()
df_map_taiwan.set_index([list(range(rows_china+1, rows_china+rows_taiwan+1))], inplace=True)


"""
Combine the maps.
"""

df_map = df_map_china.append(df_map_taiwan)

"""
Import the data file.
"""

df_price = pd.read_csv(path_data_file)
df_map = pd.merge(df_map, df_price, how='left', on='region')


df_map['density_m2'] = df_map['price(million)'] / df_map['region_area_m2'] /100
df_map['density_km2'] = df_map['price(million)'] / df_map['region_area_km2'] /100
df_map['density_100km2'] = df_map['price(million)'] / df_map['region_area_100km2'] /100

df_map.replace(to_replace={'density_m2': {0: np.nan}, 'density_km2': {0: np.nan}, 'density_100km2': {0: np.nan}}, inplace=True)

"""
# Calculate Jenks natural breaks for density
breaks = nb(
    df_map[df_map['density_100km2'].notnull()]['density_100km2'].values,
    initial=300,
    k=5)
# the notnull method lets us match indices when joining
jb = pd.DataFrame({'jenks_bins': breaks.yb}, index=df_map[df_map['density_100km2'].notnull()].index)
df_map = df_map.join(jb)
df_map.jenks_bins.fillna(-1, inplace=True)
jenks_labels = ["<= {:d} establishments per 100 km$^2$".format(int(b)) for b in breaks.bins]
# jenks_labels.insert(0, 'No region (%s regions)' % len(df_map[df_map['density_km'].isnull()]))
"""

print(min(list(df_map['density_100km2'].values)))
print(max(list(df_map['density_100km2'].values)))


breaks = [0., 8., 128., 2048., 16384., 262144.] + [1e20]
df_map['jenks_bins'] = df_map['density_100km2'].apply(self_categorize, args=(breaks,))
jenks_labels = ["<= {:d} million RMB per 100 km$^2$".format(int(perc)) for perc in breaks[1:-1]]

print(jenks_labels)


print(min(list(df_map['jenks_bins'].values)))
print(max(list(df_map['jenks_bins'].values)))

#%%
"""
Plot the map.
"""

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, frame_on=False)

# use a blue colour ramp - we'll be converting it to a map using cmap()
# http://matplotlib.org/examples/color/colormaps_reference.html
cmap = plt.get_cmap('Blues')

# draw wards with grey outlines
df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#555555', lw=.2, alpha=0.75, zorder=4))
pc = PatchCollection(df_map['patches'], match_original=True)

# impose our colour map onto the patch collection
norm = Normalize()
pc.set_facecolor(cmap(norm(df_map['jenks_bins'].values)))
ax.add_collection(pc)

# Add a colour bar
cb = colorbar_index(ncolors=len(jenks_labels), cmap=cmap, shrink=0.3, labels=jenks_labels)
cb.ax.tick_params(labelsize=7)

# # Show highest densities, in descending order
# highest = '\n'.join(
#     value[1] for _, value in df_map[(df_map['jenks_bins'] == 4)][:10].sort().iterrows())
# highest = 'Most Dense Region:\n' + highest

# # Subtraction is necessary for precise y coordinate alignment
# details = cb.ax.text(
#     -1., 0 - 0.007,
#     highest,
#     ha='right', va='bottom',
#     size=5,
#     color='#555555')

# Bin method, copyright and source data info
smallprint = ax.text(
    1, 0.1,
    'Data from: www.landchina.com\nData for Taiwan not available\nPlotted by: Yuhao Zhu\nCopyright at 2017',
    ha='left', va='bottom',
    size=7,
    color='#555555',
    transform=ax.transAxes)


map_taiwan.scatter(
    [geom.x for geom in []],
    [geom.y for geom in []],
    15, marker='o', lw=.5,
    facecolor='grey', edgecolor='w',
    alpha=1.0, antialiased=True,
    label='Million RMB per square kilometers of land sold 2013 to 2016', zorder=3, ax=ax,)


# Draw a map scale
# map_china.drawmapscale(coords[0] + 0.08, coords[1] + -0.01,
#     (coords[0] + coords[2])/2, (coords[1] + coords[3])/2, length = 2000.,
#     fontsize=6, barstyle='fancy', labelstyle='simple',
#     fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
#     zorder=0, ax=ax,)

ax.set_title('China Land Leasing at County Level from 2013 to 2016')

# this will set the image width
# plt.tight_layout()
plt.subplots_adjust(left=-0.02)
fig.set_size_inches(10,8)
plt.savefig(path_output, dpi=720, alpha=True)
plt.show()