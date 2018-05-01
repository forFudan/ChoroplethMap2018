# ChoroplethMap
Draw choropleth maps.

The final target of this project is to make functions that automatically generate maps with parameters and the data set.

An example figure looks like:

![Alt text](China_landleasing/landchina_2013.png)

Python packages required for the project:

#### mpl_toolkits.basemap
basemap:        1.0.7-np113py35_0
geos:           3.5.0-0
```
brew install geos    
sudo -H pip3 install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz   
``` 
or use
```    
conda install basemap
```

#### shapely
```
pip install shapely    
```
shapely-1.6.4.post1

#### pysal
```
pip install pysal    
```
pysal-1.14.3

#### descartes
```
pip install descartes    
```
descartes-1.1.0

#### fiona
```
pip install fiona    
```
click-plugins-1.0.3 cligj-0.4.0 fiona-1.7.11.post1 munch-2.3.1

### Imports
```
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
import pysal.esda.mapclassify as mc
from descartes import PolygonPatch
import fiona
from itertools import chain
```
