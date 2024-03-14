### GIS Tool Overlay To Polygon

This Python script usess the Python Computer Vision library to take in an overlay from an image, which should be a .kml, and then generates a polygon based on it.

The rotation is interpreted a little funky, and seems to distort/offset the final polygon slightly. So an overlay without rotation works much much better.


It is able to give a preview of the area highlighting over by changing PREVIEW_MASK to True.

It is recommended to edit the variables defined after the importing of libraries (Line 28-38). 

Areas - How many disjoint areas are considered. Getting multiple ones will concatenate a polygon together. 

RBGCOLOR - The main RGB values that you should edit
SAVE_IMAGE - Save the mask, but should really know what you are saving so check by previewing the mask beforehand. Will save it as a .png

PREVIEW_MASK - Checks the area you are highlighting and have selected for you polygon.

OPACITY

DELTA

THETA

PREVIEW_MASK_ON_IMAGE

QUIT_UPON_PREVIEW


### Instructions


### Example
