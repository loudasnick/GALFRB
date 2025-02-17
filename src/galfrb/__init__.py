"""
This module serves as the initialization file for the GALFRB package.

The GALFRB package is a collection of functions and classes for analyzing and processing GALFRB data.

"""
__version__ = "0.1.0"

#from .generator import * 
#import utils as utls

# Get the path to the style file
import os
style_path = os.path.join(os.path.dirname(__file__), "../../styles", "nick_style.mplstyle")
# Apply the style
import matplotlib.pyplot as plt 
plt.style.use(style_path)