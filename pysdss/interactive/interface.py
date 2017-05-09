"""User interfaces for Stars."""
import cmd
import sys
#import pickle
#from pathlib import Path

class TextInterface(cmd.Cmd):
    """Text-based interface for Stars"""

    def __init__(self):
        super().__init__(completekey = 'tab')
        print ("\nWelcome to the Stars command-line\n")
        print ("main commands:")
        print("\tquit\tto quit the command-line")
        print("\thelp\tto get the list of commands")		
        self.prompt = 'Stars> '

    def do_workspace(self, args):
        """Set the input/output folder.\nArguments\n\tabsolute path for the working folder\n"""

    def do_inputs(self, args):
        """\nthis will set the inputs: raster path, shape path, field with the class value
        \npaths are absolute or relative to the workspace
        \nthis will check if the features are inside the raster, also if the coordinate systems of raster and vectors are the same\n"""

    def do_train(self, arg):
        """define training/validation data
        \nthis will define and save to disk training/validation data
        \nthe arguments are:percentage of validation data,training type,training error
        \ntraining type: mean  - this will use the polygon mean for the training data, validation will be both mean and single pixel values for the validation polygons
        \ntraining type: pixel - this will use the single polygon pixels for the training data, validation will be both mean and single pixel values for the validation polygons
        \ntraining number error - if we train by pixels we may not be able to correctly  get the exact percentage of validation data when we separate training and validation polygons; this number set the maximum allowed difference between the correct number of pixels and the defined number of pixels, the program will try to stay below this number\n"""

    def do_load(self, arg):
        """load from disk existing training validation data"""


    def do_quit(self, arg):
        """Exit Stars"""
        sys.exit(0)