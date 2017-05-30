"""Application to enhance data
"""
from py_eddy_tracker import EddyParser
from py_eddy_tracker.dataset import EddyDataset



class EddyUpdateParser(EddyParser):
    def __init__(self, *args, **kwargs):
        super(EddyUpdateParser, self).__init__(*args, **kwargs)
        self.add_argument(
            'input_file',
            help='EddyObservation/EddyTracking NetCDF File')
        self.add_argument(
            'output_file',
            help='NetCDF File to store new data')
        group = self.add_argument_group('Fields available')
        group.add_argument(
            '--all',
            action='store_true',
            help='Add all fields available')
        group.add_argument(
            '--dist', '--distance_between_observations',
            action='store_true',
            help='Add Distance fields between to consecutive'
                 ' observation from the same track')
    
def eddy_update():
    parser = EddyUpdateParser()
    opts = parser.parse_args()

    # Create an object which represent Dataset
    handler = EddyDataset.load(opts.input_file)

    print handler._extract(None)


    # Add distance fields
    if opts.all or opts.dist:
        handler.add_distance()

    # Save Dataset
    handler.write(opts.output_file)
