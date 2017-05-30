# -*- coding: utf-8 -*-
"""Class to manage data at the input of eddy detection and general
function gridded data
"""

class InputData(object):
    
    __slots__ = (
        '_area'
        )
    
    @area.setter
    def area(self, value):
        self._area = value
    
    @property
    def area(self):
        return self._area

    def view(self, offset=0):
        pass

    @property
    def geostrophic_velocity(self):
        pass

    def filtering(self, x_value, y_value):
        pass


class RegularGrid(InputData):
    __slots__ = ()


class StucturedGrid(InputData):
    __slots__ = ()
