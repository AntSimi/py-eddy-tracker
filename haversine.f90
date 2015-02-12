module haversine
!
! ===========================================================================
! This file is part of py-eddy-tracker.
! 
!     py-eddy-tracker is free software: you can redistribute it and/or modify
!     it under the terms of the GNU General Public License as published by
!     the Free Software Foundation, either version 3 of the License, or
!     (at your option) any later version.
! 
!     py-eddy-tracker is distributed in the hope that it will be useful,
!     but WITHOUT ANY WARRANTY; without even the implied warranty of
!     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!     GNU General Public License for more details.
! 
!     You should have received a copy of the GNU General Public License
!     along with py-eddy-tracker.  If not, see <http://www.gnu.org/licenses/>.
! 
! Copyright (c) 2014 by Evan Mason
! Email: emason@imedea.uib-csic.es
!
! ===========================================================================
!
!    To compile for f2py do following in terminal:
!    f2py -m haversine -h haversine.pyf haversine.f90  --overwrite-signature
!    f2py -c --fcompiler=gfortran haversine.pyf haversine.f90
!
!    If you have ifort on your system, change 'gfortran' to 'intelem'!
!
!   Version 1.4.2
!
!===========================================================================
implicit none
real(kind=8), parameter :: erad = 6371315.0
real(kind=8), parameter :: d2r = 0.017453292519943295 ! == pi / 180.

contains

!------------------------------------------------------------------------------

    subroutine distance_matrix(xa, xb, dist)
!------------------------------------------------------------------------------

      real(kind=8), dimension(:,:), intent(in)    :: xa, xb
      integer(kind=8)                             :: i, j, m, n
      real(kind=8)                                :: thedist
      real(kind=8), dimension(:,:), intent(inout) :: dist

      m = size(xa, 1)
      n = size(xb, 1)
!       write (*,*) 'm,n', m, n

!     Loop over empty dist matrix and fill
      do j = 1, m
        do i = 1, n
          call get_haversine(xa(j,1), xa(j,2), xb(i,1), xb(i,2), thedist)
          dist(j,i) = thedist
!           write (*,*) 'dist(j,i)', dist(j,i)
        enddo
      enddo
      dist = dist * erad
!       write (*,*) 'dist', dist
    end subroutine distance_matrix


!------------------------------------------------------------------------------
    subroutine distance_vector(lon1, lat1, lon2, lat2, dist)
!------------------------------------------------------------------------------
      
      real(kind=8), dimension(:), intent(in) :: lon1, lat1, lon2, lat2
      integer(kind=8)                        :: i, m
      real(kind=8)             :: thedist
      real(kind=8), dimension(:), intent(inout) :: dist
      
      m = size(lon1)
      
!     Loop over empty dist matrix and fill
      do i = 1, m
        call get_haversine(lon1(i), lat1(i), lon2(i), lat2(i), thedist)
        dist(i) = thedist
      enddo
      dist = dist * erad
    
    end subroutine distance_vector

    
!------------------------------------------------------------------------------
    subroutine distance(lon1, lat1, lon2, lat2, thedist)
!------------------------------------------------------------------------------
      
      real(kind=8), intent(in) :: lon1, lat1, lon2, lat2
      real(kind=8), intent(out) :: thedist
!
      call get_haversine(lon1, lat1, lon2, lat2, thedist)
      thedist = thedist * erad
    
    end subroutine distance

    
!------------------------------------------------------------------------------
    subroutine get_haversine(lon1, lat1, lon2, lat2, thedist)
!------------------------------------------------------------------------------
!
      real(kind=8), intent(in)  :: lon1, lat1, lon2, lat2
      real(kind=8)              :: lt1, lt2, dlat, dlon
      real(kind=8)              :: a
      real(kind=8), intent(out) :: thedist
!
      dlat = d2r * (lat2 - lat1)
      dlon = d2r * (lon2 - lon1)
      lt1 = d2r * lat1
      lt2 = d2r * lat2
!
      a = sin(0.5 * dlon) * sin(0.5 * dlon)
      a = a * cos(lt1) * cos(lt2)
      a = a + (sin(0.5 * dlat) * sin(0.5 * dlat))
      thedist = 2 * atan2(sqrt(a), sqrt(1 - a))
!
    end subroutine get_haversine


!------------------------------------------------------------------------------
    subroutine waypoint_vector(lonin, latin, anglein, distin, lon, lat)
!------------------------------------------------------------------------------
      
      real(kind=8), dimension(:), intent(in) :: lonin, latin, anglein, distin
      integer(kind=8)                        :: i, m
      real(kind=8)             :: thelon, thelat
      real(kind=8), dimension(:), intent(inout) :: lon, lat
!       external :: waypoint

      m = size(lonin)
      
!     Loop over empty dist matrix and fill
      do i = 1, m
        call get_waypoint(lonin(i), latin(i), anglein(i), distin(i), thelon, thelat)
        lon(i) = thelon
        lat(i) = thelat
      enddo
    
    end subroutine waypoint_vector


!------------------------------------------------------------------------------
    subroutine get_waypoint(lonin, latin, anglein, distin, thelon, thelat)
!------------------------------------------------------------------------------
!
      implicit none
      real(kind=8), intent(in)  :: lonin, latin, anglein, distin
      real(kind=8)              :: ln1, lt1, angle, dr
      real(kind=8), intent(out) :: thelon, thelat
!
      dr = distin / erad ! angular distance
      thelon = d2r * lonin
      thelat = d2r * latin
      angle = d2r * anglein
      
      lt1 = asin(sin(thelat) * cos(dr) + cos(thelat) * sin(dr) * cos(angle))
      ln1 = atan2(sin(angle) * sin(dr) * cos(thelat), cos(dr) &
                                      - sin(thelat) * sin(lt1))
      ln1 = ln1 + thelon
      thelat = lt1 / d2r
      thelon = ln1 / d2r
!
    end subroutine get_waypoint

    

    
!------------------------------------------------------------------------------
end module haversine
!------------------------------------------------------------------------------
