! haversine_distmat.f90
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
!    f2py -m haversine_distmat -h haversine_distmat.pyf haversine_distmat.f90  --overwrite-signature
!    f2py -c --fcompiler=gfortran haversine_distmat.pyf haversine_distmat.f90
!
!    If you have ifort on your system, change 'gfortran' to 'intelem'
!
!   Version 1.4.2
!
!===========================================================================


!------------------------------------------------------------------------------

    subroutine haversine_distmat(xa, xb, dist)
!------------------------------------------------------------------------------
      implicit none
      real(kind=8), dimension(:,:), intent(in)    :: xa, xb
      real(kind=8), allocatable, dimension(:)     :: xa1, xa2, xb1, xb2
      integer(kind=8)                             :: i, j, m, n
      real(kind=8)                                :: thedist, d2r
      real(kind=8), parameter                     :: erad = 6371315.0
      real(kind=8), dimension(:,:), intent(inout) :: dist
      external :: haversine
      !write (*,*) 'aaa'
      d2r = atan2(0.,-1.)/ 180. ! atan2(0.,-1.) == pi
      m = size(xa, 1)
      n = size(xb, 1)
      allocate(xa1(m))
      allocate(xa2(m))
      allocate(xb1(n))
      allocate(xb2(n))
      xa1 = xa(:,1)
      xa2 = xa(:,2)
      xb1 = xb(:,1)
      xb2 = xb(:,2)
      !write (*,*) 'bbb', xa1

!     Loop over empty dist matrix and fill
      do j = 1, m
        do i = 1, n
          call haversine(xa1(j), xa2(j), xb1(i), xb2(i), d2r, thedist)
          dist(j,i) = thedist
        enddo
      enddo
      deallocate(xa1)
      deallocate(xa2)
      deallocate(xb1)
      deallocate(xb2)
      dist = dist * erad
    end subroutine haversine_distmat


!------------------------------------------------------------------------------
    subroutine haversine_distvec(lon1, lat1, lon2, lat2, dist)
!------------------------------------------------------------------------------
      
      implicit none
      real(kind=8), dimension(:), intent(in) :: lon1, lat1, lon2, lat2
      integer(kind=8)                        :: i, m
      real(kind=8)             :: thedist, d2r
      real(kind=8), parameter  :: erad = 6371315.0
      real(kind=8), dimension(:), intent(inout) :: dist
      external :: haversine

      d2r = atan2(0.,-1.)/ 180. ! atan2(0.,-1.) == pi
      m = size(lon1)
      
!     Loop over empty dist matrix and fill
      do i = 1, m
        call haversine(lon1(i), lat1(i), lon2(i), lat2(i), d2r, thedist)
        dist(i) = thedist
      enddo
      dist = dist * erad
    
    end subroutine haversine_distvec


!------------------------------------------------------------------------------
    subroutine haversine_dist(lon1, lat1, lon2, lat2, thedist)
!------------------------------------------------------------------------------
      
      implicit none
      real(kind=8), intent(in) :: lon1, lat1, lon2, lat2
      real(kind=8)             :: d2r
      real(kind=8), parameter  :: erad = 6371315.0
      real(kind=8), intent(out) :: thedist
      external :: haversine
      
      d2r = atan2(0.,-1.)/ 180. ! atan2(0.,-1.) == pi
      call haversine(lon1, lat1, lon2, lat2, d2r, thedist)
      thedist = thedist * erad
    
    end subroutine haversine_dist


!------------------------------------------------------------------------------
    subroutine haversine(lon1, lat1, lon2, lat2, d2r, thedist)
!------------------------------------------------------------------------------
!
      implicit none
      real(kind=8), intent(in)  :: lon1, lat1, lon2, lat2, d2r
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
    end subroutine haversine


!------------------------------------------------------------------------------
    subroutine waypoint_vec(lonin, latin, anglein, distin, lon, lat)
!------------------------------------------------------------------------------
      
      implicit none
      real(kind=8), dimension(:), intent(in) :: lonin, latin, anglein, distin
      integer(kind=8)                        :: i, m
      real(kind=8)             :: thelon, thelat, d2r
      real(kind=8), dimension(:), intent(inout) :: lon, lat
      external :: waypoint

      d2r = atan2(0.,-1.)/ 180. ! atan2(0.,-1.) == pi
      m = size(lonin)
      
!     Loop over empty dist matrix and fill
        do i = 1, m
          call waypoint(lonin(i), latin(i), anglein(i), distin(i), d2r, thelon, thelat)
          lon(i) = thelon
          lat(i) = thelat
        enddo
    
    end subroutine waypoint_vec


!------------------------------------------------------------------------------
    subroutine waypoint(lonin, latin, anglein, distin, d2r, thelon, thelat)
!------------------------------------------------------------------------------
!
      implicit none
      real(kind=8), intent(in)  :: lonin, latin, anglein, distin, d2r
      real(kind=8)              :: ln1, lt1, angle, d_r
      real(kind=8), parameter   :: erad = 6371315.0
      real(kind=8), intent(out) :: thelon, thelat
!
      d_r = distin / erad ! angular distance
      thelon = d2r * lonin
      thelat = d2r * latin
      angle = d2r * anglein
      
      lt1 = asin(sin(thelat) * cos(d_r) + cos(thelat) * sin(d_r) * cos(angle))
      ln1 = atan2(sin(angle) * sin(d_r) * cos(thelat), cos(d_r) - sin(thelat) * sin(lt1))
      ln1 = ln1 + thelon
      thelat = lt1 / d2r
      thelon = ln1 / d2r
!
    end subroutine waypoint


!      
! !------------------------------------------------------------------------------
!     subroutine concat_arrays(a, b)
! !------------------------------------------------------------------------------
!       implicit none
!       real(kind=8), dimension(:) :: a
!       real(kind=8), dimension(:) :: b
!       real(kind=8), dimension(:), allocatable :: c
! !
!       allocate(c(size(a)+size(b)))
!       c(1:size(a)) = a
!       c(size(a)+1:size(a)+size(b)) = b
!  
!     end subroutine concat_arrays








