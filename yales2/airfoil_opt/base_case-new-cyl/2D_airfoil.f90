!-----------------------------------------------------------------------------------
! 2D_cylinder program
!-----------------------------------------------------------------------------------

!=================================================================
program main

      use yales2_m

      implicit none

      ! ------------------------
      character(len=LEN_MAX) :: inputfile
      ! ------------------------

      inputfile = "2D_airfoil.in"

      ! ------------------------
      ! run
      call run_yales2(inputfile)

end program main
!=================================================================

!=================================================================
subroutine initialize_data()

      use yales2_m

      implicit none

      ! ------------------------
      ! ------------------------
      type(grid_t), pointer :: grid
      type(el_grp_t), pointer :: el_grp
      type (data_t), pointer :: u_ptr,z_ptr
      type (r1_t), pointer :: z
      type (r2_t), pointer :: u
      integer :: n,i
      ! ------------------------

      grid => solver%first_grid
      call find_data(grid%first_data,"U",u_ptr)
      call find_data(grid%first_data,"Z",z_ptr)

      ! ------------------------
      ! u initialization
      if (.not.solver%restarted_with_solution) then
         do n=1,grid%nel_grps
            el_grp => grid%el_grps(n)%ptr
            z => z_ptr%r1_ptrs(n)%ptr
            u => u_ptr%r2_ptrs(n)%ptr
            do i=1,el_grp%nnode
               u%val(1,i) = 0.15_WP
               u%val(2,i) = 0.0_WP
               u%val(3:grid%ndim,i) = 0.0_WP
               z%val(i) = 1.0_WP
            end do
         end do
      end if

end subroutine initialize_data
!=================================================================

!=================================================================
subroutine temporal_loop_preproc()

      use yales2_m

      implicit none

!      ! ------------------------
!      type(grid_t), pointer :: grid
!      type (data_t), pointer :: u_ptr,x_ptr
!      integer :: i,bndnb
!      real(WP) :: x(3)
!      logical :: done
!      type(boundary_t), pointer :: bndptr
!      type(bnd_data_t), pointer :: u_ref,x_bnd
!      ! ------------------------
!
!      ! pointers
!      grid => solver%first_grid
!      call find_data(grid%first_data,"U",u_ptr)
!
!      ! change the reference velocity on the inlet "x0" at each iteration
!      call find_boundary(grid%first_boundary,"x0",res_ptr=bndptr,rank=bndnb)
!      u_ref => u_ptr%bnd_ref_ptrs(bndnb)%ptr
!      do i=1,bndptr%nnode
!         x(1:grid%ndim) = bndptr%x_node%val(1:grid%ndim,i)
!         u_ref%r2%val(1:grid%ndim,i) = 0.0_WP
!         u_ref%r2%val(1,i) = 0.15_WP*cos(solver%total_time*2.0_WP*pi/1.5_WP)
!      end do

end subroutine temporal_loop_preproc
!=================================================================

!=================================================================
subroutine temporal_loop_postproc()

      use yales2_m

      implicit none

      ! ------------------------
      ! ------------------------

end subroutine temporal_loop_postproc
!=================================================================
