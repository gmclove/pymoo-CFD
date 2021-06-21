!-----------------------------------------------------------------------------------
! 2D_af program
!-----------------------------------------------------------------------------------

!=================================================================
program main


      use yales2_m

      implicit none

      ! ------------------------
      character(len=LEN_MAX) :: inputfile
      ! ------------------------

      inputfile = "2D_af.in"

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
      type (data_t), pointer :: u_ptr
      type (r2_t), pointer :: u
      integer :: n,i
      real(WP) :: x(2)
      ! ------------------------

      grid => solver%first_grid
      call find_data(grid%first_data,"U",u_ptr)

      ! ------------------------
      ! u initialization
      if (.not.solver%restarted_with_solution) then
         do n=1,grid%nel_grps
            el_grp => grid%el_grps(n)%ptr
            u => u_ptr%r2_ptrs(n)%ptr
            do i=1,el_grp%nnode
               x(1:grid%ndim) = el_grp%x_node%val(1:grid%ndim,i)
               u%val(1:grid%ndim,i) = 0.0_WP
               u%val(1,i) = 2.0_WP
            end do
         end do
      end if

end subroutine initialize_data
!=================================================================

!=================================================================
subroutine temporal_loop_preproc()

      use yales2_m

      implicit none

      ! ------------------------
      ! type(grid_t), pointer :: grid
      ! type(actuator_set_t), pointer :: actuator_set,vawt_ptr
      ! type(actuator_t), pointer :: actuator
      ! logical :: vawt_found
      ! integer :: iblade
      ! real(WP) :: pitch_angle_velocity_max,target_aoa,mean_aoa
      ! ! ------------------------
      !
      ! ! pointers
      ! grid => solver%first_grid
      ! actuator_set => grid%first_actuator_set
      !
      ! pitch_angle_velocity_max=1.0_WP*deg2rad ! Y2 need a velocity in rad/second
      ! target_aoa=4.0_WP*deg2rad ! target aoa in rad
      !
      ! ! Exemple of dummy Individual pitch control based on a target blade mean Angle of attack
      ! ! Here if AoA>target the pitching velocity is set to a the maximum positive value
      ! ! And if AoA<target the pitching velocity is set to a the maximum negative value
      ! call find_actuator_set(actuator_set,'VAWT_1',vawt_ptr,vawt_found)
      ! if (vawt_found) then
      !    actuator => vawt_ptr%first_actuator
      !    do iblade = 1, vawt_ptr%nblade
      !       mean_aoa=sum(actuator%aoa%val(1:actuator%ndim))/real(actuator%ndim,WP)
      !       if (mean_aoa-target_aoa>0.0_WP) then
      !           actuator%d_pitch_angle=pitch_angle_velocity_max
      !       else
      !          actuator%d_pitch_angle=-pitch_angle_velocity_max
      !       end if
      !       actuator => actuator%next
      !    end do
      ! end if

end subroutine temporal_loop_preproc
!=================================================================

!=================================================================
subroutine temporal_loop_postproc()

      use yales2_m

      implicit none

      ! ------------------------
      ! type(grid_t), pointer :: grid
      ! type(actuator_set_t), pointer :: actuator_set
      ! integer :: i
      ! ! ------------------------
      !
      ! ! pointers
      ! grid => solver%first_grid
      !
      ! actuator_set => grid%first_actuator_set
      ! do i=1,count_actuator_set(actuator_set)
      !
      !    call show_rotor_force_and_power(solver,actuator_set)
      !
      !    actuator_set => actuator_set%next
      ! end do

end subroutine temporal_loop_postproc
!=================================================================
