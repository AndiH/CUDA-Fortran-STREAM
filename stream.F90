! ------------------------------------------------------------------------------
! CUDA Fortran F90 variant of the STREAM benchmark (originally: John D McCalpin)
! Programmer: Andreas Herten
! ------------------------------------------------------------------------------
! This program measures the memory transfer rates in GB/s for the four simple
! kernels of the STREAM benchmark __on the GPU__ using CUDA Fortran.
! 
! The code lives at https://github.com/AndiH/CUDA-Fortran-STREAM
! See also http://www.streambench.org, especially the associated license
! ------------------------------------------------------------------------------
! Notes
!   - The dp_kind is selectable, as it is in the project I'm using the code for
!   - CUDA events are used for timings, since they seem easy to handle
!   - Timings: Max and Avg values are not printed, yet
!   - Verification of Results: Also not done yet!
!   - Makefile is given with PGFortran, which should be the only compiler
!     capable to generate CUDA code from Fortran
!   - I actually know very little Fortran. Feel free to correct something on
!     Github!
! ------------------------------------------------------------------------------

! The following is a ugly hack, but the line is very close to being too long
#define CUDA_SUCCESS 0
#define CUDACALL__(e,fmt,c) \
e=c; \
if(e/=CUDA_SUCCESS) \
write(*,fmt) "CUDA Error ",e," in ",__FILE__,":",__LINE__,": ",trim(cudaGetErrorString(e))," (",#c,")"
#define CUDACALL(c) CUDACALL__(gpuStatus,fmt,c)

module debug
	character(len=27) :: fmt = "(A,I0,A,A,A,I0,A,A,A,A,A,A)"
end module debug

module datatypes
	integer, parameter :: DP=13, dp_kind=selected_real_kind(DP)
end module datatypes

module gpu
	use datatypes
	contains
	attributes(global) subroutine init(array, value, length)
		! INIT
		! Initialize values on GPU device (also functions as warmup)
		! >> array_j = value
		implicit none
		real(kind=dp_kind), dimension(length), intent(inout) :: array
		real(kind=dp_kind), intent(in), value :: value
		integer(kind=8), intent(in), value :: length
		integer :: index

		index = blockDim%x * (blockIdx%x - 1) + threadIdx%x

		if (index <= length) &
			array(index) = value
	end subroutine init

	attributes(global) subroutine copy(lhs, rhs, length)
		! STREAM COPY
		! Copy values from one array into another
		! >> lhs_j = rhs_j
		implicit none
		real(kind=dp_kind), dimension(length), intent(out) :: lhs
		real(kind=dp_kind), dimension(length), intent(in) :: rhs
		integer(kind=8), intent(in), value :: length
		integer :: index

		index = blockDim%x * (blockIdx%x - 1) + threadIdx%x
		if (index <= length) &
			lhs(index) = rhs(index)
	end subroutine copy

	attributes(global) subroutine scale(lhs, rhs, scalar, length)
		! STREAM SCALE
		! Scale entry of an array and copy to new array
		! >> lhs_j = scalar * rhs_j
		implicit none
		real(kind=dp_kind), dimension(length), intent(out) :: lhs
		real(kind=dp_kind), dimension(length), intent(in) :: rhs
		real(kind=dp_kind), intent(in), value :: scalar
		integer(kind=8), intent(in), value :: length
		integer :: index

		index = blockDim%x * (blockIdx%x - 1) + threadIdx%x
		if (index <= length) &
			lhs(index) = scalar*rhs(index)
	end subroutine scale

	attributes(global) subroutine add(lhs, rhsl, rhsr, length)
		! STREAM ADD
		! Add two arrays, put values into other array
		! >> lhs_j = rhsl_j + rhsr_j
		implicit none
		real(kind=dp_kind), dimension(length), intent(out) :: lhs
		real(kind=dp_kind), dimension(length), intent(in) :: rhsl, rhsr
		integer(kind=8), intent(in), value :: length
		integer :: index

		index = blockDim%x * (blockIdx%x - 1) + threadIdx%x
		if (index <= length) &
			lhs(index) = rhsl(index) + rhsr(index)
	end subroutine add

	attributes(global) subroutine triad(lhs, rhsl, rhsr, scalar, length)
		! STREAM TRIAD
		! Multiply entry of array by scalar, add to other array's entry, put values into third array
		! >> lhs_j = rhsl_j + scalar * rhsr_j
		implicit none
		real(kind=dp_kind), dimension(length), intent(out) :: lhs
		real(kind=dp_kind), dimension(length), intent(in) :: rhsl, rhsr
		real(kind=dp_kind), intent(in), value :: scalar
		integer(kind=8), intent(in), value :: length
		integer :: index

		index = blockDim%x * (blockIdx%x - 1) + threadIdx%x
		if (index <= length) &
			lhs(index) = rhsl(index) + scalar * rhsr(index)
	end subroutine triad

end module gpu


program stream
	use datatypes
	use cudafor
	use gpu
	use debug
	implicit none

	integer(kind=8), parameter :: N = 250000000  ! Lengths of data arrays
	real(kind=dp_kind) :: value, scalar  ! Helper variables
	real(kind=dp_kind), dimension(:), allocatable :: a, d_a, b, d_b, c, d_c  ! Data arrays
	attributes(device) :: d_a, d_b, d_c  ! Prefixed with d_ for device
	type(dim3) :: nBlocks, nThreads  ! Launch this many blocks and threads
	type(cudaEvent) :: startEvent, stopEvent  ! Using CUDA events for timing
	real, dimension(4) :: time  ! Array of minimal times for each experiment
	real :: tempTime  ! Helper value for timings
	integer :: gpuStatus  ! Holds CUDA error values
	integer(kind=8), dimension(4) :: bytes
	integer, parameter :: ntimes = 10  ! Repeat measurements this often
	integer :: i  ! Loop run variables
	character(len=30) :: outputformat  ! FORMAT string for Fortran output

	bytes = (/	2 * sizeof(value) * N, &
				2 * sizeof(value) * N, &
				3 * sizeof(value) * N, &
				3 * sizeof(value) * N /)

	allocate(d_a(N))
	allocate(d_b(N))
	allocate(d_c(N))

	CUDACALL( cudaEventCreate(startEvent) )
	CUDACALL( cudaEventCreate(stopEvent) )

	nThreads = dim3(192, 1, 1)
	nBlocks = dim3(N/nThreads%x, 1, 1)
	if (mod(N, nThreads%x) /= 0) &
		nBlocks%x = nBlocks%x + 1

	value = 1.1
	call init<<<nBlocks,nThreads>>>(d_a, value, N)
	value = 2.2
	call init<<<nBlocks,nThreads>>>(d_b, value, N)
	value = 3.3
	call init<<<nBlocks,nThreads>>>(d_c, value, N)

	scalar = 0.25 * value

	do i = 1, ntimes
		CUDACALL( cudaEventRecord(startEvent, 0) )
		call copy<<<nBlocks, nThreads>>>(d_a, d_b, N)
		CUDACALL( cudaEventRecord(stopEvent, 0) )
		CUDACALL( cudaEventSynchronize(stopEvent) )
		CUDACALL( cudaEventElapsedTime(tempTime, startEvent, stopEvent) ) ! in ms
		if ((i == 1) .OR. (tempTime < time(1))) then
			time(1) = tempTime
		end if

		CUDACALL( cudaEventRecord(startEvent, 0) )
		call scale<<<nBlocks, nThreads>>>(d_a, d_b, scalar, N)
		CUDACALL( cudaEventRecord(stopEvent, 0) )
		CUDACALL( cudaEventSynchronize(stopEvent) )
		CUDACALL( cudaEventElapsedTime(tempTime, startEvent, stopEvent) ) ! in ms
		if ((i == 1) .OR. (tempTime < time(2))) then
			time(2) = tempTime
		end if

		CUDACALL( cudaEventRecord(startEvent, 0) )
		call add<<<nBlocks, nThreads>>>(d_a, d_b, d_c, N)
		CUDACALL( cudaEventRecord(stopEvent, 0) )
		CUDACALL( cudaEventSynchronize(stopEvent) )
		CUDACALL( cudaEventElapsedTime(tempTime, startEvent, stopEvent) ) ! in ms
		if ((i == 1) .OR. (tempTime < time(3))) then
			time(3) = tempTime
		end if

		CUDACALL( cudaEventRecord(startEvent, 0) )
		call triad<<<nBlocks, nThreads>>>(d_a, d_b, d_c, scalar, N)
		CUDACALL( cudaEventRecord(stopEvent, 0) )
		CUDACALL( cudaEventSynchronize(stopEvent) )
		CUDACALL( cudaEventElapsedTime(tempTime, startEvent, stopEvent) )! in ms
		if ((i == 1) .OR. (tempTime < time(4))) then
			time(4) = tempTime
		end if
	end do


	write(*, "(A15, I3, A32)") "Ran benchmarks ", ntimes, " times. Showing minimum results."
	write(*,*) "-----------------------------------------------"
	write(*, "(A10, 4x, A11, 4x, A9)") "Experiment", "Rate / GB/s", "Time / ms"
	outputformat = "(A10, 4x, F11.3, 4x, F9.2)"
	write(*, outputformat) "Copy", bytes(1)/(time(1)/1000.)/1024/1024/1024, time(1)
	write(*, outputformat) "Scale", bytes(2)/(time(2)/1000.)/1024/1024/1024, time(2)
	write(*, outputformat) "Add", bytes(3)/(time(3)/1000.)/1024/1024/1024, time(3)
	write(*, outputformat) "Triad", bytes(4)/(time(4)/1000.)/1024/1024/1024, time(4)

end program stream
