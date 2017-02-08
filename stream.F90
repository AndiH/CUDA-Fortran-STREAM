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

! The following CUDA_SUCCESS definition is a ugly hack, but the line is very close to being too long
#define CUDA_SUCCESS 0
#define CUDACALL__(e,fmt,c) \
e=c; \
if(e/=CUDA_SUCCESS) \
write(*,fmt) "CUDA Error ",e," in ",__FILE__,":",__LINE__,": ",trim(cudaGetErrorString(e))," (",#c,")"
#define CUDACALL(c) CUDACALL__(gpuStatus,fmt,c)

module debug
	character(len=27) :: fmt = "(A,I0,A,A,A,I0,A,A,A,A,A,A)"
	integer :: gpuStatus
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
	integer, parameter :: ntimes = 10  ! Repeat measurements this often
	real, dimension(4, ntimes) :: time  ! Array of minimal times for each experiment
	real, dimension(4) :: minTime, maxTime, avgTime
	real :: tempTime  ! Helper value for timings
	integer(kind=8), dimension(4) :: bytes
	integer :: i  ! Loop run variables
	character(len=40) :: outputformat  ! FORMAT string for Fortran output

	character(len=32) :: arg
	integer :: iArg
	logical :: csv = .false., header = .false., fullout = .false.

	do i = 1, command_argument_count()
		call get_command_argument(i, arg)

		select case (arg)
			case ('-h', '--help')
				call printHelp()
				stop
			case ('--csv')
				csv = .true.
			case ('--header')
				header = .true.
				csv = .true.
			case ('--full')
				fullout = .true.
				csv = .true.
			case default
				write(*,'(A,A,/)') 'Unrecognized option: ', arg
				call printHelp()
				stop
		end select
	end do

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
		CUDACALL( cudaGetLastError() )
		CUDACALL( cudaEventRecord(stopEvent, 0) )
		CUDACALL( cudaEventSynchronize(stopEvent) )
		CUDACALL( cudaEventElapsedTime(tempTime, startEvent, stopEvent) ) ! in ms
		time(1, i) = tempTime

		CUDACALL( cudaEventRecord(startEvent, 0) )
		call scale<<<nBlocks, nThreads>>>(d_a, d_b, scalar, N)
		CUDACALL( cudaGetLastError() )
		CUDACALL( cudaEventRecord(stopEvent, 0) )
		CUDACALL( cudaEventSynchronize(stopEvent) )
		CUDACALL( cudaEventElapsedTime(tempTime, startEvent, stopEvent) ) ! in ms
		time(2, i) = tempTime

		CUDACALL( cudaEventRecord(startEvent, 0) )
		call add<<<nBlocks, nThreads>>>(d_a, d_b, d_c, N)
		CUDACALL( cudaGetLastError() )
		CUDACALL( cudaEventRecord(stopEvent, 0) )
		CUDACALL( cudaEventSynchronize(stopEvent) )
		CUDACALL( cudaEventElapsedTime(tempTime, startEvent, stopEvent) ) ! in ms
		time(3, i) = tempTime

		CUDACALL( cudaEventRecord(startEvent, 0) )
		call triad<<<nBlocks, nThreads>>>(d_a, d_b, d_c, scalar, N)
		CUDACALL( cudaGetLastError() )
		CUDACALL( cudaEventRecord(stopEvent, 0) )
		CUDACALL( cudaEventSynchronize(stopEvent) )
		CUDACALL( cudaEventElapsedTime(tempTime, startEvent, stopEvent) )! in ms
		time(4, i) = tempTime
	end do

	call populateMinMaxAvg(time(1,:), minTime(1), maxTime(1), avgTime(1))
	call populateMinMaxAvg(time(2,:), minTime(2), maxTime(2), avgTime(2))
	call populateMinMaxAvg(time(3,:), minTime(3), maxTime(3), avgTime(3))
	call populateMinMaxAvg(time(4,:), minTime(4), maxTime(4), avgTime(4))

	if (.NOT. csv) then
		write(*, "(A, I0, A, I0, A, F0.2, A)") "Ran benchmarks ", ntimes, " times. Data array length: ", N, " => ", sizeof(value) * real(N)/1024/1024/1024, " GB"
		write(*,*) "-----------------------------------------------"
		write(*, "(A10, 4x, A16, 4x, A16, 4x, A16)") "Experiment", "Max. Rate / GB/s",  "Min. Rate / GB/s", "Avg. Rate / GB/s"
		outputformat = "(A10, 4x, F16.3, 4x, F16.3, 4x, F16.3)"
		write(*, outputformat)  "Copy", convertRate(bytes(1), minTime(1)), convertRate(bytes(1), maxTime(1)), convertRate(bytes(1), avgTime(1))
		write(*, outputformat) "Scale", convertRate(bytes(2), minTime(2)), convertRate(bytes(2), maxTime(2)), convertRate(bytes(2), avgTime(2))
		write(*, outputformat)   "Add", convertRate(bytes(3), minTime(3)), convertRate(bytes(3), maxTime(3)), convertRate(bytes(3), avgTime(3))
		write(*, outputformat) "Triad", convertRate(bytes(4), minTime(4)), convertRate(bytes(4), maxTime(4)), convertRate(bytes(4), avgTime(4))
	else
		if (fullout) then
			if (header) &
				write (*, "(A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A)") "Copy (Max)", ",", "Copy (Min)", ",", "Copy (Avg)", ",", "Scale (Max)", ",", "Scale (Min)", ",", "Scale (Avg)", ",", "Triad (Min)", ",", "Triad (Max)", ",", "Triad (Avg)"
			write (*, "(F0.3, A, F0.3, A, F0.3, A, F0.3, A, F0.3, A, F0.3, A, F0.3, A, F0.3, A, F0.3, A, F0.3)") &
				convertRate(bytes(1), minTime(1)), ",", &
				convertRate(bytes(1), maxTime(1)), ",", &
				convertRate(bytes(1), avgTime(1)), ",", &
				convertRate(bytes(2), minTime(2)), ",", &
				convertRate(bytes(2), maxTime(2)), ",", &
				convertRate(bytes(2), avgTime(2)), ",", &
				convertRate(bytes(3), minTime(3)), ",", &
				convertRate(bytes(3), maxTime(3)), ",", &
				convertRate(bytes(3), avgTime(3))
		else
			if (header) &
				write (*, "(A, A, A, A, A, A, A)") "Copy", ",", "Scale", ",", "Add", ",", "Triad"
			write(*, "(F0.3, A, F0.3, A, F0.3, A, F0.3)") &
				convertRate(bytes(1), minTime(1)), ",", &
				convertRate(bytes(2), minTime(2)), ",", &
				convertRate(bytes(3), minTime(3)), ",", &
				convertRate(bytes(4), minTime(4))
		end if
	end if

contains
	subroutine printHelp()
		write(*,"(A)") "Usage: stream.bin [OPTIONS]"
		write(*,*) ""
		write(*,"(A)") "Options:"
		write(*,*) ""
		write(*,"(A10, 4x, A)")    "  --csv", "Print output in concise CSV format. Max. rates given."
		write(*,"(14x, A)")                   "Format: Copy,Scale,Add,Triad. In GB/s. (See --header)"
		write(*,"(A10, 4x, A)")   "  --full", "Print min, max, avg rate in CSV format."
		write(*,"(14x, A)")                   "In GB/s. Implies --csv, currently."
		write(*,"(A10, 4x, A)") "  --header", "Print header above CSV values. Implies --csv."
	end subroutine printHelp

	real function convertRate(byte, time)
		implicit none
		integer(kind=8) :: byte
		real :: time
		convertRate = real(byte)/(time/1000.)/1024/1024/1024
	end function convertRate

	subroutine populateMinMaxAvg(times, min_, max_, avg_)
		implicit none
		real, dimension(:), intent(in) :: times
		real, intent(out) :: min_, max_, avg_
		real :: sum_
		integer :: i

		avg_ = sum(times)/size(times,1)
		min_ = times(1)
		max_ = times(1)
		do i = 2, size(times,1)
			if (times(i) < min_) &
				min_ = times(i)
			if (times(i) > max_) &
				max_ = times(i)
		end do
	end subroutine populateMinMaxAvg
end program stream
