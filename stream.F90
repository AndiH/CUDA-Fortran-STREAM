module datatypes
	integer, parameter :: DP=13, dp_kind=selected_real_kind(DP)
end module datatypes

module gpu
	use datatypes
	contains
	attributes(global) subroutine init(array, value, length)
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
	implicit none

	integer(kind=8), parameter :: N = 250000000
	real(kind=dp_kind) :: value, scalar
	real(kind=dp_kind), dimension(:), allocatable :: a, d_a, b, d_b, c, d_c
	attributes(device) :: d_a, d_b, d_c
	type(dim3) :: nBlocks, nThreads
	type(cudaEvent) :: startEvent, stopEvent
	real, dimension(4) :: time
	real :: tempTime
	integer :: gpuStatus
	integer(kind=8), dimension(4) :: bytes
	integer, parameter :: ntimes = 1
	integer :: i
	character(len=30) :: outputformat

	bytes = (/	2 * sizeof(value) * N, &
				2 * sizeof(value) * N, &
				3 * sizeof(value) * N, &
				3 * sizeof(value) * N /)

	allocate(a(N))
	allocate(d_a(N))
	allocate(d_b(N))
	allocate(d_c(N))

	gpuStatus = cudaEventCreate(startEvent)
	gpuStatus = cudaEventCreate(stopEvent)

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
		gpuStatus = cudaEventRecord(startEvent, 0)
		call copy<<<nBlocks, nThreads>>>(d_a, d_b, N)
		gpuStatus = cudaEventRecord(stopEvent, 0)
		gpuStatus = cudaEventSynchronize(stopEvent)
		gpuStatus = cudaEventElapsedTime(tempTime, startEvent, stopEvent) ! in ms
		if ((i == 1) .OR. (tempTime < time(1))) then
			time(1) = tempTime
		end if

		gpuStatus = cudaEventRecord(startEvent, 0)
		call scale<<<nBlocks, nThreads>>>(d_a, d_b, scalar, N)
		gpuStatus = cudaEventRecord(stopEvent, 0)
		gpuStatus = cudaEventSynchronize(stopEvent)
		gpuStatus = cudaEventElapsedTime(tempTime, startEvent, stopEvent) ! in ms
		if ((i == 1) .OR. (tempTime < time(2))) then
			time(2) = tempTime
		end if

		gpuStatus = cudaEventRecord(startEvent, 0)
		call add<<<nBlocks, nThreads>>>(d_a, d_b, d_c, N)
		gpuStatus = cudaEventRecord(stopEvent, 0)
		gpuStatus = cudaEventSynchronize(stopEvent)
		gpuStatus = cudaEventElapsedTime(tempTime, startEvent, stopEvent) ! in ms
		if ((i == 1) .OR. (tempTime < time(3))) then
			time(3) = tempTime
		end if

		gpuStatus = cudaEventRecord(startEvent, 0)
		call triad<<<nBlocks, nThreads>>>(d_a, d_b, d_c, scalar, N)
		gpuStatus = cudaEventRecord(stopEvent, 0)
		gpuStatus = cudaEventSynchronize(stopEvent)
		gpuStatus = cudaEventElapsedTime(tempTime, startEvent, stopEvent) ! in ms
		if ((i == 1) .OR. (tempTime < time(4))) then
			time(4) = tempTime
		end if
	end do
	! bytes = 2 * sizeof(value) * N
	! write(*,*) "Bytes: ", bytes
	write(*, "(A15, I3, A32)") "Ran benchmarks ", ntimes, " times. Showing minimum results."
	write(*,*) "-----------------------------------------------"
	write(*, "(A10, 4x, A11, 4x, A9)") "Experiment", "Rate / GB/s", "Time / ms"
	outputformat = "(A10, 4x, F11.3, 4x, F9.2)"
	write(*, outputformat) "Copy", bytes(1)/(time(1)/1000.)/1024/1024/1024, time(1)
	write(*, outputformat) "Scale", bytes(2)/(time(2)/1000.)/1024/1024/1024, time(2)
	write(*, outputformat) "Add", bytes(3)/(time(3)/1000.)/1024/1024/1024, time(3)
	write(*, outputformat) "Triad", bytes(4)/(time(4)/1000.)/1024/1024/1024, time(4)

	! a = d_a

	! write(*,*) a(1), sizeof(a(1))

end program stream
