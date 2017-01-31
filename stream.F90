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

end module gpu


program stream
	use datatypes
	use cudafor
	use gpu
	implicit none

	integer(kind=8), parameter :: N = 250000000
	real(kind=dp_kind) :: value
	real(kind=dp_kind), dimension(:), allocatable :: a, d_a, b, d_b
	attributes(device) :: d_a, d_b
	type(dim3) :: nBlocks, nThreads
	type(cudaEvent) :: startEvent, stopEvent
	real :: time, tempTime
	integer :: gpuStatus
	integer(kind=8) :: bytes
	integer, parameter :: ntimes

	allocate(a(N))
	allocate(d_a(N))
	allocate(d_b(N))

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

	do ntimes = 1, 10
		gpuStatus = cudaEventRecord(startEvent, 0)
		call copy<<<nBlocks,nThreads>>>(d_a, d_b, N)
		gpuStatus = cudaEventRecord(stopEvent, 0)
		gpuStatus = cudaEventSynchronize(stopEvent)
		gpuStatus = cudaEventElapsedTime(tempTime, startEvent, stopEvent) ! in ms
	! write(*,*) time/1000.
		if ((ntimes == 1) .OR. (tempTime < time)) then
			time = tempTime
		end if
		! write (*,*) "Loop", ntimes, "tempTime", tempTime, "time", time
	end do
	bytes = 2 * sizeof(value) * N
	! write(*,*) "Bytes: ", bytes
	write(*,*) "GB/s:", bytes/(time/1000.)/1024/1024/1024

	! a = d_a

	! write(*,*) a(1), sizeof(a(1))

end program stream
