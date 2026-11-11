solve:
	movl	n_iter(%rip), %eax
	testl	%eax, %eax
	jle	.L4
	xorl	%esi, %esi
	movl	$1, %edx
	xorl	%edi, %edi
	movabsq	$-8543223828751151131, %r8
.L3:
	leaq	(%rdi,%rdx), %rcx
	movq	%rdx, %rdi
	addl	$1, %esi
	movq	%rcx, %rax
	imulq	%r8
	movq	%rcx, %rax
	sarq	$63, %rax
	addq	%rcx, %rdx
	sarq	$29, %rdx
	subq	%rax, %rdx
	imulq	$1000000007, %rdx, %rax
	subq	%rax, %rcx
	movl	n_iter(%rip), %eax
	movq	%rcx, %rdx
	cmpl	%esi, %eax
	jg	.L3
	movq	%rdi, %rax
	ret
.L4:
	xorl	%edi, %edi
	movq	%rdi, %rax
	ret
	.data
n_iter:
	.long	80
