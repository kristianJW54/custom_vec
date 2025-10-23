
// NOTE: Custom implementation of a vec using unsafe
#![allow(unstable_features)]

// A Vec can be comprised of 3 levels
// Level 1: Public facing Vec the object a user interacts with - holding a buf and len for how many items are allocated
// Level 2: RawVec which holds a RawInner, a Cap for storing the capacity of a vec and a marker for the drop checker
// Level 3: RawInnerVec which handles the generic memory allocation and layout specificity

use std::alloc::{GlobalAlloc, Layout};
use std::marker::PhantomData;
use std::ptr::{NonNull};
use std::cmp::{Eq, PartialEq, Ord, PartialOrd};
use std::num::{NonZero, NonZeroUsize};

// To try to follow with the std library implementation we will define a top level Alloc Enum which can
// help with telling the Allocator if we have un-initialized memory (basically do nothing) or if
// we have zeroed memory in which we need to allocate

#[derive(Clone, Copy, Debug)]
enum AllocInit {
	Uninitialized,
	Zeroed
}

// We also need define some Layout helpers. A Layout represents the shape of a block of memory
// It's SIZE and ALIGNMENT
// How many bytes in total do we need to allocate
// and what is the alignment of each element.
// Essentially:
// We want to allocate memory for N elements of type T.
// Each element has a known size and alignment.
// Compute the total size (size * N) and keep the same alignment.
// If multiplication overflows, fail gracefully.
// Return that layout so we can tell the allocator how much to allocate.

// std achieves this with a helper using some nightly methods
// #[inline]
// fn layout_array(cap: usize, elem_layout: Layout) -> Result<Layout, TryReserveError> {
// 	elem_layout.repeat(cap).map(|(layout, _pad)| layout).map_err(|_| CapacityOverflow.into())
// }

//TODO: Implement layout helpers here...


// For ZSTs there is zero allocation but a defined alignment
// A vec of Vec<()>::new() could be size = 0 but alignment = 1
// Allocators work in bytes so they give only non-zero-sized chunks of memory if we were to pass in a
// ZST the allocator would not know what to allocate but must provision for some memory so the trick is
// we let a capacity check handle this by setting capacity to 0..isize::MAX meaning set capacity to infinite
// as it doesn't matter because we are not actually allocating any actual memory

// For capacity handling, we need a type that stores capacity in a range that's safe for pointer arithmetic.
// In Rust's standard library, this is done using a niche optimization:
// the highest bit (the most significant bit) of a `usize` is always guaranteed to be zero
// because valid pointer offsets must be <= isize::MAX.
//
// Std can safely reserve that top bit as an unused niche.
// The compiler can then use that bit to represent `Option<Cap>::None` without
// increasing the size of the type (i.e., `Option<Cap>` stays the same size as `Cap`).
//
// In effect:
//   - All valid capacities are in 0..=isize::MAX (top bit = 0)
//   - `Option<Cap>::None` is represented by setting the top bit = 1
//   - If the top bit is ever set for a real capacity, we've overflowed and must fail
//
// This trick keeps `RawVec` compact and pointer-offset–safe without needing
// an extra tag field for `Option<Cap>`.

// See [https://doc.rust-lang.org/src/alloc/raw_vec/mod.rs.html#40]
// Since we cannot replicate this in std without using internals, we must just take the
// Extra memory for Option<> instead of 8 bytes it will be 16 bytes.

// We define a global capacity overflow panic to call whenever we encounter this
#[track_caller]
fn capacity_overflow() -> ! {
	panic!("capacity overflow");
}


#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
#[repr(transparent)]
struct CustomUsize(usize);

impl CustomUsize {

	pub const MAX: usize = isize::MAX as usize;

	// SAFETY: initializing type with `rustc_layout_scalar_valid_range` attr is unsafe

	pub const unsafe fn new(value: usize) -> Option<Self> {
		// Safety: we checked upper bound.
		if value <= 0x7FFF_FFFF_FFFF_FFFF{
			Some(Self(value))
		} else {
			None
		}
	}

	/// Unsafe constructor.
	///
	///# Safety
	///Caller must guarantee `val <= isize::MAX`.
	pub const unsafe fn new_unchecked(value: usize) -> Self {
		Self(value)
	}

	pub const fn get(self) -> usize {
		self.0
	}

}

type Cap = CustomUsize;

const ZERO_CAP: Cap = unsafe { Cap::new_unchecked(0) };

unsafe fn new_cap<T>(cap: usize) -> Cap {
	if std::mem::size_of::<T>() == 0 {
		ZERO_CAP
	} else {
		match unsafe { Cap::new(cap) } {
			Some(cap) => cap,
			None => capacity_overflow()
		}
	}
}


/// ## Level 3
///
/// # RawInnerVec
///
/// Is generic over the Allocator.
/// The standard library uses this separation for operations which require only the layout
///
/// We leave the type to the layer above in RawVec
///
/// At the moment, Allocator and Global is unstable so we can just use placeholder for now and continue
/// with global allocator
#[derive(Debug)]
struct RawInnerVec<A> {
	// We have ownership of this raw byte buffer. It's untyped and safe for all T and can be
	// cast later. Is always NonNull
	ptr: NonNull<u8>, // The allocator allocated based on raw bytes and not on typed memory hence u8
	cap: Cap, // Use custom Cap type to enforce 0..isize::MAX checks
	_alloc: PhantomData<A>,
}

// InnerVec will cast *mut u8 bytes to *mut T when using it

///# Level 2
#[derive(Debug)]
struct InnerVec<T, A> {
	inner: RawInnerVec<A>,
	_marker: PhantomData<T>, // To signal to the drop checker that there is a type here to check
}

// Level 1
struct MyVec<T, A> {
	buf: InnerVec<T, A>,
	len: usize,
}

// IMPL BLOCK for RawInnerVec

impl<A> RawInnerVec<A> {
	#[inline]
	// A constructor usually does not allocate, so here we want to initialize and empty buffer with cap = 0
	// and a dangling pointer. We also need to store the alloc
	const fn new_in(align: usize) -> Self {
		// We need to define a NonNull pointer without provenance, meaning the pointer cannot be used
		// for memory access as we have not yet allocated memory
		// NonNull::dangling() provides an abstracted function for this but the alignment is always known when we
		// have the T type.
		// For this layer, we pass down the alignment from the type so we must align ourselves

		// Std does this by using its internal Alignment implementation [https://doc.rust-lang.org/src/core/ptr/alignment.rs.html#13]
		// Looking through the docs it seems that for a RawInnerVec only a usize is needed which is derived from
		// using Alignment -> Alignment::of::<T>() which gives a NonZero<usize> to pass to Layout

		// align_of_val also gives this [https://doc.rust-lang.org/src/core/mem/mod.rs.html#509-512]
		// So to avoid using an internal structure and to avoid implementing my own Alignment struct
		// I will pass in a usize derived from align_of_val() which I can wrap in a NonZero

		// SAFETY: RawVec uses align_of_val which checks for correct alignment and non-zero so we are safe to
		// use new_unchecked here
		let nsu = unsafe { NonZeroUsize::new_unchecked(align) };
		let ptr = NonNull::without_provenance(nsu);
		// For alloc - we use PhantomData as placeholder for when Allocator internals become standard and we can
		// pass in a custom allocation for memory
		Self { ptr, cap: ZERO_CAP, _alloc: PhantomData }

	}

	const fn with_capacity(cap: usize, align: )

	// Getters
	#[inline]
	const fn capacity(&self, elem_size: usize) -> usize {
		if elem_size == 0 { usize::MAX } else { self.cap.0 }
	}

}

//NOTE: std library uses two implement blocks for an InnerVec. The first being a Global allocator
// almost like a default, using the default global allocator and its functions
// The second being generic over A an Allocator which can be passed in.

// IMPL BLOCK for Default Global Allocator InnerVec
// For a default allocator we can use the free functions in alloc. [https://doc.rust-lang.org/alloc/alloc/index.html#functions]
// This may be a better way to not use unstable Global struct and simply invoke our own simple struct type which implements Allocator using the
// free functions in alloc?
impl<T, A> InnerVec<T, A> {
	pub fn new(t: T) -> Self {
		// TODO: Will use a new_in but for now just skip
		// Also we are borrowing T here?
		Self { inner: RawInnerVec::new_in(align_of_val(&t)), _marker: PhantomData }
	}



	// Getters
	const fn capacity(&self) -> usize {
		self.inner.capacity(size_of::<T>())
	}
}



#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_zst_capacity() {
		let zst = ();
		let mv: InnerVec<(), ()> = InnerVec::new(zst);
		assert_eq!(mv.capacity(), 18446744073709551615);

		let l: Layout = Layout::new::<i32>();


	}

	#[test]
	fn test_simple() {
		println!("this is my vec");

		// Testing ZSTs
		let v: Vec<()> = Vec::new();
		println!("this is my vec {:?}", v.capacity());

	}

	#[test]
	fn test_valid_capacity() {
		let c = unsafe { new_cap::<u8>(100) };
		assert_eq!(c.get(), 100);
	}

	#[test]
	fn test_overflow_panics() {
		let too_big = (isize::MAX as usize) + 1;
		let result = std::panic::catch_unwind(|| unsafe { new_cap::<u8>(too_big) });
		assert!(result.is_err(), "expected panic for cap overflow");
	}

	#[test]
	fn test_max_value_ok() {
		let c = unsafe { new_cap::<u8>(isize::MAX as usize) };
		assert_eq!(c.get(), isize::MAX as usize);
	}

	#[test]
	fn test_option_niche_size() {
		use std::mem::size_of;
		assert_ne!(size_of::<Option<CustomUsize>>(), size_of::<CustomUsize>(),
				   "Option<CustomUsize> should be the same size (niche optimization active)");
	}

	#[test]
	fn test_inner_vec_new() {

		let size = 10;
		let v: InnerVec<i32, ()> = InnerVec::new(size);

		assert!(v.inner.cap.get() == 0);

	}

}