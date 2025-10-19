
// Custom Alignment

use std::alloc::Layout;
use std::num::NonZeroUsize;
use std::ptr::Alignment;

//NOTE: Because Layout is std library stable
// We can implement our own simple version of Alignment to plug into Layout which takes usize and uses that with
// the internal implementation of Alignment which it stores

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MyAlignment(NonZeroUsize);

#[test]
fn test_alignment() {
	let n: usize = 32;
	let a = Alignment::new(n).unwrap();

	let l = Layout::from_size_align(n,a.as_usize()).unwrap();
}