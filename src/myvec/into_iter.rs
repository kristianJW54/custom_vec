use std::marker::PhantomData;
use std::ptr::NonNull;
use std::{mem, slice};

pub struct IntoIter<T, A> {

    // We use pub(super) here so that it is only public to this module
    // buf is a NonNull<T> here as we want to give a NonNull pointer to the start of the block
    // This does not move and, we use this to deallocate and, so we can reconstruct RawVec later to drop
    pub(super) buf: NonNull<T>,
    // Ptr, therefore, is mutable and is what we increment to move through the block.
    pub(super) ptr: NonNull<T>,
    // Specify the capacity
    pub(super) cap: usize,
    // We don't have a custom allocator but, we do need phantom data for the drop checker
    // So I combine the two - and later if we have an alloc we can add a separate field and get to keep the bounds
    pub(super) _alloc: PhantomData<A>,
    // We specify an end pointer which is the last element in the block offset by len
    pub(super) end: *const T,


}

impl<T, A> IntoIter<T, A> {

    // Simple as_slice method for returning the rest of the MyVec as slice
    fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.buf.as_ptr(), self.end.offset_from(self.buf.as_ptr()) as usize) }
    }

    // We could also have a mutable slice version

}

impl<T, A> Iterator for IntoIter<T, A> {
    type Item = T;


    fn next(&mut self) -> Option<T> {

        let ptr = if mem::size_of::<T>() == 0 {

            // If the current start ptr is the same as end we have no next and must return None
            if self.ptr.as_ptr() == self.end as *mut T {
                return None;
            }
            // we must leave the ptr as it because it is Ox1 dangling
            // to adjust out len we, therefore, must reduce our end
            self.end = self.end.wrapping_byte_sub(1);
            self.ptr
        } else {

            if self.ptr.as_ptr() == self.end as *mut T {
                //if self.ptr == non_null!(self.end, T) {
                // unsafe { *((&raw const $place) as *const NonNull<$t>) }
                return None;
            }

            let old_ptr = self.ptr;
            self.ptr = unsafe { old_ptr.add(1) };
            old_ptr
        };
        Some( unsafe { ptr.read() } )
    }
}