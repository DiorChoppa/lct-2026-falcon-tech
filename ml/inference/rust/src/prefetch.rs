//! A single CPU producer, with no queued batch beyond its current preparation.
use crate::{Error, Result};
use std::{sync::mpsc::sync_channel, thread};

/// A zero-capacity channel permits exactly one producer-owned future result.
/// Drop the receiver before joining on every consumer exit, including errors.
pub(crate) fn ordered<T: Send>(
    prepared: impl Iterator<Item = Result<T>> + Send,
    mut consume: impl FnMut(T) -> Result<bool>,
) -> Result<()> {
    thread::scope(|scope| {
        let (sender, receiver) = sync_channel(0);
        let producer = thread::Builder::new().name("lct-next-batch".into()).spawn_scoped(scope, move || {
            for item in prepared {
                let failed = item.is_err();
                if sender.send(item).is_err() || failed { break; }
            }
        })?;
        let result = (|| {
            while let Ok(item) = receiver.recv() {
                if !consume(item?)? { break; }
            }
            Ok(())
        })();
        drop(receiver);
        let joined = producer.join().map_err(|_| Error::Invalid("CPU preparation producer panicked".into()));
        result.and(joined)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

    struct Tracked(usize, Arc<AtomicUsize>);
    impl Drop for Tracked { fn drop(&mut self) { self.1.fetch_sub(1, Ordering::SeqCst); } }

    #[test]
    fn ordered_delivery_has_at_most_current_plus_one_future_and_drops_on_stop() {
        let alive = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let producer_alive = alive.clone();
        let producer_maximum = maximum.clone();
        let prepared = (0..100).map(move |index| {
            let n = producer_alive.fetch_add(1, Ordering::SeqCst) + 1;
            producer_maximum.fetch_max(n, Ordering::SeqCst);
            Ok(Tracked(index, producer_alive.clone()))
        });
        let mut seen = Vec::new();
        ordered(prepared, |item| {
            seen.push(item.0);
            std::thread::sleep(std::time::Duration::from_millis(1));
            Ok(seen.len() < 7)
        }).unwrap();
        assert_eq!(seen, (0..7).collect::<Vec<_>>());
        assert!(maximum.load(Ordering::SeqCst) <= 2);
        assert_eq!(alive.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn preparation_error_is_ordered_and_consumer_error_does_not_deadlock() {
        let mut seen = Vec::new();
        let error = ordered((0..8).map(|i| if i == 3 { Err(Error::Invalid("prepare3".into())) } else { Ok(i) }), |i| {
            seen.push(i); Ok(true)
        }).unwrap_err();
        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(error.to_string(), "prepare3");
        let error = ordered((0..100).map(Ok), |_| Err(Error::Invalid("consumer".into()))).unwrap_err();
        assert_eq!(error.to_string(), "consumer");
    }

    #[test]
    fn producer_panic_is_reported_after_join_and_empty_stream_succeeds() {
        assert!(ordered(std::iter::empty::<Result<usize>>(), |_| unreachable!()).is_ok());
        let prepared = (0..1).map(|_| -> Result<usize> { panic!("fixture"); });
        assert!(ordered(prepared, |_| Ok(true)).unwrap_err().to_string().contains("producer panicked"));
    }
}
