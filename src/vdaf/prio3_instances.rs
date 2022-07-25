//! Implementation of the Prio3 instances from [[draft-irtf-cfrg-vdaf-01]].
//! These use the internal PRG.
//!
use crate::field::{Field128, Field64};
#[cfg(feature = "multithreaded")]
use crate::flp::gadgets::ParallelSumMultithreaded;
use crate::flp::gadgets::{BlindPolyEval, ParallelSum};
use crate::flp::types::{Count, CountVec, Histogram, Sum};
use crate::vdaf::prg::PrgAes128;
use crate::vdaf::prio3::check_num_aggregators;
use crate::vdaf::prio3::Prio3;
use crate::vdaf::VdafError;
#[cfg(feature = "multithreaded")]
use std::marker::PhantomData;

/// The count type. Each measurement is an integer in `[0,2)` and the aggregate result is the sum.
pub type Prio3Aes128Count = Prio3<Count<Field64>, PrgAes128, 16>;

impl Prio3Aes128Count {
    /// Construct an instance of Prio3Aes128Count with the given number of aggregators.
    pub fn new_aes128_count(num_aggregators: u8) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        Ok(Prio3::new(num_aggregators, Count::new())?)
    }
}

/// The count-vector type. Each measurement is a vector of integers in `[0,2)` and the aggregate is
/// the element-wise sum.
pub type Prio3Aes128CountVec =
    Prio3<CountVec<Field128, ParallelSum<Field128, BlindPolyEval<Field128>>>, PrgAes128, 16>;

impl Prio3Aes128CountVec {
    /// Construct an instance of Prio3Aes1238CountVec with the given number of aggregators. `len`
    /// defines the length of each measurement.
    pub fn new_aes128_count_vec(num_aggregators: u8, len: usize) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        Ok(Prio3::new(num_aggregators, CountVec::new(len))?)
    }
}

/// Like [`Prio3Aes128CountVec`] except this type uses multithreading to improve sharding and
/// preparation time. Note that the improvement is only noticeable for very large input lengths,
/// e.g., 201 and up. (Your system's mileage may vary.)
#[cfg(feature = "multithreaded")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
pub type Prio3Aes128CountVecMultithreaded = Prio3<
    CountVec<Field128, ParallelSumMultithreaded<Field128, BlindPolyEval<Field128>>>,
    PrgAes128,
    16,
>;

#[cfg(feature = "multithreaded")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreaded")))]
impl Prio3Aes128CountVecMultithreaded {
    /// Construct an instance of Prio3Aes1238CountVecMultithreaded with the given number of
    /// aggregators. `len` defines the length of each measurement.
    pub fn new_aes128_count_vec_multithreaded(
        num_aggregators: u8,
        len: usize,
    ) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        Ok(Prio3::new(num_aggregators, CountVec::new(len))?)
    }
}

/// The sum type. Each measurement is an integer in `[0,2^bits)` for some `0 < bits < 64` and the
/// aggregate is the sum.
pub type Prio3Aes128Sum = Prio3<Sum<Field128>, PrgAes128, 16>;

impl Prio3Aes128Sum {
    /// Construct an instance of Prio3Aes128Sum with the given number of aggregators and required
    /// bit length. The bit length must not exceed 64.
    pub fn new_aes128_sum(num_aggregators: u8, bits: u32) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        if bits > 64 {
            return Err(VdafError::Uncategorized(format!(
                "bit length ({}) exceeds limit for aggregate type (64)",
                bits
            )));
        }

        Ok(Prio3::new(num_aggregators, Sum::new(bits as usize)?)?)
    }
}
/// The histogram type. Each measurement is an unsigned integer and the result is a histogram
/// representation of the distribution. The bucket boundaries are fixed in advance.
pub type Prio3Aes128Histogram = Prio3<Histogram<Field128>, PrgAes128, 16>;

impl Prio3Aes128Histogram {
    /// Constructs an instance of Prio3Aes128Histogram with the given number of aggregators and
    /// desired histogram bucket boundaries.
    pub fn new_aes128_histogram(num_aggregators: u8, buckets: &[u64]) -> Result<Self, VdafError> {
        check_num_aggregators(num_aggregators)?;

        let buckets = buckets.iter().map(|bucket| *bucket as u128).collect();

        Ok(Prio3::new(
            num_aggregators,
            Histogram::<Field128>::new(buckets)?,
        )?)
    }
}

#[cfg(tests)]
mod tests {
    use super::*;

    #[test]
    fn test_prio3_count() {
        let prio3 = Prio3::new_aes128_count(2).unwrap();

        assert_eq!(run_vdaf(&prio3, &(), [1, 0, 0, 1, 1]).unwrap(), 3);

        let mut verify_key = [0; 16];
        thread_rng().fill(&mut verify_key[..]);
        let nonce = b"This is a good nonce.";

        let input_shares = prio3.shard(&0).unwrap();
        run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares).unwrap();

        let input_shares = prio3.shard(&1).unwrap();
        run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares).unwrap();

        test_prepare_state_serialization(&prio3, &1).unwrap();
    }

    #[test]
    fn test_prio3_sum() {
        let prio3 = Prio3::new_aes128_sum(3, 16).unwrap();

        assert_eq!(
            run_vdaf(&prio3, &(), [0, (1 << 16) - 1, 0, 1, 1]).unwrap(),
            (1 << 16) + 1
        );

        let mut verify_key = [0; 16];
        thread_rng().fill(&mut verify_key[..]);
        let nonce = b"This is a good nonce.";

        let mut input_shares = prio3.shard(&1).unwrap();
        input_shares[0].joint_rand_param.as_mut().unwrap().blind.0[0] ^= 255;
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let mut input_shares = prio3.shard(&1).unwrap();
        input_shares[0]
            .joint_rand_param
            .as_mut()
            .unwrap()
            .seed_hint
            .0[0] ^= 255;
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let mut input_shares = prio3.shard(&1).unwrap();
        assert_matches!(input_shares[0].input_share, Share::Leader(ref mut data) => {
            data[0] += Field128::one();
        });
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        let mut input_shares = prio3.shard(&1).unwrap();
        assert_matches!(input_shares[0].proof_share, Share::Leader(ref mut data) => {
                data[0] += Field128::one();
        });
        let result = run_vdaf_prepare(&prio3, &verify_key, &(), nonce, input_shares);
        assert_matches!(result, Err(VdafError::Uncategorized(_)));

        test_prepare_state_serialization(&prio3, &1).unwrap();
    }

    #[test]
    fn test_prio3_histogram() {
        let prio3 = Prio3::new_aes128_histogram(2, &[0, 10, 20]).unwrap();

        assert_eq!(
            run_vdaf(&prio3, &(), [0, 10, 20, 9999]).unwrap(),
            vec![1, 1, 1, 1]
        );
        assert_eq!(run_vdaf(&prio3, &(), [0]).unwrap(), vec![1, 0, 0, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [5]).unwrap(), vec![0, 1, 0, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [10]).unwrap(), vec![0, 1, 0, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [15]).unwrap(), vec![0, 0, 1, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [20]).unwrap(), vec![0, 0, 1, 0]);
        assert_eq!(run_vdaf(&prio3, &(), [25]).unwrap(), vec![0, 0, 0, 1]);
        test_prepare_state_serialization(&prio3, &23).unwrap();
    }
}
