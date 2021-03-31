// Copyright (c) 2021 The Authors
// SPDX-License-Identifier: MPL-2.0

//! Finite field arithmetic for any field GF(p) for which p < 2^126.

#[cfg(test)]
use rand::{prelude::*, Rng};

/// This structure represents the parameters of a finite field GF(p) for which p < 2^126.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FieldParameters {
    /// The prime modulus `p`.
    pub p: u128,
    /// `p * 2`.
    pub p2: u128,
    /// `mu = -p^(-1) mod 2^64`.
    pub mu: u64,
    /// `r2 = (2^128)^2 mod p`.
    pub r2: u128,
    /// The generator of a multiplicative subgroup of order `num_roots`. The value is mapped to the
    /// Montgomeryh domain.
    pub g: u128,
    /// The order of the multiplicaitve subgroup of generated by `g`.
    pub num_roots: u128,
}

pub(crate) const FP32: FieldParameters = FieldParameters {
    p: 4293918721, // 32-bit prime
    p2: 8587837442,
    mu: 17302828673139736575,
    r2: 1676699750,
    g: 1074114499,
    num_roots: 1 << 20,
};

pub(crate) const FP64: FieldParameters = FieldParameters {
    p: 15564440312192434177, // 64-bit prime
    p2: 31128880624384868354,
    mu: 15564440312192434175,
    r2: 13031533328350459868,
    g: 8693478717884812021,
    num_roots: 1 << 59,
};

pub(crate) const FP80: FieldParameters = FieldParameters {
    p: 779190469673491460259841, // 80-bit prime
    p2: 1558380939346982920519682,
    mu: 18446744073709551615,
    r2: 699883506621195336351723,
    g: 470015708362303528848629,
    num_roots: 1 << 72,
};

pub(crate) const FP126: FieldParameters = FieldParameters {
    p: 74769074762901517850839147140769382401, // 126-bit prime
    p2: 149538149525803035701678294281538764802,
    mu: 18446744073709551615,
    r2: 27801541991839173768379182336352451464,
    g: 63245316532470582112420298384754157617,
    num_roots: 1 << 118,
};

impl FieldParameters {
    /// Addition.
    pub fn add(&self, x: u128, y: u128) -> u128 {
        let (z, carry) = x.wrapping_add(y).overflowing_sub(self.p2);
        let m = 0u128.wrapping_sub(carry as u128);
        z.wrapping_add(m & self.p2)
    }

    /// Subtraction.
    pub fn sub(&self, x: u128, y: u128) -> u128 {
        let (z, carry) = x.overflowing_sub(y);
        let m = 0u128.wrapping_sub(carry as u128);
        z.wrapping_add(m & self.p2)
    }

    /// Multiplication of field elements in the Montgomery domain. This uses the REDC algorithm
    /// described
    /// [here](https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/S0025-5718-1985-0777282-X.pdfA).
    ///
    /// Example usage:
    /// assert_eq!(fp.from_elem(fp.mul(fp.elem(23), fp.elem(2))), 46);
    pub fn mul(&self, x: u128, y: u128) -> u128 {
        let x = [lo64(x), hi64(x)];
        let y = [lo64(y), hi64(y)];
        let p = [lo64(self.p), hi64(self.p)];
        let mut zz = [0; 4];
        let mut result: u128;
        let mut carry: u128;
        let mut hi: u128;
        let mut lo: u128;
        let mut cc: u128;

        // Integer multiplication
        result = x[0] * y[0];
        carry = hi64(result);
        zz[0] = lo64(result);
        result = x[0] * y[1];
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        zz[2] = lo64(result);

        result = x[1] * y[0];
        hi = hi64(result);
        lo = lo64(result);
        result = zz[1] + lo;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        carry = lo64(result);

        result = x[1] * y[1];
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        lo = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        hi = lo64(result);
        result = zz[2] + lo;
        zz[2] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        zz[3] = lo64(result);

        // Reduction
        let w = self.mu.wrapping_mul(zz[0] as u64);
        result = p[0] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = zz[0] + lo;
        zz[0] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        carry = lo64(result);

        result = p[1] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        lo = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        hi = lo64(result);
        result = zz[1] + lo;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = zz[2] + hi + cc;
        zz[2] = lo64(result);
        cc = hi64(result);
        result = zz[3] + cc;
        zz[3] = lo64(result);

        let w = self.mu.wrapping_mul(zz[1] as u64);
        result = p[0] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = zz[1] + lo;
        zz[1] = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        carry = lo64(result);

        result = p[1] * (w as u128);
        hi = hi64(result);
        lo = lo64(result);
        result = lo + carry;
        lo = lo64(result);
        cc = hi64(result);
        result = hi + cc;
        hi = lo64(result);
        result = zz[2] + lo;
        zz[2] = lo64(result);
        cc = hi64(result);
        result = zz[3] + hi + cc;
        zz[3] = lo64(result);

        zz[2] | (zz[3] << 64)
    }

    /// Modular exponentiation, i.e., `x^exp (mod p)` where `p` is the modulus. Note that the
    /// runtime of this algorithm is linear in the bit length of `exp`.
    pub fn pow(&self, x: u128, exp: u128) -> u128 {
        let mut t = self.elem(1);
        for i in (0..128 - exp.leading_zeros()).rev() {
            t = self.mul(t, t);
            if (exp >> i) & 1 != 0 {
                t = self.mul(t, x);
            }
        }
        t
    }

    /// Modular inversion, i.e., x^-1 (mod p) where `p` is the modulu. Note that the runtime of
    /// this algorithm is linear in the bit length of `p`.
    pub fn inv(&self, x: u128) -> u128 {
        self.pow(x, self.p - 2)
    }

    /// Negation, i.e., `-x (mod p)` where `p` is the modulus.
    pub fn neg(&self, x: u128) -> u128 {
        self.sub(0, x)
    }

    /// Maps an integer to its internal representation. Field elements are mapped to the Montgomery
    /// domain in order to carry out field arithmetic.
    ///
    /// Example usage:
    /// let integer = 1; // Standard integer representation
    /// let elem = fp.elem(integer); // Internal representation in the Montgomery domain
    /// assert_eq!(elem, 2564090464);
    pub fn elem(&self, x: u128) -> u128 {
        modp(self.mul(x, self.r2), self.p)
    }

    /// Returns a random field element mapped.
    #[cfg(test)]
    pub fn rand_elem<R: Rng + ?Sized>(&self, rng: &mut R) -> u128 {
        let uniform = rand::distributions::Uniform::from(0..self.p);
        self.elem(uniform.sample(rng))
    }

    /// Maps a field element to its representation as an integer.
    ///
    /// Example usage:
    /// let elem = 2564090464; // Internal representation in the Montgomery domain
    /// let integer = fp.from_elem(elem); // Standard integer representation
    /// assert_eq!(integer, 1);
    pub fn from_elem(&self, x: u128) -> u128 {
        modp(self.mul(x, 1), self.p)
    }

    /// Returns the number of bytes required to encode field elements.
    #[cfg(test)] // This code is only used by tests for now.
    pub fn size(&self) -> usize {
        (16 - (self.p.leading_zeros() / 8)) as usize
    }

    #[cfg(test)]
    pub fn new(p: u128, g: u128, num_roots: u128) -> Result<FieldParameters, &'static str> {
        use modinverse::modinverse;
        use num_bigint::{BigInt, ToBigInt};

        let err_modulus_too_large = "p > 2^126";
        if let Some(x) = p.checked_next_power_of_two() {
            if x > 1 << 126 {
                return Err(err_modulus_too_large);
            }
        } else {
            return Err(err_modulus_too_large);
        }

        let mu = match modinverse((-(p as i128)).rem_euclid(1 << 64), 1 << 64) {
            Some(mu) => mu as u64,
            None => return Err("inverse of -p (mod 2^64) is undefined"),
        };

        let big_p = &p.to_bigint().unwrap();
        let big_r: &BigInt = &(&(BigInt::from(1) << 128) % big_p);
        let big_r2: &BigInt = &(&(big_r * big_r) % big_p);
        let mut it = big_r2.iter_u64_digits();
        let mut r2 = 0;
        r2 |= it.next().unwrap() as u128;
        if let Some(x) = it.next() {
            r2 |= (x as u128) << 64;
        }

        let mut fp = FieldParameters {
            p: p,
            p2: p << 1,
            mu: mu,
            r2: r2,
            g: 0,
            num_roots: num_roots,
        };

        fp.g = fp.elem(g);
        Ok(fp)
    }
}

fn lo64(x: u128) -> u128 {
    x & ((1 << 64) - 1)
}

fn hi64(x: u128) -> u128 {
    x >> 64
}

fn modp(x: u128, p: u128) -> u128 {
    let (z, carry) = x.overflowing_sub(p);
    let m = 0u128.wrapping_sub(carry as u128);
    z.wrapping_add(m & p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use modinverse::modinverse;
    use num_bigint::ToBigInt;

    struct TestFieldParametersData {
        fp: FieldParameters,  // The paramters being tested
        expected_p: u128,     // Expected fp.p
        expected_g: u128,     // Expected fp.from_elem(fp.g)
        expected_order: u128, // Expect fp.from_elem(fp.pow(fp.g, expected_order)) == 1
        expected_size: usize, // expected fp.size()
    }

    #[test]
    fn test_fp() {
        let test_fps = vec![
            TestFieldParametersData {
                fp: FP32,
                expected_p: 4293918721,
                expected_g: 3925978153,
                expected_order: 1 << 20,
                expected_size: 4,
            },
            TestFieldParametersData {
                fp: FP64,
                expected_p: 15564440312192434177,
                expected_g: 7450580596923828125,
                expected_order: 1 << 59,
                expected_size: 8,
            },
            TestFieldParametersData {
                fp: FP80,
                expected_p: 779190469673491460259841,
                expected_g: 41782115852031095118226,
                expected_order: 1 << 72,
                expected_size: 10,
            },
            TestFieldParametersData {
                fp: FP126,
                expected_p: 74769074762901517850839147140769382401,
                expected_g: 43421413544015439978138831414974882540,
                expected_order: 1 << 118,
                expected_size: 16,
            },
        ];

        for t in test_fps.into_iter() {
            //  Check that the field parameters have been constructed properly.
            assert_eq!(
                t.fp,
                FieldParameters::new(t.expected_p, t.expected_g, t.expected_order).unwrap(),
                "error for GF({})",
                t.expected_p,
            );

            // Check that the field element size is computed correctly.
            assert_eq!(
                t.fp.size(),
                t.expected_size,
                "error for GF({})",
                t.expected_p
            );

            // Check that the generator has the correct order.
            assert_eq!(t.fp.from_elem(t.fp.pow(t.fp.g, t.expected_order)), 1);

            // Test arithmetic using the field parameters.
            test_arithmetic(t.fp);
        }
    }

    fn test_arithmetic(fp: FieldParameters) {
        let mut rng = rand::thread_rng();
        let big_p = &fp.p.to_bigint().unwrap();
        for _ in 0..100 {
            let x = fp.rand_elem(&mut rng);
            let y = fp.rand_elem(&mut rng);
            let big_x = &fp.from_elem(x).to_bigint().unwrap();
            let big_y = &fp.from_elem(y).to_bigint().unwrap();

            // Test addition.
            let got = fp.add(x, y);
            let want = (big_x + big_y) % big_p;
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);

            // Test subtraction.
            let got = fp.sub(x, y);
            let want = if big_x >= big_y {
                big_x - big_y
            } else {
                big_p - big_y + big_x
            };
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);

            // Test multiplication.
            let got = fp.mul(x, y);
            let want = (big_x * big_y) % big_p;
            assert_eq!(fp.from_elem(got).to_bigint().unwrap(), want);

            // Test inversion.
            let got = fp.inv(x);
            let want = modinverse(fp.from_elem(x) as i128, fp.p as i128).unwrap();
            assert_eq!(fp.from_elem(got) as i128, want);
            assert_eq!(fp.from_elem(fp.mul(got, x)), 1);

            // Test negation.
            let got = fp.neg(x);
            let want = (-(fp.from_elem(x) as i128)).rem_euclid(fp.p as i128);
            assert_eq!(fp.from_elem(got) as i128, want);
            assert_eq!(fp.from_elem(fp.add(got, x)), 0);
        }
    }
}