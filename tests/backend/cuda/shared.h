/**
 * @file shared.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Shared test utilities for CUDA unit tests.
 * @version 0.1
 * @date 2026-08-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <string>

namespace cuda_test {

  /**
   * @brief Accumulates element-wise near-equality checks over a large loop
   * without flooding test output with one failure message per element.
   *
   * Individual mismatches are only reported (via EXPECT_NEAR) for the first
   * `maxFailures` occurrences; beyond that they're only counted. finalize()
   * fails the test if the total mismatch count exceeds `maxFailures`, so a
   * handful of elements can be off (e.g. due to floating point
   * non-associativity between the CPU and GPU code paths) without the whole
   * comparison being tossed out.
   *
   * Usage:
   *   cuda_test::NumericalStabilityChecker checker(1e-4);
   *   for (int i = 0; i < n; i++) {
   *     checker.check(actual[i], expected[i], "index " + std::to_string(i));
   *   }
   *   checker.finalize();
   */
  struct NumericalStabilityChecker final {
  private:
    double tolerance;
    int maxFailures;
    int failures = 0;
    int total = 0;

  public:
    explicit NumericalStabilityChecker(double tolerance, int maxFailures = 5)
        : tolerance(tolerance), maxFailures(maxFailures) {}

    void check(double actual, double expected, const std::string& context = "") {
      total++;
      if (std::abs(actual - expected) > tolerance) {
        failures++;
        if (failures <= maxFailures) {
          EXPECT_NEAR(actual, expected, tolerance) << context;
        }
      }
    }

    // Call once after the loop; fails the test if too many elements mismatched.
    void finalize() const {
      EXPECT_LE(failures, maxFailures)
          << failures << " / " << total << " elements exceeded tolerance " << tolerance
          << " (max allowed failures: " << maxFailures << ")";
    }
  };

}
