/**
 * @file test_data_structures.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2026-06-25
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "data_modeling/tensor.h"
#include "data_modeling/matmul_tile.h"

using namespace std;

TEST(TensorOpsTest, TestCtor) {
  auto t = Tensor({2, 2}, {2.0, 3.0, 4.0, 5.0}, Device::CPU);

  ASSERT_EQ(t.getDims(), Dimension({2, 2}));
  ASSERT_EQ(t.getDevice(), Device::CPU);
  ASSERT_TRUE(!t.getRequiresGrad());

  ASSERT_NEAR(t.get(0, 0), 2.0, 1e-5);
  ASSERT_NEAR(t.get(0, 1), 3.0, 1e-5);
  ASSERT_NEAR(t.get(1, 0), 4.0, 1e-5);
  ASSERT_NEAR(t.get(1, 1), 5.0, 1e-5);
}

static constexpr float kSrc4x4[16] = {
   1,  2,  3,  4,
   5,  6,  7,  8,
   9, 10, 11, 12,
  13, 14, 15, 16,
};

// --- loadLeft ---
TEST(MatmulTileTest, LoadLeft_FullTileAtOrigin) {
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.loadLeft(kSrc4x4, 0, 0, 4, 4);
  EXPECT_FLOAT_EQ(tile.left[0],  1.f);
  EXPECT_FLOAT_EQ(tile.left[1],  2.f);
  EXPECT_FLOAT_EQ(tile.left[2],  5.f);
  EXPECT_FLOAT_EQ(tile.left[3],  6.f);
}

// Load a full 2x2 left tile at offset (row=1, col=2).
// Covers rows 1-2, cols 2-3: { 7, 8, 11, 12 }
TEST(MatmulTileTest, LoadLeft_FullTileAtOffset) {
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.loadLeft(kSrc4x4, 1, 2, 4, 4);
  EXPECT_FLOAT_EQ(tile.left[0],  7.f);
  EXPECT_FLOAT_EQ(tile.left[1],  8.f);
  EXPECT_FLOAT_EQ(tile.left[2], 11.f);
  EXPECT_FLOAT_EQ(tile.left[3], 12.f);
}

// Load a 2x3 left tile from a 4x4 matrix at origin.
// Covers rows 0-1, cols 0-2: { 1, 2, 3, 5, 6, 7 }
TEST(MatmulTileTest, LoadLeft_RectangularTile) {
  matmul::MatmulTile<float, 2, 3, 2> tile;
  tile.loadLeft(kSrc4x4, 0, 0, 4, 4);
  EXPECT_FLOAT_EQ(tile.left[0], 1.f);
  EXPECT_FLOAT_EQ(tile.left[1], 2.f);
  EXPECT_FLOAT_EQ(tile.left[2], 3.f);
  EXPECT_FLOAT_EQ(tile.left[3], 5.f);
  EXPECT_FLOAT_EQ(tile.left[4], 6.f);
  EXPECT_FLOAT_EQ(tile.left[5], 7.f);
}

// Tile is wider than the available columns — only 1 column remains at col=3.
// For a 2x2 tile starting at (0, 3) in a 4x4 matrix: col 3 only.
// Expected: left[0]=4, left[1]=0(pad), left[2]=8, left[3]=0(pad)
TEST(MatmulTileTest, LoadLeft_PartialColumns) {
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.loadLeft(kSrc4x4, 0, 3, 4, 4);
  EXPECT_FLOAT_EQ(tile.left[0],  4.f);
  EXPECT_FLOAT_EQ(tile.left[1],  0.f);  // zero-padded (tile is 2 wide, only 1 col available)
  EXPECT_FLOAT_EQ(tile.left[2],  8.f);
  EXPECT_FLOAT_EQ(tile.left[3],  0.f);
}

// Tile is taller than the available rows — only 1 row remains at row=3.
// For a 2x2 tile starting at (3, 0) in a 4x4 matrix.
// Expected: left[0]=13, left[1]=14, left[2]=0(pad), left[3]=0(pad)
TEST(MatmulTileTest, LoadLeft_PartialRows) {
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.loadLeft(kSrc4x4, 3, 0, 4, 4);
  EXPECT_FLOAT_EQ(tile.left[0], 13.f);
  EXPECT_FLOAT_EQ(tile.left[1], 14.f);
  EXPECT_FLOAT_EQ(tile.left[2],  0.f);  // zero-padded (tile is 2 tall, only 1 row available)
  EXPECT_FLOAT_EQ(tile.left[3],  0.f);
}

// --- loadRight ---

// Load a full 2x2 right tile from the top-left corner.
// Expected: { 1, 2, 5, 6 }
TEST(MatmulTileTest, LoadRight_FullTileAtOrigin) {
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.loadRight(kSrc4x4, 0, 0, 4, 4);
  EXPECT_FLOAT_EQ(tile.right[0],  1.f);
  EXPECT_FLOAT_EQ(tile.right[1],  2.f);
  EXPECT_FLOAT_EQ(tile.right[2],  5.f);
  EXPECT_FLOAT_EQ(tile.right[3],  6.f);
}

// Load a full 2x2 right tile at offset (row=2, col=1).
// Covers rows 2-3, cols 1-2: { 10, 11, 14, 15 }
TEST(MatmulTileTest, LoadRight_FullTileAtOffset) {
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.loadRight(kSrc4x4, 2, 1, 4, 4);
  EXPECT_FLOAT_EQ(tile.right[0], 10.f);
  EXPECT_FLOAT_EQ(tile.right[1], 11.f);
  EXPECT_FLOAT_EQ(tile.right[2], 14.f);
  EXPECT_FLOAT_EQ(tile.right[3], 15.f);
}

// Load a 3x2 right tile (TileK=3 rows, TileN=2 cols) from a 4x4 matrix at (1, 2).
// Covers rows 1-3, cols 2-3: { 7, 8, 11, 12, 15, 16 }
TEST(MatmulTileTest, LoadRight_RectangularTile) {
  matmul::MatmulTile<float, 2, 3, 2> tile;
  tile.loadRight(kSrc4x4, 1, 2, 4, 4);
  EXPECT_FLOAT_EQ(tile.right[0],  7.f);
  EXPECT_FLOAT_EQ(tile.right[1],  8.f);
  EXPECT_FLOAT_EQ(tile.right[2], 11.f);
  EXPECT_FLOAT_EQ(tile.right[3], 12.f);
  EXPECT_FLOAT_EQ(tile.right[4], 15.f);
  EXPECT_FLOAT_EQ(tile.right[5], 16.f);
}

// Tile is wider than remaining columns at col=3 (only 1 col available for TileN=2).
// For a 2x2 right tile (TileK=2 rows, TileN=2 cols) starting at (0, 3).
// Expected: right[0]=4, right[1]=0(pad), right[2]=8, right[3]=0(pad)
TEST(MatmulTileTest, LoadRight_PartialColumns) {
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.loadRight(kSrc4x4, 0, 3, 4, 4);
  EXPECT_FLOAT_EQ(tile.right[0],  4.f);
  EXPECT_FLOAT_EQ(tile.right[1],  0.f);
  EXPECT_FLOAT_EQ(tile.right[2],  8.f);
  EXPECT_FLOAT_EQ(tile.right[3],  0.f);
}

// --- addResult ---

// Add a full 2x2 result tile into a zero-initialised 4x4 dst at origin.
// result = { 10, 20,
//             30, 40 }
// After add: dst[0,0]=10, dst[0,1]=20, dst[1,0]=30, dst[1,1]=40; rest 0.
TEST(MatmulTileTest, AddResult_FullTileAtOrigin) {
  float dst[16] = {};
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.result = {10.f, 20.f, 30.f, 40.f};
  tile.addResult(dst, 0, 0, 4, 4);
  EXPECT_FLOAT_EQ(dst[0],  10.f);
  EXPECT_FLOAT_EQ(dst[1],  20.f);
  EXPECT_FLOAT_EQ(dst[4],  30.f);
  EXPECT_FLOAT_EQ(dst[5],  40.f);
  // rest must be untouched
  EXPECT_FLOAT_EQ(dst[2],   0.f);
  EXPECT_FLOAT_EQ(dst[6],   0.f);
}

// Verify += semantics: pre-existing dst values are preserved and accumulated.
TEST(MatmulTileTest, AddResult_Accumulates) {
  float dst[16];
  for(int i = 0; i < 16; i++) dst[i] = static_cast<float>(i + 1); // 1..16
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.result = {1.f, 1.f, 1.f, 1.f};
  tile.addResult(dst, 0, 0, 4, 4);
  EXPECT_FLOAT_EQ(dst[0],  2.f);   // 1+1
  EXPECT_FLOAT_EQ(dst[1],  3.f);   // 2+1
  EXPECT_FLOAT_EQ(dst[4],  6.f);   // 5+1
  EXPECT_FLOAT_EQ(dst[5],  7.f);   // 6+1
  EXPECT_FLOAT_EQ(dst[2],  3.f);   // unchanged
  EXPECT_FLOAT_EQ(dst[6],  7.f);   // unchanged
}

// Add a 2x2 result tile into a zero-initialised 4x4 dst at offset (row=1, col=2).
// result = { 100, 200,
//             300, 400 }
// After add: dst[1*4+2]=100, dst[1*4+3]=200, dst[2*4+2]=300, dst[2*4+3]=400.
TEST(MatmulTileTest, AddResult_FullTileAtOffset) {
  float dst[16] = {};
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.result = {100.f, 200.f, 300.f, 400.f};
  tile.addResult(dst, 1, 2, 4, 4);
  EXPECT_FLOAT_EQ(dst[1*4+2], 100.f);
  EXPECT_FLOAT_EQ(dst[1*4+3], 200.f);
  EXPECT_FLOAT_EQ(dst[2*4+2], 300.f);
  EXPECT_FLOAT_EQ(dst[2*4+3], 400.f);
  EXPECT_FLOAT_EQ(dst[0],       0.f);  // untouched
  EXPECT_FLOAT_EQ(dst[1*4+1],   0.f);  // untouched
}

// Tile extends past the last column (col=3, TileN=2 → only col 3 within bounds).
// result = { 10, 20,
//             30, 40 }
// Only the first element of each tile row is within the matrix.
// After add: dst[3]=10, dst[7]=30; result[1] and result[3] are out-of-bounds, not written.
TEST(MatmulTileTest, AddResult_PartialColumns) {
  float dst[16] = {};
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.result = {10.f, 20.f, 30.f, 40.f};
  tile.addResult(dst, 0, 3, 4, 4);
  EXPECT_FLOAT_EQ(dst[3],  10.f);
  EXPECT_FLOAT_EQ(dst[7],  30.f);
  EXPECT_FLOAT_EQ(dst[0],   0.f);  // untouched
  EXPECT_FLOAT_EQ(dst[4],   0.f);  // untouched
}

// Tile extends past the last row (row=3, TileM=2 → only row 3 within bounds).
// result = { 10, 20,
//             30, 40 }
// Only the first tile row is within the matrix.
// After add: dst[3*4+0]=10, dst[3*4+1]=20; result[2] and result[3] not written.
TEST(MatmulTileTest, AddResult_PartialRows) {
  float dst[16] = {};
  matmul::MatmulTile<float, 2, 2, 2> tile;
  tile.result = {10.f, 20.f, 30.f, 40.f};
  tile.addResult(dst, 3, 0, 4, 4);
  EXPECT_FLOAT_EQ(dst[3*4+0], 10.f);
  EXPECT_FLOAT_EQ(dst[3*4+1], 20.f);
  EXPECT_FLOAT_EQ(dst[0],      0.f);  // untouched
  EXPECT_FLOAT_EQ(dst[1],      0.f);  // untouched
}