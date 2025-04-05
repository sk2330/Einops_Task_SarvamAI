from Einops_module import rearrange, parse_axes, apply_operations
import numpy as np

if __name__ == "__main__":
    # Test 1: Simple transposition
    x1 = np.random.rand(2, 3, 4, 5)
    y1 = rearrange(x1, 'a b c d -> a d b c')
    assert y1.shape == (2, 5, 3, 4)
    assert np.array_equal(y1, np.transpose(x1, (0, 3, 1, 2)))
    print("Test 1: Simple transposition - PASSED")

    # Test 2: Splitting dimensions
    x2 = np.random.rand(2, 6, 8, 3)
    y2 = rearrange(x2, 'b (h h2) (w w2) c -> b h w h2 w2 c', h2=2, w2=4)
    assert y2.shape == (2, 3, 2, 2, 4, 3)
    print("Test 2: Splitting dimensions - PASSED")

    # Test 3: Merging dimensions
    x3 = np.random.rand(2, 3, 4, 5)
    y3 = rearrange(x3, 'b h w c -> b (h w) c')
    assert y3.shape == (2, 12, 5)
    print("Test 3: Merging dimensions - PASSED")

    # Test 4: Using ellipsis
    x4 = np.random.rand(2, 3, 4, 5, 6)
    y4 = rearrange(x4, '... h w c -> ... (h w) c')
    assert y4.shape == (2, 3, 20, 6)

    assert y4.shape == (2, 3, 20, 6)
    print("Test 4: Ellipsis handling - PASSED")

    # Test 5: Numpy stack comparison
    x5 = np.random.rand(2, 3, 4, 1)
    y5 = rearrange(x5, 'b h w c -> h w c b')
    numpy_result = np.stack(x5, axis=3)
    assert y5.shape == numpy_result.shape
    print("Test 5: Numpy stack comparison - PASSED")

    # Test 6: Error handling - Invalid pattern
    try:
        rearrange(x1, 'a b c d')  # Missing arrow
        assert False, "Should have raised ValueError for invalid pattern"
    except ValueError as e:
        assert "Invalid pattern" in str(e)
    print("Test 6: Invalid pattern detection - PASSED")

    # Test 7: Error handling - Mismatched tensor shape
    try:
        rearrange(x1, 'a b c d e -> a b c d e')  # Too many dimensions
        assert False, "Should have raised ValueError for mismatched dimensions"
    except ValueError as e:
        assert "dimensions" in str(e)
    print("Test 7: Mismatched dimensions detection - PASSED")

    # Test 8: Error handling - Missing axes_lengths
    try:
        rearrange(x2, 'b (h h2) (w w2) c -> b h w c', h2=2)  # Missing w2
        assert False, "Should have raised ValueError for missing axes_lengths"
    except ValueError as e:
        assert "unspecified" in str(e) or "Missing" in str(e)
    print("Test 8: Missing axes_lengths detection - PASSED")

    # Test 9: Error handling - Incompatible dimensions
    try:
        rearrange(x2, 'b (h h2) (w w2) c -> b h w c', h2=5, w2=5)  # Can't divide 6 by 5
        assert False, "Should have raised ValueError for incompatible dimensions"
    except ValueError as e:
        assert "split" in str(e) or "evenly" in str(e)
    print("Test 9: Incompatible dimensions detection - PASSED")

    # Test 10: List input handling
    x10 = [np.random.rand(3, 4, 1) for _ in range(2)]
    y10 = rearrange(x10, 'b h w c -> h w c b')
    assert y10.shape == (3, 4, 1, 2)
    print("Test 10: List input handling - PASSED")

    print("\nAll tests passed successfully!")