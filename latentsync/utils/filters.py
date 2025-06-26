import numpy as np
import scipy.signal
import logging
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def _plot_savgol_debug_data(
    original_series_data: np.ndarray,
    filtered_series_data: np.ndarray,
    m_idx: int,
    c_idx: int,
    n_dimension_label: str,
    value_label: str,
    window_length: int,
    polyorder: int,
    save_path: str,
    filename_prefix: str,
):
    """Helper function to plot and save Savitzky-Golay debug data."""
    try:
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.plot(
            original_series_data,
            label=f"Original Data (M:{m_idx}, C:{c_idx})",
            marker=".",
        )
        plt.plot(
            filtered_series_data,
            label=f"Filtered Data (Win:{window_length}, Poly:{polyorder})",
            marker=".",
        )

        plt.title(f"Savitzky-Golay Filter: M_idx={m_idx}, C_idx={c_idx}")
        plt.xlabel(n_dimension_label)
        plt.ylabel(value_label)
        plt.legend()
        plt.grid(True)

        plot_filename = f"{filename_prefix}_m{m_idx}_c{c_idx}_win{window_length}_poly{polyorder}.png"
        full_save_path = os.path.join(save_path, plot_filename)

        plt.savefig(full_save_path)
        logger.debug(f"Saved Savitzky-Golay debug plot to: {full_save_path}")
    except Exception as e:
        logger.error(
            f"Failed to generate or save Savitzky-Golay debug plot: {e}", exc_info=True
        )
    finally:
        plt.close()


# logger definition moved up


def apply_savgol_filter(
    points: np.ndarray,
    window_length: int,
    polyorder: int,
    axis: int = 0,
    debug_plot: bool = False,
    debug_plot_indices: Optional[Tuple[int, int]] = (0, 0),
    debug_save_path: str = "debug_plots/savgol",
    debug_plot_filename_prefix: str = "savgol_plot",
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to a 3D array of points (N, M, C).
    N: Number of frames/samples
    M: Number of points per frame/sample
    C: Number of coordinates per point (e.g., 2 for x, y)

    The filter is applied along the specified axis (typically axis 0 for time-series data).

    Args:
        points (np.ndarray): Input array of shape (N, M, C).
        window_length (int): The length of the filter window (must be a positive odd integer).
        polyorder (int): The order of the polynomial used to fit the samples.
                         Must be less than window_length.
        axis (int): The axis of the array along which the filter is to be applied.
        debug_plot (bool): If True, generates and saves a plot of data before and after filtering.
        debug_plot_indices (Optional[Tuple[int, int]]): A tuple (m_idx, c_idx) specifying
                                                       the M and C dimension indices to plot.
                                                       The plot will show values across the N dimension.
                                                       Defaults to (0, 0). If None, plotting is skipped.
        debug_save_path (str): Directory path to save the debug plots.
        debug_plot_filename_prefix (str): Prefix for the saved plot filenames.

    Returns:
        np.ndarray: The smoothed array with the same shape as input.
                    Returns the original points if an error occurs or if window_length
                    is too large for the number of samples along the specified axis.
    """
    if not isinstance(points, np.ndarray):
        logger.error("Input 'points' must be a NumPy array. Got %s", type(points))
        raise TypeError("Input 'points' must be a NumPy array.")

    if points.ndim != 3:
        logger.error(
            "Input 'points' must be a 3D array (N, M, C). Got %s dimensions.",
            points.ndim,
        )
        raise ValueError(f"Input 'points' must be a 3D array. Got shape {points.shape}")

    logger.debug(
        "Applying Savitzky-Golay filter with window: %s, polyorder: %s to points of shape: %s along axis: %s",
        window_length,
        polyorder,
        points.shape,
        axis,
    )

    if not isinstance(window_length, int) or window_length <= 0:
        logger.error(
            "Savitzky-Golay window_length must be a positive integer. Got: %s",
            window_length,
        )
        raise ValueError("Savitzky-Golay window_length must be a positive integer.")

    if not isinstance(polyorder, int) or polyorder < 0:
        logger.error(
            "Savitzky-Golay polyorder must be a non-negative integer. Got: %s",
            polyorder,
        )
        raise ValueError("Savitzky-Golay polyorder must be a non-negative integer.")

    if window_length % 2 == 0:
        logger.error(
            "Savitzky-Golay window_length must be an odd integer. Got: %s",
            window_length,
        )
        raise ValueError("Savitzky-Golay window_length must be an odd integer.")

    if polyorder >= window_length:
        logger.error(
            "Savitzky-Golay polyorder must be less than window_length. Got polyorder: %s, window_length: %s",
            polyorder,
            window_length,
        )
        raise ValueError("Savitzky-Golay polyorder must be less than window_length.")

    if points.shape[axis] < window_length:
        logger.warning(
            "Number of samples (%s) along axis %s is less than Savitzky-Golay window_length (%s). "
            "Returning original points.",
            points.shape[axis],
            axis,
            window_length,
        )
        output_points = points.copy()
        if debug_plot and debug_plot_indices is not None:
            # Plot original vs original as filter is skipped
            m_idx, c_idx = debug_plot_indices
            if 0 <= m_idx < points.shape[1] and 0 <= c_idx < points.shape[2]:
                # Assuming axis 0 is the primary N-dim for plotting context,
                # even if filter_axis is different.
                # If filter_axis is not 0, this plot might be less intuitive
                # but still shows the data along the first dimension.
                # For this specific case (filter skipped), original_series and filtered_series are the same.
                if axis == 0:
                    original_series = points[:, m_idx, c_idx]
                    filtered_series_to_plot = output_points[:, m_idx, c_idx]
                elif axis == 1:  # N, M, C -> M is filter axis, plot along M
                    original_series = points[
                        m_idx, :, c_idx
                    ]  # A bit of a misuse of m_idx here for N index
                    filtered_series_to_plot = output_points[m_idx, :, c_idx]
                elif axis == 2:  # N, M, C -> C is filter axis, plot along C
                    original_series = points[
                        m_idx, c_idx, :
                    ]  # Misuse of m_idx, c_idx for N, M indices
                    filtered_series_to_plot = output_points[m_idx, c_idx, :]
                else:  # Fallback or error, plot along N
                    original_series = points[:, m_idx, c_idx]
                    filtered_series_to_plot = output_points[:, m_idx, c_idx]

                _plot_savgol_debug_data(
                    original_series_data=original_series,
                    filtered_series_data=filtered_series_to_plot,  # Plotting original against itself
                    m_idx=m_idx,
                    c_idx=c_idx,
                    n_dimension_label=f"Sample Index (Filter Axis {axis} - Skipped)",
                    value_label="Value",
                    window_length=window_length,
                    polyorder=polyorder,
                    save_path=debug_save_path,
                    filename_prefix=f"{debug_plot_filename_prefix}_skipped",
                )
        return output_points

    try:
        # Reshape points to (N, M*C) if axis is 0, or transpose and reshape accordingly
        # This is to apply filter to each coordinate series independently.
        # For (N, M, C) and axis=0, we want to filter N samples for each of M*C series.

        num_samples = points.shape[axis]

        # Create an output array
        smoothed_points = np.empty_like(points)

        if axis == 0:  # Filter along N (frames)
            # Iterate over M points and C coordinates
            for i in range(points.shape[1]):  # M
                for j in range(points.shape[2]):  # C
                    smoothed_points[:, i, j] = scipy.signal.savgol_filter(
                        points[:, i, j],
                        window_length=window_length,
                        polyorder=polyorder,
                        axis=0,  # Filter along the first axis of this 1D slice
                        mode="mirror",
                    )
        elif (
            axis == 1
        ):  # Filter along M (points per frame) - less common for this use case
            for i in range(points.shape[0]):  # N
                for j in range(points.shape[2]):  # C
                    smoothed_points[i, :, j] = scipy.signal.savgol_filter(
                        points[i, :, j],
                        window_length=window_length,
                        polyorder=polyorder,
                        axis=0,  # Filter along the first axis of this 1D slice
                        mode="mirror",
                    )
        elif axis == 2:  # Filter along C (coordinates) - unlikely for this use case
            for i in range(points.shape[0]):  # N
                for j in range(points.shape[1]):  # M
                    smoothed_points[i, j, :] = scipy.signal.savgol_filter(
                        points[i, j, :],
                        window_length=window_length,
                        polyorder=polyorder,
                        axis=0,  # Filter along the first axis of this 1D slice
                        mode="mirror",
                    )
        else:
            logger.error(
                "Unsupported axis for filtering: %s. Must be 0, 1, or 2.", axis
            )
            raise ValueError(
                f"Unsupported axis for filtering: {axis}. Must be 0, 1, or 2."
            )

        logger.debug(
            "Savitzky-Golay filter applied. Output shape: %s", smoothed_points.shape
        )

        if debug_plot and debug_plot_indices is not None:
            m_idx, c_idx = debug_plot_indices
            if 0 <= m_idx < points.shape[1] and 0 <= c_idx < points.shape[2]:
                # Determine the correct slice for plotting based on the filter axis
                if axis == 0:  # Filtered along N, plot along N
                    original_series_for_plot = points[:, m_idx, c_idx]
                    filtered_series_for_plot = smoothed_points[:, m_idx, c_idx]
                    n_label = "Sample Index (N-dim)"
                elif axis == 1:  # Filtered along M, plot along M
                    # For plotting, we'll pick the first frame (N=0) if M is the filter axis
                    # User provides m_idx (for M) and c_idx (for C)
                    # This means debug_plot_indices.m_idx is actually the N-index for this plot
                    # and debug_plot_indices.c_idx is the C-index for this plot
                    # The plot will be points[debug_plot_indices.m_idx, :, debug_plot_indices.c_idx]
                    plot_n_index = debug_plot_indices[
                        0
                    ]  # Use m_idx from tuple as N index
                    plot_c_index = debug_plot_indices[
                        1
                    ]  # Use c_idx from tuple as C index
                    if (
                        0 <= plot_n_index < points.shape[0]
                        and 0 <= plot_c_index < points.shape[2]
                    ):
                        original_series_for_plot = points[plot_n_index, :, plot_c_index]
                        filtered_series_for_plot = smoothed_points[
                            plot_n_index, :, plot_c_index
                        ]
                        n_label = (
                            f"Point Index (M-dim, N={plot_n_index}, C={plot_c_index})"
                        )
                    else:  # Fallback if indices are out of bounds for this interpretation
                        original_series_for_plot = points[:, 0, 0]  # Default plot
                        filtered_series_for_plot = smoothed_points[:, 0, 0]
                        n_label = "Sample Index (N-dim, Fallback)"
                elif axis == 2:  # Filtered along C, plot along C
                    # User provides m_idx (for M) and c_idx (for C)
                    # This means debug_plot_indices.m_idx is N-index
                    # and debug_plot_indices.c_idx is M-index
                    plot_n_index = debug_plot_indices[0]
                    plot_m_index = debug_plot_indices[1]
                    if (
                        0 <= plot_n_index < points.shape[0]
                        and 0 <= plot_m_index < points.shape[1]
                    ):
                        original_series_for_plot = points[plot_n_index, plot_m_index, :]
                        filtered_series_for_plot = smoothed_points[
                            plot_n_index, plot_m_index, :
                        ]
                        n_label = f"Coordinate Index (C-dim, N={plot_n_index}, M={plot_m_index})"
                    else:  # Fallback
                        original_series_for_plot = points[:, 0, 0]
                        filtered_series_for_plot = smoothed_points[:, 0, 0]
                        n_label = "Sample Index (N-dim, Fallback)"
                else:  # Should not happen given prior validation, but as a fallback
                    original_series_for_plot = points[:, m_idx, c_idx]
                    filtered_series_for_plot = smoothed_points[:, m_idx, c_idx]
                    n_label = "Sample Index (N-dim, Fallback)"

                _plot_savgol_debug_data(
                    original_series_data=original_series_for_plot,
                    filtered_series_data=filtered_series_for_plot,
                    m_idx=m_idx,
                    c_idx=c_idx,  # These are the user-provided indices
                    n_dimension_label=n_label,
                    value_label="Value",
                    window_length=window_length,
                    polyorder=polyorder,
                    save_path=debug_save_path,
                    filename_prefix=debug_plot_filename_prefix,
                )
        return smoothed_points

    except Exception as e:
        logger.error("Error during Savitzky-Golay filtering: %s", e, exc_info=True)
        output_points_on_error = points.copy()
        if debug_plot and debug_plot_indices is not None:
            # Plot original vs original as filter failed
            m_idx, c_idx = debug_plot_indices
            if 0 <= m_idx < points.shape[1] and 0 <= c_idx < points.shape[2]:
                # Consistent with the 'skipped' case, attempt to plot along N-dim by default
                # or adapt based on 'axis' if feasible.
                if axis == 0:
                    original_series = points[:, m_idx, c_idx]
                elif axis == 1:
                    original_series = points[
                        m_idx, :, c_idx
                    ]  # m_idx interpreted as N index
                elif axis == 2:
                    original_series = points[m_idx, c_idx, :]  # m_idx as N, c_idx as M
                else:
                    original_series = points[:, m_idx, c_idx]

                _plot_savgol_debug_data(
                    original_series_data=original_series,
                    filtered_series_data=original_series,  # Plotting original against itself
                    m_idx=m_idx,
                    c_idx=c_idx,
                    n_dimension_label=f"Sample Index (Filter Axis {axis} - Error Occurred)",
                    value_label="Value",
                    window_length=window_length,
                    polyorder=polyorder,
                    save_path=debug_save_path,
                    filename_prefix=f"{debug_plot_filename_prefix}_error",
                )
        return output_points_on_error


if __name__ == "__main__":
    # Basic test and example usage
    logging.basicConfig(level=logging.DEBUG)

    # Example: 5 frames, 3 points per frame, 2 coordinates (x, y)
    test_points = np.array(
        [
            [[1, 10], [2, 20], [3, 30]],  # Frame 1
            [[1.5, 11], [2.5, 21], [3.5, 31]],  # Frame 2
            [[0.5, 9], [1.5, 19], [2.5, 29]],  # Frame 3
            [[1.2, 10.5], [2.2, 20.5], [3.2, 30.5]],  # Frame 4
            [[1.8, 11.5], [2.8, 21.5], [3.8, 31.5]],  # Frame 5
        ],
        dtype=float,
    )

    # Add some noise
    noise = np.random.rand(*test_points.shape) * 0.5
    noisy_points = test_points + noise

    logger.info("Original noisy points:\n%s", noisy_points)

    # Apply filter
    # Test case 1: Valid parameters
    try:
        smoothed = apply_savgol_filter(
            noisy_points,
            window_length=3,
            polyorder=1,
            axis=0,
            debug_plot=True,
            debug_plot_indices=(
                0,
                0,
            ),  # Plot for the first point (M=0), first coordinate (C=0)
            debug_save_path="debug_output/filters_test",  # Custom path for test plots
            debug_plot_filename_prefix="tc1_savgol",
        )
        logger.info("Smoothed points (win=3, poly=1, debug_plot=True):\n%s", smoothed)
        logger.info(
            "Check 'debug_output/filters_test' directory for tc1_savgol_m0_c0_win3_poly1.png"
        )
    except Exception as e:
        logger.error("Test case 1 failed: %s", e, exc_info=True)

    # Test case 2: Window too large (also with debug plot for skipped case)
    try:
        smoothed_win_large = apply_savgol_filter(
            noisy_points,
            window_length=7,  # Window is larger than number of frames (5)
            polyorder=2,
            axis=0,
            debug_plot=True,
            debug_plot_indices=(0, 1),  # Plot for M=0, C=1
            debug_save_path="debug_output/filters_test",
            debug_plot_filename_prefix="tc2_savgol_skipped",
        )
        logger.info(
            "Smoothed points (win=7, poly=2, debug_plot=True) - should be original due to window size:\n%s",
            smoothed_win_large,
        )
        logger.info(
            "Check 'debug_output/filters_test' directory for tc2_savgol_skipped_m0_c1_win7_poly2.png (should show original data twice)"
        )
        assert np.array_equal(smoothed_win_large, noisy_points), (
            "Should return original if window too large"
        )
    except Exception as e:
        logger.error("Test case 2 failed: %s", e, exc_info=True)

    # Test case 3: Invalid polyorder
    try:
        apply_savgol_filter(noisy_points, window_length=3, polyorder=3, axis=0)
    except ValueError as e:
        logger.info("Test case 3 (invalid polyorder) passed with ValueError: %s", e)
    except Exception as e:
        logger.error("Test case 3 failed unexpectedly: %s", e)

    # Test case 4: Even window length
    try:
        apply_savgol_filter(noisy_points, window_length=4, polyorder=1, axis=0)
    except ValueError as e:
        logger.info("Test case 4 (even window) passed with ValueError: %s", e)
    except Exception as e:
        logger.error("Test case 4 failed unexpectedly: %s", e)

    # Test case 5: Not enough dimensions
    try:
        apply_savgol_filter(
            np.array([1, 2, 3, 4, 5]), window_length=3, polyorder=1, axis=0
        )
    except ValueError as e:
        logger.info("Test case 5 (wrong dimensions) passed with ValueError: %s", e)
    except Exception as e:
        logger.error("Test case 5 failed unexpectedly: %s", e)

    # Test case 6: Non-NumPy input
    try:
        apply_savgol_filter([[1, 2], [3, 4]], window_length=3, polyorder=1, axis=0)
    except TypeError as e:
        logger.info("Test case 6 (non-numpy input) passed with TypeError: %s", e)
    except Exception as e:
        logger.error("Test case 6 failed unexpectedly: %s", e)