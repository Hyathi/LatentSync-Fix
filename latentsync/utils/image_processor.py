# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union, Optional
from .affine_transform import AlignRestore
from .face_detector import FaceDetector
from rich import print
from .filters import apply_savgol_filter
from scipy import signal
from scipy.ndimage import gaussian_filter1d


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device, dtype=torch.float32)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        if device == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=device)

    def _get_landmarks(self, image: torch.Tensor) -> np.ndarray:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            raise RuntimeError("Face not detected")

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])
        
        return landmarks3

    def affine_transform(self, image: torch.Tensor) -> np.ndarray:
        landmarks3 = self._get_landmarks(image)

        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix

    def affine_transform_from_landmarks(self, image: torch.Tensor, landmarks3: np.ndarray) -> np.ndarray:
        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values

    def affine_transform_video_with_metadata(self, video_frames: np.ndarray,
                                           enhanced_smoothing: bool = True,
                                           landmark_smoothing_params: Optional[dict] = None,
                                           matrix_smoothing_params: Optional[dict] = None):
        """
        Apply affine transformation to video frames with enhanced smoothing while returning metadata.

        This method applies enhanced smoothing to landmarks and affine matrices to reduce
        high-frequency vibrations in the output video while maintaining the required
        return format for the lipsync pipeline.

        Args:
            video_frames: Input video frames as numpy array of shape (N, H, W, C)
            enhanced_smoothing: Whether to apply enhanced smoothing (default: True)
            landmark_smoothing_params: Optional parameters for landmark smoothing
            matrix_smoothing_params: Optional parameters for matrix smoothing

        Returns:
            tuple: (faces, boxes, affine_matrices)
                - faces: Torch tensor of aligned face crops (N, C, H, W)
                - boxes: List of bounding box coordinates for each frame
                - affine_matrices: List of affine transformation matrices
        """
        # Create a VideoProcessor instance to handle the enhanced processing
        video_processor = VideoProcessor(resolution=self.resolution, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Delegate to the VideoProcessor's implementation
        return video_processor.affine_transform_video_with_metadata(
            video_frames, enhanced_smoothing, landmark_smoothing_params, matrix_smoothing_params
        )


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)
        self.previous_affine_matrices = None

    def _collect_landmarks(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            landmarks = self.image_processor._get_landmarks(frame)
            results.append(landmarks)
        return video_frames, results

    def _smooth_landmarks_multi_stage(self, landmarks: np.ndarray,
                                    primary_window: int = 15,
                                    primary_poly: int = 2,
                                    secondary_window: int = 7,
                                    secondary_poly: int = 1,
                                    gaussian_sigma: float = 1.0) -> np.ndarray:
        """
        Apply multi-stage smoothing to landmarks for better vibration reduction.

        Args:
            landmarks: Array of shape (N, M, C) where N=frames, M=landmarks, C=coordinates
            primary_window: Window length for primary Savitzky-Golay filter
            primary_poly: Polynomial order for primary filter
            secondary_window: Window length for secondary filter
            secondary_poly: Polynomial order for secondary filter
            gaussian_sigma: Standard deviation for Gaussian smoothing

        Returns:
            Smoothed landmarks array
        """
        # Stage 1: Primary Savitzky-Golay filter
        smoothed = apply_savgol_filter(landmarks, primary_window, primary_poly)

        # Stage 2: Secondary Savitzky-Golay filter with smaller window
        if len(landmarks) >= secondary_window:
            smoothed = apply_savgol_filter(smoothed, secondary_window, secondary_poly)

        # Stage 3: Gaussian smoothing for high-frequency noise reduction
        for i in range(landmarks.shape[1]):  # For each landmark
            for j in range(landmarks.shape[2]):  # For each coordinate
                smoothed[:, i, j] = gaussian_filter1d(smoothed[:, i, j], sigma=gaussian_sigma)

        return smoothed

    def _smooth_affine_matrices(self, affine_matrices: list,
                              window_length: int = 9,
                              poly_order: int = 2,
                              gaussian_sigma: float = 0.8) -> list:
        """
        Apply smoothing to affine transformation matrices to reduce high-frequency vibrations.

        Args:
            affine_matrices: List of 2x3 affine transformation matrices
            window_length: Window length for Savitzky-Golay filter
            poly_order: Polynomial order for filter
            gaussian_sigma: Standard deviation for Gaussian smoothing

        Returns:
            List of smoothed affine matrices
        """
        if len(affine_matrices) < window_length:
            return affine_matrices

        # Convert to numpy array for processing, ensuring consistent shape
        matrices_list = []
        for m in affine_matrices:
            if hasattr(m, 'cpu'):
                matrix = m.cpu().numpy()
            else:
                matrix = m

            # Ensure matrix is 2D [2, 3] for processing
            if matrix.ndim == 3 and matrix.shape[0] == 1:  # [1, 2, 3] -> [2, 3]
                matrix = matrix.squeeze(0)
            matrices_list.append(matrix)

        matrices_array = np.stack(matrices_list)

        # Smooth each element of the transformation matrix
        smoothed_matrices = np.zeros_like(matrices_array)

        for i in range(matrices_array.shape[1]):  # 2 rows
            for j in range(matrices_array.shape[2]):  # 3 columns
                # Apply Savitzky-Golay filter
                if len(matrices_array) >= window_length:
                    smoothed_matrices[:, i, j] = signal.savgol_filter(
                        matrices_array[:, i, j],
                        window_length,
                        poly_order,
                        mode='mirror'
                    )
                else:
                    smoothed_matrices[:, i, j] = matrices_array[:, i, j]

                # Apply Gaussian smoothing
                smoothed_matrices[:, i, j] = gaussian_filter1d(
                    smoothed_matrices[:, i, j],
                    sigma=gaussian_sigma
                )

        # Return matrices with consistent [2, 3] shape
        return [torch.from_numpy(matrix) for matrix in smoothed_matrices]

    def _apply_temporal_consistency_filter(self, frames: list, alpha: float = 0.7) -> list:
        """
        Apply temporal consistency filtering to reduce frame-to-frame variations.

        Args:
            frames: List of processed frame tensors
            alpha: Blending factor for temporal smoothing (0.0 = no smoothing, 1.0 = maximum smoothing)

        Returns:
            Temporally smoothed frames
        """
        if len(frames) <= 1 or alpha <= 0:
            return frames

        smoothed_frames = [frames[0]]  # First frame remains unchanged

        for i in range(1, len(frames)):
            # Blend current frame with previous smoothed frame
            current_frame = frames[i].float()
            prev_frame = smoothed_frames[i-1].float()

            # Apply temporal blending
            blended_frame = alpha * prev_frame + (1 - alpha) * current_frame
            smoothed_frames.append(blended_frame.to(frames[i].dtype))

        return smoothed_frames

    def affine_transform_video_with_metadata(self, video_frames: np.ndarray,
                                           enhanced_smoothing: bool = True,
                                           landmark_smoothing_params: Optional[dict] = None,
                                           matrix_smoothing_params: Optional[dict] = None):
        """
        Apply affine transformation to video frames with enhanced smoothing, returning faces, boxes, and matrices.

        This method is designed for use in pipelines that need access to the transformation metadata
        (like lipsync_pipeline.py) while still benefiting from enhanced smoothing.

        Args:
            video_frames: Input video frames as numpy array
            enhanced_smoothing: Whether to use enhanced multi-stage smoothing
            landmark_smoothing_params: Parameters for landmark smoothing
            matrix_smoothing_params: Parameters for affine matrix smoothing

        Returns:
            tuple: (faces, boxes, affine_matrices)
                - faces: torch.Tensor of transformed faces
                - boxes: list of bounding boxes
                - affine_matrices: list of affine transformation matrices
        """
        # Collect landmarks from frames
        landmarks = []
        for frame in video_frames:
            try:
                landmarks3 = self.image_processor._get_landmarks(frame)
                landmarks.append(landmarks3)
            except RuntimeError:
                # Fallback: use previous landmarks or default
                if landmarks:
                    landmarks3 = landmarks[-1]
                else:
                    # Create default landmarks (center of frame)
                    h, w = frame.shape[:2]
                    landmarks3 = np.array([[w//2, h//2], [w//2, h//2], [w//2, h//2]], dtype=np.float32)
                landmarks.append(landmarks3)

        landmarks = np.stack(landmarks)

        # Set default parameters
        if landmark_smoothing_params is None:
            landmark_smoothing_params = {
                'primary_window': 15,
                'primary_poly': 2,
                'secondary_window': 7,
                'secondary_poly': 1,
                'gaussian_sigma': 1.0
            }

        if matrix_smoothing_params is None:
            matrix_smoothing_params = {
                'window_length': 9,
                'poly_order': 2,
                'gaussian_sigma': 0.8
            }

        # Apply enhanced landmark smoothing
        if enhanced_smoothing:
            landmarks = self._smooth_landmarks_multi_stage(landmarks, **landmark_smoothing_params)
        else:
            landmarks = apply_savgol_filter(landmarks, 15, 2)

        # Collect faces, boxes, and affine matrices
        faces = []
        boxes = []
        affine_matrices = []

        for frame, landmarks3 in zip(video_frames, landmarks):
            # Use the smoothed landmarks for transformation
            face, box, affine_matrix = self.image_processor.affine_transform_from_landmarks(frame, landmarks3)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        # Convert faces to tensor and return
        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    def affine_transform_video_smooth(self, video_path,
                                    enhanced_smoothing: bool = True,
                                    landmark_smoothing_params: Optional[dict] = None,
                                    matrix_smoothing_params: Optional[dict] = None):
        """
        Apply affine transformation to video with enhanced smoothing to reduce vibrations.

        Args:
            video_path: Path to input video
            enhanced_smoothing: Whether to use enhanced multi-stage smoothing
            landmark_smoothing_params: Parameters for landmark smoothing
            matrix_smoothing_params: Parameters for affine matrix smoothing

        Returns:
            Processed video frames as numpy array
        """
        video_frames, landmarks = self._collect_landmarks(video_path)
        landmarks = np.stack(landmarks)

        # Set default parameters
        if landmark_smoothing_params is None:
            landmark_smoothing_params = {
                'primary_window': 15,
                'primary_poly': 2,
                'secondary_window': 7,
                'secondary_poly': 1,
                'gaussian_sigma': 1.0
            }

        if matrix_smoothing_params is None:
            matrix_smoothing_params = {
                'window_length': 9,
                'poly_order': 2,
                'gaussian_sigma': 0.8
            }

        # Apply enhanced landmark smoothing
        if enhanced_smoothing:
            landmarks = self._smooth_landmarks_multi_stage(landmarks, **landmark_smoothing_params)
        else:
            landmarks = apply_savgol_filter(landmarks, 15, 2)

        # Collect affine matrices for additional smoothing
        results = []
        affine_matrices = []

        for frame, landmarks3 in zip(video_frames, landmarks):
            face, _, affine_matrix = self.image_processor.affine_transform_from_landmarks(frame, landmarks3)
            results.append(face)
            affine_matrices.append(affine_matrix)

        results = torch.stack(results)
        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results

    def affine_transform_video_ultra_smooth(self, video_path,
                                          landmark_params: Optional[dict] = None,
                                          matrix_params: Optional[dict] = None,
                                          temporal_alpha: float = 0.3) -> np.ndarray:
        """
        Ultra-smooth affine transformation with maximum vibration reduction.

        This method combines multiple smoothing techniques:
        1. Multi-stage landmark smoothing
        2. Affine matrix smoothing
        3. Temporal consistency filtering
        4. High-quality interpolation

        Args:
            video_path: Path to input video
            landmark_params: Custom parameters for landmark smoothing
            matrix_params: Custom parameters for matrix smoothing
            temporal_alpha: Temporal consistency blending factor

        Returns:
            Ultra-smooth processed video frames
        """
        # Set optimized default parameters for maximum smoothness
        if landmark_params is None:
            landmark_params = {
                'primary_window': 21,      # Larger window for stronger smoothing
                'primary_poly': 3,         # Higher polynomial order for better fitting
                'secondary_window': 11,    # Secondary smoothing
                'secondary_poly': 2,
                'gaussian_sigma': 1.5      # Stronger Gaussian smoothing
            }

        if matrix_params is None:
            matrix_params = {
                'window_length': 13,       # Larger window for matrix smoothing
                'poly_order': 3,           # Higher polynomial order
                'gaussian_sigma': 1.2      # Stronger Gaussian smoothing
            }

        # Apply enhanced smoothing
        video_frames, landmarks = self._collect_landmarks(video_path)
        landmarks = np.stack(landmarks)

        # Multi-stage landmark smoothing
        landmarks = self._smooth_landmarks_multi_stage(landmarks, **landmark_params)

        # Collect transformations
        results = []
        affine_matrices = []

        for frame, landmarks3 in zip(video_frames, landmarks):
            face, _, affine_matrix = self.image_processor.affine_transform_from_landmarks(frame, landmarks3)
            results.append(face)
            affine_matrices.append(affine_matrix)

        # Smooth affine matrices
        if len(affine_matrices) > 1:
            smoothed_matrices = self._smooth_affine_matrices(affine_matrices, **matrix_params)

            # Re-apply with smoothed matrices
            results = []
            for frame, landmarks3, smooth_matrix in zip(video_frames, landmarks, smoothed_matrices):
                face = self._apply_smoothed_transform(frame, landmarks3, smooth_matrix)
                results.append(face)

        # Apply temporal consistency filtering
        if temporal_alpha > 0:
            results = self._apply_temporal_consistency_filter(results, temporal_alpha)

        results = torch.stack(results)
        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results

    def _apply_smoothed_transform(self, image: torch.Tensor, landmarks3: np.ndarray, affine_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply affine transformation using a pre-computed smoothed affine matrix.

        Args:
            image: Input image tensor
            landmarks3: Facial landmarks (not used in this version, kept for compatibility)
            affine_matrix: Pre-computed smoothed affine transformation matrix

        Returns:
            Transformed face tensor
        """
        # Convert image to appropriate format
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = rearrange(image.to(device=self.image_processor.restorer.device,
                                 dtype=self.image_processor.restorer.dtype),
                         "h w c -> c h w").unsqueeze(0)

        # Ensure affine matrix is on the correct device and has the right shape
        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix)

        affine_matrix = affine_matrix.to(device=self.image_processor.restorer.device,
                                       dtype=self.image_processor.restorer.dtype)

        # Add batch dimension only if needed (should be [B, 2, 3])
        if affine_matrix.dim() == 2:  # Shape [2, 3] -> [1, 2, 3]
            affine_matrix = affine_matrix.unsqueeze(0)
        elif affine_matrix.dim() == 4:  # Shape [1, 1, 2, 3] -> [1, 2, 3]
            affine_matrix = affine_matrix.squeeze(1)

        # Apply the transformation using kornia
        import kornia
        cropped_face = kornia.geometry.transform.warp_affine(
            image,
            affine_matrix,
            (self.image_processor.restorer.face_size[1], self.image_processor.restorer.face_size[0]),
            mode="bicubic",  # Use bicubic instead of bilinear for better quality
            padding_mode="fill",
            fill_value=self.image_processor.restorer.fill_value,
        )

        # Resize and format output
        cropped_face = rearrange(cropped_face.squeeze(0), "c h w -> h w c").cpu().numpy().astype(np.uint8)
        face = cv2.resize(cropped_face,
                         (self.image_processor.resolution, self.image_processor.resolution),
                         interpolation=cv2.INTER_CUBIC)  # Use cubic interpolation for smoother results
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            frame, _, _ = self.image_processor.affine_transform(frame)
            results.append(frame)
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")

    # Original method
    print("Processing with original smoothing...")
    video_frames_original = video_processor.affine_transform_video_smooth(
        "assets/demo2_video.mp4", enhanced_smoothing=False
    )
    write_video("output_original.mp4", video_frames_original, fps=25)

    # Enhanced smoothing method
    print("Processing with enhanced smoothing...")
    video_frames_enhanced = video_processor.affine_transform_video_smooth(
        "assets/demo2_video.mp4", enhanced_smoothing=True
    )
    write_video("output_enhanced.mp4", video_frames_enhanced, fps=25)

    # Ultra-smooth method for maximum vibration reduction
    print("Processing with ultra-smooth method...")
    video_frames_ultra = video_processor.affine_transform_video_ultra_smooth(
        "assets/demo2_video.mp4"
    )
    write_video("output_ultra_smooth.mp4", video_frames_ultra, fps=25)

    print("Processing complete! Compare the three output videos to see the improvements.")
