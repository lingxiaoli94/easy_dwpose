from typing import Callable, Dict, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import torch
from huggingface_hub import hf_hub_download

from easy_dwpose.body_estimation import Wholebody, resize_image, inference_detector, inference_pose
from easy_dwpose.draw import draw_openpose


class DWposeDetector:
    def __init__(self, device: str = "сpu"):
        hf_hub_download("RedHash/DWPose", "yolox_l.onnx", local_dir="./checkpoints")
        hf_hub_download("RedHash/DWPose", "dw-ll_ucoco_384.onnx", local_dir="./checkpoints")
        self.pose_estimation = Wholebody(
            device=device, model_det="checkpoints/yolox_l.onnx", model_pose="checkpoints/dw-ll_ucoco_384.onnx"
        )

    def _format_pose(self, candidates, scores, width, height):
        num_candidates, _, locs = candidates.shape

        candidates[..., 0] /= float(width)
        candidates[..., 1] /= float(height)

        bodies = candidates[:, :18].copy()
        bodies = bodies.reshape(num_candidates * 18, locs)

        body_scores = scores[:, :18]
        for i in range(len(body_scores)):
            for j in range(len(body_scores[i])):
                if body_scores[i][j] > 0.3:
                    body_scores[i][j] = int(18 * i + j)
                else:
                    body_scores[i][j] = -1

        faces = candidates[:, 24:92]
        faces_scores = scores[:, 24:92]

        # Extract hand keypoints and scores
        left_hands = candidates[:, 92:113]
        right_hands = candidates[:, 113:]
        left_hands_scores = scores[:, 92:113]
        right_hands_scores = scores[:, 113:]
        
        # Create masks for valid hand detections (average score above threshold)
        hand_confidence_threshold = 0.4  # High threshold to filter out false detections
        valid_left_hands = np.mean(left_hands_scores, axis=1) > hand_confidence_threshold
        valid_right_hands = np.mean(right_hands_scores, axis=1) > hand_confidence_threshold
        
        # Filter hands based on confidence
        filtered_left_hands = left_hands[valid_left_hands] if np.any(valid_left_hands) else np.empty((0, 21, 2))
        filtered_right_hands = right_hands[valid_right_hands] if np.any(valid_right_hands) else np.empty((0, 21, 2))
        filtered_left_scores = left_hands_scores[valid_left_hands] if np.any(valid_left_hands) else np.empty((0, 21))
        filtered_right_scores = right_hands_scores[valid_right_hands] if np.any(valid_right_hands) else np.empty((0, 21))
        
        # Stack the filtered hands
        hands = np.vstack([filtered_left_hands, filtered_right_hands]) if (len(filtered_left_hands) > 0 or len(filtered_right_hands) > 0) else np.empty((0, 21, 2))
        hands_scores = np.vstack([filtered_left_scores, filtered_right_scores]) if (len(filtered_left_scores) > 0 or len(filtered_right_scores) > 0) else np.empty((0, 21))

        pose = dict(
            bodies=bodies,
            body_scores=body_scores,
            hands=hands,
            hands_scores=hands_scores,
            faces=faces,
            faces_scores=faces_scores,
        )

        return pose

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray],
        detect_resolution: int = 512,
        draw_pose: Optional[Callable] = draw_openpose,
        output_type: str = "pil",
        **kwargs,
    ) -> Union[PIL.Image.Image, np.ndarray, Dict]:
        if type(image) != np.ndarray:
            image = np.array(image.convert("RGB"))

        image = image.copy()
        original_height, original_width, _ = image.shape

        image = resize_image(image, target_resolution=detect_resolution)
        height, width, _ = image.shape

        # Detect human bounding boxes
        det_result = inference_detector(self.pose_estimation.session_det, image)
        
        # Check if any valid human detections were found
        if len(det_result) == 0:
            # No humans detected, return empty pose or blank image
            if not draw_pose:
                # Create empty pose structure with zero arrays
                empty_pose = {
                    "bodies": np.zeros((0, 18, 3)),
                    "body_scores": np.zeros((0, 18)),
                    "hands": np.zeros((0, 21, 2)),
                    "hands_scores": np.zeros((0, 21)),
                    "faces": np.zeros((0, 68, 2)),
                    "faces_scores": np.zeros((0, 68)),
                }
                return empty_pose
            else:
                # Return blank image
                blank_image = np.zeros((height, width, 3), dtype=np.uint8)
                blank_image = cv2.resize(blank_image, (original_width, original_height), cv2.INTER_LANCZOS4)
                if output_type == "pil":
                    blank_image = PIL.Image.fromarray(blank_image)
                return blank_image
        
        # Proceed with pose estimation if humans were detected
        # Use the detection results we already have instead of calling self.pose_estimation(image)
        # which would detect humans again
        keypoints, scores = inference_pose(self.pose_estimation.session_pose, det_result, image)
        
        # Process keypoints as done in Wholebody.__call__
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info
        
        candidates, scores = keypoints_info[..., :2], keypoints_info[..., 2]
        pose = self._format_pose(candidates, scores, width, height)

        if not draw_pose:
            return pose

        pose_image = draw_pose(pose, height=height, width=width, **kwargs)
        pose_image = cv2.resize(pose_image, (original_width, original_height), cv2.INTER_LANCZOS4)

        if output_type == "pil":
            pose_image = PIL.Image.fromarray(pose_image)
        elif output_type == "np":
            pass
        else:
            raise ValueError("output_type should be 'pil' or 'np'")

        return pose_image
