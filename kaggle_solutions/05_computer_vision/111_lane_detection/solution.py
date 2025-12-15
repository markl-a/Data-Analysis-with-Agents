"""
Autonomous Driving Lane Detection

Detect road lane lines using image processing and machine learning
techniques for autonomous driving and ADAS applications.

Dataset: https://www.kaggle.com/datasets/manideep1108/tusimple
Difficulty: ⭐⭐⭐ Advanced Level
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LaneDetector:
    """Lane detection using classical computer vision techniques."""

    def __init__(self, img_height: int = 720, img_width: int = 1280):
        self.img_height = img_height
        self.img_width = img_width

        # ROI vertices (region of interest)
        self.roi_vertices = np.array([
            [(0, img_height),
             (img_width * 0.45, img_height * 0.6),
             (img_width * 0.55, img_height * 0.6),
             (img_width, img_height)]
        ], dtype=np.int32)

    def create_sample_image(self) -> np.ndarray:
        """Create a synthetic road image with lane lines."""
        # Create base road image
        img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # Sky (gradient blue)
        for y in range(self.img_height // 2):
            intensity = int(200 + 55 * y / (self.img_height // 2))
            img[y, :] = [intensity, intensity - 50, intensity - 100]

        # Road (gray asphalt)
        road_start = self.img_height // 2
        for y in range(road_start, self.img_height):
            img[y, :] = [80, 80, 80]

        # Draw lane lines (white dashed lines)
        # Left lane
        for y in range(road_start, self.img_height):
            progress = (y - road_start) / (self.img_height - road_start)
            left_x = int(self.img_width * 0.2 + progress * self.img_width * 0.25)

            # Dashed line pattern
            if (y // 30) % 2 == 0:
                for dx in range(-3, 4):
                    if 0 <= left_x + dx < self.img_width:
                        img[y, left_x + dx] = [255, 255, 255]

        # Right lane
        for y in range(road_start, self.img_height):
            progress = (y - road_start) / (self.img_height - road_start)
            right_x = int(self.img_width * 0.8 - progress * self.img_width * 0.25)

            # Dashed line pattern
            if (y // 30) % 2 == 0:
                for dx in range(-3, 4):
                    if 0 <= right_x + dx < self.img_width:
                        img[y, right_x + dx] = [255, 255, 255]

        # Add some noise
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int32) + noise - 10, 0, 255).astype(np.uint8)

        return img

    def rgb_to_gray(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB to grayscale."""
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    def gaussian_blur(self, img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur."""
        # Simple box blur approximation
        from scipy import ndimage
        return ndimage.uniform_filter(img.astype(float), size=kernel_size).astype(np.uint8)

    def sobel_edge(self, img: np.ndarray) -> np.ndarray:
        """Compute Sobel edge detection."""
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Convolve
        from scipy import ndimage
        grad_x = ndimage.convolve(img.astype(float), sobel_x)
        grad_y = ndimage.convolve(img.astype(float), sobel_y)

        # Magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return (magnitude / magnitude.max() * 255).astype(np.uint8)

    def canny_edge(self, img: np.ndarray, low_threshold: int = 50,
                   high_threshold: int = 150) -> np.ndarray:
        """Simplified Canny edge detection."""
        # Blur
        blurred = self.gaussian_blur(img, 5)

        # Sobel edges
        edges = self.sobel_edge(blurred)

        # Double threshold
        strong = edges > high_threshold
        weak = (edges >= low_threshold) & (edges <= high_threshold)

        # Simple hysteresis
        result = np.zeros_like(edges)
        result[strong] = 255

        return result

    def apply_roi_mask(self, img: np.ndarray) -> np.ndarray:
        """Apply region of interest mask."""
        mask = np.zeros_like(img)

        # Fill ROI polygon
        vertices = self.roi_vertices[0]
        from matplotlib.path import Path
        y_coords, x_coords = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
        path = Path(vertices)
        mask_flat = path.contains_points(points)
        mask = mask_flat.reshape(img.shape).astype(np.uint8) * 255

        return img & mask

    def hough_lines(self, edge_img: np.ndarray, threshold: int = 50,
                    min_line_length: int = 50, max_line_gap: int = 150) -> List[Tuple]:
        """Detect lines using Hough transform."""
        lines = []
        height, width = edge_img.shape

        # Hough space
        diag_len = int(np.sqrt(height**2 + width**2))
        thetas = np.deg2rad(np.arange(-90, 90))
        rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)

        # Accumulator
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

        # Edge points
        y_idxs, x_idxs = np.nonzero(edge_img)

        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            for t_idx, theta in enumerate(thetas):
                rho = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len
                if 0 <= rho < len(rhos):
                    accumulator[rho, t_idx] += 1

        # Find peaks
        for rho_idx, theta_idx in zip(*np.where(accumulator > threshold)):
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Line endpoints
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            lines.append((x1, y1, x2, y2))

        return lines[:20]  # Limit number of lines

    def separate_lines(self, lines: List[Tuple]) -> Tuple[List, List]:
        """Separate left and right lane lines."""
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line
            if x2 - x1 == 0:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Filter by slope
            if abs(slope) < 0.5:
                continue

            if slope < 0:  # Left lane (negative slope in image coords)
                left_lines.append(line)
            else:  # Right lane
                right_lines.append(line)

        return left_lines, right_lines

    def average_lines(self, lines: List[Tuple]) -> Optional[Tuple]:
        """Average multiple lines into one."""
        if not lines:
            return None

        x1s, y1s, x2s, y2s = [], [], [], []
        for x1, y1, x2, y2 in lines:
            x1s.append(x1)
            y1s.append(y1)
            x2s.append(x2)
            y2s.append(y2)

        return (int(np.mean(x1s)), int(np.mean(y1s)),
                int(np.mean(x2s)), int(np.mean(y2s)))

    def draw_lanes(self, img: np.ndarray, left_line: Optional[Tuple],
                   right_line: Optional[Tuple]) -> np.ndarray:
        """Draw detected lanes on image."""
        result = img.copy()

        # Draw lane lines
        if left_line:
            x1, y1, x2, y2 = left_line
            self._draw_thick_line(result, x1, y1, x2, y2, color=[0, 255, 0], thickness=5)

        if right_line:
            x1, y1, x2, y2 = right_line
            self._draw_thick_line(result, x1, y1, x2, y2, color=[0, 255, 0], thickness=5)

        # Draw lane area
        if left_line and right_line:
            pts = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ])

            # Fill polygon with semi-transparent green
            overlay = result.copy()
            from matplotlib.path import Path
            y_coords, x_coords = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
            path = Path(pts)
            mask = path.contains_points(points).reshape(img.shape[:2])
            overlay[mask] = [0, 255, 0]
            result = ((result * 0.7) + (overlay * 0.3)).astype(np.uint8)

        return result

    def _draw_thick_line(self, img: np.ndarray, x1: int, y1: int,
                         x2: int, y2: int, color: List[int], thickness: int = 3) -> None:
        """Draw a thick line on the image."""
        # Bresenham's line algorithm with thickness
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        if dx > dy:
            err = dx / 2
            while x != x2:
                for t in range(-thickness, thickness + 1):
                    if 0 <= y + t < img.shape[0] and 0 <= x < img.shape[1]:
                        img[y + t, x] = color
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2
            while y != y2:
                for t in range(-thickness, thickness + 1):
                    if 0 <= y < img.shape[0] and 0 <= x + t < img.shape[1]:
                        img[y, x + t] = color
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

    def detect_lanes(self, img: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Full lane detection pipeline."""
        # Convert to grayscale
        gray = self.rgb_to_gray(img)

        # Edge detection
        edges = self.canny_edge(gray)

        # Apply ROI mask
        masked_edges = self.apply_roi_mask(edges)

        # Hough lines
        lines = self.hough_lines(masked_edges)

        # Separate and average lines
        left_lines, right_lines = self.separate_lines(lines)
        left_lane = self.average_lines(left_lines)
        right_lane = self.average_lines(right_lines)

        # Draw result
        result = self.draw_lanes(img, left_lane, right_lane)

        info = {
            'num_lines_detected': len(lines),
            'left_lines': len(left_lines),
            'right_lines': len(right_lines),
            'left_lane': left_lane,
            'right_lane': right_lane
        }

        return result, info


class LaneDetectionEvaluator:
    """Evaluate lane detection performance."""

    def __init__(self):
        self.detector = LaneDetector()
        self.results: List[Dict] = []

    def generate_test_set(self, n_samples: int = 10) -> List[np.ndarray]:
        """Generate synthetic test images."""
        images = []
        for _ in range(n_samples):
            img = self.detector.create_sample_image()
            # Add random variations
            brightness = np.random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            images.append(img)
        return images

    def evaluate(self, images: List[np.ndarray]) -> pd.DataFrame:
        """Evaluate detection on test images."""
        import pandas as pd

        results = []
        for i, img in enumerate(images):
            result_img, info = self.detector.detect_lanes(img)

            # Calculate metrics
            left_detected = info['left_lane'] is not None
            right_detected = info['right_lane'] is not None
            both_detected = left_detected and right_detected

            results.append({
                'Image': i + 1,
                'Lines Detected': info['num_lines_detected'],
                'Left Lane': '✓' if left_detected else '✗',
                'Right Lane': '✓' if right_detected else '✗',
                'Both Lanes': '✓' if both_detected else '✗'
            })

            self.results.append({
                'image': img,
                'result': result_img,
                'info': info
            })

        return pd.DataFrame(results)

    def plot_results(self, output_dir: str = '.') -> None:
        """Visualize detection results."""
        n_samples = min(4, len(self.results))

        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
        fig.suptitle('Lane Detection Results', fontsize=16)

        for i in range(n_samples):
            result = self.results[i]

            # Original
            axes[i, 0].imshow(result['image'])
            axes[i, 0].set_title(f'Original Image {i+1}')
            axes[i, 0].axis('off')

            # Edges
            gray = self.detector.rgb_to_gray(result['image'])
            edges = self.detector.canny_edge(gray)
            axes[i, 1].imshow(edges, cmap='gray')
            axes[i, 1].set_title('Edge Detection')
            axes[i, 1].axis('off')

            # Detection result
            axes[i, 2].imshow(result['result'])
            info = result['info']
            axes[i, 2].set_title(f'Lane Detection (L:{info["left_lines"]}, R:{info["right_lines"]})')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/lane_detection_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/lane_detection_results.png")
        plt.close()

    def plot_pipeline(self, output_dir: str = '.') -> None:
        """Visualize the full pipeline."""
        if not self.results:
            return

        img = self.results[0]['image']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Lane Detection Pipeline', fontsize=16)

        # Step 1: Original
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')

        # Step 2: Grayscale
        gray = self.detector.rgb_to_gray(img)
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('2. Grayscale')
        axes[0, 1].axis('off')

        # Step 3: Gaussian Blur
        blurred = self.detector.gaussian_blur(gray)
        axes[0, 2].imshow(blurred, cmap='gray')
        axes[0, 2].set_title('3. Gaussian Blur')
        axes[0, 2].axis('off')

        # Step 4: Edge Detection
        edges = self.detector.canny_edge(gray)
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('4. Edge Detection')
        axes[1, 0].axis('off')

        # Step 5: ROI Mask
        masked = self.detector.apply_roi_mask(edges)
        axes[1, 1].imshow(masked, cmap='gray')
        axes[1, 1].set_title('5. Region of Interest')
        axes[1, 1].axis('off')

        # Step 6: Final Result
        axes[1, 2].imshow(self.results[0]['result'])
        axes[1, 2].set_title('6. Lane Detection')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/lane_detection_pipeline.png', dpi=300, bbox_inches='tight')
        print(f"Pipeline saved to {output_dir}/lane_detection_pipeline.png")
        plt.close()


# Need pandas for DataFrame
import pandas as pd


def main():
    """Main execution."""
    print("=" * 70)
    print("AUTONOMOUS DRIVING LANE DETECTION")
    print("=" * 70)

    evaluator = LaneDetectionEvaluator()

    # Generate test images
    print("\nGenerating test images...")
    images = evaluator.generate_test_set(n_samples=8)
    print(f"Generated {len(images)} test images")

    # Evaluate
    print("\nRunning lane detection...")
    results = evaluator.evaluate(images)

    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)
    print(results.to_string(index=False))

    # Calculate summary
    total = len(results)
    both_detected = (results['Both Lanes'] == '✓').sum()
    success_rate = both_detected / total * 100

    print(f"\nSuccess Rate (Both Lanes): {success_rate:.1f}%")

    # Visualize
    evaluator.plot_results()
    evaluator.plot_pipeline()

    print("\n" + "=" * 70)
    print("LANE DETECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
