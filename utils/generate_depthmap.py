
    def _generate_depth_from_lidar(self, pcd_cam: np.ndarray, P: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Project LiDAR points to image plane and create a depth map. Returns [H, W] float32."""
        # Convert lidar points to homogeneous coordinates
        assert P.shape == (3, 4), f"Projection matrix P must be of shape (3, 4), got {P.shape}"

        if pcd_cam.shape[1] == 3:
            pcd_cam = np.hstack((pcd_cam, np.ones((pcd_cam.shape[0], 1))))  # (N,4)

        pcd_image = pcd_cam @ P.T
        
        # Filter out points behind camera (depth <= 0)
        valid_mask = pcd_image[:, 2] > 0
        pcd_image = pcd_image[valid_mask]
        
        # Normalize to get pixel coordinates
        pcd_image[:, 0] /= pcd_image[:, 2]
        pcd_image[:, 1] /= pcd_image[:, 2]

        # Filter points within image bounds
        img_h, img_w = img_shape[0], img_shape[1]
        within_bounds_mask = (
            (pcd_image[:, 0] >= 0) & (pcd_image[:, 0] < img_w) &
            (pcd_image[:, 1] >= 0) & (pcd_image[:, 1] < img_h)
        )
        pcd_image = pcd_image[within_bounds_mask]
        
        depth_map = np.zeros(target_size, dtype=np.float32)
        return depth_map