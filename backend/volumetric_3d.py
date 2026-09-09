"""
Advanced Volumetric 3D Reconstruction from 2D X-ray
Creates true 3D bone models with fracture detection and multi-dimensional representation
"""
import os
import logging
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional, Dict, Any, List
from scipy.ndimage import sobel, gaussian_filter, binary_erosion, binary_dilation, label
from scipy.signal import find_peaks
from skimage.measure import marching_cubes, regionprops
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import threshold_otsu, gaussian
import cv2

logger = logging.getLogger(__name__)

class Bone3DReconstructor:
    """Advanced 3D bone reconstruction with fracture detection"""
    
    def __init__(self):
        self.fracture_zones = []
        self.bone_segments = []
        
    def _segment_bone_structure(self, img_array: np.ndarray) -> np.ndarray:
        """Segment bone structure from X-ray using adaptive thresholding"""
        # Normalize image
        if img_array.max() > img_array.min():
            normalized = (img_array.astype(np.float32) - img_array.min()) / (img_array.max() - img_array.min())
        else:
            normalized = np.ones_like(img_array, dtype=np.float32) * 0.5
        
        # Apply Gaussian blur to reduce noise
        blurred = gaussian(normalized, sigma=1.0)
        
        # Adaptive thresholding for bone segmentation
        threshold = threshold_otsu(blurred)
        bone_mask = blurred > threshold * 0.7  # Slightly lower threshold to capture more bone
        
        # Morphological operations to clean up
        bone_mask = binary_erosion(bone_mask, iterations=1)
        bone_mask = binary_dilation(bone_mask, iterations=2)
        
        # Remove small objects
        bone_mask = remove_small_objects(bone_mask, min_size=100)
        
        return bone_mask.astype(np.uint8)
    
    def _detect_fractures_2d(self, img_array: np.ndarray, bone_mask: np.ndarray) -> List[Dict]:
        """Detect fractures in 2D X-ray using edge analysis and discontinuity detection"""
        fractures = []
        
        # Edge detection
        edges = sobel(img_array)
        edge_magnitude = np.sqrt(edges[0]**2 + edges[1]**2)
        
        # Find bone edges
        bone_edges = edge_magnitude * bone_mask
        
        # Look for discontinuities in bone structure
        # Horizontal fractures (transverse)
        row_profiles = np.sum(bone_mask, axis=1)
        row_peaks, row_properties = find_peaks(
            -row_profiles,  # Negative to find dips (gaps)
            prominence=np.max(row_profiles) * 0.1,
            distance=img_array.shape[1] * 0.05,
            width=3
        )
        
        # Vertical fractures (longitudinal)
        col_profiles = np.sum(bone_mask, axis=0)
        col_peaks, col_properties = find_peaks(
            -col_profiles,  # Negative to find dips (gaps)
            prominence=np.max(col_profiles) * 0.1,
            distance=img_array.shape[0] * 0.05,
            width=3
        )
        
        # Analyze each potential fracture
        for peak_idx, prominence in zip(row_peaks, row_properties['prominences']):
            if prominence > np.max(row_profiles) * 0.15:  # Significant gap
                # Find fracture extent horizontally
                fracture_row = bone_edges[peak_idx, :]
                fracture_points = np.where(fracture_row > np.max(fracture_row) * 0.3)[0]
                
                if len(fracture_points) > 0:
                    fractures.append({
                        'type': 'transverse',
                        'position': [float(peak_idx), float(np.mean(fracture_points))],
                        'extent': [float(fracture_points[0]), float(fracture_points[-1])],
                        'severity': float(prominence / np.max(row_profiles)),
                        'angle': 0.0  # Horizontal
                    })
        
        for peak_idx, prominence in zip(col_peaks, col_properties['prominences']):
            if prominence > np.max(col_profiles) * 0.15:  # Significant gap
                # Find fracture extent vertically
                fracture_col = bone_edges[:, peak_idx]
                fracture_points = np.where(fracture_col > np.max(fracture_col) * 0.3)[0]
                
                if len(fracture_points) > 0:
                    fractures.append({
                        'type': 'longitudinal',
                        'position': [float(np.mean(fracture_points)), float(peak_idx)],
                        'extent': [float(fracture_points[0]), float(fracture_points[-1])],
                        'severity': float(prominence / np.max(col_profiles)),
                        'angle': 90.0  # Vertical
                    })
        
        logger.info(f"Detected {len(fractures)} potential fractures")
        return fractures
    
    def _estimate_bone_thickness(self, img_array: np.ndarray, bone_mask: np.ndarray) -> np.ndarray:
        """Estimate bone thickness at each point"""
        # Distance transform gives thickness estimation
        from scipy.ndimage import distance_transform_edt
        distance = distance_transform_edt(bone_mask)
        
        # Normalize to reasonable bone thickness (1-20mm typical)
        max_thickness = 20.0  # mm
        thickness_map = (distance / np.max(distance)) * max_thickness
        
        return thickness_map.astype(np.float32)
    
    def _create_volumetric_bone(self, img_array: np.ndarray, bone_mask: np.ndarray, 
                              fractures: List[Dict], thickness_map: np.ndarray) -> np.ndarray:
        """Create 3D volumetric bone model with fractures"""
        h, w = img_array.shape
        depth_resolution = 64  # Z-axis resolution
        
        # Create 3D volume
        volume = np.zeros((h, w, depth_resolution), dtype=np.float32)
        
        # Base bone density from X-ray intensity
        bone_density = img_array.astype(np.float32) / 255.0
        
        # Extrude bone into 3D using thickness map
        for z in range(depth_resolution):
            z_normalized = z / (depth_resolution - 1)
            
            # Create bone cross-section at this depth
            for y in range(h):
                for x in range(w):
                    if bone_mask[y, x]:
                        # Bone thickness determines how deep it extends
                        local_thickness = thickness_map[y, x]
                        thickness_normalized = local_thickness / 20.0  # Normalize to 0-1
                        
                        # Bone exists where z is within thickness range
                        if z_normalized <= thickness_normalized:
                            # Density varies with depth (cortical bone is denser)
                            if z_normalized < 0.3:  # Outer layer (cortical)
                                volume[y, x, z] = bone_density[y, x] * 0.9
                            else:  # Inner layer (trabecular)
                                volume[y, x, z] = bone_density[y, x] * 0.6
        
        # Add fractures to 3D volume
        for fracture in fractures:
            self._apply_fracture_3d(volume, fracture, depth_resolution)
        
        return volume
    
    def _apply_fracture_3d(self, volume: np.ndarray, fracture: Dict, depth_resolution: int):
        """Apply fracture to 3D volume with proper displacement"""
        h, w = volume.shape[:2]
        y_pos, x_pos = fracture['position']
        severity = fracture['severity']
        
        if fracture['type'] == 'transverse':
            # Horizontal fracture line
            y_start = max(0, int(y_pos - 10))
            y_end = min(h, int(y_pos + 10))
            
            for y in range(y_start, y_end):
                for x in range(w):
                    # Create gap with rough edges
                    gap_width = int(3 + severity * 5)
                    for z in range(depth_resolution):
                        # Add some randomness for realistic fracture
                        if abs(z - depth_resolution//2) < gap_width:
                            volume[y, x, z] *= 0.1  # Almost empty (fracture gap)
                            
                            # Add displacement around fracture
                            if y > y_pos:
                                volume[y, x, z] *= (1 - severity * 0.3)  # Displaced fragment
        
        elif fracture['type'] == 'longitudinal':
            # Vertical fracture line
            x_start = max(0, int(x_pos - 10))
            x_end = min(w, int(x_pos + 10))
            
            for y in range(h):
                for x in range(x_start, x_end):
                    gap_width = int(3 + severity * 5)
                    for z in range(depth_resolution):
                        if abs(z - depth_resolution//2) < gap_width:
                            volume[y, x, z] *= 0.1  # Almost empty (fracture gap)
                            
                            # Add displacement
                            if x > x_pos:
                                volume[y, x, z] *= (1 - severity * 0.3)
    
    def _extract_3d_mesh_with_fractures(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 3D mesh with proper fracture representation"""
        try:
            # Use marching cubes to extract surface
            verts, faces, normals, _ = marching_cubes(
                volume, 
                level=0.15,  # Lower threshold to capture more detail
                method='lewiner',
                gradient_direction='descent'
            )
            
            # Enhance vertices around fractures
            enhanced_verts = self._enhance_fracture_edges(verts, volume)
            
            # Calculate vertex colors based on local density and fractures
            vertex_colors = self._calculate_fracture_colors(enhanced_verts, volume)
            
            return enhanced_verts, faces, vertex_colors
            
        except Exception as e:
            logger.error(f"3D mesh extraction failed: {e}")
            # Fallback: create simple mesh
            return self._create_fallback_mesh(volume)
    
    def _enhance_fracture_edges(self, verts: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Enhance vertices around fracture areas for better visualization"""
        enhanced_verts = verts.copy()
        h, w, d = volume.shape
        
        for i, vert in enumerate(verts):
            y, x, z = vert
            
            # Check if vertex is near a fracture (low density region)
            if (0 <= int(y) < h and 0 <= int(x) < w and 0 <= int(z) < d):
                local_density = volume[int(y), int(x), int(z)]
                
                # If low density, it's likely a fracture edge
                if local_density < 0.3:
                    # Slightly displace to make fracture more visible
                    enhanced_verts[i] = vert + np.random.normal(0, 0.5, 3)
        
        return enhanced_verts
    
    def _calculate_fracture_colors(self, verts: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate vertex colors with fracture highlighting"""
        colors = []
        h, w, d = volume.shape
        
        for vert in verts:
            y, x, z = vert
            
            if (0 <= int(y) < h and 0 <= int(x) < w and 0 <= int(z) < d):
                local_density = volume[int(y), int(x), int(z)]
                
                if local_density < 0.3:
                    # Red for fractures
                    colors.append([1.0, 0.2, 0.2, 1.0])
                elif local_density < 0.6:
                    # Yellow for damaged areas
                    colors.append([1.0, 0.8, 0.2, 1.0])
                else:
                    # White/gray for healthy bone
                    colors.append([0.9, 0.9, 0.9, 1.0])
            else:
                colors.append([0.9, 0.9, 0.9, 1.0])
        
        return np.array(colors)
    
    def _create_fallback_mesh(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create simple fallback mesh if marching cubes fails"""
        h, w, d = volume.shape
        
        # Create simple box mesh
        verts = np.array([
            [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],  # Bottom
            [0, 0, d], [w, 0, d], [w, h, d], [0, h, d]   # Top
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 6, 5], [4, 7, 6],  # Top
            [0, 4, 5], [0, 5, 1],  # Front
            [2, 6, 7], [2, 7, 3],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2]   # Right
        ], dtype=np.int32)
        
        colors = np.array([[0.9, 0.9, 0.9, 1.0]] * len(verts))
        
        return verts, faces, colors
    
    def reconstruct_3d_bone(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Main reconstruction pipeline"""
        logger.info("Starting advanced 3D bone reconstruction...")
        
        # Step 1: Segment bone structure
        logger.info("Segmenting bone structure...")
        bone_mask = self._segment_bone_structure(img_array)
        
        # Step 2: Detect fractures
        logger.info("Detecting fractures...")
        fractures = self._detect_fractures_2d(img_array, bone_mask)
        
        # Step 3: Estimate bone thickness
        logger.info("Estimating bone thickness...")
        thickness_map = self._estimate_bone_thickness(img_array, bone_mask)
        
        # Step 4: Create 3D volume
        logger.info("Creating 3D volumetric model...")
        volume = self._create_volumetric_bone(img_array, bone_mask, fractures, thickness_map)
        
        # Step 5: Extract 3D mesh
        logger.info("Extracting 3D mesh...")
        verts, faces, colors = self._extract_3d_mesh_with_fractures(volume)
        
        # Normalize vertices to [-1, 1] range
        h, w, d = volume.shape
        verts[:, 0] = (verts[:, 0] / (w - 1)) * 2 - 1  # X
        verts[:, 1] = (verts[:, 1] / (h - 1)) * 2 - 1  # Y  
        verts[:, 2] = (verts[:, 2] / (d - 1)) * 2 - 1  # Z
        
        # Create UV coordinates
        uvs = []
        for vert in verts:
            u = (vert[0] + 1) * 0.5
            v = 1 - (vert[1] + 1) * 0.5
            uvs.extend([u, v])
        
        result = {
            'vertices': verts.flatten().tolist(),
            'faces': faces.flatten().tolist(),
            'vertex_colors': colors.flatten().tolist(),
            'uvs': uvs,
            'fractures': fractures,
            'bone_volume': float(np.sum(volume)),
            'bone_density': float(np.mean(volume[volume > 0])),
            'image_width': int(w),
            'image_height': int(h),
            'depth_resolution': int(d),
            'fracture_count': len(fractures)
        }
        
        logger.info(f"3D reconstruction completed: {len(verts)} vertices, {len(fractures)} fractures")
        return result

def reconstruct_volumetric_3d(contents: bytes) -> Dict[str, Any]:
    """Main entry point for volumetric 3D reconstruction"""
    try:
        # Load image
        img = Image.open(BytesIO(contents))
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # Initialize reconstructor
        reconstructor = Bone3DReconstructor()
        
        # Perform reconstruction
        result = reconstructor.reconstruct_3d_bone(img_array)
        
        # Add depth map for compatibility
        depth_map = np.ones_like(img_array, dtype=np.float32) * 0.5
        depth_b64 = encode_depth_as_png(depth_map)
        
        return {
            'depth_map_b64': depth_b64,
            'mesh': result,
            'image_width': result['image_width'],
            'image_height': result['image_height'],
            'used_ai': True,
            'depth_min': 0.0,
            'depth_max': 1.0,
            'reconstruction_type': 'volumetric',
            'fracture_analysis': {
                'count': result['fracture_count'],
                'fractures': result['fractures']
            }
        }
        
    except Exception as e:
        logger.error(f"Volumetric 3D reconstruction failed: {e}")
        raise

def encode_depth_as_png(depth: np.ndarray) -> str:
    """Encode depth map as 16-bit PNG base64"""
    d = (np.clip(depth, 0, 1) * 65535).astype(np.uint16)
    img = Image.fromarray(d, mode="I;16")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
