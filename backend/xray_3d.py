"""
X-ray to 3D reconstruction module — production-grade pipeline.
- GPU-accelerated depth: DPT_Large (best quality) or MiDaS_small fallback.
- Bilateral smoothing for artifact-free depth.
- Marching cubes isosurface for a single, smooth 3D mesh (no jagged slabs).
"""
import os
import logging
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional, Dict, Any, List
from scipy.ndimage import sobel, gaussian_filter
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# GPU depth model (DPT_Large for best quality when GPU available)
_depth_model = None
_depth_transform = None
_depth_available = False
_depth_use_dpt = False  # True = DPT_Large, False = MiDaS_small


def _ensure_depth_model(use_best: bool = True):
    """Lazy-load best available depth model on GPU, fallback CPU/small model."""
    global _depth_model, _depth_transform, _depth_available, _depth_use_dpt
    if _depth_available:
        return True
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_dpt = use_best and (str(device) == "cuda")  # DPT only when GPU
        logger.info(f"Loading depth model on device: {device}, use_dpt: {use_dpt}")
        
        if use_dpt:
            try:
                logger.info("Attempting to load DPT_Large model...")
                _depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
                _depth_model.to(device).eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                _depth_transform = midas_transforms.dpt_transform
                _depth_use_dpt = True
                logger.info("DPT_Large depth model loaded on GPU for best 3D quality")
            except Exception as e:
                logger.warning(f"DPT_Large failed, falling back to MiDaS_small: {e}")
                use_dpt = False
        
        if not use_dpt:
            logger.info("Loading MiDaS_small model...")
            _depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            _depth_model.to(device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            _depth_transform = midas_transforms.small_transform
            logger.info(f"MiDaS depth model loaded on device: {device}")
        
        _depth_available = True
        return True
    except Exception as e:
        logger.error(f"Failed to load any depth model: {str(e)}")
        _depth_available = False
        return False


def _depth_heuristic(img_array: np.ndarray) -> np.ndarray:
    """
    X-ray heuristic depth: brighter (bone/dense) = closer, darker (air) = farther.
    Uses intensity + edge strength for plausible 3D relief.
    """
    # Normalize to [0, 1]
    if img_array.max() > img_array.min():
        gray = (img_array.astype(np.float32) - img_array.min()) / (img_array.max() - img_array.min())
    else:
        gray = np.ones_like(img_array, dtype=np.float32) * 0.5

    # Edges (structure) should also contribute to "closeness"
    dy = sobel(gray, axis=0)
    dx = sobel(gray, axis=1)
    edge = np.sqrt(dx**2 + dy**2)
    if edge.max() > 0:
        edge = edge / edge.max()

    # Depth = intensity + edge boost (inverted so high = front for X-ray)
    depth = 0.7 * gray + 0.3 * edge
    depth = gaussian_filter(depth, sigma=1.5)  # Smooth
    if depth.max() > depth.min():
        depth = (depth - depth.min()) / (depth.max() - depth.min())
    return depth.astype(np.float32)


def _depth_learned(img_array: np.ndarray, img_pil: Image.Image) -> Optional[np.ndarray]:
    """Run GPU/CPU depth model (DPT_Large or MiDaS); returns normalized depth map or None."""
    if not _ensure_depth_model(use_best=True):
        logger.warning("Depth model not available, using heuristic depth")
        return None
    try:
        import torch

        if len(img_array.shape) == 2:
            img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
        else:
            img_rgb = img_array
        img_pil_rgb = Image.fromarray(img_rgb.astype(np.uint8))
        batch = _depth_transform(img_pil_rgb).unsqueeze(0)
        device = next(_depth_model.parameters()).device
        batch = batch.to(device)

        logger.info(f"Running depth inference on device: {device}")
        with torch.no_grad():
            prediction = _depth_model(batch)
            if prediction.dim() == 3:
                prediction = prediction.unsqueeze(1)
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=img_array.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy().astype(np.float32)
        if depth.max() > depth.min():
            depth = 1.0 - (depth - depth.min()) / (depth.max() - depth.min())
        else:
            depth = np.ones_like(depth) * 0.5
        logger.info(f"Depth inference completed, shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
        return depth
    except Exception as e:
        logger.error(f"Depth model inference failed: {str(e)}")
        return None


def _smooth_depth_bilateral(depth: np.ndarray, d: int = 5, sigma_color: float = 0.05, sigma_space: float = 5.0) -> np.ndarray:
    """Smooth depth while preserving edges (no jagged extrusion)."""
    try:
        import cv2
        depth_u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(depth_u8, d, sigma_color * 255, sigma_space)
        return (smoothed.astype(np.float32) / 255.0)
    except Exception:
        return gaussian_filter(depth, sigma=2.0)


def compute_depth_map(contents: bytes, use_ai: bool = True, smooth: bool = True) -> Tuple[np.ndarray, bool]:
    """
    Compute a single-channel depth map from X-ray image bytes.
    Uses DPT_Large on GPU when available for best quality; optional bilateral smoothing.
    """
    img = Image.open(BytesIO(contents))
    img_gray = img.convert("L")
    img_array = np.array(img_gray)

    depth = None
    used_ai = False
    if use_ai:
        depth = _depth_learned(img_array, img)
        if depth is not None:
            used_ai = True
    if depth is None:
        depth = _depth_heuristic(img_array)

    if smooth and depth is not None:
        depth = _smooth_depth_bilateral(depth)

    return depth, used_ai


def _compute_edge_damage_map(img_array: np.ndarray) -> np.ndarray:
    """Compute normalized edge strength (high = potential fracture/boundary). Used for damage highlighting."""
    if img_array.max() > img_array.min():
        gray = (img_array.astype(np.float32) - img_array.min()) / (img_array.max() - img_array.min())
    else:
        gray = np.ones_like(img_array, dtype=np.float32) * 0.5
    dy = sobel(gray, axis=0)
    dx = sobel(gray, axis=1)
    edge = np.sqrt(dx**2 + dy**2)
    edge = gaussian_filter(edge, sigma=0.8)
    if edge.max() > 0:
        edge = edge / edge.max()
    return edge.astype(np.float32)


def depth_map_to_mesh_data(
    depth: np.ndarray,
    height: int,
    width: int,
    depth_scale: float = 80.0,
    grid_step: int = 4,
) -> Dict[str, Any]:
    """
    Convert depth map to a compact mesh representation for the frontend.
    grid_step: sample every N pixels to keep vertex count manageable.
    Returns dict with vertices (flat list), dimensions, and mesh dimensions.
    """
    h, w = depth.shape
    step = max(1, grid_step)
    rows = (h - 1) // step + 1
    cols = (w - 1) // step + 1

    vertices = []
    for i in range(rows):
        for j in range(cols):
            y = min(i * step, h - 1)
            x = min(j * step, w - 1)
            z = float(depth[y, x]) * depth_scale
            # Normalized x,y in [-1, 1] for centering in 3D view
            nx = (j / max(cols - 1, 1)) * 2 - 1
            ny = -((i / max(rows - 1, 1)) * 2 - 1)
            vertices.extend([nx, ny, z])

    # Build faces (two triangles per quad)
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a = i * cols + j
            b = a + 1
            c = a + cols
            d = c + 1
            faces.extend([a, c, b, b, c, d])

    return {
        "vertices": vertices,
        "faces": faces,
        "rows": rows,
        "cols": cols,
        "depth_scale": depth_scale,
        "image_width": w,
        "image_height": h,
    }


def depth_map_to_thick_volume_mesh(
    depth: np.ndarray,
    img_array: np.ndarray,
    depth_scale: float = 0.5,
    thickness: float = 0.25,
    grid_step: int = 3,
) -> Dict[str, Any]:
    """
    Build a proper 3D solid (thick volume) from the depth map: front face, back face, and closed sides.
    Uses normalized coordinates in [-1,1] for x,y and depth for z.
    Also computes edge-based vertex colors for damage/abnormality highlighting.
    """
    h, w = depth.shape
    step = max(1, grid_step)
    rows = (h - 1) // step + 1
    cols = (w - 1) // step + 1
    edge_map = _compute_edge_damage_map(img_array)

    # Scale depth into a reasonable range (z forward = positive)
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max <= d_min:
        d_max = d_min + 1.0
    depth_norm = (depth - d_min) / (d_max - d_min)

    vertices: List[float] = []
    uvs: List[float] = []
    vertex_colors: List[float] = []
    faces: List[int] = []

    def ix(i: int, j: int) -> int:
        return i * cols + j

    # ---- Front face (facing camera, z = depth)
    base = 0
    for i in range(rows):
        for j in range(cols):
            y = min(i * step, h - 1)
            x = min(j * step, w - 1)
            z = float(depth_norm[y, x]) * depth_scale
            nx = (j / max(cols - 1, 1)) * 2 - 1
            ny = -((i / max(rows - 1, 1)) * 2 - 1)
            vertices.extend([nx, ny, z])
            uvs.extend([j / max(cols - 1, 1), 1 - i / max(rows - 1, 1)])
            e = float(edge_map[y, x])
            # Red tint where edge is strong (potential damage/boundary)
            r = min(1.0, 0.7 + 0.3 * e)
            g = max(0.0, 1.0 - 0.8 * e)
            b = max(0.0, 1.0 - 0.8 * e)
            vertex_colors.extend([r, g, b])
    for i in range(rows - 1):
        for j in range(cols - 1):
            a = base + ix(i, j)
            b = base + ix(i, j + 1)
            c = base + ix(i + 1, j)
            d = base + ix(i + 1, j + 1)
            faces.extend([a, c, b, b, c, d])
    n_front = rows * cols
    base_back = n_front

    # ---- Back face (same x,y, z = front_z - thickness)
    for i in range(rows):
        for j in range(cols):
            y = min(i * step, h - 1)
            x = min(j * step, w - 1)
            z = float(depth_norm[y, x]) * depth_scale - thickness
            nx = (j / max(cols - 1, 1)) * 2 - 1
            ny = -((i / max(rows - 1, 1)) * 2 - 1)
            vertices.extend([nx, ny, z])
            uvs.extend([j / max(cols - 1, 1), 1 - i / max(rows - 1, 1)])
            vertex_colors.extend([1.0, 1.0, 1.0])
    for i in range(rows - 1):
        for j in range(cols - 1):
            a = base_back + ix(i, j)
            b = base_back + ix(i, j + 1)
            c = base_back + ix(i + 1, j)
            d = base_back + ix(i + 1, j + 1)
            # Back face winding so normal points backward
            faces.extend([a, b, c, c, b, d])
    n_back = rows * cols

    # ---- Side faces (connect front and back edges): each quad = 4 new vertices, 2 tris
    def add_side_quad(f0: int, f1: int, b0: int, b1: int):
        o = len(vertices) // 3
        vertices.extend([
            vertices[f0 * 3], vertices[f0 * 3 + 1], vertices[f0 * 3 + 2],
            vertices[f1 * 3], vertices[f1 * 3 + 1], vertices[f1 * 3 + 2],
            vertices[b1 * 3], vertices[b1 * 3 + 1], vertices[b1 * 3 + 2],
            vertices[b0 * 3], vertices[b0 * 3 + 1], vertices[b0 * 3 + 2],
        ])
        uvs.extend([0, 0, 1, 0, 1, 1, 0, 1])
        for _ in range(4):
            vertex_colors.extend([1.0, 1.0, 1.0])
        faces.extend([o, o + 1, o + 2, o, o + 2, o + 3])

    # Bottom edge (i = rows-1)
    for j in range(cols - 1):
        i = rows - 1
        add_side_quad(base + ix(i, j), base + ix(i, j + 1), base_back + ix(i, j), base_back + ix(i, j + 1))
    # Top edge (i = 0)
    for j in range(cols - 1):
        i = 0
        add_side_quad(base + ix(i, j), base + ix(i, j + 1), base_back + ix(i, j), base_back + ix(i, j + 1))
    # Left edge (j = 0)
    for i in range(rows - 1):
        j = 0
        add_side_quad(base + ix(i, j), base + ix(i + 1, j), base_back + ix(i, j), base_back + ix(i + 1, j))
    # Right edge (j = cols-1)
    for i in range(rows - 1):
        j = cols - 1
        add_side_quad(base + ix(i, j), base + ix(i + 1, j), base_back + ix(i, j), base_back + ix(i + 1, j))

    return {
        "vertices": vertices,
        "faces": faces,
        "uvs": uvs,
        "vertex_colors": vertex_colors,
        "rows": rows,
        "cols": cols,
        "image_width": w,
        "image_height": h,
        "thick": True,
    }


def encode_depth_as_png(depth: np.ndarray) -> str:
    """Encode depth map as 16-bit PNG base64 for optional frontend use."""
    d = (np.clip(depth, 0, 1) * 65535).astype(np.uint16)
    img = Image.fromarray(d, mode="I;16")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_volume_and_marching_cubes(
    img_array: np.ndarray,
    depth: np.ndarray,
    depth_res: int = 48,
    sigma_z: float = 0.08,
    isolevel: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Build 3D volume from X-ray + depth (Gaussian falloff along z), then extract
    a single smooth isosurface via marching cubes. Returns verts (V,3), faces (F,3), h, w.
    """
    from skimage.measure import marching_cubes

    h, w = depth.shape
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max <= d_min:
        d_max = d_min + 1.0
    depth_norm = (depth - d_min) / (d_max - d_min)

    if img_array.max() > img_array.min():
        intensity = (img_array.astype(np.float32) - img_array.min()) / (img_array.max() - img_array.min())
    else:
        intensity = np.ones((h, w), dtype=np.float32) * 0.5

    # Volume: shape (H, W, D). value = intensity(i,j) * gaussian in z centered at depth(i,j)
    volume = np.zeros((h, w, depth_res), dtype=np.float32)
    z_axis = np.linspace(0, 1, depth_res, dtype=np.float32)
    for k in range(depth_res):
        z = z_axis[k]
        # Gaussian weight centered at depth_norm(i,j)
        w_z = np.exp(-0.5 * ((z - depth_norm) ** 2) / (sigma_z ** 2))
        volume[:, :, k] = intensity * w_z

    try:
        verts, faces, normals, _ = marching_cubes(volume, level=isolevel, method="lewiner", gradient_direction="descent")
    except Exception as e:
        logger.warning("Marching cubes failed (%s), using fallback slab mesh", e)
        return None, None, h, w

    # Normalize verts to [-1, 1] for x,y and similar scale for z; flip so front faces camera
    # verts are in (row, col, depth_slice) = (y, x, z)
    vy, vx, vz = verts[:, 0], verts[:, 1], verts[:, 2]
    nx = (vx / max(w - 1, 1)) * 2 - 1
    ny = -((vy / max(h - 1, 1)) * 2 - 1)
    nz = (vz / max(depth_res - 1, 1)) * 0.6 - 0.1  # z in ~[-0.1, 0.5]
    verts_norm = np.stack([nx, ny, nz], axis=1).astype(np.float32)

    return verts_norm, faces, h, w


def _marching_cubes_mesh_to_output(
    verts: np.ndarray,
    faces: np.ndarray,
    h: int,
    w: int,
    img_array: np.ndarray,
) -> Dict[str, Any]:
    """Convert marching cubes verts/faces to frontend mesh format with UVs and vertex colors."""
    edge_map = _compute_edge_damage_map(img_array)
    vertices: List[float] = verts.flatten().tolist()
    uvs: List[float] = []
    vertex_colors: List[float] = []
    for i in range(verts.shape[0]):
        x, y, z = verts[i, 0], verts[i, 1], verts[i, 2]
        # UV: map from [-1,1] back to image coords
        u = (x + 1) * 0.5
        v = 1 - (y + 1) * 0.5  # flip v
        px = int(np.clip(u * (w - 1), 0, w - 1))
        py = int(np.clip(v * (h - 1), 0, h - 1))
        uvs.extend([u, v])
        e = float(edge_map[py, px])
        r = min(1.0, 0.75 + 0.25 * e)
        g = max(0.0, 1.0 - 0.7 * e)
        b = max(0.0, 1.0 - 0.7 * e)
        vertex_colors.extend([r, g, b])
    faces_flat: List[int] = faces.flatten().tolist()
    return {
        "vertices": vertices,
        "faces": faces_flat,
        "uvs": uvs,
        "vertex_colors": vertex_colors,
        "image_width": w,
        "image_height": h,
        "thick": True,
        "rows": 0,
        "cols": 0,
    }


def reconstruct_3d(contents: bytes, use_ai: bool = True, grid_step: int = 4) -> Dict[str, Any]:
    """
    Production 3D reconstruction: GPU depth (DPT_Large when available), bilateral smoothing,
    and a single smooth isosurface via marching cubes for an ultra-realistic 3D model.
    Falls back to thick slab mesh if marching cubes fails.
    """
    logger.info(f"Starting 3D reconstruction with AI={use_ai}, grid_step={grid_step}")
    
    try:
        img = Image.open(BytesIO(contents))
        img_gray = img.convert("L")
        img_array = np.array(img_gray)
        logger.info(f"Image loaded: shape={img_array.shape}, mode={img.mode}")

        depth, used_ai = compute_depth_map(contents, use_ai=use_ai, smooth=True)
        h, w = depth.shape
        logger.info(f"Depth map computed: shape={depth.shape}, used_ai={used_ai}, range=[{depth.min():.3f}, {depth.max():.3f}]")

        # Try smooth isosurface first (marching cubes)
        logger.info("Attempting marching cubes reconstruction...")
        verts, faces, _, _ = _build_volume_and_marching_cubes(
            img_array,
            depth,
            depth_res=48,
            sigma_z=0.08,
            isolevel=0.35,
        )
        
        if verts is not None and faces is not None and len(verts) > 0 and len(faces) > 0:
            logger.info(f"Marching cubes succeeded: verts={len(verts)}, faces={len(faces)}")
            mesh_data = _marching_cubes_mesh_to_output(verts, faces, h, w, img_array)
        else:
            logger.warning("Marching cubes failed, using fallback thick volume mesh")
            # Fallback: smooth thick volume (no jagged sides)
            mesh_data = depth_map_to_thick_volume_mesh(
                depth,
                img_array,
                depth_scale=0.5,
                thickness=0.2,
                grid_step=max(2, grid_step),
            )
            logger.info(f"Thick volume mesh created: vertices={len(mesh_data['vertices'])//3}, faces={len(mesh_data['faces'])}")

        depth_b64 = encode_depth_as_png(depth)
        
        result = {
            "depth_map_b64": depth_b64,
            "mesh": mesh_data,
            "image_width": w,
            "image_height": h,
            "used_ai": used_ai,
            "depth_min": float(np.min(depth)),
            "depth_max": float(np.max(depth)),
        }
        
        logger.info(f"3D reconstruction completed successfully: used_ai={used_ai}, vertices={len(mesh_data['vertices'])//3}")
        return result
        
    except Exception as e:
        logger.error(f"3D reconstruction failed: {str(e)}")
        # Create a minimal fallback mesh
        try:
            img = Image.open(BytesIO(contents))
            img_gray = img.convert("L")
            img_array = np.array(img_gray)
            h, w = img_array.shape
            
            # Create a simple flat mesh as ultimate fallback
            depth = np.ones((h, w), dtype=np.float32) * 0.5
            mesh_data = depth_map_to_thick_volume_mesh(
                depth,
                img_array,
                depth_scale=0.1,
                thickness=0.05,
                grid_step=8,  # Very coarse for performance
            )
            
            depth_b64 = encode_depth_as_png(depth)
            
            result = {
                "depth_map_b64": depth_b64,
                "mesh": mesh_data,
                "image_width": w,
                "image_height": h,
                "used_ai": False,
                "depth_min": 0.0,
                "depth_max": 1.0,
                "error": str(e)
            }
            
            logger.warning(f"Used fallback mesh due to error: {str(e)}")
            return result
            
        except Exception as fallback_error:
            logger.error(f"Even fallback mesh failed: {str(fallback_error)}")
            raise HTTPException(status_code=500, detail=f"3D reconstruction completely failed: {str(fallback_error)}")
