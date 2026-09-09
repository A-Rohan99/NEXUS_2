import requests
import base64
import json

# Test the 3D reconstruction endpoint
def test_3d_reconstruction():
    # Create a simple test image (1x1 black pixel)
    test_image_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    
    # Send to the 3D reconstruction endpoint
    files = {'image': ('test.png', test_image_data, 'image/png')}
    data = {
        'use_ai': 'true',
        'grid_step': '4'
    }
    
    try:
        response = requests.post('http://localhost:8000/v1/nexus/xray/reconstruct-3d', 
                             files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 3D Reconstruction successful!")
            print(f"   - Used AI: {result.get('used_ai', False)}")
            print(f"   - Image size: {result.get('image_width', 0)}x{result.get('image_height', 0)}")
            print(f"   - Mesh vertices: {len(result.get('mesh', {}).get('vertices', [])) // 3}")
            print(f"   - Mesh faces: {len(result.get('mesh', {}).get('faces', [])) // 3}")
            print(f"   - Depth range: {result.get('depth_min', 0):.3f} - {result.get('depth_max', 0):.3f}")
            return True
        else:
            print(f"❌ 3D Reconstruction failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing 3D Reconstruction API...")
    test_3d_reconstruction()
