import requests
import base64
import json
import numpy as np
from PIL import Image
import io

# Create a more realistic test X-ray image
def create_test_xray():
    # Create a 256x256 grayscale image that looks like an X-ray
    img_array = np.zeros((256, 256), dtype=np.uint8)
    
    # Add some bone-like structures (brighter areas)
    # Vertical bone
    img_array[50:200, 100:156] = 180
    
    # Add some texture/noise to make it more realistic
    noise = np.random.normal(0, 20, (256, 256))
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Add some darker areas (soft tissue)
    img_array[80:120, 110:146] = 120
    img_array[140:180, 110:146] = 120
    
    # Convert to PIL Image and then to base64
    img = Image.fromarray(img_array, mode='L')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return base64.b64decode(img_str)

def test_3d_reconstruction():
    print("Creating realistic test X-ray image...")
    test_image_data = create_test_xray()
    
    # Send to the 3D reconstruction endpoint
    files = {'image': ('test_xray.png', test_image_data, 'image/png')}
    data = {
        'use_ai': 'true',
        'grid_step': '4'
    }
    
    try:
        print("Sending request to 3D reconstruction API...")
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
            
            # Check mesh properties
            mesh = result.get('mesh', {})
            if 'vertex_colors' in mesh:
                print(f"   - Has vertex colors: {len(mesh['vertex_colors']) // 3}")
            if 'uvs' in mesh:
                print(f"   - Has UV coordinates: {len(mesh['uvs']) // 2}")
            
            return True
        else:
            print(f"❌ 3D Reconstruction failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing 3D Reconstruction API with realistic X-ray...")
    success = test_3d_reconstruction()
    if success:
        print("\n🎉 3D Reconstruction is working correctly!")
        print("You can now test it in the browser at http://localhost:4200")
        print("1. Upload an X-ray image")
        print("2. Click on 'True 3D' mode")
        print("3. The 3D model should appear in the viewer")
    else:
        print("\n❌ 3D Reconstruction test failed")
