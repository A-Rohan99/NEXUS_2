import requests
import base64
import json
import numpy as np
from PIL import Image
import io

# Create a test X-ray with a fracture
def create_fracture_xray():
    # Create a 256x256 grayscale image with a bone and fracture
    img_array = np.zeros((256, 256), dtype=np.uint8)
    
    # Add bone structure (vertical femur-like bone)
    img_array[50:200, 100:156] = 180  # Main bone
    
    # Add a fracture (horizontal gap in the bone)
    fracture_y = 125
    fracture_width = 3
    img_array[fracture_y-fracture_width:fracture_y+fracture_width, 100:156] = 50  # Dark gap
    
    # Add some displacement around fracture
    img_array[fracture_y+fracture_width:fracture_y+10, 102:154] = 160  # Slightly displaced fragment
    
    # Add texture/noise
    noise = np.random.normal(0, 15, (256, 256))
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image and then to base64
    img = Image.fromarray(img_array, mode='L')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return base64.b64decode(img_str)

def test_volumetric_3d():
    print("Creating test X-ray with fracture...")
    test_image_data = create_fracture_xray()
    
    # Send to the volumetric 3D reconstruction endpoint
    files = {'image': ('fracture_xray.png', test_image_data, 'image/png')}
    
    try:
        print("Sending request to volumetric 3D reconstruction API...")
        response = requests.post('http://localhost:8000/v1/nexus/xray/reconstruct-volumetric-3d', 
                             files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Volumetric 3D Reconstruction successful!")
            print(f"   - Reconstruction type: {result.get('reconstruction_type', 'unknown')}")
            print(f"   - Image size: {result.get('image_width', 0)}x{result.get('image_height', 0)}")
            print(f"   - Mesh vertices: {len(result.get('mesh', {}).get('vertices', [])) // 3}")
            print(f"   - Mesh faces: {len(result.get('mesh', {}).get('faces', [])) // 3}")
            print(f"   - Fracture count: {result.get('fracture_analysis', {}).get('count', 0)}")
            
            fractures = result.get('fracture_analysis', {}).get('fractures', [])
            for i, fracture in enumerate(fractures):
                print(f"   - Fracture {i+1}: {fracture.get('type', 'unknown')} at position {fracture.get('position', 'unknown')}, severity: {fracture.get('severity', 0):.2f}")
            
            return True
        else:
            print(f"❌ Volumetric 3D Reconstruction failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Volumetric 3D Reconstruction API...")
    success = test_volumetric_3d()
    if success:
        print("\n🎉 Volumetric 3D Reconstruction is working correctly!")
        print("You can now test it in the browser at http://localhost:4200")
        print("1. Upload an X-ray image")
        print("2. Click on '🦴 Volumetric 3D' mode")
        print("3. The 3D bone model with fractures should appear")
    else:
        print("\n❌ Volumetric 3D Reconstruction test failed")
