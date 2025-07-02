from textureGenPipeline import MaterialMVPPipeline, MaterialMVPConfig

try:
    from utils.torchvision_fix import apply_fix

    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


if __name__ == "__main__":

    max_num_view = 6
    resolution = 512

    conf = MaterialMVPConfig(max_num_view, resolution)
    pipe = MaterialMVPPipeline(conf)
    output_mesh_path = pipe(mesh_path="test_examples/mesh.glb", image_path="test_examples/image.png")
    print(f"Output mesh path: {output_mesh_path}")
