import urllib.request

print("Downloading target person image...")
# Target image (pose/background)
urllib.request.urlretrieve(
    "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=1024&h=1024&fit=crop",
    "target_person.jpg"
)

print("Downloading source face image...")
# Source face (identity to swap in)
urllib.request.urlretrieve(
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=1024&h=1024&fit=crop",
    "source_face.jpg"
)

print("Done! Saved 'target_person.jpg' and 'source_face.jpg' successfully.")