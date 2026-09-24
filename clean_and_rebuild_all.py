import json
import struct

def clean_miku_glb():
    path = 'z:/home/edwintai/github/3dmodel/miku_modular.glb'
    print(f"Cleaning {path}...")

    with open(path, 'rb') as f:
        magic, version, length = struct.unpack('<4sII', f.read(12))
        c_len, c_type = struct.unpack('<I4s', f.read(8))
        g = json.loads(f.read(c_len).decode('utf-8'))
        b_len, b_type = struct.unpack('<I4s', f.read(8))
        b = bytearray(f.read(b_len))

    bad_names = {'Hair_Bob', 'Hair_Hime', 'Hair_Parted', 'Hair_Spiky', 'Hair_Twintails', 'Hair_Miku_Short'}
    bad_indices = set()
    for i, n in enumerate(g['nodes']):
        if n.get('name') in bad_names:
            bad_indices.add(i)
            n['name'] = f'_unused_{i}'
            if 'mesh' in n:
                del n['mesh']

    for n in g['nodes']:
        if 'children' in n:
            n['children'] = [c for c in n['children'] if c not in bad_indices]

    # Re-pack GLB
    new_json_bytes = json.dumps(g, separators=(',', ':')).encode('utf-8')
    pad_json = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b' ' * pad_json

    total_len = 12 + 8 + len(new_json_bytes) + 8 + len(b)

    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', magic, version, total_len))
        f.write(struct.pack('<I4s', len(new_json_bytes), c_type))
        f.write(new_json_bytes)
        f.write(struct.pack('<I4s', len(b), b_type))
        f.write(b)

    print(f"Successfully cleaned {path} (removed {len(bad_indices)} bad nodes)!")

if __name__ == '__main__':
    clean_miku_glb()
