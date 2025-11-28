import UnityPy, os, sys

data_dir = r"C:\Program Files\10Botics\CoDrone Simulator Installer\CoDrone Simulator_Data"
# If your Data folder path differs, change data_dir above.

exts = (".assets", "resources.assets", "sharedassets0.assets", ".resource")
for root, _, files in os.walk(data_dir):
    for f in files:
        if any(f.lower().endswith(e) for e in exts):
            path = os.path.join(root, f)
            try:
                env = UnityPy.load(path)
            except Exception:
                continue
            found = []
            for obj in env.objects:
                try:
                    data = obj.read()
                    if data.type.name == "GameObject":
                        found.append(data.name)
                except Exception:
                    pass
            if found:
                print(f"--- {path} ---")
                for name in sorted(set(found)):
                    print(name)